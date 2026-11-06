"""Frozen-ROI experiments that isolate tiled detection and longer DETR training.

The completed detection_roi_detr_v1 directory is read-only. All caches, new
training state and reports belong to the separate bbox improvement directory.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import gzip
import json
import os
from pathlib import Path
import zipfile

os.environ.setdefault('TORCH_CUDNN_V8_API_DISABLED', '1')
import cv2
import numpy as np
from pycocotools import mask as mu
import torch

import filament_detr as detr
import roi_filament_pipeline as roi
from roi_filament_core import RoiFilamentNet, prepare_image
from native_instance_data import load_cohort
from native_instance_training import atomic_json, digest, identity, device_for, configure_cuda

BASE_MANIFEST = '45cae00116599984add596946a47d8ef1b1ebe979ec57baf8ad7b7c831573715'
BASE_ROI_SHA = '4d1a97c0baf3b7d820754de141d68f1f56cffb0f5fb1b06be6f1b93a9a90ad7f'
BASE_DETR_SHA = '4ea4a09244b24b00ac61a9618a11327a31f7156bcf279e7da87a1ea81b6f6565'
FROZEN = ('roi_filament_pipeline.py', 'roi_filament_core.py', 'filament_detr.py',
          'native_instance_data.py', 'native_instance_training.py', 'native_instance_model.py')
DEFAULTS = dict(tile_size=1024, tile_stride=768, tile_nms=[.5,.7], tile_max_det=200,
    device='cuda', cache_root='/content/solar_bbox_improvement_cache',
    yolo_thresholds=[.05,.1,.2,.3], detr_thresholds=[.05,.3,.6,.8,.9],
    minimum_areas=[150,300], minimum_quality=.6, added_epochs=30,
    continuation_lr=1e-5, continuation_backbone_lr=1e-6,
    continuation_image_size=1280,continuation_batch=2,continuation_accumulate=2,
    continuation_precision='bf16')


def normalize_config(config):
    cfg=dict(DEFAULTS,**config)
    for key in ('base_root','output_root'):
        if not cfg.get(key): raise ValueError(key+' is required')
    base,out,cache=(Path(cfg[k]).resolve() for k in ('base_root','output_root','cache_root'))
    if base==out or base in out.parents or out in base.parents:
        raise ValueError('The completed base experiment and new output must be disjoint')
    if any(x==cache or x in cache.parents or cache in x.parents for x in (base,out)):
        raise ValueError('Disposable cache must be separate from both persistent experiments')
    for key in ('tile_size','tile_stride','tile_max_det','added_epochs','continuation_image_size','continuation_batch','continuation_accumulate'):
        if not isinstance(cfg[key],int) or cfg[key]<1:raise ValueError('Invalid '+key)
    if cfg['tile_size']>2048 or cfg['tile_stride']>cfg['tile_size']:
        raise ValueError('Tiles must cover the original image without gaps')
    if not cfg['tile_nms'] or any(not 0<t<1 for t in cfg['tile_nms']):raise ValueError('Invalid tile NMS')
    for key in ('yolo_thresholds','detr_thresholds'):
        if not cfg[key] or any(not .05<=t<=1 for t in cfg[key]):raise ValueError('Invalid '+key)
    if not cfg['minimum_areas'] or any(not isinstance(a,int) or a<1 for a in cfg['minimum_areas']):raise ValueError('Invalid minimum areas')
    if not 0<=cfg['minimum_quality']<=1:raise ValueError('Invalid ROI quality')
    if cfg['continuation_image_size']%32 or cfg['continuation_precision'] not in ('bf16','fp16','fp32'):
        raise ValueError('Invalid continuation resolution/precision')
    cfg.update(base_root=str(base),output_root=str(out),cache_root=str(cache))
    return cfg


def source_hashes():
    names=FROZEN+('bbox_probe_pipeline.py','bbox_detr_continuation.py')
    return {name:digest(Path(__file__).with_name(name)) for name in names}


def tile_windows(size=2048,tile=1024,stride=768):
    if not 0<stride<=tile<=size:raise ValueError('Invalid tile geometry')
    positions=list(range(0,size-tile+1,stride))
    if positions[-1]!=size-tile:positions.append(size-tile)
    return [(x,y,x+tile,y+tile) for y in positions for x in positions]


def native_boxes(rows,window):
    """Map local pixel xyxy to native coordinates, retaining seam-edge boxes."""
    x,y,x2,y2=window;w,h=x2-x,y2-y
    output=[]
    for row in rows:
        box=np.asarray(row['box'],float);score=float(row['score'])
        if box.shape!=(4,) or not np.isfinite(box).all() or not np.isfinite(score) or not 0<=score<=1:
            raise ValueError('Invalid tile detector output')
        box=np.clip(box,[0,0,0,0],[w,h,w,h])+[x,y,x,y]
        if box[2]<=box[0] or box[3]<=box[1]:continue
        output.append(dict(box=box.tolist(),score=score))
    return roi.checked_boxes(output)


def suppress_boxes(rows,threshold=.5,max_det=200):
    """Stable class-agnostic IoU NMS; overlapping distinct boxes are not fused."""
    rows=roi.checked_boxes(rows);order=sorted(range(len(rows)),key=lambda i:(-rows[i]['score'],i))
    boxes=np.asarray([r['box'] for r in rows],float).reshape(-1,4)
    areas=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]);chosen=[]
    for index in order:
        if chosen:
            lo=np.maximum(boxes[index,:2],boxes[chosen,:2]);hi=np.minimum(boxes[index,2:],boxes[chosen,2:])
            inter=np.maximum(hi-lo,0).prod(1);overlap=inter/(areas[index]+areas[chosen]-inter)
            if (overlap>threshold).any():continue
        chosen.append(index)
        if len(chosen)>=max_det:break
    return [rows[i] for i in chosen]


def _encode_indices(indices):
    """Encode sorted native2048 column-major foreground without dense allocation."""
    values=np.asarray(indices,dtype=np.int64)
    if not len(values):raise ValueError('Cannot encode empty instance')
    starts=np.r_[True,np.diff(values)!=1];first=values[starts];last=values[np.r_[starts[1:],True]]
    lengths=last-first+1;gaps=np.r_[first[0],first[1:]-last[:-1]-1]
    runs=np.stack([gaps,lengths],axis=1).ravel().tolist();runs.append(int(2048**2-last[-1]-1))
    return mu.frPyObjects(dict(size=[2048,2048],counts=runs),2048,2048)


class MaskSelector:
    """Exact frozen NMS/ownership semantics, reusing decoded sparse pixels."""
    def __init__(self,candidates,nms=.5):
        self.rows=candidates;self.nms=nms
        self.order=sorted(range(len(candidates)),key=lambda i:-candidates[i]['selection_score'])
        masks=[x['rle'] for x in candidates]
        self.areas=np.asarray(mu.area(masks)) if masks else np.zeros(0)
        self.overlap=np.asarray(mu.iou(masks,masks,[0]*len(masks))) if masks else np.zeros((0,0))
        self.indices=[np.flatnonzero(mu.decode(mask).ravel(order='F')) for mask in masks];self.cache={}
    def __call__(self,confidence,area,quality):
        eligible=tuple(i for i in self.order if self.rows[i]['box_score']>=confidence and self.rows[i].get('roi_quality',1.)>=quality and self.areas[i]>=area)
        key=(eligible,area)
        if key in self.cache:return self.cache[key]
        chosen=[]
        for i in eligible:
            if chosen and self.overlap[i,chosen].max()>self.nms:continue
            chosen.append(i)
        occupied=np.zeros(2048**2,bool);masks=[]
        for i in chosen:
            old=self.indices[i];new=old[~occupied[old]]
            if len(new)<area:continue
            occupied[new]=True
            masks.append(self.rows[i]['rle'] if len(new)==len(old) else _encode_indices(new))
        self.cache[key]=masks;return masks


def infer_yolo_preprocessed(model,image,cfg,imgsz):
    prediction=model.predict(source=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR),imgsz=imgsz,
        conf=cfg['proposal_confidence'],iou=cfg['yolo_nms_iou'],max_det=cfg['yolo_max_det'],
        device=0 if str(cfg['device']) in ('cuda','cuda:0','0') else cfg['device'],verbose=False)[0]
    boxes=prediction.boxes
    return [] if boxes is None else [dict(box=b,score=s) for b,s in zip(boxes.xyxy.cpu().tolist(),boxes.conf.cpu().tolist())]


def infer_detr_preprocessed(model,image,cfg):
    """Input is already corrected native/tile uint8. Never run radial correction here."""
    if image.dtype!=np.uint8 or image.ndim!=2:raise ValueError('Corrected grayscale uint8 crop required')
    device=next(model.parameters()).device;h,w=image.shape
    pixels=detr._resized_normalized(image.astype(np.float32)/255.,cfg['detr_image_size'])[None].to(device)
    precision=cfg.get('detector_precision','fp16' if cfg.get('detr_amp',True) else 'fp32')
    if precision not in ('fp16','bf16','fp32'):raise ValueError('Invalid detector inference precision')
    if precision!='fp32' and device.type!='cuda':raise RuntimeError('Mixed precision inference requires CUDA; CPU fixtures must explicitly select fp32')
    if precision=='bf16' and device.type=='cuda' and not torch.cuda.is_bf16_supported():raise RuntimeError('Select an A100 GPU for BF16 inference')
    with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.bfloat16 if precision=='bf16' else torch.float16,
            enabled=bool(precision!='fp32' and device.type=='cuda')):
        result=model(pixel_values=pixels)
    scores=result.logits.float().softmax(-1)[0,:,:-1].max(-1).values.cpu().numpy()
    boxes=result.pred_boxes.float()[0].cpu().numpy();output=[]
    for index in sorted(range(len(scores)),key=lambda i:(-float(scores[i]),i)):
        if scores[index]<cfg.get('detr_confidence',.05):continue
        cx,cy,bw,bh=map(float,boxes[index])
        box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]
        if not np.isfinite(box).all():continue
        output.extend(native_boxes([dict(box=box,score=float(scores[index]))],(0,0,w,h)))
        if len(output)>=cfg.get('detr_max_det',100):break
    return output


def load_base(config):
    """Verify base artifacts without writing or executing anything in that directory."""
    cfg=normalize_config(config);base=Path(cfg['base_root']);out=Path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True)
    original=json.loads((base/'experiment_config.json').read_text());old=roi.config_defaults(original)
    strict=old.get('strict',True)
    if Path(old['output_root']).resolve()!=base:raise ValueError('Base experiment path differs from original config')
    pointer=json.loads((base/'current_code.json').read_text())
    if digest(base/'experiment_config.json')!=pointer['config_sha256']:raise ValueError('Base config changed')
    manifest=base/'code'/pointer['version']/'source_manifest.json'
    if digest(manifest)!=pointer['manifest_sha256']:raise ValueError('Base code manifest changed')
    if strict and pointer['manifest_sha256']!=BASE_MANIFEST:raise ValueError('Expected the reviewed completed30+30 experiment')
    for name,sha in json.loads(manifest.read_text()).items():
        if Path(name).name!=name:raise ValueError('Invalid source manifest path')
        if digest(manifest.parent/name)!=sha:raise ValueError('Saved original source changed: '+name)
        if name in FROZEN and digest(Path(__file__).with_name(name))!=sha:raise ValueError('Bundled frozen helper differs: '+name)
    old['device']=cfg['device'];old['cache_root']=cfg['cache_root']
    cohort=load_cohort(old)
    names=sorted({g['file_name'] for g in cohort['train_groups']}|set(cohort['validation_names']))
    stamps={name:digest(Path(old['images'])/name) for name in names}
    rd=json.loads((base/'roi/training_identity.json').read_text());dd=json.loads((base/'detr/training_identity.json').read_text())
    if identity(rd['protocol'])!=rd['identity'] or identity(dd['protocol'])!=dd['identity']:
        raise ValueError('Base training identity checksum mismatch')
    current=roi.roi_protocol(old,cohort,stamps);saved=rd['protocol']['static']
    if {k:v for k,v in current.items() if k!='runtime'}!={k:v for k,v in saved.items() if k!='runtime'}:
        raise ValueError('ROI source/data/settings differ from the completed experiment')
    dp=dd['protocol']
    if dp['images']!=stamps:raise ValueError('Base DETR image pixels changed')
    roi_path=base/'roi'/f"epoch{old['roi_epochs']:03d}.pt";detr_path=base/'detr'/f"epoch{old['detr_epochs']:03d}.pt"
    roi_sha,detr_sha=digest(roi_path),digest(detr_path)
    if strict and (old['roi_epochs']!=30 or old['detr_epochs']!=30 or roi_sha!=BASE_ROI_SHA or detr_sha!=BASE_DETR_SHA):
        raise ValueError('Use the reviewed completed ROI30 and DETR30 checkpoints')
    if strict and digest(Path(old['yolo']))!=roi.YOLO_SHA:raise ValueError('Original YOLO checkpoint changed')
    roi_state=torch.load(roi_path,map_location='cpu',weights_only=False)
    if (roi_state.get('identity'),roi_state.get('completed_epoch'))!=(rd['identity'],old['roi_epochs']):
        raise ValueError('ROI checkpoint identity mismatch')
    del roi_state
    current_detr_runtime=detr.runtime_versions()
    # Old GPU architecture was not recorded. Even identical package versions
    # do not establish identical inference: rerun all before/after controls on
    # the current GPU, preserving original caches only as historical evidence.
    runtime_equal=(saved['runtime']==current['runtime'] and dp['runtime']==current_detr_runtime)
    reuse=False
    protocol=dict(version=1,base_config_sha256=digest(base/'experiment_config.json'),base_roi_sha256=roi_sha,
        base_detr_sha256=detr_sha,base_yolo_sha256=digest(Path(old['yolo'])),base_roi_identity=rd['identity'],
        base_detr_identity=dd['identity'],source_hashes=source_hashes(),metadata=cohort['input_sha256'],images=stamps,
        settings={k:cfg[k] for k in DEFAULTS if k not in ('device','cache_root')},
        tile_yolo_size=cfg.get('tile_yolo_size',cfg['tile_size']),tile_detr_size=cfg.get('tile_detr_size',old.get('detr_image_size',800)),
        current_runtime=current['runtime'],current_detr_runtime=current_detr_runtime,
        base_roi_runtime=saved['runtime'],base_detr_runtime=dp['runtime'],base_runtime_equal=runtime_equal,reuse_base_cache=reuse,
        current_gpu=torch.cuda.get_device_name(device_for(cfg['device'])) if str(cfg['device']).startswith('cuda') and torch.cuda.is_available() else None)
    run_id=identity(protocol);guard=out/'study_identity.json'
    if guard.exists() and json.loads(guard.read_text())['identity']!=run_id:
        raise ValueError('BBox study sources/data/settings/runtime changed; preserve it and use a new output folder')
    atomic_json(guard,dict(identity=run_id,protocol=protocol));atomic_json(out/'source_images.json',stamps)
    return dict(config=cfg,base_config=old,cohort=cohort,stamps=stamps,roi_checkpoint=roi_path,
        detr_checkpoint=detr_path,roi_sha=roi_sha,detr_sha=detr_sha,reuse_base_cache=reuse,identity=run_id,protocol=protocol)


def cached_full_proposals(ctx,source):
    if not ctx['reuse_base_cache']:return None
    base=Path(ctx['config']['base_root']);cfg=ctx['base_config'];matches=[]
    for path in (base/'proposals'/source).glob('*/validation/manifest.json'):
        doc=json.loads(path.read_text())
        if source=='yolo':
            p=doc.get('protocol',{})
            valid=(p.get('checkpoint_sha256')==ctx['protocol']['base_yolo_sha256'] and p.get('confidence')==cfg['proposal_confidence'])
        else:valid=doc.get('checkpoint_sha256')==ctx['detr_sha'] and doc.get('confidence')==.05
        if valid:matches.append(path.parent)
    if len(matches)>1:raise ValueError('Ambiguous completed base proposal cache')
    if not matches:return None
    predictions={}
    for name in ctx['cohort']['validation_names']:
        path=matches[0]/(Path(name).stem+'.json.gz')
        if not path.exists():return None
        if source=='yolo':record=roi.read_record(path)
        else:
            record=json.loads(gzip.decompress(path.read_bytes()))
            if detr.fingerprint(record['proposals'])!=record['proposal_sha256']:raise ValueError('Base DETR proposal checksum mismatch')
        if record['image_sha256']!=ctx['stamps'][name]:raise ValueError('Base proposal image mismatch')
        predictions[Path(name).stem]=roi.checked_boxes(record['proposals'])
    return predictions


def cached_full_roi(ctx,source):
    if not ctx['reuse_base_cache']:return None
    base=Path(ctx['config']['base_root']);latest=base/'latest_evaluation.json'
    if not latest.exists():return None
    locator=json.loads(latest.read_text());name=Path(locator['report']).parent.name
    folder=base/'evaluation'/name;path=folder/'report.json'
    if not path.exists():return None
    report=json.loads(path.read_text())
    if report.get('roi_checkpoint_sha256')!=ctx['roi_sha'] or report.get('identity')!=locator['identity']:raise ValueError('Base ROI evaluation identity differs')
    values={}
    for image in ctx['cohort']['validation_names']:
        stem=Path(image).stem;p=folder/'records'/(stem+'.json.gz')
        if not p.exists():return None
        record=roi.read_record(p)
        if record['identity']!=report['identity'] or record['image_sha256']!=ctx['stamps'][image]:raise ValueError('Base ROI record differs')
        if source not in record['actual']:return None
        values[stem]=record['actual'][source]
    return values


def detector_model(ctx,source):
    if source=='yolo':
        from ultralytics import YOLO
        return YOLO(ctx['base_config']['yolo'])
    saved=torch.load(ctx['detr_checkpoint'],map_location='cpu',weights_only=False)
    document=json.loads((Path(ctx['config']['base_root'])/'detr/training_identity.json').read_text())
    if (saved.get('identity'),saved.get('completed_epoch'))!=(document['identity'],ctx['base_config']['detr_epochs']):
        raise ValueError('Original DETR checkpoint identity differs')
    model=detr.build_detector(ctx['base_config'],pretrained=False,model_config=saved['model_config']).to(device_for(ctx['config']['device']))
    model.load_state_dict(saved['model'])
    from bbox_detr_continuation import install_fp32_bbox_head
    install_fp32_bbox_head(model)
    model.eval();return model


def collect_probe_proposals(ctx,include_tiles=True):
    cfg=ctx['config'];old=ctx['base_config'];out=Path(cfg['output_root']);variants={};metadata={}
    windows=tile_windows(tile=cfg['tile_size'],stride=cfg['tile_stride'])
    for source in ('yolo','detr'):
        stage=old['detr_epochs'] if source=='detr' else 'original'
        full_name=f'{source}{stage}_full';full=cached_full_proposals(ctx,source);tiled={};model=None
        protocol=dict(study=ctx['identity'],source=source,stage=stage,windows=windows if include_tiles else [],
                      coordinate_system='native2048 xyxy',correction='whole image once before crops')
        proposal_id=identity(protocol);folder=out/'proposals'/proposal_id[:16]
        for number,name in enumerate(ctx['cohort']['validation_names'],1):
            stem=Path(name).stem;path=folder/(stem+'.json.gz')
            if path.exists():
                record=roi.read_record(path)
                if (record['identity'],record['image_sha256'])!=(proposal_id,ctx['stamps'][name]):raise ValueError('Probe proposal cache differs')
            else:
                existing=full[stem] if full is not None else None
                if existing is not None and not include_tiles:
                    record=dict(identity=proposal_id,image_sha256=ctx['stamps'][name],full=existing,tiled=[])
                else:
                    if model is None:model=detector_model(ctx,source)
                    image=prepare_image(cv2.imread(str(Path(old['images'])/name),cv2.IMREAD_GRAYSCALE),old['preprocessing'])
                    if existing is None:
                        existing=(infer_yolo_preprocessed(model,image,old,old['yolo_imgsz']) if source=='yolo'
                                  else infer_detr_preprocessed(model,image,dict(old,detector_precision=cfg['continuation_precision'])))
                    rows=[]
                    if include_tiles:
                        for window in windows:
                            x1,y1,x2,y2=window;patch=image[y1:y2,x1:x2]
                            local=(infer_yolo_preprocessed(model,patch,old,cfg.get('tile_yolo_size',cfg['tile_size'])) if source=='yolo'
                                   else infer_detr_preprocessed(model,patch,dict(old,detr_image_size=cfg.get('tile_detr_size',old.get('detr_image_size',800)),detector_precision=cfg['continuation_precision'])))
                            rows.extend(native_boxes(local,window))
                    record=dict(identity=proposal_id,image_sha256=ctx['stamps'][name],full=roi.checked_boxes(existing),tiled=rows)
                roi.write_record(path,record)
            tiled[stem]=record
            if number%10==0 or number==len(ctx['cohort']['validation_names']):print(f'BBox {source}: {number}/{len(ctx["cohort"]["validation_names"])} photos, {len(windows) if include_tiles else 0} tile calls/photo',flush=True)
        variants[full_name]={stem:row['full'] for stem,row in tiled.items()}
        metadata[full_name]=dict(source=source,stage=stage,mode='full',base_roi_source=source if full is not None else None,
            detector_input_size=old.get('detr_image_size',800) if source=='detr' else old['yolo_imgsz'],
            precision=cfg['continuation_precision'] if source=='detr' else 'ultralytics_auto',bbox_head_precision='fp32' if source=='detr' else None,
            old_cache_is_historical_reference=full is not None,
            old_cache_note='Original cache GPU architecture was not recorded; the new resolution-matched control is used for continuation comparison')
        if include_tiles:
            for nms in cfg['tile_nms']:
                key=f'{source}{stage}_full_tiles_nms{nms:g}'
                variants[key]={stem:suppress_boxes(row['full']+row['tiled'],nms,cfg['tile_max_det']) for stem,row in tiled.items()}
                metadata[key]=dict(source=source,stage=stage,mode='full_plus_tiles',tile_nms=nms,tile_max_det=cfg['tile_max_det'],tile_count=len(windows),
                    tile_detector_input_size=cfg.get('tile_detr_size',old.get('detr_image_size',800)) if source=='detr' else cfg.get('tile_yolo_size',cfg['tile_size']),
                    precision=cfg['continuation_precision'] if source=='detr' else 'ultralytics_auto',bbox_head_precision='fp32' if source=='detr' else None)
        atomic_json(folder/'manifest.json',dict(identity=proposal_id,protocol=protocol,variants=[k for k in variants if metadata[k]['source']==source]))
        if model is not None:del model;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    return variants,metadata


def collect_resolution_control(ctx):
    """Run old DETR weights at the new resolution/precision before more training."""
    cfg,old=ctx['config'],ctx['base_config'];size=cfg['continuation_image_size'];precision=cfg['continuation_precision']
    name=f"detr{old['detr_epochs']}_full{size}_{precision}_control"
    settings=dict(old,detr_image_size=size,detector_precision=precision)
    key=identity(dict(study=ctx['identity'],checkpoint=ctx['detr_sha'],resolution=size,precision=precision))
    folder=Path(cfg['output_root'])/'proposals'/key[:16];proposals={};model=None
    for number,image_name in enumerate(ctx['cohort']['validation_names'],1):
        stem=Path(image_name).stem;path=folder/(stem+'.json.gz')
        if path.exists():
            record=roi.read_record(path)
            if (record['identity'],record['image_sha256'])!=(key,ctx['stamps'][image_name]):raise ValueError('Resolution-control cache changed')
        else:
            if model is None:model=detector_model(ctx,'detr')
            corrected=prepare_image(cv2.imread(str(Path(old['images'])/image_name),cv2.IMREAD_GRAYSCALE),old['preprocessing'])
            rows=infer_detr_preprocessed(model,corrected,settings)
            record=dict(identity=key,image_sha256=ctx['stamps'][image_name],proposals=rows)
            roi.write_record(path,record)
        proposals[stem]=roi.checked_boxes(record['proposals'])
        if number%10==0 or number==len(ctx['cohort']['validation_names']):print(f'DETR{old["detr_epochs"]} resolution control {size}/{precision}: {number}/{len(ctx["cohort"]["validation_names"])}',flush=True)
    if model is not None:del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    metadata=dict(source='detr',stage=old['detr_epochs'],mode='full_resolution_control',detector_input_size=size,
                  precision=precision,bbox_head_precision='fp32',checkpoint_sha256=ctx['detr_sha'],additional_epochs=0,
                  purpose='Same old weights at continuation resolution/precision; separates inference resolution from additional learning')
    atomic_json(folder/'manifest.json',dict(identity=key,metadata=metadata))
    return {name:proposals},{name:metadata}


def bbox_metrics(truth,mapping,proposals,thresholds):
    """Per-rater availability and score-ordered one-to-one box precision/recall."""
    result={}
    for confidence in thresholds:
        rows={str(t):dict(tp=0,fp=0,fn=0,GT=0,available_GT=0,matched_iou_sum=0.,small_GT=0,small_matched=0,small_available=0) for t in (.5,.75)}
        for sid,gt in truth.items():
            selected=sorted((p for p in proposals[mapping[sid]] if p['score']>=confidence),key=lambda p:-p['score'])
            boxes=np.asarray([[p['box'][0],p['box'][1],p['box'][2]-p['box'][0],p['box'][3]-p['box'][1]] for p in selected],float).reshape(-1,4)
            matrix=np.asarray(mu.iou(np.asarray(mu.toBbox(gt),float),boxes,[0]*len(boxes))) if selected else np.zeros((len(gt),0))
            small=np.asarray(mu.area(gt))<670
            for cutoff in (.5,.75):
                row=rows[str(cutoff)];used=set();matched=[]
                for index in range(len(selected)):
                    options=[g for g in range(len(gt)) if g not in used and matrix[g,index]>cutoff]
                    if options:
                        best=max(options,key=lambda g:matrix[g,index]);used.add(best);matched.append(float(matrix[best,index]))
                row['tp']+=len(used);row['fp']+=len(selected)-len(used);row['fn']+=len(gt)-len(used);row['GT']+=len(gt)
                row['available_GT']+=int((matrix>cutoff).any(1).sum());row['matched_iou_sum']+=sum(matched)
                row['small_GT']+=int(small.sum());row['small_matched']+=sum(bool(small[g]) for g in used)
                row['small_available']+=int((small&(matrix>cutoff).any(1)).sum())
        for row in rows.values():
            row.update(recall=row['tp']/row['GT'] if row['GT'] else None,
                precision=row['tp']/(row['tp']+row['fp']) if row['tp']+row['fp'] else None,
                availability=row['available_GT']/row['GT'] if row['GT'] else None,
                mean_matched_bbox_iou=row['matched_iou_sum']/row['tp'] if row['tp'] else None)
        result[str(confidence)]=dict(by_iou=rows,physical_proposals=sum(sum(p['score']>=confidence for p in values) for values in proposals.values()))
    return dict(scope='Box diagnostics on separate annotator groups; strict IoU>.5/.75; not mask PQ or COCO AP',thresholds=result)


def evaluate_variants(ctx,variants,metadata,label):
    cfg=ctx['config'];old=ctx['base_config'];cohort=ctx['cohort'];out=Path(cfg['output_root'])
    if not variants or set(variants)!=set(metadata):raise ValueError('Missing variant metadata')
    stems={Path(n).stem for n in cohort['validation_names']}
    for rows in variants.values():
        if set(rows)!=stems:raise ValueError('Every detector must cover exact validation photos')
    evaluation_id=identity(dict(study=ctx['identity'],roi=ctx['roi_sha'],variants=variants,metadata=metadata))
    folder=out/'evaluation'/(label+'_'+evaluation_id[:16]);folder.mkdir(parents=True,exist_ok=True)
    report_path=folder/'report.json'
    if report_path.exists():
        report=json.loads(report_path.read_text())
        if report.get('identity')!=evaluation_id:raise ValueError('Evaluation identity changed')
        if any(not (folder/name).is_file() or digest(folder/name)!=sha for name,sha in report['csv_sha256'].items()):raise ValueError('Completed evaluation CSV changed')
        return report_path
    predictions={key:{f'box{t:g}_area{a}':{} for t in cfg[metadata[key]['source']+'_thresholds'] for a in cfg['minimum_areas']} for key in variants}
    base_cached={s:cached_full_roi(ctx,s) for s in ('yolo','detr')}
    model=None;device=device_for(cfg['device']);configure_cuda()
    for number,name in enumerate(cohort['validation_names'],1):
        stem=Path(name).stem;record_path=folder/'records'/(stem+'.json.gz')
        if record_path.exists():
            record=roi.read_record(record_path)
            if (record['identity'],record['image_sha256'])!=(evaluation_id,ctx['stamps'][name]):raise ValueError('Evaluation record changed')
        else:
            cache_path=out/'roi_box_cache'/ctx['identity'][:16]/(stem+'.json.gz')
            cache=roi.read_record(cache_path) if cache_path.exists() else dict(identity=ctx['identity'],image_sha256=ctx['stamps'][name],boxes={})
            if (cache['identity'],cache['image_sha256'])!=(ctx['identity'],ctx['stamps'][name]):raise ValueError('Per-box ROI cache changed')
            image=None;actual={}
            for key,props in variants.items():
                source=metadata[key]['source']
                if metadata[key].get('base_roi_source') and base_cached[source] is not None:
                    actual[key]=base_cached[source][stem];continue
                rows=[]
                for proposal in props[stem]:
                    box_id=identity(proposal['box'])
                    if box_id not in cache['boxes']:
                        if model is None:
                            saved=torch.load(ctx['roi_checkpoint'],map_location='cpu',weights_only=False)
                            model=RoiFilamentNet(encoder=old['roi_encoder'],encoder_weights=None).to(device)
                            model.load_state_dict(saved['model']);model.eval();del saved
                        if image is None:image=prepare_image(cv2.imread(str(Path(old['images'])/name),cv2.IMREAD_GRAYSCALE),old['preprocessing'])
                        inferred=roi.infer_rois(model,image,[dict(box=proposal['box'],score=1.)],old,device)
                        cache['boxes'][box_id]=roi.json_ready(inferred[0] if inferred else None)
                    item=cache['boxes'][box_id]
                    if item is not None:rows.append(dict(item,box=proposal['box'],box_id=box_id,box_score=proposal['score'],selection_score=proposal['score']*item['roi_quality']))
                actual[key]=rows
                roi.write_record(cache_path,cache)
            record=dict(identity=evaluation_id,image_sha256=ctx['stamps'][name],actual=actual,actual_boxes={k:v[stem] for k,v in variants.items()})
            roi.write_record(record_path,record)
        for key in variants:
            candidates=roi.decode_serialized_candidates(record['actual'][key])
            select=MaskSelector(candidates,old['mask_nms_iou'])
            for threshold in cfg[metadata[key]['source']+'_thresholds']:
                for area in cfg['minimum_areas']:
                    predictions[key][f'box{threshold:g}_area{area}'][stem]=select(threshold,area,cfg['minimum_quality'])
        print(f'Frozen ROI evaluation {label}: {number}/{len(cohort["validation_names"])} {name}',flush=True)
    if model is not None:del model;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    results={};diagnostics={}
    for key,settings in predictions.items():
        results[key]={}
        for setting,pred in settings.items():
            name=key+'_'+setting+'.csv';roi.write_predictions(folder/name,pred)
            results[key][setting]=roi.score_groups(cohort['truth'],cohort['mapping'],pred)
        diagnostics[key]=bbox_metrics(cohort['truth'],cohort['mapping'],variants[key],cfg[metadata[key]['source']+'_thresholds'])
        best=max(results[key],key=lambda setting:results[key][setting]['score']['pq'])
        print(key,'best DEVELOPMENT setting',best,results[key][best]['score'],flush=True)
    report=dict(status='complete',identity=evaluation_id,label=label,study_identity=ctx['identity'],
        scope='Repeated fixed91 development; threshold-selected maxima are descriptive, not test/LB scores',
        roi_checkpoint_sha256=ctx['roi_sha'],roi_epochs=old['roi_epochs'],roi_retrained=False,
        variants=metadata,results=results,bbox_diagnostics=diagnostics,
        cohort=cohort['counts'],raw_candidates_before_quality_gate=True,
        preprocessing='Whole2048 image corrected once, then cropped; crop never gets radial correction',
        prediction_protocol=dict(mask_threshold=old['mask_threshold'],mask_nms=old['mask_nms_iou'],quality=cfg['minimum_quality'],areas=cfg['minimum_areas']),
        csv_sha256={p.name:digest(p) for p in folder.glob('*.csv')})
    atomic_json(report_path,report);return report_path


def export_results(config,label):
    root=Path(config['output_root']);destination=root/'exports';destination.mkdir(parents=True,exist_ok=True)
    common=[];records=[]
    for path in root.rglob('*'):
        if not path.is_file() or destination in path.parents or path.name=='latest_results.json' or '.pending' in path.name or 'roi_box_cache' in path.parts:continue
        if path.suffix not in ('.json','.csv','.py','.md','.txt','.gz'):continue
        relative=path.relative_to(root)
        # Prediction CSVs and large per-variant reports must be partitioned,
        # not repeated in every part; otherwise 64+ CSVs overflow common data.
        (common if len(relative.parts)==1 or relative.parts[0]=='code' else records).append((path,str(relative)))
    entries={name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p,name in common+records}
    base=sum(p.stat().st_size+len(n)*2+256 for p,n in common)+len(json.dumps(entries))*2+65536
    budget=26*1024*1024
    if base>budget//2:raise ValueError('Summary exceeds export budget')
    parts=[];part=[];size=base
    for path,name in sorted(records):
        amount=path.stat().st_size+len(name)*2+512
        if base+amount>budget:raise ValueError('One result exceeds export budget')
        if part and size+amount>budget:parts.append(part);part=[];size=base
        part.append((path,name));size+=amount
    parts.append(part);export_id=identity(entries);archives=[]
    for index,part in enumerate(parts,1):
        path=destination/f'bbox_{label}_{export_id[:12]}_part{index:02d}.zip';pending=path.with_suffix('.pending')
        with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as z:
            for source,name in sorted(common+part):z.write(source,name)
            z.writestr('export_manifest.json',json.dumps(dict(export_id=export_id,label=label,part=index,parts_total=len(parts),entries=entries),indent=2))
        if pending.stat().st_size>28*1024*1024:raise ValueError('Export exceeds28MiB')
        pending.replace(path);archives.append(str(path))
    return archives


def run(config,phase='all',run_until_added=30):
    if phase not in ('probes','extend','all'):raise ValueError('Invalid phase')
    cfg=normalize_config(config)
    if not 1<=run_until_added<=cfg['added_epochs']:raise ValueError('Requested extra epoch exceeds fixed new-stage budget')
    out=Path(cfg['output_root']);out.mkdir(parents=True,exist_ok=True)
    lock_path=Path('/tmp')/('bbox_improvement_'+identity(str(out))[:16]+'.lock')
    with lock_path.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another cell is using this bbox study')
        ctx=load_base(cfg);base_epoch=ctx['base_config']['detr_epochs']
        result=dict(status='running',phase=phase,reports=[],archives=[],interim_exports=[],roi_retrained=False,
                    maximum_additional_training_epochs=0 if phase=='probes' else cfg['added_epochs'])
        atomic_json(out/'result.json',result)
        variants,metadata=collect_probe_proposals(ctx,include_tiles=phase in ('probes','all'))
        control,control_metadata=collect_resolution_control(ctx);variants.update(control);metadata.update(control_metadata)
        label='probes' if phase in ('probes','all') else 'baseline'
        result['reports'].append(str(evaluate_variants(ctx,variants,metadata,label)))
        result['status']='complete' if phase=='probes' else 'probes_complete_training_pending'
        atomic_json(out/'result.json',result)
        result['archives']=export_results(cfg,label)
        atomic_json(out/'latest_results.json',result)
        print('Completed '+label+' result ZIPs:',json.dumps(result['archives']),flush=True)
        if phase in ('extend','all'):
            from bbox_detr_continuation import train_extension,export_extension_proposals
            ecfg=dict(cfg,output_root=str(out/'detr_extension'))
            targets=sorted({min(15,cfg['added_epochs']),cfg['added_epochs'],run_until_added})
            targets=[value for value in targets if value<=run_until_added]
            for added in targets:
                checkpoint=out/'detr_extension'/f'epoch{base_epoch+added:03d}.pt'
                trained=train_extension(ecfg,run_until_added=added) if not checkpoint.exists() else dict(checkpoint=str(checkpoint))
                props=export_extension_proposals(ecfg,checkpoint=trained['checkpoint'],role='validation')
                key=f'detr{base_epoch+added}_full';stage_variants={key:props}
                stage_metadata={key:dict(source='detr',stage=base_epoch+added,mode='full',additional_epochs=added,
                    checkpoint_sha256=digest(Path(trained['checkpoint'])),roi_fixed_checkpoint_sha256=ctx['roi_sha'],
                    detector_input_size=cfg['continuation_image_size'],precision=cfg['continuation_precision'],bbox_head_precision='fp32')}
                report=evaluate_variants(ctx,stage_variants,stage_metadata,f'extended{base_epoch+added}')
                result['reports'].append(str(report));result['added_epochs_completed']=added;result['cumulative_detr_epochs']=base_epoch+added
                result['status']=('complete' if run_until_added==cfg['added_epochs'] else 'stage_complete') if added==targets[-1] else 'stage_complete_training_pending'
                result['interim_exports'].append(dict(label=label,archives=result['archives']))
                result['archives']=[]  # Current export paths are assigned after its content hash is known.
                atomic_json(out/'result.json',result)
                stage_archives=export_results(cfg,f'extended{base_epoch+added}')
                result['archives']=stage_archives;label=f'extended{base_epoch+added}'
                atomic_json(out/'latest_results.json',result)
                print(f'Completed DETR{base_epoch+added} result ZIPs:',json.dumps(stage_archives),flush=True)
        result['status']='complete' if phase=='probes' or run_until_added==cfg['added_epochs'] else 'stage_complete'
        atomic_json(out/'result.json',result);atomic_json(out/'latest_results.json',result)
        return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--phase',choices=('probes','extend','all'),default='all')
    parser.add_argument('--run-until-added',type=int,default=30)
    args=parser.parse_args();value=run(json.loads(Path(args.config).read_text()),args.phase,args.run_until_added)
    print(json.dumps(value,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
