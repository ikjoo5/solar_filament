"""Detection -> box-conditioned ROI segmentation, fixed-split development.

Default all: existing YOLO boxes + thirty DETR epochs + thirty ROI epochs.
GT-box oracle is reported per annotator and never presented as deployable PQ.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import gc
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import zipfile

os.environ.setdefault('TORCH_CUDNN_V8_API_DISABLED', '1')
import cv2
import numpy as np
from pycocotools import mask as mu
import torch
from torch.utils.data import DataLoader

from native_instance_data import load_cohort, annotation_rle
from native_instance_training import (atomic_json, atomic_torch, capture_rng, restore_rng,
    seed_everything, cpu_copy, device_for, configure_cuda, digest, identity, input_image_hashes)

DEFAULTS = dict(roi_epochs=30, roi_encoder='resnet34', roi_pretrained=True,
    roi_size=384, roi_context=1.5, roi_batch=4, roi_accumulate=1,
    roi_steps_per_epoch=256, roi_lr=1e-4, roi_weight_decay=.01,
    seed=42, device='cuda', amp=True, preprocessing='limb', strict=True,
    proposal_confidence=.05, yolo_imgsz=2048, yolo_max_det=100,
    yolo_nms_iou=.7, eval_box_thresholds=[.05,.30], eval_min_areas=[150,300],
    mask_threshold=.5, roi_min_quality=.6, mask_nms_iou=.5,
    detr_epochs=30, cache_root='/content/solar_detection_roi_cache')
ROI_KEYS = tuple(k for k in DEFAULTS if k not in ('device','cache_root','detr_epochs'))
SOURCES = ('roi_filament_pipeline.py','roi_filament_core.py','filament_detr.py',
           'native_instance_data.py','native_instance_training.py')
YOLO_SHA = '81c2a27d95821b5ede283169c0da8c08d360ecc4c2a60a1b9c6baf96d76bcccc'


def config_defaults(config):
    cfg = dict(DEFAULTS, **config)
    for key in ('roi_epochs','roi_size','roi_batch','roi_accumulate','roi_steps_per_epoch','detr_epochs'):
        if not isinstance(cfg[key], int) or cfg[key]<1: raise ValueError(key+' must be positive integer')
    if cfg['roi_size']%32 or cfg['roi_encoder'] not in ('resnet18','resnet34','resnet50'):
        raise ValueError('ROI size must be multiple of32; choose a supported ResNet encoder')
    if cfg['roi_context']<1 or not 0<cfg['roi_lr']<.1: raise ValueError('Invalid ROI context/lr')
    if not 0<cfg['proposal_confidence']<=min(cfg['eval_box_thresholds'])<=1:
        raise ValueError('Export confidence must cover evaluated detector thresholds')
    if not 0<cfg.get('detr_confidence',.05)<=min(cfg['eval_box_thresholds']):
        raise ValueError('DETR export confidence must cover evaluated detector thresholds')
    if any(not 0<=x<=1 for x in cfg['eval_box_thresholds']) or not cfg['eval_box_thresholds']:
        raise ValueError('Invalid detector score thresholds')
    if not cfg['eval_min_areas'] or any(not isinstance(x,int) or x<1 for x in cfg['eval_min_areas']):
        raise ValueError('Positive minimum mask areas required')
    if not 0<cfg['mask_threshold']<1 or not 0<=cfg['roi_min_quality']<=1:
        raise ValueError('Invalid mask thresholds')
    output,cache=Path(cfg['output_root']).resolve(),Path(cfg['cache_root']).resolve()
    if output==cache or output in cache.parents or cache in output.parents:
        raise ValueError('Persistent output and disposable cache must be disjoint')
    return cfg


def source_hashes():
    return {name:digest(Path(__file__).with_name(name)) for name in SOURCES}


def versions():
    names=('torch','torchvision','numpy','scipy','segmentation-models-pytorch','timm','pycocotools','ultralytics')
    result={name:importlib.metadata.version(name) for name in names}
    result.update(opencv=cv2.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    return result


def json_ready(value):
    if isinstance(value,bytes): return value.decode('ascii')
    if isinstance(value,dict): return {k:json_ready(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_ready(x) for x in value]
    if isinstance(value,np.generic): return value.item()
    return value


def write_record(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=json_ready(payload);envelope=dict(payload=payload,sha256=identity(payload))
    pending=path.with_name(path.name+'.pending')
    try: pending.write_bytes(gzip.compress(json.dumps(envelope,separators=(',',':'),allow_nan=False).encode()));pending.replace(path)
    finally: pending.unlink(missing_ok=True)


def read_record(path):
    envelope=json.loads(gzip.decompress(Path(path).read_bytes()))
    if envelope.get('sha256')!=identity(envelope.get('payload')): raise ValueError('Result cache checksum mismatch: '+str(path))
    return envelope['payload']


def checked_boxes(rows):
    result=[]
    for row in rows:
        box=np.asarray(row['box'],float);score=float(row['score'])
        if box.shape!=(4,) or not np.isfinite(box).all() or not np.isfinite(score) or not 0<=score<=1:
            raise ValueError('Invalid original-coordinate bbox/score')
        if not (0<=box[0]<box[2]<=2048 and 0<=box[1]<box[3]<=2048):
            raise ValueError('bbox must be within original2048 image')
        result.append(dict(box=box.tolist(),score=score))
    return result


def export_yolo_proposals(config,role,cohort=None,image_hashes=None):
    """Export actual regressed boxes, including proposals with empty masks."""
    from roi_filament_core import prepare_image
    cfg=config_defaults(config);cohort=cohort or load_cohort(cfg)
    if role not in ('train','validation'): raise ValueError('Invalid proposal role')
    checkpoint=Path(cfg['yolo']);checkpoint_sha=digest(checkpoint)
    if cfg['strict'] and checkpoint_sha!=YOLO_SHA: raise ValueError('Use the reviewed original YOLO checkpoint')
    stamps=image_hashes or input_image_hashes(cfg,cohort)
    names=sorted({g['file_name'] for g in cohort['train_groups']}) if role=='train' else cohort['validation_names']
    protocol=dict(version=1,checkpoint_sha256=checkpoint_sha,code=source_hashes(),image_sha256={n:stamps[n] for n in names},
                  role=role,preprocessing=cfg['preprocessing'],confidence=cfg['proposal_confidence'],
                  imgsz=cfg['yolo_imgsz'],max_det=cfg['yolo_max_det'],iou=cfg['yolo_nms_iou'],
                  source='actual Results.boxes.xyxy; no mask-area gate',views=['base'])
    proposal_id=identity(protocol);folder=Path(cfg['output_root'])/'proposals/yolo'/proposal_id[:16]/role
    folder.mkdir(parents=True,exist_ok=True);result={};model=None
    try:
        for number,name in enumerate(names,1):
            stem=Path(name).stem;dest=folder/(stem+'.json.gz')
            if dest.exists():
                record=read_record(dest)
                if (record.get('identity'),record.get('image_sha256'))!=(proposal_id,stamps[name]):
                    raise ValueError('Wrong YOLO proposal cache identity')
            else:
                if model is None:
                    configure_cuda()
                    from ultralytics import YOLO
                    model=YOLO(str(checkpoint))
                pixels=cv2.imread(str(Path(cfg['images'])/name),cv2.IMREAD_GRAYSCALE)
                image=prepare_image(pixels,cfg['preprocessing'])
                prediction=model.predict(source=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR),imgsz=cfg['yolo_imgsz'],
                    conf=cfg['proposal_confidence'],iou=cfg['yolo_nms_iou'],max_det=cfg['yolo_max_det'],
                    device=0 if str(cfg['device']) in ('cuda','cuda:0','0') else cfg['device'],verbose=False)[0]
                boxes=prediction.boxes
                rows=[] if boxes is None else [dict(box=box,score=score) for box,score in zip(boxes.xyxy.cpu().tolist(),boxes.conf.cpu().tolist())]
                record=dict(identity=proposal_id,image_sha256=stamps[name],file_name=name,proposals=checked_boxes(rows))
                write_record(dest,record)
            result[stem]=checked_boxes(record['proposals'])
            if number%25==0 or number==len(names):print(f'YOLO actualbbox {role} {number}/{len(names)}',flush=True)
    finally:
        if model is not None:del model;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    atomic_json(folder/'manifest.json',dict(identity=proposal_id,protocol=protocol,images=len(names),proposal_count=sum(map(len,result.values()))))
    return result


def roi_protocol(cfg,cohort,image_hashes):
    return dict(settings={key:cfg[key] for key in ROI_KEYS},sources=source_hashes(),runtime=versions(),
                metadata=cohort['input_sha256'],image_sha256=image_hashes,
                initial_weights='generic ImageNet ResNet encoder' if cfg['roi_pretrained'] else 'random',
                epoch_unit='roi_steps_per_epoch minibatches of sampled single-object crops',
                validation_selects_checkpoint=False,bitwise_GPU_trajectory_guaranteed=False)


def train_roi(config,proposals_by_image,run_until=None,corpus_label='yolo_train_only'):
    from roi_filament_core import RoiFilamentNet,RoiTrainingDataset,roi_loss
    cfg=config_defaults(config);cap=cfg['roi_epochs'] if run_until is None else int(run_until)
    if not 1<=cap<=cfg['roi_epochs']:raise ValueError('ROI stage outside fixed schedule')
    cohort=load_cohort(cfg);stamps=input_image_hashes(cfg,cohort)
    train_stems={Path(g['file_name']).stem for g in cohort['train_groups']}
    if not set(proposals_by_image)<=train_stems:raise ValueError('Validation/test proposal cannot enter ROI training corpus')
    proposals={stem:checked_boxes(rows) for stem,rows in proposals_by_image.items()}
    static=roi_protocol(cfg,cohort,stamps)
    fixed=dict(static=static,proposal_corpus_sha256=identity(proposals),corpus_label=corpus_label)
    run_id=identity(fixed);root=Path(cfg['output_root'])/'roi';root.mkdir(parents=True,exist_ok=True)
    protocol_path=root/'training_identity.json';last=root/'last_full.pt';stage=root/f'epoch{cap:03d}.pt'
    if protocol_path.exists() and json.loads(protocol_path.read_text())['identity']!=run_id:
        raise ValueError('ROI training code/data/proposal corpus changed; preserve existing model and use a new output root')
    if last.exists() and not protocol_path.exists():raise ValueError('Unidentified ROI checkpoint cannot resume')
    atomic_json(protocol_path,dict(identity=run_id,protocol=fixed))
    if stage.exists():
        saved=torch.load(stage,map_location='cpu',weights_only=False)
        if (saved.get('identity'),saved.get('completed_epoch'))!=(run_id,cap):raise ValueError('Wrong ROI stage identity')
        return dict(checkpoint=str(stage),checkpoint_sha256=digest(stage),identity=run_id,epochs_completed=cap,fixed_final_epoch=cfg['roi_epochs'],status='complete' if cap==cfg['roi_epochs'] else 'stage_complete')
    saved=torch.load(last,map_location='cpu',weights_only=False) if last.exists() else None
    if saved and (saved.get('identity')!=run_id or saved['completed_epoch']>cap):raise ValueError('Wrong/later ROI checkpoint')
    configure_cuda();device=device_for(cfg['device']);seed_everything(cfg['seed'])
    model=RoiFilamentNet(encoder=cfg['roi_encoder'],encoder_weights='imagenet' if cfg['roi_pretrained'] and saved is None else None).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['roi_lr'],weight_decay=cfg['roi_weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e:.1+.9*(1+math.cos(math.pi*e/cfg['roi_epochs']))/2)
    scaler=torch.amp.GradScaler('cuda',enabled=bool(cfg['amp'] and device.type=='cuda'))
    history=[];start=0
    if saved:
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler']);restore_rng(saved['rng']);history=saved['history'];start=saved['completed_epoch']
        if len(history)!=start:raise ValueError('Inconsistent completed ROI history')
    dataset=RoiTrainingDataset(cohort['train_groups'],cfg['images'],proposals_by_image=proposals,
        crop_size=cfg['roi_size'],context=cfg['roi_context'],steps_per_epoch=cfg['roi_steps_per_epoch'],
        batch_size=cfg['roi_batch'],seed=cfg['seed'],preprocessing=cfg['preprocessing'],
        cache_dir=Path(cfg['cache_root'])/'source_images',source_sha256=stamps)
    print(f'ROI {cfg["roi_encoder"]}: completed{start} -> requested{cap} / final{cfg["roi_epochs"]}',flush=True)
    for epoch in range(start,cap):
        epoch_seed=cfg['seed']+10007*epoch;seed_everything(epoch_seed);dataset.set_epoch(epoch)
        loader=DataLoader(dataset,batch_size=cfg['roi_batch'],shuffle=False,num_workers=0,
                          generator=torch.Generator().manual_seed(epoch_seed),pin_memory=device.type=='cuda')
        if len(loader)!=cfg['roi_steps_per_epoch']:raise ValueError('ROI sampled minibatch budget changed')
        model.train();optimizer.zero_grad(set_to_none=True);losses={};batches=attempts=successes=0;started=time.monotonic()
        for index,(inputs,targets) in enumerate(loader):
            # RoiFilamentNet owns grayscale normalization; prompt stays binary.
            inputs,targets=inputs.to(device),targets.to(device)
            group_start=(index//cfg['roi_accumulate'])*cfg['roi_accumulate']
            group_length=min(cfg['roi_accumulate'],len(loader)-group_start)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=bool(cfg['amp'] and device.type=='cuda')):
                logits=model(inputs);loss,terms=roi_loss(logits,targets)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite ROI loss; prior completed epoch retained')
            scaler.scale(loss/group_length).backward();batches+=1
            for key,value in terms.items():losses[key]=losses.get(key,0.)+float(value.detach().cpu() if isinstance(value,torch.Tensor) else value)
            losses['total']=losses.get('total',0.)+float(loss.detach().cpu())
            if (index+1)%cfg['roi_accumulate']==0 or index+1==len(loader):
                before=scaler.get_scale();scaler.step(optimizer);scaler.update();attempts+=1;successes+=int(scaler.get_scale()>=before);optimizer.zero_grad(set_to_none=True)
            if batches%32==0 or batches==len(loader):print(f'ROI epoch{epoch+1}/{cfg["roi_epochs"]} {batches}/{len(loader)} loss{losses["total"]/batches:.5f}',flush=True)
        if batches!=len(loader):raise RuntimeError('Partial ROI epoch cannot become a completed checkpoint')
        scheduler.step()
        history.append(dict(epoch=epoch+1,losses={k:v/batches for k,v in losses.items()},minibatches=batches,
                            sampled_crops=batches*cfg['roi_batch'],optimizer_attempts=attempts,optimizer_updates=successes,
                            amp_skipped_updates=attempts-successes,next_lr=scheduler.get_last_lr(),seconds=time.monotonic()-started))
        payload=dict(schema_version=1,identity=run_id,completed_epoch=epoch+1,fixed_final_epoch=cfg['roi_epochs'],
                     config={k:cfg[k] for k in ROI_KEYS},model=cpu_copy(model.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),
                     scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=capture_rng(),history=history)
        atomic_torch(last,payload);atomic_json(root/'history.json',history)
        atomic_json(root/'progress.json',dict(completed_epoch=epoch+1,fixed_final_epoch=cfg['roi_epochs']))
    if start==cap:payload=saved
    compact=dict(payload)
    for key in ('optimizer','scheduler','scaler','rng'):compact[key]=None
    atomic_torch(stage,compact);del model;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return dict(checkpoint=str(stage),checkpoint_sha256=digest(stage),identity=run_id,epochs_completed=cap,
                fixed_final_epoch=cfg['roi_epochs'],status='complete' if cap==cfg['roi_epochs'] else 'stage_complete')


def mask_predictions(candidates,box_threshold,min_area,nms_iou=.5,minimum_quality=0.):
    """Confidence order + mask NMS + ownership, always disjoint final masks."""
    eligible=[x for x in candidates if x['box_score']>=box_threshold and x.get('roi_quality',1.)>=minimum_quality
              and int(mu.area(x['rle']))>=min_area]
    eligible.sort(key=lambda x:-x['selection_score']);selected=[]
    for item in eligible:
        rle=item['rle']
        if selected and np.asarray(mu.iou([rle],selected,[0]*len(selected))).max()>nms_iou:continue
        selected.append(rle)
    owner=np.zeros((2048,2048),bool);result=[]
    for rle in selected:
        mask=mu.decode(rle).astype(bool)&~owner
        if mask.sum()<min_area:continue
        owner|=mask;result.append(mu.encode(np.asfortranarray(mask.astype(np.uint8))))
    if result and int(mu.area(mu.merge(result)))!=sum(map(int,mu.area(result))):raise AssertionError('Final ROI instance overlap')
    return result


def infer_rois(model,image,proposals,cfg,device):
    from roi_filament_core import crop_roi,restore_mask
    rows=checked_boxes(proposals);result=[]
    model.eval()
    with torch.inference_mode():
        for proposal in rows:
            if proposal['score']<cfg['proposal_confidence']:continue
            inputs,_,transform=crop_roi(image,proposal['box'],context=cfg['roi_context'],size=cfg['roi_size'])
            inputs=torch.from_numpy(np.ascontiguousarray(inputs)).float()[None].to(device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=bool(cfg['amp'] and device.type=='cuda')):
                probability=model(inputs).sigmoid()[0,0].float().cpu().numpy()
            foreground=probability>=cfg['mask_threshold']
            quality=float(probability[foreground].mean()) if foreground.any() else 0.
            rle=restore_mask(probability,transform,threshold=cfg['mask_threshold'])
            if rle is not None and int(mu.area(rle)):
                result.append(dict(rle=rle,box_score=proposal['score'],roi_quality=quality,selection_score=proposal['score']*quality))
    return result


def decode_serialized_candidates(rows):
    return [dict(row,rle=dict(size=row['rle']['size'],counts=row['rle']['counts'].encode() if isinstance(row['rle']['counts'],str) else row['rle']['counts'])) for row in rows]


def score_groups(truth,mapping,predictions,per_group=False):
    rows=[];small_total=small_hits=0
    for sid,gt in sorted(truth.items()):
        if not gt:continue
        pred=predictions[sid if per_group else mapping[sid]]
        matrix=np.asarray(mu.iou(gt,pred,[0]*len(pred))) if pred else np.zeros((len(gt),0));hits=matrix>.5
        rows.append(dict(group_id=sid,image=mapping[sid],tp_pairs=int(hits.sum()),fp=int((~hits.any(0)).sum()),
                         fn=int((~hits.any(1)).sum()),matched_iou_sum=float(matrix[hits].sum())))
        small=np.asarray(mu.area(gt))<670;small_total+=int(small.sum());small_hits+=int((small&hits.any(1)).sum())
    total={k:sum(row[k] for row in rows) for k in ('tp_pairs','fp','fn','matched_iou_sum')};den=total['tp_pairs']+.5*(total['fp']+total['fn'])
    total.update(pq=total['matched_iou_sum']/den if den else 0.,sq=total['matched_iou_sum']/total['tp_pairs'] if total['tp_pairs'] else 0.,
                 rq=total['tp_pairs']/den if den else 0.,physical_images=len(set(mapping.values())),annotation_groups=len(rows),
                 small_GT=dict(total=small_total,matched=small_hits,recall=small_hits/small_total if small_total else None))
    return dict(score=total,per_group=rows)


def write_predictions(path,predictions):
    path=Path(path);pending=path.with_name(path.name+'.pending')
    with pending.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['filament_id','segmentation_rle'])
        for stem,masks in predictions.items():
            for index,rle in enumerate(masks):writer.writerow([f'{stem}_{index}',rle['counts'].decode() if isinstance(rle['counts'],bytes) else rle['counts']])
    pending.replace(path)


def rectangle_rle(box):
    x1,y1,x2,y2=map(float,box)
    return mu.frPyObjects([[x1,y1,x2,y1,x2,y2,x1,y2]],2048,2048)[0]


def bbox_diagnostics(truth,mapping,proposals,context):
    from roi_filament_core import _geometry
    totals={str(t):dict(GT=0,bbox_iou50=0,expanded_crop_coverage95=0,proposal_count=0) for t in (.05,.30)}
    for sid,gt in truth.items():
        if not gt:continue
        gt_boxes=np.asarray(mu.toBbox(gt),float)
        for threshold in (.05,.30):
            chosen=[p for p in proposals[mapping[sid]] if p['score']>=threshold]
            boxes=np.asarray([[p['box'][0],p['box'][1],p['box'][2]-p['box'][0],p['box'][3]-p['box'][1]] for p in chosen],float).reshape(-1,4)
            matrix=np.asarray(mu.iou(gt_boxes,boxes,[0]*len(boxes))) if len(boxes) else np.zeros((len(gt),0))
            crops=[]
            for p in chosen:
                # Use exactly the ROI transform, including its square context
                # and integer edge rounding; rectangular expansion differs.
                geometry=_geometry(p['box'],context,32)
                left,top,side=(geometry[k] for k in ('left','top','side'))
                crops.append(rectangle_rle([max(0,left),max(0,top),min(2048,left+side),min(2048,top+side)]))
            coverage=np.asarray(mu.iou(gt,crops,[1]*len(crops))) if crops else np.zeros((len(gt),0))
            row=totals[str(threshold)];row['GT']+=len(gt);row['bbox_iou50']+=int((matrix>.5).any(1).sum());row['expanded_crop_coverage95']+=int((coverage>=.95).any(1).sum());row['proposal_count']+=len(chosen)
    for row in totals.values():
        row['proposal_recall']=row['bbox_iou50']/row['GT'] if row['GT'] else None
        row['crop_coverage']=row['expanded_crop_coverage95']/row['GT'] if row['GT'] else None
    return dict(scope='Actual regressed detector boxes; GT availability/crop coverage is not achievable instance recall',
                crop_geometry='Actual shared square-context ROI transform, clipped to image',thresholds=totals)


def validate_roi_stage(cfg,cohort,stamps,checkpoint,cap):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    document=json.loads((Path(cfg['output_root'])/'roi/training_identity.json').read_text());fixed=document['protocol']
    if identity(fixed)!=document['identity'] or (saved.get('identity'),saved.get('completed_epoch'))!=(document['identity'],cap):
        raise ValueError('Wrong ROI evaluation checkpoint')
    if saved.get('fixed_final_epoch')!=cfg['roi_epochs'] or not isinstance(saved.get('model'),dict) or not saved['model']:
        raise ValueError('Incomplete or wrong fixed-schedule ROI checkpoint')
    if roi_protocol(cfg,cohort,stamps)!=fixed['static']:raise ValueError('ROI evaluation source/data/settings changed')
    return saved,fixed


def evaluate_pipeline(config,proposal_sources,roi_checkpoint=None,run_until=None):
    from roi_filament_core import RoiFilamentNet,prepare_image
    cfg=config_defaults(config);cap=cfg['roi_epochs'] if run_until is None else int(run_until)
    root=Path(cfg['output_root']);checkpoint=Path(roi_checkpoint or root/'roi'/f'epoch{cap:03d}.pt')
    cohort=load_cohort(cfg);stamps=input_image_hashes(cfg,cohort)
    saved,fixed=validate_roi_stage(cfg,cohort,stamps,checkpoint,cap)
    stems={Path(name).stem for name in cohort['validation_names']}
    sources={name:{stem:checked_boxes(rows) for stem,rows in props.items()} for name,props in proposal_sources.items()}
    if any(set(props)!=stems for props in sources.values()):raise ValueError('Detector sources must cover exact validation photos')
    evaluate_id=identity(dict(checkpoint=digest(checkpoint),static=fixed['static'],proposal_sources=sources,
                              settings={k:cfg[k] for k in ('eval_box_thresholds','eval_min_areas','mask_threshold','roi_min_quality','mask_nms_iou')}))
    folder=root/'evaluation'/f'roi{cap:03d}_{evaluate_id[:16]}';folder.mkdir(parents=True,exist_ok=True)
    methods={name:{f'box{threshold:g}_area{area}':{} for threshold in cfg['eval_box_thresholds'] for area in cfg['eval_min_areas']} for name in sources}
    oracle={f'area{area}':{} for area in cfg['eval_min_areas']};model=None;device=device_for(cfg['device']);configure_cuda()
    groups_by_stem={stem:[] for stem in stems}
    for sid in cohort['truth']:groups_by_stem[cohort['mapping'][sid]].append(sid)
    for number,name in enumerate(cohort['validation_names'],1):
        stem=Path(name).stem;path=folder/'records'/(stem+'.json.gz')
        if path.exists():
            record=read_record(path)
            if (record.get('identity'),record.get('image_sha256'))!=(evaluate_id,stamps[name]):raise ValueError('Wrong ROI prediction record')
        else:
            if model is None:
                model=RoiFilamentNet(encoder=cfg['roi_encoder'],encoder_weights=None).to(device);model.load_state_dict(saved['model']);model.eval()
            image=prepare_image(cv2.imread(str(Path(cfg['images'])/name),cv2.IMREAD_GRAYSCALE),cfg['preprocessing'])
            actual={source:infer_rois(model,image,props[stem],cfg,device) for source,props in sources.items()}
            oracle_rows={}
            for sid in groups_by_stem[stem]:
                boxes=[dict(box=[float(x),float(y),float(x+w),float(y+h)],score=1.) for x,y,w,h in mu.toBbox(cohort['truth'][sid])]
                oracle_rows[sid]=infer_rois(model,image,boxes,cfg,device)
            record=dict(identity=evaluate_id,image_sha256=stamps[name],actual=actual,oracle_per_rater=oracle_rows)
            write_record(path,record)
        for source in sources:
            candidates=decode_serialized_candidates(record['actual'][source])
            for threshold in cfg['eval_box_thresholds']:
                for area in cfg['eval_min_areas']:methods[source][f'box{threshold:g}_area{area}'][stem]=mask_predictions(candidates,threshold,area,cfg['mask_nms_iou'],cfg['roi_min_quality'])
        for sid,rows in record['oracle_per_rater'].items():
            candidates=decode_serialized_candidates(rows)
            for area in cfg['eval_min_areas']:oracle[f'area{area}'][sid]=mask_predictions(candidates,0.,area,cfg['mask_nms_iou'],cfg['roi_min_quality'])
        print(f'ROI full-image evaluation {number}/{len(cohort["validation_names"])} {name}',flush=True)
    if model is not None:del model;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    results={}
    for source,settings in methods.items():
        results[source]={}
        for setting,predictions in settings.items():
            write_predictions(folder/f'{source}_{setting}.csv',predictions)
            results[source][setting]=score_groups(cohort['truth'],cohort['mapping'],predictions)
            print(source,setting,results[source][setting]['score'],flush=True)
    oracle_results={setting:score_groups(cohort['truth'],cohort['mapping'],predictions,per_group=True) for setting,predictions in oracle.items()}
    report=dict(status='complete',scope='Repeated fixed91 development; no LB/freshOOF claim',identity=evaluate_id,
        roi_checkpoint_sha256=digest(checkpoint),roi_epochs_completed=cap,roi_fixed_final_epoch=cfg['roi_epochs'],
        roi_training_corpus_label=fixed['corpus_label'],actual_detector_results=results,
        gt_bbox_oracle=dict(scope='Uses each rater own GT boxes; diagnostic only, not deployable or submission PQ',results=oracle_results),
        detector_diagnostics={name:bbox_diagnostics(cohort['truth'],cohort['mapping'],proposals,cfg['roi_context']) for name,proposals in sources.items()},
        annotation_groups=len(cohort['truth']),physical_images=len(stems),GT=sum(map(len,cohort['truth'].values())),
        threshold_search=False,validation_selected_training_checkpoint=False,raw_candidates_before_quality_gate=True,
        prediction_csv_sha256={p.name:digest(p) for p in folder.glob('*.csv')})
    atomic_json(folder/'report.json',report);atomic_json(root/'latest_evaluation.json',dict(report=str(folder/'report.json'),identity=evaluate_id))
    return report


def export_results(config):
    root=Path(config['output_root']);destination=root/'exports';destination.mkdir(parents=True,exist_ok=True)
    common=[];records=[]
    for path in root.rglob('*'):
        if not path.is_file() or destination in path.parents or '.pending' in path.name:continue
        if path.name=='latest_results.json' or path.suffix not in ('.json','.csv','.py','.md','.txt','.gz'):continue
        (records if path.suffix=='.gz' else common).append((path,str(path.relative_to(root))))
    entries={name:dict(bytes=path.stat().st_size,sha256=digest(path)) for path,name in common+records}
    budget=26*1024*1024;base=sum(p.stat().st_size+len(n)*2+256 for p,n in common)+len(json.dumps(entries))*2+65536
    if base>budget//2:raise ValueError('Summary metadata exceeds export budget')
    parts=[];part=[];size=base
    for path,name in sorted(records):
        amount=path.stat().st_size+len(name)*2+512
        if base+amount>budget:raise ValueError('One record exceeds multipart budget')
        if part and size+amount>budget:parts.append(part);part=[];size=base
        part.append((path,name));size+=amount
    parts.append(part);export_id=identity(entries);archives=[]
    for index,part in enumerate(parts,1):
        path=destination/f'detection_roi_{export_id[:12]}_part{index:02d}.zip';pending=path.with_suffix('.pending')
        with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as z:
            for source,name in sorted(common+part):z.write(source,name)
            z.writestr('export_manifest.json',json.dumps(dict(export_id=export_id,part=index,parts_total=len(parts),entries=entries),indent=2))
        if pending.stat().st_size>28*1024*1024:raise ValueError('Export exceeds28MiB')
        pending.replace(path);archives.append(str(path))
    atomic_json(root/'latest_results.json',dict(archives=archives,parts=len(parts),export_id=export_id));return archives


def merge_training_proposals(first,second):
    return {stem:first.get(stem,[])+second.get(stem,[]) for stem in sorted(set(first)|set(second))}


def run(config,phase='all',roi_until=None,detr_until=None):
    cfg=config_defaults(config);roi_cap=cfg['roi_epochs'] if roi_until is None else int(roi_until);detr_cap=cfg['detr_epochs'] if detr_until is None else int(detr_until)
    if phase not in ('all','yolo_roi','compare_detr','evaluate'):raise ValueError('Invalid phase')
    if not 1<=roi_cap<=cfg['roi_epochs'] or not 1<=detr_cap<=cfg['detr_epochs']:raise ValueError('Requested stage outside fixed final schedule')
    root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    lock_path=Path('/tmp')/('detection_roi_'+identity(str(root.resolve()))[:16]+'.lock')
    with lock_path.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another cell is using this experiment in this runtime')
        if phase in ('compare_detr','evaluate') and not (root/'roi'/f'epoch{roi_cap:03d}.pt').is_file():
            raise FileNotFoundError('Complete the requested ROI epoch before this phase; no detector training was started')
        roi_identity=root/'roi/training_identity.json'
        if phase in ('all','yolo_roi') and roi_identity.exists():
            previous=json.loads(roi_identity.read_text())['protocol']['corpus_label']
            expected='mixed_yolo_detr_train' if phase=='all' else 'yolo_train_only'
            if previous!=expected:
                raise ValueError('Existing ROI training corpus is '+previous+'; use compare_detr/evaluate or a new output folder before changing training corpus')
        cohort=load_cohort(cfg);stamps=input_image_hashes(cfg,cohort)
        if phase in ('compare_detr','evaluate'):
            saved,_=validate_roi_stage(cfg,cohort,stamps,root/'roi'/f'epoch{roi_cap:03d}.pt',roi_cap)
            del saved
        result=dict(status='running',phase=phase,roi_fixed_final_epoch=cfg['roi_epochs'],detr_fixed_final_epoch=cfg['detr_epochs'],
                    actual_training_budget='all: oneDETR30 + oneROI30 bydefault; existingYOLO isnottrained',archives=[])
        atomic_json(root/'result.json',result)
        validation={'yolo':export_yolo_proposals(cfg,'validation',cohort,stamps)}
        train_proposals=None
        if phase in ('all','yolo_roi'):train_proposals=export_yolo_proposals(cfg,'train',cohort,stamps)
        if phase in ('all','compare_detr'):
            from filament_detr import train_detr,export_proposals
            result['detr_training']=train_detr(cfg,detr_cap);atomic_json(root/'result.json',result)
            validation['detr']=export_proposals(cfg,checkpoint=result['detr_training']['checkpoint'],role='validation')
            if phase=='all' and detr_cap<cfg['detr_epochs']:
                # Otherwise ROI10 would learn a DETR10 negative corpus, then
                # fail identity checks when DETR30 changes that corpus later.
                result['detector_diagnostics']={name:bbox_diagnostics(cohort['truth'],cohort['mapping'],rows,cfg['roi_context']) for name,rows in validation.items()}
                result['status']='detr_stage_complete_roi_not_started'
                result['reason']='Finish the fixed DETR schedule before constructing the immutable mixed ROI corpus'
                print('DETR stage complete. ROI has not started; finish DETR final epoch before ROI training.',flush=True)
                atomic_json(root/'result.json',result);result['archives']=export_results(cfg);atomic_json(root/'result.json',result)
                return result
            if phase=='all':train_proposals=merge_training_proposals(train_proposals,export_proposals(cfg,checkpoint=result['detr_training']['checkpoint'],role='train'))
        elif phase=='evaluate' and (root/'detr'/f'epoch{detr_cap:03d}.pt').is_file():
            from filament_detr import export_proposals
            validation['detr']=export_proposals(cfg,checkpoint=str(root/'detr'/f'epoch{detr_cap:03d}.pt'),role='validation')
        if phase in ('all','yolo_roi'):
            result['roi_training']=train_roi(cfg,train_proposals,roi_cap,'mixed_yolo_detr_train' if phase=='all' else 'yolo_train_only');atomic_json(root/'result.json',result)
        else:
            checkpoint=root/'roi'/f'epoch{roi_cap:03d}.pt'
            if not checkpoint.is_file():raise FileNotFoundError('Complete ROI stage first: '+str(checkpoint))
            result['roi_training']=dict(checkpoint=str(checkpoint),epochs_completed=roi_cap,fixed_final_epoch=cfg['roi_epochs'])
        result['evaluation']=evaluate_pipeline(cfg,validation,result['roi_training']['checkpoint'],roi_cap)
        complete=roi_cap==cfg['roi_epochs'] and (phase in ('yolo_roi','evaluate') or detr_cap==cfg['detr_epochs'])
        result['status']='complete' if complete else 'stage_complete_final_training_remaining'
        atomic_json(root/'result.json',result);result['archives']=export_results(cfg);atomic_json(root/'result.json',result)
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--phase',choices=('all','yolo_roi','compare_detr','evaluate'),default='all')
    p.add_argument('--roi-until',type=int);p.add_argument('--detr-until',type=int)
    args=p.parse_args();result=run(json.loads(Path(args.config).read_text()),args.phase,args.roi_until,args.detr_until)
    print(json.dumps({k:result.get(k) for k in ('status','phase','roi_training','detr_training','archives')},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
