"""Real-only native U-Net experiments with completed-epoch resume and fixed PQ.

One epoch is a declared crop-sampling budget, not a complete pass over photos.
No YOLO predictions, test labels, synthetic data, or old native weights are used.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
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
import random
import time
import zipfile

os.environ.setdefault('TORCH_CUDNN_V8_API_DISABLED', '1')
import cv2
import numpy as np
from pycocotools import mask as mu
import torch
from torch import nn
from torch.utils.data import DataLoader

DEFAULTS = dict(encoder='resnet34', head_mode='instance', epochs=30, crop=768,
    batch=2, accumulate=2, steps_per_epoch=256, lr=1e-4, weight_decay=.01,
    pretrained=True, seed=42, device='cuda', amp=True, preprocessing='limb',
    input_mean=.449, input_std=.226, threshold=.5, eval_min_areas=[150, 300],
    overlap=.25, tta=False, cache_root='/content/solar_native_instance_cache', strict=True)
FIXED_KEYS = tuple(k for k in DEFAULTS if k not in ('device', 'cache_root'))
SOURCE_FILES = ('native_instance_training.py', 'native_instance_model.py', 'native_instance_data.py')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.pending')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.pending')
    try:
        torch.save(value, temporary); temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def configure_cuda():
    """Keep ordinary cuDNN; route only CUDA depthwise conv through fallback."""
    torch.backends.cudnn.benchmark = False
    if not getattr(nn.Conv2d._conv_forward, '_solar_native_safe', False):
        original = nn.Conv2d._conv_forward
        def safe(self, inputs, weight, bias):
            if inputs.is_cuda and self.groups > 1 and self.groups == self.in_channels:
                with torch.backends.cudnn.flags(enabled=False):
                    return original(self, inputs, weight, bias)
            return original(self, inputs, weight, bias)
        safe._solar_native_safe = True
        nn.Conv2d._conv_forward = safe


def normalize_config(config):
    cfg = dict(DEFAULTS, **config)
    for name in ('epochs', 'crop', 'batch', 'accumulate', 'steps_per_epoch'):
        if not isinstance(cfg[name], int) or cfg[name] < 1:
            raise ValueError(name+' must be a positive integer')
    if cfg['crop'] % 32 or cfg['crop'] > 2048:
        raise ValueError('crop must be a multiple of32 and <=2048')
    if cfg['head_mode'] not in ('instance', 'semantic'):
        raise ValueError('Unknown head_mode')
    if cfg['encoder'] not in ('resnet18', 'resnet34', 'tu-convnext_tiny'):
        raise ValueError('Unknown encoder')
    if not 0 < cfg['lr'] < .1 or not 0 <= cfg['overlap'] < 1:
        raise ValueError('Invalid lr/overlap')
    if not 0 < cfg['threshold'] < 1 or cfg['input_std'] <= 0:
        raise ValueError('Invalid threshold/normalization')
    if not cfg['eval_min_areas'] or any(not isinstance(x, int) or x < 1 for x in cfg['eval_min_areas']):
        raise ValueError('Positive eval_min_areas required')
    if Path(cfg['cache_root']).resolve() == Path(cfg['output_root']).resolve():
        raise ValueError('Local cache and persistent output must be separate')
    return cfg


def runtime_versions():
    result = {}
    for package in ('torch', 'torchvision', 'numpy', 'scipy', 'segmentation-models-pytorch', 'timm', 'pycocotools'):
        result[package] = importlib.metadata.version(package)
    return dict(result, opencv=cv2.__version__, torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())


def source_hashes():
    return {name: digest(Path(__file__).with_name(name)) for name in SOURCE_FILES}


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(value):
    random.setstate(value['python']); np.random.set_state(value['numpy']); torch.set_rng_state(value['torch'])
    if value['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value['cuda'])


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def cpu_copy(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_copy(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_copy(v) for v in value)
    return value


def device_for(value):
    result = torch.device('cuda:0' if str(value) == '0' else value)
    if result.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a Colab T4 GPU runtime')
    return result


def input_image_hashes(cfg, cohort):
    names = sorted({row['file_name'] for row in cohort['train_groups']} | set(cohort['validation_names']))
    result = {}
    for number, name in enumerate(names, 1):
        result[name] = digest(Path(cfg['images'])/name)
        if number % 100 == 0 or number == len(names):
            print(f'원본 이미지 해시 검사 {number}/{len(names)}', flush=True)
    atomic_json(Path(cfg['output_root'])/'source_images.json', result)
    return result


def fixed_identity(cfg, cohort):
    return dict(settings={key: cfg[key] for key in FIXED_KEYS}, sources=source_hashes(),
                input_metadata=cohort['input_sha256'], image_sha256=input_image_hashes(cfg, cohort),
                runtime=runtime_versions(), selection='fixed completed epoch; no validation checkpoint selection',
                initial_weights='generic ImageNet encoder' if cfg['pretrained'] else 'random initialization',
                epoch_unit='steps_per_epoch minibatches of sampled crops', bitwise_GPU_trajectory_guaranteed=False)


def train(config, run_until=None):
    from native_instance_data import load_cohort, NativeInstanceDataset
    from native_instance_model import NativeInstanceNet, loss_fn
    cfg = normalize_config(config); cap = cfg['epochs'] if run_until is None else int(run_until)
    if not 1 <= cap <= cfg['epochs']:
        raise ValueError('run-until must be within fixed final epochs')
    configure_cuda(); device = device_for(cfg['device'])
    cohort = load_cohort(cfg); fixed = fixed_identity(cfg, cohort); run_id = identity(fixed)
    root = Path(cfg['output_root']); root.mkdir(parents=True, exist_ok=True)
    identity_path = root/'training_identity.json'; last = root/'last.pt'; stage = root/f'epoch{cap:03d}.pt'
    if identity_path.exists() and json.loads(identity_path.read_text())['identity'] != run_id:
        raise ValueError('Training data/code/settings/runtime changed; use a new output folder')
    if last.exists() and not identity_path.exists():
        raise ValueError('Unidentified last.pt preserved; cannot resume')
    atomic_json(identity_path, dict(identity=run_id, protocol=fixed))
    if stage.exists():
        completed = torch.load(stage, map_location='cpu', weights_only=False)
        if (completed.get('identity'), completed.get('completed_epoch')) != (run_id, cap):
            raise ValueError('Saved stage has a different identity/epoch')
        return dict(status='complete' if cap == cfg['epochs'] else 'stage_complete',
                    epochs_completed=cap, fixed_final_epoch=cfg['epochs'], checkpoint=str(stage),
                    checkpoint_sha256=digest(stage), identity=run_id)
    saved = torch.load(last, map_location='cpu', weights_only=False) if last.exists() else None
    if saved and (saved.get('identity') != run_id or saved.get('completed_epoch', -1) > cap):
        raise ValueError('Cannot restore a wrong or later checkpoint into this stage')
    seed_everything(cfg['seed'])
    model = NativeInstanceNet(encoder=cfg['encoder'], encoder_weights='imagenet' if cfg['pretrained'] and saved is None else None,
                              head_mode=cfg['head_mode']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: .1+.9*(1+math.cos(math.pi*epoch/cfg['epochs']))/2)
    scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg['amp'] and device.type == 'cuda'))
    history = []
    start = 0
    if saved:
        required = {'optimizer', 'scheduler', 'scaler', 'rng', 'model', 'history'}
        if not required <= set(saved): raise ValueError('Full completed-epoch resume state required')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler']); restore_rng(saved['rng'])
        history = saved['history']; start = saved['completed_epoch']
        if len(history) != start or (start and history[-1]['epoch'] != start):
            raise ValueError('Completed checkpoint history is inconsistent')
    dataset = NativeInstanceDataset(cohort['train_groups'], cfg['images'], crop=cfg['crop'],
        steps_per_epoch=cfg['steps_per_epoch'], batch_size=cfg['batch'], seed=cfg['seed'],
        training=True, preprocessing=cfg['preprocessing'],
        cache_dir=Path(cfg['cache_root'])/'source_images', source_sha256=fixed['image_sha256'])
    amp = bool(cfg['amp'] and device.type == 'cuda')
    print(f"단일 {cfg['encoder']} U-Net/{cfg['head_mode']}: 완료{start}→이번{cap}, 실제 최종{cfg['epochs']}epoch", flush=True)
    print(f"epoch당 {cfg['steps_per_epoch']} minibatches = {cfg['steps_per_epoch']*cfg['batch']} sampled crops; 전체 사진1회 통과 단위가 아닙니다.", flush=True)
    for epoch in range(start, cap):
        seed_everything(cfg['seed']+10007*epoch); dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(cfg['seed']+10007*epoch)
        loader = DataLoader(dataset, batch_size=cfg['batch'], shuffle=False, num_workers=0,
                            generator=generator, pin_memory=device.type == 'cuda')
        if len(loader) != cfg['steps_per_epoch']:
            raise ValueError('Dataset length differs from declared minibatch budget')
        model.train(); optimizer.zero_grad(set_to_none=True)
        total = {}; attempts = successes = batches = 0; started = time.monotonic()
        for index, (inputs, instances) in enumerate(loader):
            inputs = (inputs.to(device, non_blocking=True)-cfg['input_mean'])/cfg['input_std']
            # Target geometry is constructed on CPU; avoid moving IDs to GPU
            # only for loss_fn to copy them straight back to NumPy.
            group_start = (index//cfg['accumulate'])*cfg['accumulate']
            group_length = min(cfg['accumulate'], len(loader)-group_start)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(inputs); loss, terms = loss_fn(logits, instances, head_mode=cfg['head_mode'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; previous completed epoch retained')
            scaler.scale(loss/group_length).backward()
            for key, value in terms.items(): total[key] = total.get(key, 0.)+float(value.detach().cpu() if isinstance(value,torch.Tensor) else value)
            total['total_loss'] = total.get('total_loss',0.)+float(loss.detach().cpu())
            batches += 1
            if (index+1) % cfg['accumulate'] == 0 or index+1 == len(loader):
                before = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                attempts += 1; successes += int(scaler.get_scale() >= before)
                optimizer.zero_grad(set_to_none=True)
            if (index+1) % 32 == 0 or index+1 == len(loader):
                print(f"epoch{epoch+1}/{cfg['epochs']} train{index+1}/{len(loader)} loss={total['total_loss']/batches:.5f}", flush=True)
        if batches != len(loader): raise RuntimeError('Incomplete epoch is not a completed checkpoint')
        scheduler.step()
        history.append(dict(epoch=epoch+1, losses={key:value/batches for key,value in total.items()},
                            minibatches=batches, sampled_crops=batches*cfg['batch'], optimizer_attempts=attempts,
                            optimizer_updates=successes, amp_skipped_updates=attempts-successes,
                            next_lr=scheduler.get_last_lr(), seconds=time.monotonic()-started))
        payload = dict(schema_version=1, identity=run_id, completed_epoch=epoch+1, fixed_final_epoch=cfg['epochs'],
                       config={key:cfg[key] for key in DEFAULTS}, model=cpu_copy(model.state_dict()),
                       optimizer=cpu_copy(optimizer.state_dict()), scheduler=scheduler.state_dict(),
                       scaler=scaler.state_dict(), rng=capture_rng(), history=history)
        atomic_torch(last,payload); atomic_json(root/'history.json',history)
        atomic_json(root/'progress.json',dict(completed_epoch=epoch+1,fixed_final_epoch=cfg['epochs'],identity=run_id))
    if start == cap:
        payload = saved
    compact = dict(payload)
    for key in ('optimizer','scheduler','scaler','rng'): compact[key] = None
    atomic_torch(stage,compact)
    del model; gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()
    return dict(status='complete' if cap == cfg['epochs'] else 'stage_complete', epochs_completed=cap,
                fixed_final_epoch=cfg['epochs'], checkpoint=str(stage), checkpoint_sha256=digest(stage),identity=run_id)


def tile_positions(length,crop,overlap):
    if crop >= length:return [0]
    stride=max(1,int(round(crop*(1-overlap))))
    return sorted(set(range(0,length-crop+1,stride))|{length-crop})


def predict_image(model,pixels,cfg,device):
    """Shared deterministic preprocessing and normalization for train/inference."""
    from native_instance_data import preprocess_image
    corrected=preprocess_image(pixels,cfg['preprocessing'])
    if corrected.dtype != np.float32 or not np.isfinite(corrected).all() or corrected.min()<0 or corrected.max()>1:
        raise ValueError('Shared preprocessing must return float32 pixels in[0,1]')
    height,width=corrected.shape;crop=cfg['crop'];padded=np.pad(corrected,((0,max(0,crop-height)),(0,max(0,crop-width))),mode='reflect')
    sum_probability=np.zeros((3,*padded.shape),np.float32);weight=np.zeros(padded.shape,np.float32)
    amp=bool(cfg['amp'] and device.type=='cuda')
    model.eval()
    with torch.inference_mode():
        for top in tile_positions(padded.shape[0],crop,cfg['overlap']):
            for left in tile_positions(padded.shape[1],crop,cfg['overlap']):
                patch=padded[top:top+crop,left:left+crop]
                inputs=torch.from_numpy(np.ascontiguousarray(patch)).float()[None,None].to(device)
                # Shared preprocessing and dataset both return float32 in[0,1].
                inputs=(inputs-cfg['input_mean'])/cfg['input_std']
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                    probabilities=model(inputs).sigmoid()
                    if cfg['tta']:
                        probabilities=(probabilities+model(inputs.flip(-1)).sigmoid().flip(-1))/2
                sum_probability[:,top:top+crop,left:left+crop]+=probabilities[0].float().cpu().numpy()
                weight[top:top+crop,left:left+crop]+=1
    if not np.all(weight>0):raise AssertionError('Tile grid did not cover the full image')
    result=(sum_probability/weight)[...,:height,:width]
    if not np.isfinite(result).all():raise FloatingPointError('Invalid inference probabilities')
    return result


def score_predictions(truth,mapping,predictions):
    rows=[];small_total=small_hits=0
    for sid,gt in sorted(truth.items()):
        pred=predictions[mapping[sid]]
        if not gt:continue
        iou=np.asarray(mu.iou(gt,pred,[0]*len(pred))) if pred else np.zeros((len(gt),0))
        hits=iou>.5
        row=dict(group_id=sid,image=mapping[sid],tp_pairs=int(hits.sum()),
                 fp=int((~hits.any(0)).sum()),fn=int((~hits.any(1)).sum()),matched_iou_sum=float(iou[hits].sum()))
        rows.append(row)
        small=np.asarray(mu.area(gt))<670;small_total+=int(small.sum());small_hits+=int((small&hits.any(1)).sum())
    total={key:sum(row[key] for row in rows) for key in ('tp_pairs','fp','fn','matched_iou_sum')}
    den=total['tp_pairs']+.5*(total['fp']+total['fn'])
    total.update(pq=total['matched_iou_sum']/den if den else 0.,sq=total['matched_iou_sum']/total['tp_pairs'] if total['tp_pairs'] else 0.,
                 rq=total['tp_pairs']/den if den else 0.,physical_images=len(set(mapping.values())),annotation_groups=len(rows),
                 small_GT=dict(total=small_total,matched=small_hits,recall=small_hits/small_total if small_total else None))
    return dict(score=total,per_group=rows)


def write_csv(path,predictions):
    with Path(path).open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['filament_id','segmentation_rle'])
        for stem,masks in predictions.items():
            for index,rle in enumerate(masks):
                counts=rle['counts'].decode('ascii') if isinstance(rle['counts'],bytes) else rle['counts']
                writer.writerow([f'{stem}_{index}',counts])


def atomic_gzip_json(path,value):
    def ready(x):
        if isinstance(x,bytes):return x.decode('ascii')
        if isinstance(x,dict):return {k:ready(v) for k,v in x.items()}
        if isinstance(x,list):return [ready(v) for v in x]
        return x
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);pending=path.with_name(path.name+'.pending')
    try:pending.write_bytes(gzip.compress(json.dumps(ready(value),separators=(',',':'),allow_nan=False).encode()));pending.replace(path)
    finally:pending.unlink(missing_ok=True)


def evaluate(config,checkpoint=None,run_until=None):
    from native_instance_data import load_cohort
    from native_instance_model import NativeInstanceNet,decode_instances
    cfg=normalize_config(config);cap=cfg['epochs'] if run_until is None else int(run_until)
    if not 1 <= cap <= cfg['epochs']:raise ValueError('run-until must be within fixed final epochs')
    root=Path(cfg['output_root']);checkpoint=Path(checkpoint or root/f'epoch{cap:03d}.pt')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    fixed=json.loads((root/'training_identity.json').read_text())
    if (saved.get('identity'),saved.get('completed_epoch'))!=(fixed['identity'],cap):
        raise ValueError('Wrong evaluation stage or experiment identity')
    if saved['fixed_final_epoch']!=cfg['epochs']:raise ValueError('Wrong fixed final schedule')
    cohort=load_cohort(cfg)
    # Evaluate-only must also reject source, metadata and input pixel changes.
    if identity(fixed_identity(cfg,cohort))!=fixed['identity']:
        raise ValueError('Evaluation data/code/settings/runtime differ from saved training')
    device=device_for(cfg['device']);configure_cuda()
    evaluation_id=identity(dict(training=saved['identity'],checkpoint_sha256=digest(checkpoint),sources=source_hashes(),
                                threshold=cfg['threshold'],areas=cfg['eval_min_areas'],overlap=cfg['overlap'],tta=cfg['tta']))
    folder=root/'validation'/f'epoch{cap:03d}_{evaluation_id[:12]}';folder.mkdir(parents=True,exist_ok=True)
    methods=['connected','watershed'] if cfg['head_mode']=='instance' else ['connected']
    keys=[f'{mode}_area{area}' for mode in methods for area in cfg['eval_min_areas']]
    predictions={key:{} for key in keys};local=Path(cfg['cache_root'])/evaluation_id;local.mkdir(parents=True,exist_ok=True)
    model=None
    for number,name in enumerate(cohort['validation_names'],1):
        stem=Path(name).stem;path=Path(cfg['images'])/name;stamp=fixed['protocol']['image_sha256'][name]
        record_path=folder/'records'/(stem+'.json.gz');record=None
        if record_path.exists():
            record=json.loads(gzip.decompress(record_path.read_bytes()))
            expected=record.get('rles_sha256');actual=identity(record.get('predictions'))
            if (record.get('evaluation_identity'),record.get('image_sha256'),expected)!=(evaluation_id,stamp,actual):
                raise ValueError('Invalid saved evaluation record: '+stem)
        if record is None:
            if model is None:
                model=NativeInstanceNet(encoder=cfg['encoder'],encoder_weights=None,head_mode=cfg['head_mode']).to(device)
                model.load_state_dict(saved['model']);model.eval()
            local_path=local/(stem+'.npz')
            probabilities=None
            if local_path.exists():
                with np.load(local_path,allow_pickle=False) as cache:
                    if str(cache['image_sha256'])!=stamp:raise ValueError('Local probability image identity changed')
                    stored=cache['probability']
                    if str(cache['probability_sha256'])!=hashlib.sha256(stored.tobytes()).hexdigest():
                        raise ValueError('Local probability cache checksum changed')
                    probabilities=stored.astype(np.float32)
            if probabilities is None:
                pixels=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
                if pixels is None:raise ValueError('Unreadable validation image: '+str(path))
                if cfg['strict'] and pixels.shape!=(2048,2048):raise ValueError('Original2048 image required')
                probabilities=predict_image(model,pixels,cfg,device)
                # Decode the same float16 representation on first run and resume.
                probabilities=probabilities.astype(np.float16).astype(np.float32)
                pending=local_path.with_name(local_path.name+'.pending')
                with pending.open('wb') as stream:np.savez_compressed(stream,probability=probabilities.astype(np.float16),image_sha256=stamp,probability_sha256=hashlib.sha256(probabilities.astype(np.float16).tobytes()).hexdigest())
                pending.replace(local_path)
            selected={}
            for mode in methods:
                for area in cfg['eval_min_areas']:
                    # Split acceptance depends on min_area; independently decode
                    # each area so small children can fall back to their parent.
                    masks=decode_instances(probabilities,threshold=cfg['threshold'],min_area=area,mode=mode)
                    if masks and int(mu.area(mu.merge(masks)))!=sum(map(int,mu.area(masks))):raise AssertionError('Decoded instance overlap')
                    selected[f'{mode}_area{area}']=masks
            serialized={key:[dict(size=list(r['size']),counts=r['counts'].decode('ascii') if isinstance(r['counts'],bytes) else r['counts']) for r in masks] for key,masks in selected.items()}
            record=dict(evaluation_identity=evaluation_id,image_sha256=stamp,predictions=serialized,rles_sha256=identity(serialized))
            atomic_gzip_json(record_path,record)
        for key in keys:
            predictions[key][stem]=[dict(size=r['size'],counts=r['counts'].encode('ascii') if isinstance(r['counts'],str) else r['counts']) for r in record['predictions'][key]]
        print(f'validation [{number}/{len(cohort["validation_names"])}] {name}',flush=True)
    if model is not None:del model;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    results={}
    for key,pred in predictions.items():
        write_csv(folder/(key+'.csv'),pred);results[key]=score_predictions(cohort['truth'],cohort['mapping'],pred)
        print(key,results[key]['score'],flush=True)
    report=dict(status='complete',scope='Original fixed616/91 development; no fresh707-image OOF or public LB claim',
                evaluation_identity=evaluation_id,epochs_completed=cap,fixed_final_epoch=cfg['epochs'],
                model=dict(encoder=cfg['encoder'],head_mode=cfg['head_mode']),results=results,
                primary_min_area=300,threshold_search=False,semantic_watershed_disabled=cfg['head_mode']=='semantic',
                validation_selects_training_checkpoint=False,checkpoint_sha256=digest(checkpoint),
                evaluation_root=str(folder),prediction_csv_sha256={p.name:digest(p) for p in folder.glob('*.csv')})
    atomic_json(folder/'report.json',report);atomic_json(root/'latest_evaluation.json',dict(report=str(folder/'report.json'),identity=evaluation_id))
    return report


def export_results(config):
    root=Path(config['output_root']);destination=root/'exports';destination.mkdir(parents=True,exist_ok=True)
    common=[];records=[]
    for path in root.rglob('*'):
        if not path.is_file() or destination in path.parents or path.name.endswith('.pending'):continue
        relative=path.relative_to(root)
        if path.suffix not in ('.json','.csv','.py','.md','.txt','.gz'):continue
        if path.name=='latest_results.json':continue
        (records if path.suffix=='.gz' else common).append((path,str(relative)))
    entries={name:dict(bytes=path.stat().st_size,sha256=digest(path)) for path,name in common+records}
    common_size=sum(path.stat().st_size+len(name)*2+256 for path,name in common)
    budget=26*1024*1024;base_size=common_size+len(json.dumps(entries))*2+65536
    if base_size>budget//2:raise ValueError('Summary export metadata exceeds budget')
    parts=[];current=[];size=base_size
    for path,name in sorted(records):
        amount=path.stat().st_size+len(name)*2+512
        if base_size+amount>budget:raise ValueError('One result record exceeds multipart budget')
        if current and size+amount>budget:parts.append(current);current=[];size=base_size
        current.append((path,name));size+=amount
    parts.append(current);export_id=identity(entries);archives=[]
    for index,part in enumerate(parts,1):
        archive=destination/f'native_instance_{export_id[:12]}_part{index:02d}.zip';pending=archive.with_suffix('.pending')
        with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as z:
            for path,name in sorted(common+part):z.write(path,name)
            z.writestr('export_manifest.json',json.dumps(dict(export_id=export_id,part=index,parts_total=len(parts),entries=entries),indent=2))
        if pending.stat().st_size>28*1024*1024:raise ValueError('Export part exceeds28MiB')
        pending.replace(archive);archives.append(str(archive))
    atomic_json(root/'latest_results.json',dict(archives=archives,export_id=export_id,parts=len(archives)))
    return archives


def run(config,phase='all',run_until=None):
    cfg=normalize_config(config);cap=cfg['epochs'] if run_until is None else int(run_until)
    if phase not in ('all','train','evaluate'):raise ValueError('Invalid phase')
    if not 1 <= cap <= cfg['epochs']:raise ValueError('run-until must be within fixed final epochs')
    root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    lock_path=Path('/tmp')/('native_instance_'+identity(str(root.resolve()))[:16]+'.lock')
    with lock_path.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another cell is using this experiment in this runtime')
        result=dict(status='running',phase=phase,epochs_completed=None,fixed_final_epoch=cfg['epochs'],archives=[])
        atomic_json(root/'result.json',result)
        if phase!='evaluate':result.update(train(cfg,cap))
        else:
            checkpoint=root/f'epoch{cap:03d}.pt'
            result.update(checkpoint=str(checkpoint),epochs_completed=cap)
        if phase in ('all','evaluate'):
            result['evaluation']=evaluate(cfg,result['checkpoint'],cap)
        result['status']='complete' if cap==cfg['epochs'] else 'stage_complete_final_training_remaining'
        if phase=='train':result['status']='training_complete_evaluation_pending' if cap==cfg['epochs'] else 'training_stage_complete'
        atomic_json(root/'result.json',result)
        result['archives']=export_results(cfg);atomic_json(root/'result.json',result)
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--phase',choices=('all','train','evaluate'),default='all');p.add_argument('--run-until',type=int)
    args=p.parse_args();config=json.loads(Path(args.config).read_text());result=run(config,args.phase,args.run_until)
    print(json.dumps({key:result.get(key) for key in ('status','epochs_completed','fixed_final_epoch','checkpoint','archives')},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
