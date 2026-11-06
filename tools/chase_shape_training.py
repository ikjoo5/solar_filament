"""Fresh best84-recipe training with GONG + CHASE-shape synthetic images.

The original best84 completed 100 epochs and selected epoch 84. This runner starts
from an ImageNet encoder/new decoder, not from that solar checkpoint. T4 batch 1
and accumulation 4 preserve 400 crops and 100 updates per epoch; FP16 replaces BF16.
The user-requested synthetic probability is 50%, versus 20% in original best84.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import zipfile

os.environ.setdefault('TORCH_CUDNN_V8_API_DISABLED', '1')
import cv2
import numpy as np
from pycocotools import mask as mu
import torch

import solar_a100_train_standalone as base
from chase_best84_training import configure_cuda
from chase_independent_decode import _rle

VERSION = 'chase_shape_best84_recipe_v1'
BASELINE_SHA = 'ae6b209f72355fd52d2cb07182e69028e97c703a1977ad1e7b38b27cf8cebe69'
INPUT_SHA = {
    'annotations': '5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1',
    'train_split': 'aa0733424815731c5c108ada91e45525eb17ca3a42b377bf19e3089d86aede7e',
    'val_split': '54c6a9aac5a3b70b3ecc6721109a8b0fe8ec020e90cb56e3848fccc92eadc61f',
}
DEFAULTS = dict(encoder='tu-convnext_tiny', pretrained=True, epochs=100,
    crop=1024, batch=1, accumulate=4, steps_per_epoch=400, lr=1e-4,
    weight_decay=.01, minimum_lr_ratio=.05, clip_grad_norm=1., spine_weight=.1,
    synthetic_ratio=.5, warmup_epochs=3, real_finish_epochs=6,
    include_empty_synthetic=True, seed=42, workers=0, amp_dtype='float16',
    preprocessing='limb_only', validate_every=2, thresholds=[.35,.5,.65],
    min_areas=[100,200,300], overlap=.25, validation_tile=1024, infer_batch=1, device='cuda',
    initialization='imagenet_encoder_new_decoder')
SOURCE_FILES = ('chase_shape_training.py', 'chase_shape_synthesis.py',
    'solar_a100_train_standalone.py', 'chase_best84_training.py',
    'chase_transfer_data.py', 'targeted_synthetic_data.py',
    'chase_independent_decode.py')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def serial(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [serial(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(serial(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending')
    try:
        temporary.write_text(json.dumps(serial(value),indent=2,allow_nan=False)+'\n')
        temporary.replace(path)
    finally: temporary.unlink(missing_ok=True)


def write_envelope(path, identity, payload):
    atomic_json(path,dict(identity=identity,payload=payload,payload_sha256=fingerprint(payload)))


def read_envelope(path, identity):
    value=json.loads(Path(path).read_text())
    if value.get('identity')!=identity or fingerprint(value.get('payload'))!=value.get('payload_sha256'):
        raise ValueError('Cached metadata identity/checksum differs: '+str(path))
    return value['payload']


class _HashWriter:
    def __init__(self, stream): self.stream,self.hash=stream,hashlib.sha256()
    def write(self, value):
        result=self.stream.write(value)
        self.hash.update(value)
        return result
    def flush(self): return self.stream.flush()


def save_checkpoint(path, payload):
    """One immutable generation; pointer publication is a separate commit."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending')
    try:
        with temporary.open('wb') as stream:
            writer=_HashWriter(stream)
            torch.save(payload,writer)
            writer.flush()
            checksum=writer.hash.hexdigest()
        temporary.replace(path)
    finally: temporary.unlink(missing_ok=True)
    return dict(path=str(path.relative_to(path.parent.parent)),sha256=checksum)


def load_checkpoint(output, record):
    path=(Path(output)/record['path']).resolve()
    if not path.is_relative_to(Path(output).resolve()) or digest(path)!=record['sha256']:
        raise ValueError('Checkpoint path/checksum differs: '+str(path))
    return torch.load(path,map_location='cpu',weights_only=True)


def normalize_config(value):
    cfg=dict(DEFAULTS,**value)
    differences={k:cfg[k] for k in DEFAULTS if cfg[k]!=DEFAULTS[k]}
    if differences:
        raise ValueError('This entry fixes the declared best84-plus-50%-synthetic recipe; changed settings: '+str(differences))
    if cfg.get('initial_checkpoint') or cfg.get('resume_dir'):
        raise ValueError('Fresh ImageNet initialization only; resume is automatic in output_root/train.')
    for key in ('images','annotations','train_split','val_split','chase_root','chase_manifest','cache_root','output_root'):
        if not cfg.get(key): raise ValueError('Missing configuration path: '+key)
        cfg[key]=str(Path(cfg[key]).expanduser().resolve())
    if Path(cfg['cache_root']).is_relative_to(Path('/content/drive')):
        raise ValueError('Place the image/cache directory on local Colab disk, not Drive.')
    for key in ('baseline_checkpoint','manual_exclusions','clean_background','test_images'):
        if cfg.get(key): cfg[key]=str(Path(cfg[key]).expanduser().resolve())
    if cfg.get('clean_backgrounds'):
        cfg['clean_backgrounds']=[str(Path(p).expanduser().resolve()) for p in cfg['clean_backgrounds']]
    cfg.update(samples_per_epoch=400,effective_batch=4,updates_per_epoch=100,
        comparison='Best84 architecture/optimizer/schedule; synthetic probability raised from 0.2 to 0.5, new synthesis and T4 execution.',
        original_synthetic_ratio=.2,
        sampling='400 random image-first crops per epoch; not a 616-image full pass.',
        loss='Foreground BCE+Dice +0.1 balanced genuine-spine BCE; synthetic spine loss disabled.')
    return cfg


class VerifiedImageStore:
    """Hash-bound limb cache, independent of changing source file timestamps."""
    def __init__(self, images, cache, expected_hashes):
        self.images,self.cache=Path(images),Path(cache)
        self.hashes=dict(expected_hashes)
        self.cache.mkdir(parents=True,exist_ok=True)
        self.preprocessing_sha=digest(base.__file__)
    def load(self, filename):
        if filename not in self.hashes or Path(filename).name!=filename:
            raise ValueError('Unknown image in verified image store')
        key=fingerprint(dict(source=self.hashes[filename],preprocessing=self.preprocessing_sha))
        target=self.cache/(key+'.png'); metadata=self.cache/(key+'.json')
        if target.exists() and metadata.exists():
            saved=read_envelope(metadata,key)
            if digest(target)!=saved['png_sha256']:
                raise ValueError('Processed image cache checksum differs: '+str(target))
            image=cv2.imread(str(target),cv2.IMREAD_GRAYSCALE)
        else:
            source=self.images/filename
            if digest(source)!=self.hashes[filename]:
                raise ValueError('Original image changed after preflight: '+str(source))
            raw=cv2.imread(str(source),cv2.IMREAD_GRAYSCALE)
            if raw is None or raw.shape!=(2048,2048) or raw.dtype!=np.uint8:
                raise ValueError('Expected native2048 grayscale image: '+str(source))
            image=base.limb(raw)
            pending=target.with_name(key+'.pending.png')
            if not cv2.imwrite(str(pending),image): raise OSError('Image cache write failed')
            pending.replace(target)
            write_envelope(metadata,key,dict(png_sha256=digest(target)))
        if image is None or image.shape!=(2048,2048) or image.dtype!=np.uint8:
            raise ValueError('Malformed processed image cache')
        return image


class RLEMixedCropDataset(base.MixedCropDataset):
    """Exact original sampler/augmentation; canonical RLE synthetic targets."""
    def _domain(self,data,allowed,synthetic,exclude_blank):
        if synthetic:
            for ann in data['annotations']:
                encoded=_rle(ann['segmentation'])
                area=int(mu.area(encoded)); bbox=list(map(float,mu.toBbox(encoded)))
                if area<=0 or area!=ann.get('area') or not np.allclose(bbox,ann.get('bbox',[]),rtol=0,atol=0):
                    raise ValueError('Canonical synthetic mask area/bbox differs')
        return super()._domain(data,allowed,synthetic,exclude_blank)
    def _target(self,domain_name,group):
        domain=self.domains[domain_name]; anns=domain['annotations'].get(group,[])
        key=fingerprint(dict(version=VERSION,domain=domain_name,image=domain['images'][group],annotations=anns))
        image_path=self.cache/(key+'.png') if self.cache else None
        meta_path=self.cache/(key+'.json') if self.cache else None
        if image_path is not None and image_path.exists() and meta_path.exists():
            record=read_envelope(meta_path,key)
            if digest(image_path)!=record['png_sha256']:
                raise ValueError('Target cache checksum differs')
            target=cv2.imread(str(image_path),cv2.IMREAD_UNCHANGED)
            if target is None or target.shape!=(2048,2048,3) or target.dtype!=np.uint8 or not np.isin(target,[0,1]).all():
                raise ValueError('Malformed target cache')
            if not np.all(target[:,:,2]==target[0,0,2]): raise ValueError('Invalid sample spine validity')
            return target[:,:,0],target[:,:,1],bool(target[0,0,2])
        fg=np.zeros((2048,2048),np.uint8); spine=np.zeros_like(fg)
        valid=not domain['synthetic']
        for ann in anns:
            if domain['synthetic']:
                fg |= mu.decode(_rle(ann['segmentation']))
            else:
                fg |= base.annotation_mask(ann,False)
                xy=base.genuine_spine(ann)
                if xy is None: raise ValueError('Missing genuine GONG spine')
                cv2.polylines(spine,[np.rint(xy).astype(np.int32)],False,1,thickness=3)
        spine &= fg
        if image_path is not None:
            target=np.stack([fg,spine,np.full_like(fg,int(valid))],axis=-1)
            pending=image_path.with_name(key+'.pending.png')
            if not cv2.imwrite(str(pending),target,[cv2.IMWRITE_PNG_COMPRESSION,3]):
                raise OSError('Target cache write failed')
            pending.replace(image_path)
            write_envelope(meta_path,key,dict(png_sha256=digest(image_path)))
        return fg,spine,valid


def runtime_identity(device):
    value=dict(python=sys.version.split()[0],device=str(device),torch=torch.__version__,
        cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        numpy=np.__version__,opencv=cv2.__version__)
    for name in ('segmentation-models-pytorch','timm','scipy','pycocotools'):
        try:value[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:value[name]='absent'
    if device.type=='cuda':
        value['gpu']=torch.cuda.get_device_name(device)
        value['capability']=list(torch.cuda.get_device_capability(device))
    return value


def source_identity(cfg,prepared,device):
    source={key:digest(cfg[key]) for key in INPUT_SHA}
    if source!=INPUT_SHA: raise ValueError('Frozen GONG annotations/splits differ')
    code={name:digest(Path(__file__).with_name(name)) for name in SOURCE_FILES}
    image_hashes={}
    data=json.loads(Path(cfg['annotations']).read_text())
    for index,name in enumerate(sorted({r['file_name'] for r in data['images']}),1):
        if Path(name).name!=name:raise ValueError('Unexpected annotation image path')
        image_hashes[name]=digest(Path(cfg['images'])/name)
        if index%100==0: print(f'GONG source verification {index}/707',flush=True)
    if len(image_hashes)!=707:raise ValueError('Expected707 GONG source observations')
    settings={key:cfg[key] for key in DEFAULTS}
    identity=dict(version=VERSION,settings=settings,sources=source,code=code,
        synthesis_identity=prepared['identity'],synthetic_annotations_sha256=digest(prepared['annotations_path']),
        source_images=image_hashes,runtime=runtime_identity(device))
    return identity,image_hashes


def prepare_dataset(cfg,prepared,image_hashes):
    data=json.loads(Path(cfg['annotations']).read_text())
    base.TRAIN=Path(cfg['train_split']);base.VAL=Path(cfg['val_split']);base.IMAGES=Path(cfg['images'])
    train_ids,val_names,base.GT=base.prepare_split(data)
    synthetic=json.loads(Path(prepared['annotations_path']).read_text())
    syn_hashes={}
    for row in synthetic['images']:
        name=row['file_name']
        if Path(name).name!=name:raise ValueError('Unsafe synthetic filename')
        syn_hashes[name]=digest(Path(prepared['images_dir'])/name)
    if not syn_hashes:raise ValueError('No synthetic images prepared')
    cache=Path(cfg['cache_root'])/'native_training'/prepared['identity']
    real_store=VerifiedImageStore(cfg['images'],cache/'real_images',image_hashes)
    syn_store=VerifiedImageStore(prepared['images_dir'],cache/'synthetic_images',syn_hashes)
    dataset=RLEMixedCropDataset(data,train_ids,real_store,synthetic, syn_store,
        crop=cfg['crop'],length=cfg['batch']*cfg['steps_per_epoch'],seed=cfg['seed'],
        synthetic_ratio=cfg['synthetic_ratio'],warmup_epochs=cfg['warmup_epochs'],
        real_finish_epochs=cfg['real_finish_epochs'],total_epochs=cfg['epochs'],
        cache_dir=cache/'targets',include_empty_synthetic=True,augment=True)
    return dataset,real_store,val_names


def _publish_results(root,output,state,cfg,identity,prepared=None):
    status='complete' if state['epoch']==cfg['epochs'] else 'paused'
    payload=dict(status=status,epochs_completed=state['epoch'],epochs_target=cfg['epochs'],
        best_epoch=state['best_epoch'],best_development_pq=state['best_pq'],
        initialization=cfg['initialization'],scope='Fixed development validation, not public LB.',
        original_synthetic_probability=.2,synthetic_probability=cfg['synthetic_ratio'],
        identity=identity,output_root=str(root),
        last_checkpoint=str(output/state['last']['path']) if state.get('last') else None,
        best_checkpoint=str(output/state['best']['path']) if state.get('best') else None)
    atomic_json(output/'summary.json',payload)
    archive=root/'chase_shape_native_results.zip'; temporary=archive.with_suffix('.pending.zip')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.suffix in ('.json','.csv'):
                z.write(path,'train/'+path.name)
        for name in SOURCE_FILES:
            path=Path(__file__).with_name(name)
            if path.is_file():z.write(path,'code/'+name)
        if prepared:
            for name in ('statistics_path',):
                path=Path(prepared[name]) if prepared.get(name) else None
                if path and path.is_file() and path.stat().st_size<=2_000_000:
                    z.write(path,'data_summary/'+path.name)
            # Keep the result upload small while retaining directly reviewable previews.
            preview_bytes=0
            for index,value in enumerate(prepared.get('preview_paths',[])[:8]):
                preview=Path(value)
                if preview.is_file() and preview_bytes+preview.stat().st_size<=8_000_000:
                    z.write(preview,f'previews/{index:02d}_'+preview.name)
                    preview_bytes+=preview.stat().st_size
            atomic=serial({k:v for k,v in prepared.items() if k!='records'})
            z.writestr('prepared_summary.json',json.dumps(atomic,indent=2))
    temporary.replace(archive)
    payload.update(archive=str(archive),archive_sha256=digest(archive))
    atomic_json(root/'latest_result.json',payload)
    return payload


def _run_epochs(cfg,dataset,model_factory,validator,output,identity,*,run_until=None,prepared=None):
    """Internal testable engine; production constraints live in normalize_config."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True);root=output.parent
    device=torch.device(cfg['device']);amp=torch.float16 if device.type=='cuda' else None
    target=cfg['epochs'] if run_until is None else int(run_until)
    if not 1<=target<=cfg['epochs']:raise ValueError('run-until must lie within the unchanged schedule')
    state_path=output/'state.json'
    state=read_envelope(state_path,identity) if state_path.exists() else None
    if state is None and list((output/'checkpoints').glob('*.pt')):
        print('Uncommitted checkpoint generations found; no completed epoch pointer exists.',flush=True)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(cfg['seed'])
    model=model_factory(state is None).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'],eta_min=cfg['lr']*cfg['minimum_lr_ratio'])
    scaler=torch.amp.GradScaler('cuda',enabled=amp is not None)
    history=[];best_pq=-1.;best_epoch=0;start=1;best_record=None
    if state:
        checkpoint=load_checkpoint(output,state['last'])
        if checkpoint['identity']!=identity or checkpoint['epoch']!=state['epoch']:
            raise ValueError('Checkpoint and committed epoch identity differ')
        if len(checkpoint['history'])!=state['epoch'] or checkpoint['history'][-1]['epoch']!=state['epoch']:
            raise ValueError('Checkpoint history does not match completed epoch')
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if checkpoint['best_epoch']!=state['best_epoch'] or checkpoint['best_pq']!=state['best_pq']:
            raise ValueError('Best checkpoint metadata differs from committed state')
        history=checkpoint['history'];best_pq=float(state['best_pq']);best_epoch=int(state['best_epoch'])
        best_record=state.get('best');start=state['epoch']+1
        if best_record:
            best=load_checkpoint(output,best_record)
            if best['identity']!=identity or best['epoch']!=best_epoch:raise ValueError('Best checkpoint mismatch')
            base.export_best_predictions(output,best)
            del best
        base.restore_rng(checkpoint['rng_state']);del checkpoint
    atomic_json(output/'config.json',cfg)
    print(f'Fresh ImageNet recipe; completed {start-1}; run through {target}; fixed schedule {cfg["epochs"]}.',flush=True)
    for epoch in range(start,target+1):
        dataset.epoch=epoch;loader=base.make_epoch_loader(dataset,cfg,device)
        model.train();optimizer.zero_grad(set_to_none=True)
        losses=[];attempts=successes=0;started=time.monotonic()
        domains=Counter(dataset.sample_record(i)['domain'] for i in range(len(dataset))) if hasattr(dataset,'sample_record') else {}
        for step,(x,y,valid) in enumerate(loader):
            x,y,valid=(v.to(device,non_blocking=True) for v in (x,y,valid))
            group_start=(step//cfg['accumulate'])*cfg['accumulate']
            divisor=min(cfg['accumulate'],len(loader)-group_start)
            with torch.autocast(device_type=device.type,dtype=amp or torch.float32,enabled=amp is not None):
                logits=model(x)
                loss=base.mixed_segmentation_loss(logits,y,valid,cfg['spine_weight'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss; previous epoch checkpoint preserved')
            scaler.scale(loss/divisor).backward();losses.append(float(loss.detach()))
            if (step+1)%cfg['accumulate']==0 or step+1==len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad_norm'])
                previous=scaler.get_scale();scaler.step(optimizer);scaler.update()
                successes+=int(scaler.get_scale()>=previous);attempts+=1
                optimizer.zero_grad(set_to_none=True)
            if (step+1)%25==0 or step+1==len(loader):
                print(f'epoch {epoch}/{cfg["epochs"]} crop batch {step+1}/{len(loader)} loss {losses[-1]:.5f}',flush=True)
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),lr=float(optimizer.param_groups[0]['lr']),
            microbatches=len(loader),optimizer_updates=successes,optimizer_update_attempts=attempts,
            amp_skipped_updates=attempts-successes,sampled_domains=dict(domains),
            synthetic_probability=float(dataset.synthetic_probability()) if hasattr(dataset,'synthetic_probability') else None,
            elapsed_seconds=time.monotonic()-started)
        scheduler.step()
        new_best=None
        if epoch==1 or epoch%cfg['validate_every']==0 or epoch==cfg['epochs']:
            best,grid,predictions=validator(model,device,amp,cfg)
            row.update(validation=best,validation_grid=grid)
            fixed=[r for r in grid if r['threshold']==.5 and r['min_area']==100]
            if fixed:row['fixed_limb_threshold05_area100']=fixed[0]
            if best['pq']>best_pq:
                best_pq,best_epoch=float(best['pq']),epoch
                new_best=dict(schema_version=1,identity=identity,model_state_dict=model.state_dict(),
                    config=cfg.copy(),epoch=epoch,validation=best,validation_grid=grid,
                    validation_predictions=predictions)
                best_record=save_checkpoint(output/'checkpoints'/f'best_epoch{epoch:03}.pt',new_best)
        history.append(row)
        payload=dict(schema_version=1,identity=identity,model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict(),
            scaler_state_dict=scaler.state_dict(),config=cfg.copy(),epoch=epoch,history=history,
            best_pq=best_pq,best_epoch=best_epoch,rng_state=base.capture_rng())
        last_record=save_checkpoint(output/'checkpoints'/f'last_epoch{epoch:03}.pt',payload)
        state=dict(epoch=epoch,last=last_record,best=best_record,best_pq=best_pq,best_epoch=best_epoch)
        # Publishing this one small pointer commits all files of the completed epoch.
        write_envelope(state_path,identity,state)
        if new_best is not None:base.export_best_predictions(output,new_best)
        atomic_json(output/'history.json',history)
        keep={state[k]['path'] for k in ('last','best') if state.get(k)}
        for path in (output/'checkpoints').glob('*.pt'):
            if str(path.relative_to(output)) not in keep:path.unlink()
        print(json.dumps(row),flush=True)
        _publish_results(root,output,state,cfg,identity,prepared)
        del payload,new_best,loader;gc.collect()
    if state is None:raise RuntimeError('No completed epoch state')
    atomic_json(output/'history.json',history)
    result=_publish_results(root,output,state,cfg,identity,prepared)
    del model,optimizer,scheduler,scaler
    if device.type=='cuda':torch.cuda.empty_cache()
    return result


def evaluate_baseline(cfg,store,names,output,identity):
    if not cfg.get('baseline_checkpoint'):
        return dict(status='not_requested',note='Historical scores are not matched T4 evaluation.')
    path=Path(cfg['baseline_checkpoint'])
    if not path.is_file() or digest(path)!=BASELINE_SHA:raise ValueError('Original best84 evaluation checkpoint missing or SHA differs')
    key=fingerprint(dict(training_identity=identity,checkpoint_sha256=BASELINE_SHA,evaluation='same_limb_grid_no_training'))
    target=output/'baseline_metrics.json'
    if target.exists():return read_envelope(target,key)
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('epoch')!=84:raise ValueError('Baseline checkpoint is not epoch84')
    model=base.build_model(cfg['encoder'],pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True);del checkpoint
    device=torch.device(cfg['device']);model=model.to(device)
    print('Evaluating original best84 only; its weights do not initialize the new run.',flush=True)
    best,grid,_=base.validate_native(model,store,names,device,torch.float16,cfg)
    value=dict(status='complete',checkpoint_sha256=BASELINE_SHA,best=best,grid=grid,
        input='limb_only',scope='Same-runtime fixed development grid; baseline evaluation only, no training.')
    write_envelope(target,key,value)
    del model;torch.cuda.empty_cache();return value


def run(config,phase='all',run_until=None):
    cfg=normalize_config(config)
    cv2.setNumThreads(2)
    root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    output=root/'train';output.mkdir(exist_ok=True)
    with (root/'.training.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another process is preparing/training this experiment')
        from chase_shape_synthesis import prepare_synthesis
        prepared=prepare_synthesis(cfg)
        atomic_json(root/'prepared_summary.json',prepared)
        if phase=='prepare':
            result=dict(status='prepared',output_root=str(root),**prepared)
            atomic_json(root/'latest_result.json',result);return result
        if phase not in ('all','train'):raise ValueError('Unsupported phase')
        if not torch.cuda.is_available():raise RuntimeError('Select a Colab T4 GPU runtime for training')
        configure_cuda();cv2.setNumThreads(2)
        device=torch.device(cfg['device'])
        identity_payload,image_hashes=source_identity(cfg,prepared,device)
        identity=fingerprint(identity_payload)
        identity_path=output/'identity.json'
        if identity_path.exists():
            if read_envelope(identity_path,identity)!=identity_payload:raise ValueError('Training protocol changed')
        else:
            if (output/'state.json').exists():raise ValueError('Missing training identity for an existing run')
            write_envelope(identity_path,identity,identity_payload)
        dataset,store,names=prepare_dataset(cfg,prepared,image_hashes)
        cfg['data_domains']=dataset.domain_info
        evaluate_baseline(cfg,store,names,output,identity)
        def validator(model,device,amp,settings):
            return base.validate_native(model,store,names,device,amp,settings)
        return _run_epochs(cfg,dataset,lambda fresh:base.build_model(cfg['encoder'],pretrained=fresh),
            validator,output,identity,run_until=run_until,prepared=prepared)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--phase',choices=('prepare','train','all'),default='all')
    parser.add_argument('--run-until',type=int)
    args=parser.parse_args()
    result=run(json.loads(args.config.read_text()),args.phase,args.run_until)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
