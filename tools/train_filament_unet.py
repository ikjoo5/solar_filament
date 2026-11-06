#!/usr/bin/env python3
"""Train a native-resolution foreground/spine U-Net using existing JPEGs.

This is an experiment runner, not a promise of PQ>0.5. Checkpoints are selected
by the supplied global all-pairs PQ on the declared development validation.
Images and annotation files are never modified. Each run needs a new directory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time
import uuid

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from convert_annotations import convert_data
from evaluate_yolo_baseline import preprocess_limb_only, read_split, select_validation
from filament_eval import Filament, evaluate


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def build_model(encoder='tu-convnext_tiny', pretrained=True):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights='imagenet' if pretrained else None,
                    in_channels=3, classes=2, activation=None)


def model_input(image: np.ndarray) -> np.ndarray:
    """Grayscale is replicated, then normalized using ImageNet RGB statistics."""
    x = np.repeat(image[..., None], 3, axis=2).astype(np.float32)/255.
    return ((x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1)


def apply_d4(image, target, k, flip):
    image = np.rot90(image, k, axes=(-2,-1))
    target = np.rot90(target, k, axes=(-2,-1))
    if flip:
        image, target = image[..., ::-1], target[..., ::-1]
    return np.ascontiguousarray(image), np.ascontiguousarray(target)


def crop_origin(height, width, crop, rng, bbox=None):
    if crop > min(height,width) or crop <= 0:
        raise ValueError('crop must fit inside the native image')
    if bbox is None:
        return int(rng.integers(height-crop+1)), int(rng.integers(width-crop+1))
    x,y,w,h = bbox
    cx,cy = x+w/2, y+h/2
    left = int(np.clip(round(cx-crop/2+rng.uniform(-crop/4,crop/4)),0,width-crop))
    top = int(np.clip(round(cy-crop/2+rng.uniform(-crop/4,crop/4)),0,height-crop))
    return top,left


class ImageStore:
    """Cache only processed uint8 images on disk, keyed by source stat/version.

    The stat signature prevents routine stale reuse; run metadata also records
    the source JPEG SHA-256 on first use. No dense annotation cache is created.
    """
    def __init__(self, images, cache, preprocessing):
        self.images, self.cache = Path(images), Path(cache)
        self.preprocessing = preprocessing
        self.cache.mkdir(parents=True,exist_ok=True)

    def load(self, filename):
        source = self.images/filename
        stat = source.stat()
        signature = hashlib.sha256(f'{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{self.preprocessing}:v1'.encode()).hexdigest()
        target = self.cache/f'{signature}.png'
        if target.is_file():
            image = cv2.imread(str(target),cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (2048,2048):
                raise ValueError(f'invalid processed cache: {target}')
            return image
        image = cv2.imread(str(source),cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048,2048) or image.dtype != np.uint8:
            raise ValueError(f'expected native uint8 2048-square JPEG: {source}')
        if self.preprocessing == 'limb_only':
            image = preprocess_limb_only(image)
        temporary = self.cache/f'{signature}.{uuid.uuid4().hex}.png'
        if not cv2.imwrite(str(temporary),image):
            raise OSError(f'failed writing image cache {temporary}')
        temporary.replace(target)
        metadata = dict(file_name=filename,source_sha256=digest(source),preprocessing=self.preprocessing)
        (self.cache/f'{signature}.json').write_text(json.dumps(metadata),encoding='utf-8')
        return image


class CropDataset(Dataset):
    def __init__(self, data, ids, store, crop=768, length=400, seed=42,
                 small_probability=.5, positive_probability=.25, augment=True):
        self.store,self.crop,self.length,self.seed = store,crop,length,seed
        self.small_probability,self.positive_probability = small_probability,positive_probability
        self.augment,self.epoch = augment,0
        self.images = {i['id']:i for i in data['images'] if i['id'] in set(ids)}
        self.variants = defaultdict(list)
        for i in self.images.values():
            self.variants[i['file_name']].append(i['id'])
        self.files = sorted(self.variants)
        if not self.files:
            raise ValueError('training split has no physical images')
        self.annotations = defaultdict(list)
        for ann in data['annotations']:
            if ann['image_id'] in self.images:
                self.annotations[ann['image_id']].append(ann)
        areas = [a['area'] for group in self.annotations.values() for a in group]
        if not areas:
            raise ValueError('training split has no annotations')
        self.small_area = float(np.quantile(areas,.25))
        if not 0 <= small_probability <= 1 or not 0 <= positive_probability <= 1 or small_probability+positive_probability > 1:
            raise ValueError('invalid crop sampling probabilities')

    def __len__(self):
        return self.length

    def sample_record(self,index):
        # Image-first sampling keeps an image's probability independent of how
        # many annotator variants it has. Determinism also permits worker tests.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,int(index)]))
        filename = self.files[int(rng.integers(len(self.files)))]
        variants = self.variants[filename]
        group = variants[int(rng.integers(len(variants)))]
        anns = self.annotations[group]
        choice = rng.random()
        candidates = [a for a in anns if a['area'] <= self.small_area] if choice < self.small_probability else anns
        bbox = None
        if choice < self.small_probability+self.positive_probability and candidates:
            bbox = candidates[int(rng.integers(len(candidates)))]['bbox']
        elif choice < self.small_probability and anns:
            bbox = anns[int(rng.integers(len(anns)))]['bbox']
        top,left = crop_origin(2048,2048,self.crop,rng,bbox)
        return filename,group,top,left,rng

    def __getitem__(self,index):
        filename,group,top,left,rng = self.sample_record(index)
        image = self.store.load(filename)[top:top+self.crop,left:left+self.crop].copy()
        anns = self.annotations[group]
        parts = [mu.merge(mu.frPyObjects(a['segmentation'],2048,2048)) for a in anns]
        foreground = mu.decode(mu.merge(parts))[top:top+self.crop,left:left+self.crop] if parts else np.zeros_like(image)
        spine = np.zeros_like(image)
        for ann in anns:
            xy = np.asarray(ann['spine'],dtype=np.float64).reshape(-1,2)-[left,top]
            cv2.polylines(spine,[np.rint(xy).astype(np.int32)],False,1,thickness=3)
        # Rasterized line caps can extend beyond the annotated filament mask.
        # Keep the auxiliary target consistent with foreground supervision.
        spine *= foreground
        if self.augment:
            # Physical-image sampling is unchanged by foreground crop choice.
            values = np.clip(image.astype(np.float32)/255.,0,1)
            values = np.clip((values**rng.uniform(.9,1.1))*rng.uniform(.85,1.15),0,1)
            image = (values*255).astype(np.uint8)
        x = model_input(image)
        y = np.stack([foreground,spine]).astype(np.float32)
        if self.augment:
            x,y = apply_d4(x,y,int(rng.integers(4)),bool(rng.integers(2)))
        return torch.from_numpy(np.ascontiguousarray(x)),torch.from_numpy(np.ascontiguousarray(y))


def segmentation_loss(logits,target,spine_weight=.1):
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError('logits and target must have matching B,2,H,W shapes')
    logits,target = logits.float(),target.float()
    fg,truth = logits[:,0],target[:,0]
    probability = fg.sigmoid()
    dice = 1-((2*(probability*truth).sum((1,2))+1)/
              (probability.sum((1,2))+truth.sum((1,2))+1)).mean()
    mask_loss = F.binary_cross_entropy_with_logits(fg,truth)+dice
    line,spine = logits[:,1],target[:,1]
    loss = F.binary_cross_entropy_with_logits(line,spine,reduction='none')
    positive,negative = spine.sum(),(1-spine).sum()
    neg_loss = (loss*(1-spine)).sum()/negative.clamp_min(1)
    if positive.item() > 0:
        spine_loss = .5*((loss*spine).sum()/positive)+.5*neg_loss
    else:
        spine_loss = neg_loss
    return mask_loss+spine_weight*spine_loss


def tile_positions(size,tile,overlap=.25):
    if tile <= 0 or tile > size or not 0 <= overlap < 1:
        raise ValueError('invalid tile size or overlap')
    stride = max(1,int(round(tile*(1-overlap))))
    positions = list(range(0,size-tile+1,stride))
    if positions[-1] != size-tile:
        positions.append(size-tile)
    return positions


def blend_window(tile):
    one = np.hanning(tile).astype(np.float32)
    return np.maximum(one[:,None]*one[None,:],1e-3)


@torch.inference_mode()
def predict_foreground(model,image,device,tile=768,overlap=.25,batch=1,amp_dtype=None):
    model.eval()
    height,width = image.shape
    coordinates = [(y,x) for y in tile_positions(height,tile,overlap) for x in tile_positions(width,tile,overlap)]
    weight = blend_window(tile)
    numerator,denominator = np.zeros((height,width),np.float32),np.zeros((height,width),np.float32)
    for start in range(0,len(coordinates),batch):
        selected = coordinates[start:start+batch]
        inputs = torch.from_numpy(np.stack([model_input(image[y:y+tile,x:x+tile]) for y,x in selected])).to(device)
        with torch.autocast(device_type=device.type,dtype=amp_dtype or torch.float32,enabled=amp_dtype is not None):
            output = model(inputs)
        if output.shape[-2:] != (tile,tile) or output.shape[1] != 2:
            raise ValueError(f'decoder must output native 2-channel masks, got {tuple(output.shape)}')
        probabilities = output[:,0].float().sigmoid().cpu().numpy()
        for (y,x),p in zip(selected,probabilities):
            numerator[y:y+tile,x:x+tile] += p*weight
            denominator[y:y+tile,x:x+tile] += weight
    if np.any(denominator <= 0):
        raise ValueError('tile stitching left uncovered pixels')
    return numerator/denominator


def iter_components(probability,threshold,min_area):
    labels,count = ndimage.label(probability >= threshold,structure=np.ones((3,3),np.uint8))
    areas = np.bincount(labels.ravel())
    for i in range(1,count+1):
        if areas[i] >= min_area:
            yield labels == i


def components(probability,threshold,min_area):
    return list(iter_components(probability,threshold,min_area))


def validation_truth(data,selected):
    ids = {i['id'] for i in selected}
    subset = dict(data,images=selected,annotations=[a for a in data['annotations'] if a['image_id'] in ids])
    rows,_,conversion = convert_data(subset)
    if conversion['empty_annotator_image_groups']:
        raise ValueError('blank GT groups are unsupported by the supplied evaluator')
    truth = []
    for key,counts in rows:
        group = key.split('_',1)[0]
        area = int(mu.area({'size':[2048,2048],'counts':counts.encode('ascii')}))
        truth.append(Filament(key,group.split('-',1)[1],group,counts,area))
    return truth


def validate(model,store,filenames,truth,device,args,amp_dtype):
    predictions = {(t,a):[] for t in args.thresholds for a in args.min_areas}
    for index,filename in enumerate(filenames,1):
        probability = predict_foreground(model,store.load(filename),device,args.validation_tile or args.crop,
                                         args.overlap,args.infer_batch,amp_dtype)
        image_id = Path(filename).stem
        for t in args.thresholds:
            # Component IDs do not depend on area: filter already encoded RLEs.
            candidates = []
            for i,mask in enumerate(iter_components(probability,t,min(args.min_areas)),1):
                rle = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
                candidates.append(Filament(f'{image_id}_{i}',image_id,image_id,rle['counts'].decode('ascii'),int(mu.area(rle))))
            for area in args.min_areas:
                predictions[t,area].extend(p for p in candidates if p.area >= area)
        print(f'validation [{index}/{len(filenames)}] {filename}',flush=True)
    results = []
    for (threshold,area),predicted in predictions.items():
        score = evaluate(truth,predicted,2048,2048)
        results.append(dict(threshold=threshold,min_area=area,
                            **{k:score[k] for k in ['pq','sq','rq','tp_pairs','fp','fn']}))
    best = max(results,key=lambda r:r['pq'])
    return best,results,predictions[best['threshold'],best['min_area']]


def write_predictions(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['filament_id','segmentation_rle'])
        writer.writerows((r.filament_id,r.counts) for r in rows)


def run(args):
    if args.crop%32 or not 64 <= args.crop <= 2048:
        raise ValueError('crop must be a multiple of 32 in [64,2048]')
    if args.validation_tile and (args.validation_tile%32 or not 64 <= args.validation_tile <= 2048):
        raise ValueError('validation tile must be a multiple of 32 in [64,2048]')
    if min(args.batch,args.epochs,args.steps_per_epoch,args.accumulate,args.validate_every,args.infer_batch) <= 0:
        raise ValueError('batch/epoch/step/accumulation/validation settings must be positive')
    if (not args.thresholds or not args.min_areas or any(not 0 < t < 1 for t in args.thresholds)
            or any(a < 1 for a in args.min_areas) or not 0 <= args.overlap < 1
            or args.workers < 0 or args.lr <= 0 or args.spine_weight < 0 or args.patience < 1):
        raise ValueError('invalid mask thresholds or minimum areas')
    destination = Path(args.output_dir)
    if destination.exists():
        raise ValueError(f'use a new output directory: {destination}')
    data = json.loads(Path(args.annotations).read_text(encoding='utf-8-sig'))
    train_ids = read_split(args.train_split,data['images'])
    val_ids = read_split(args.val_split,data['images'])
    selected,selection = select_validation(data['images'],train_ids,val_ids)
    train_files = {i['file_name'] for i in data['images'] if i['id'] in set(train_ids)}
    val_files = sorted({i['file_name'] for i in selected})
    for filename in sorted(train_files|set(val_files)):
        if not (Path(args.images)/filename).is_file():
            raise ValueError(f'missing original JPEG: {filename}')
    truth = validation_truth(data,selected)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; select a GPU or use --device cpu for a tiny smoke run')
    amp_dtype = None
    if device.type == 'cuda' and not args.no_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = build_model(args.encoder,not args.no_pretrained and not args.init_checkpoint).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint,map_location='cpu',weights_only=True)
        if checkpoint['config']['encoder'] != args.encoder:
            raise ValueError('initial checkpoint encoder differs from configured encoder')
        model.load_state_dict(checkpoint['model_state_dict'])
    destination.mkdir(parents=True,exist_ok=False)
    store = ImageStore(args.images,args.cache_dir or destination/'image_cache',args.preprocessing)
    dataset = CropDataset(data,train_ids,store,args.crop,args.steps_per_epoch*args.batch,args.seed)
    loader = DataLoader(dataset,batch_size=args.batch,shuffle=False,num_workers=args.workers,
                        pin_memory=device.type=='cuda',persistent_workers=False)
    config = vars(args).copy()
    config.update(small_object_area_threshold_train_only=dataset.small_area,
                  loss='foreground BCE+Dice + spine_weight * balanced spine BCE',
                  amp_dtype=str(amp_dtype),selection=selection,
                  validation_physical_images=len(val_files),validation_gt_instances=len(truth),
                  source_sha256={k:digest(v) for k,v in [('annotations',args.annotations),('train_split',args.train_split),('val_split',args.val_split)]},
                  versions={p:importlib.metadata.version(p) for p in ['torch','segmentation-models-pytorch','timm','numpy','scipy','pycocotools']})
    if args.init_checkpoint:
        config['initial_checkpoint_sha256'] = digest(args.init_checkpoint)
    (destination/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=args.lr*.05)
    scaler = torch.amp.GradScaler('cuda',enabled=amp_dtype==torch.float16)
    history,best_pq,best_epoch,stale = [],-1.,0,0
    started = time.monotonic()
    for epoch in range(1,args.epochs+1):
        dataset.epoch = epoch
        model.train();optimizer.zero_grad(set_to_none=True)
        losses = []
        for step,(x,y) in enumerate(loader):
            x,y = x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            with torch.autocast(device_type=device.type,dtype=amp_dtype or torch.float32,enabled=amp_dtype is not None):
                logits = model(x)
            loss = segmentation_loss(logits,y,args.spine_weight)
            if not torch.isfinite(loss):
                raise ValueError(f'nonfinite loss at epoch {epoch}, step {step}')
            # Correct normalization even for the final incomplete accumulation.
            group_size = min(args.accumulate,len(loader)-(step//args.accumulate)*args.accumulate)
            scaler.scale(loss/group_size).backward()
            if (step+1)%args.accumulate==0 or step+1==len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if step==0 or (step+1)%max(1,len(loader)//10)==0:
                print(f'epoch {epoch}/{args.epochs} step {step+1}/{len(loader)} loss={losses[-1]:.5f}',flush=True)
        row = dict(epoch=epoch,train_loss=float(np.mean(losses)),lr=float(optimizer.param_groups[0]['lr']),
                   microbatches=len(loader),optimizer_updates=math.ceil(len(loader)/args.accumulate),
                   elapsed_seconds=time.monotonic()-started)
        scheduler.step()
        if epoch==1 or epoch%args.validate_every==0 or epoch==args.epochs:
            best,grid,predictions = validate(model,store,val_files,truth,device,args,amp_dtype)
            row.update(validation=best,validation_grid=grid)
            previous_best = best_pq
            if best['pq'] > best_pq:
                best_pq,best_epoch = best['pq'],epoch
                torch.save(dict(model_state_dict=model.state_dict(),config=config,epoch=epoch,
                                validation=best),destination/'best.pt')
                write_predictions(destination/'best_validation.csv',predictions)
            if best['pq'] > previous_best+args.min_delta:
                stale = 0
            else:
                stale += 1
        history.append(row)
        torch.save(dict(model_state_dict=model.state_dict(),config=config,epoch=epoch,
                        optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict(),
                        scaler_state_dict=scaler.state_dict()),destination/'last.pt')
        (destination/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)
        if stale >= args.patience:
            break
    summary = dict(status='complete',best_development_pq=best_pq,best_epoch=best_epoch,
                   target_pq=.5,target_exceeded=best_pq>.5,epochs_completed=len(history),
                   elapsed_seconds=time.monotonic()-started,config=config,
                   scope='Same development validation used for checkpoint/decoder selection; not independent CV or Kaggle LB.',
                   initialization='New training from optional ImageNet encoder; --init-checkpoint starts a new fine-tuning run, not an exact optimizer resume.')
    (destination/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['images','annotations','train-split','val-split','output-dir']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--encoder',default='tu-convnext_tiny')
    parser.add_argument('--no-pretrained',action='store_true')
    parser.add_argument('--init-checkpoint')
    parser.add_argument('--preprocessing',choices=['limb_only','raw'],default='limb_only')
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--steps-per-epoch',type=int,default=400,help='Microbatches, not optimizer updates')
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--accumulate',type=int,default=4)
    parser.add_argument('--crop',type=int,default=768)
    parser.add_argument('--validation-tile',type=int,help='Defaults to training crop; separate override for profiling/smoke runs')
    parser.add_argument('--cache-dir',help='Optional local processed-image cache, e.g. /content/filament_cache')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--spine-weight',type=float,default=.1)
    parser.add_argument('--validate-every',type=int,default=2,help='Full validation first/last and every N epochs')
    parser.add_argument('--patience',type=int,default=8,help='Non-improving full validation events')
    parser.add_argument('--min-delta',type=float,default=1e-4)
    parser.add_argument('--thresholds',type=float,nargs='+',default=[.35,.5,.65])
    parser.add_argument('--min-areas',type=int,nargs='+',default=[20,100,200])
    parser.add_argument('--overlap',type=float,default=.25)
    parser.add_argument('--infer-batch',type=int,default=1)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--no-amp',action='store_true')
    parser.add_argument('--seed',type=int,default=42)
    args = parser.parse_args(argv)
    print(json.dumps(run(args),indent=2))


if __name__ == '__main__':
    main()
