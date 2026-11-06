# Run after the current minimal SCRIPT configuration; T4/A100 GPU required.
import ast, subprocess, sys, tempfile
from pathlib import Path

PREFIX = r'''
import csv, hashlib, importlib.metadata, json, math, random, sys, time, uuid, zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import cv2, numpy as np, torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as mu
from scipy import ndimage
from scipy.interpolate import UnivariateSpline
'''

CORE = r'''
def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

def build_model(encoder='tu-convnext_tiny', pretrained=True):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights='imagenet' if pretrained else None, in_channels=3, classes=2, activation=None)

def model_input(image: np.ndarray) -> np.ndarray:
    x = np.repeat(image[..., None], 3, axis=2).astype(np.float32) / 255.0
    return ((x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)).transpose(2, 0, 1)

def apply_d4(image, target, k, flip):
    image = np.rot90(image, k, axes=(-2, -1))
    target = np.rot90(target, k, axes=(-2, -1))
    if flip:
        image, target = (image[..., ::-1], target[..., ::-1])
    return (np.ascontiguousarray(image), np.ascontiguousarray(target))

def crop_origin(height, width, crop, rng, bbox=None):
    if crop > min(height, width) or crop <= 0:
        raise ValueError('crop must fit inside the native image')
    if bbox is None:
        return (int(rng.integers(height - crop + 1)), int(rng.integers(width - crop + 1)))
    x, y, w, h = bbox
    cx, cy = (x + w / 2, y + h / 2)
    left = int(np.clip(round(cx - crop / 2 + rng.uniform(-crop / 4, crop / 4)), 0, width - crop))
    top = int(np.clip(round(cy - crop / 2 + rng.uniform(-crop / 4, crop / 4)), 0, height - crop))
    return (top, left)

class ImageStore:

    def __init__(self, images, cache, preprocessing):
        self.images, self.cache = (Path(images), Path(cache))
        self.preprocessing = preprocessing
        self.cache.mkdir(parents=True, exist_ok=True)

    def load(self, filename):
        source = self.images / filename
        stat = source.stat()
        signature = hashlib.sha256(f'{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{self.preprocessing}:v1'.encode()).hexdigest()
        target = self.cache / f'{signature}.png'
        if target.is_file():
            image = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (2048, 2048):
                raise ValueError(f'invalid processed cache: {target}')
            return image
        image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048) or image.dtype != np.uint8:
            raise ValueError(f'expected native uint8 2048-square JPEG: {source}')
        if self.preprocessing == 'limb_only':
            image = preprocess_limb_only(image)
        temporary = self.cache / f'{signature}.{uuid.uuid4().hex}.png'
        if not cv2.imwrite(str(temporary), image):
            raise OSError(f'failed writing image cache {temporary}')
        temporary.replace(target)
        metadata = dict(file_name=filename, source_sha256=digest(source), preprocessing=self.preprocessing)
        (self.cache / f'{signature}.json').write_text(json.dumps(metadata), encoding='utf-8')
        return image

class CropDataset(Dataset):

    def __init__(self, data, ids, store, crop=768, length=400, seed=42, small_probability=0.5, positive_probability=0.25, augment=True):
        self.store, self.crop, self.length, self.seed = (store, crop, length, seed)
        self.small_probability, self.positive_probability = (small_probability, positive_probability)
        self.augment, self.epoch = (augment, 0)
        self.images = {i['id']: i for i in data['images'] if i['id'] in set(ids)}
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
        self.small_area = float(np.quantile(areas, 0.25))
        if not 0 <= small_probability <= 1 or not 0 <= positive_probability <= 1 or small_probability + positive_probability > 1:
            raise ValueError('invalid crop sampling probabilities')

    def __len__(self):
        return self.length

    def sample_record(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
        filename = self.files[int(rng.integers(len(self.files)))]
        variants = self.variants[filename]
        group = variants[int(rng.integers(len(variants)))]
        anns = self.annotations[group]
        choice = rng.random()
        candidates = [a for a in anns if a['area'] <= self.small_area] if choice < self.small_probability else anns
        bbox = None
        if choice < self.small_probability + self.positive_probability and candidates:
            bbox = candidates[int(rng.integers(len(candidates)))]['bbox']
        elif choice < self.small_probability and anns:
            bbox = anns[int(rng.integers(len(anns)))]['bbox']
        top, left = crop_origin(2048, 2048, self.crop, rng, bbox)
        return (filename, group, top, left, rng)

    def __getitem__(self, index):
        filename, group, top, left, rng = self.sample_record(index)
        image = self.store.load(filename)[top:top + self.crop, left:left + self.crop].copy()
        anns = self.annotations[group]
        parts = [mu.merge(mu.frPyObjects(a['segmentation'], 2048, 2048)) for a in anns]
        foreground = mu.decode(mu.merge(parts))[top:top + self.crop, left:left + self.crop] if parts else np.zeros_like(image)
        spine = np.zeros_like(image)
        for ann in anns:
            xy = np.asarray(ann['spine'], dtype=np.float64).reshape(-1, 2) - [left, top]
            cv2.polylines(spine, [np.rint(xy).astype(np.int32)], False, 1, thickness=3)
        spine *= foreground
        if self.augment:
            values = np.clip(image.astype(np.float32) / 255.0, 0, 1)
            values = np.clip(values ** rng.uniform(0.9, 1.1) * rng.uniform(0.85, 1.15), 0, 1)
            image = (values * 255).astype(np.uint8)
        x = model_input(image)
        y = np.stack([foreground, spine]).astype(np.float32)
        if self.augment:
            x, y = apply_d4(x, y, int(rng.integers(4)), bool(rng.integers(2)))
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)))

def segmentation_loss(logits, target, spine_weight=0.1):
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError('logits and target must have matching B,2,H,W shapes')
    logits, target = (logits.float(), target.float())
    fg, truth = (logits[:, 0], target[:, 0])
    probability = fg.sigmoid()
    dice = 1 - ((2 * (probability * truth).sum((1, 2)) + 1) / (probability.sum((1, 2)) + truth.sum((1, 2)) + 1)).mean()
    mask_loss = F.binary_cross_entropy_with_logits(fg, truth) + dice
    line, spine = (logits[:, 1], target[:, 1])
    loss = F.binary_cross_entropy_with_logits(line, spine, reduction='none')
    positive, negative = (spine.sum(), (1 - spine).sum())
    neg_loss = (loss * (1 - spine)).sum() / negative.clamp_min(1)
    if positive.item() > 0:
        spine_loss = 0.5 * ((loss * spine).sum() / positive) + 0.5 * neg_loss
    else:
        spine_loss = neg_loss
    return mask_loss + spine_weight * spine_loss

def tile_positions(size, tile, overlap=0.25):
    if tile <= 0 or tile > size or (not 0 <= overlap < 1):
        raise ValueError('invalid tile size or overlap')
    stride = max(1, int(round(tile * (1 - overlap))))
    positions = list(range(0, size - tile + 1, stride))
    if positions[-1] != size - tile:
        positions.append(size - tile)
    return positions

def blend_window(tile):
    one = np.hanning(tile).astype(np.float32)
    return np.maximum(one[:, None] * one[None, :], 0.001)

@torch.inference_mode()
def predict_foreground(model, image, device, tile=768, overlap=0.25, batch=1, amp_dtype=None):
    model.eval()
    height, width = image.shape
    coordinates = [(y, x) for y in tile_positions(height, tile, overlap) for x in tile_positions(width, tile, overlap)]
    weight = blend_window(tile)
    numerator, denominator = (np.zeros((height, width), np.float32), np.zeros((height, width), np.float32))
    for start in range(0, len(coordinates), batch):
        selected = coordinates[start:start + batch]
        inputs = torch.from_numpy(np.stack([model_input(image[y:y + tile, x:x + tile]) for y, x in selected])).to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            output = model(inputs)
        if output.shape[-2:] != (tile, tile) or output.shape[1] != 2:
            raise ValueError(f'decoder must output native 2-channel masks, got {tuple(output.shape)}')
        probabilities = output[:, 0].float().sigmoid().cpu().numpy()
        for (y, x), p in zip(selected, probabilities):
            numerator[y:y + tile, x:x + tile] += p * weight
            denominator[y:y + tile, x:x + tile] += weight
    if np.any(denominator <= 0):
        raise ValueError('tile stitching left uncovered pixels')
    return numerator / denominator

def iter_components(probability, threshold, min_area):
    labels, count = ndimage.label(probability >= threshold, structure=np.ones((3, 3), np.uint8))
    areas = np.bincount(labels.ravel())
    for i in range(1, count + 1):
        if areas[i] >= min_area:
            yield (labels == i)

def score(pred):
    tp = fp = fn = 0
    total_iou = 0.0
    for group, truth in GT.items():
        predicted = pred.get(group.split('-', 1)[1], [])
        if not predicted:
            fn += len(truth)
            continue
        iou = np.asarray(mu.iou(truth, predicted, [0] * len(predicted)))
        hit = iou > 0.5
        tp += int(hit.sum())
        fp += int((~hit.any(axis=0)).sum())
        fn += int((~hit.any(axis=1)).sum())
        total_iou += float(iou[hit].sum())
    denominator = tp + 0.5 * fp + 0.5 * fn
    return dict(PQ=total_iou / denominator if denominator else 0.0, SQ=total_iou / tp if tp else 0.0, RQ=tp / denominator if denominator else 0.0, TP=tp, FP=fp, FN=fn)
'''

TRAINING = r'''

def prepare_split(data):
    by_id={row['id']:row for row in data['images']}
    assert len(by_id)==len(data['images']), '중복 annotation image ID'
    def read(path):
        with path.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
        ids={row['image_id'] for row in rows}
        assert len(ids)==len(rows) and ids<=set(by_id), 'split ID 오류'
        assert all(by_id[row['image_id']]['file_name']==row['file_name'] for row in rows), 'split 파일명 불일치'
        return ids,{row['file_name'] for row in rows}
    train_ids,train_files=read(TRAIN);val_ids,val_files=read(VAL)
    assert not train_ids&val_ids and not train_files&val_files, 'train/val 중복'
    assert val_ids=={row['id'] for row in data['images'] if row['file_name'] in val_files}, 'val annotator 누락'
    assert len(train_files)==616 and len(val_files)==91 and len(val_ids)==101
    assert all((IMAGES/name).is_file() for name in train_files|val_files), '원본 JPEG 누락'
    truth=defaultdict(list)
    for ann in data['annotations']:
        if ann['image_id'] in val_ids:
            truth[ann['image_id']].append(mu.merge(mu.frPyObjects(ann['segmentation'],2048,2048)))
    assert set(truth)==val_ids and sum(map(len,truth.values()))==708
    return train_ids,sorted(val_files),dict(truth)

def write_csv(path,predicted):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle'])
        for stem,instances in predicted.items():
            writer.writerows((f'{stem}_{i}',rle['counts'].decode('ascii')) for i,rle in enumerate(instances,1))

def validate_native(model,store,names,device,amp_dtype,cfg):
    predictions={(threshold,area):{} for threshold in (.35,.5,.65) for area in (100,200,300)}
    for index,name in enumerate(names,1):
        p=predict_foreground(model,store.load(name),device,cfg['crop'],.25,cfg['batch'],amp_dtype)
        for threshold in (.35,.5,.65):
            rles=[mu.encode(np.asfortranarray(mask.astype(np.uint8))) for mask in iter_components(p,threshold,100)]
            for area in (100,200,300):
                predictions[threshold,area][Path(name).stem]=[rle for rle in rles if int(mu.area(rle))>=area]
        print(f'validation [{index}/{len(names)}] {name}',flush=True)
    rows=[]
    for (threshold,area),predicted in predictions.items():
        metric=score(predicted)
        rows.append(dict(threshold=threshold,min_area=area,pq=metric['PQ'],sq=metric['SQ'],rq=metric['RQ'],tp_pairs=metric['TP'],fp=metric['FP'],fn=metric['FN']))
    return max(rows,key=lambda row:row['pq']),rows,predictions

def save_state(path,state):
    temporary=Path(path).with_suffix('.tmp.pt');torch.save(state,temporary);temporary.replace(path)

def package_results(output,summary):
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    archive=output.parent/f'{output.name}_native_unet_results.zip'
    temporary=archive.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.iterdir()):
            if path.suffix in ('.json','.csv','.py'):z.write(path,arcname=path.name)
    temporary.replace(archive)
    return archive

def default_config(gpu_bytes):
    large=gpu_bytes>=32*1024**3
    return dict(encoder='tu-convnext_tiny',pretrained=True,crop=1024 if large else 768,batch=2 if large else 1,accumulate=2 if large else 4,steps_per_epoch=200 if large else 400,epochs=30,validate_every=2,lr=1e-4,spine_weight=.1,seed=42,workers=2,thresholds=[.35,.5,.65],min_areas=[100,200,300],overlap=.25)

def train_native():
    global GT
    assert torch.cuda.is_available(), 'Colab 런타임에서 T4/A100 GPU를 선택하세요.'
    cv2.setNumThreads(2)
    expected={'MAGFiLO_1.0_Annotations_kaggle2026_train.json':'5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1','train_split.csv':'aa0733424815731c5c108ada91e45525eb17ca3a42b377bf19e3089d86aede7e','val_split.csv':'54c6a9aac5a3b70b3ecc6721109a8b0fe8ec020e90cb56e3848fccc92eadc61f'}
    hashes={path.name:digest(path) for path in (ANN,TRAIN,VAL)}
    assert hashes==expected, '검증했던 annotation/train/val과 다른 파일입니다.'
    data=json.loads(ANN.read_text(encoding='utf-8-sig'));train_ids,val_names,GT=prepare_split(data)
    cfg=default_config(torch.cuda.get_device_properties(0).total_memory)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    device=torch.device('cuda');amp_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    output=RUN/'native_unet_runs'/stamp;output.mkdir(parents=True,exist_ok=False)
    store=ImageStore(IMAGES,Path('/content/filament_native_cache'),'limb_only')
    dataset=CropDataset(data,train_ids,store,cfg['crop'],cfg['steps_per_epoch']*cfg['batch'],cfg['seed'],.5,.25,True)
    loader=DataLoader(dataset,batch_size=cfg['batch'],shuffle=False,num_workers=cfg['workers'],pin_memory=True,persistent_workers=False,multiprocessing_context='fork')
    cfg.update(source_sha256=hashes,train_physical_images=len(dataset.files),validation_physical_images=len(val_names),validation_annotation_groups=len(GT),validation_gt_instances=708,small_object_area_threshold_train_only=dataset.small_area,amp_dtype=str(amp_dtype),gpu=torch.cuda.get_device_name(0),preprocessing='exact existing limb_only',loss='foreground BCE+Dice +0.1 balanced spine BCE',selection='global official all-pairs PQ; strict IoU>.5',initialization='new ImageNet encoder training; not an optimizer resume',versions={p:importlib.metadata.version(p) for p in ('torch','segmentation-models-pytorch','timm','huggingface-hub','numpy','scipy','pycocotools')})
    (output/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (output/'run_source.py').write_text(Path(__file__).read_text(),encoding='utf-8')
    print(json.dumps(dict(output=str(output),config=cfg),ensure_ascii=False,indent=2),flush=True)
    model=build_model(cfg['encoder'],True).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'],eta_min=cfg['lr']*.05)
    scaler=torch.amp.GradScaler('cuda',enabled=amp_dtype==torch.float16)
    history=[];best_pq=-1.;best_epoch=0;started=time.monotonic()
    for epoch in range(1,cfg['epochs']+1):
        dataset.epoch=epoch;model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        for step,(x,y) in enumerate(loader):
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            with torch.autocast(device_type='cuda',dtype=amp_dtype): logits=model(x)
            loss=segmentation_loss(logits,y,cfg['spine_weight'])
            assert torch.isfinite(loss), f'nonfinite loss epoch{epoch} step{step}'
            group=min(cfg['accumulate'],len(loader)-(step//cfg['accumulate'])*cfg['accumulate'])
            scaler.scale(loss/group).backward()
            if (step+1)%cfg['accumulate']==0 or step+1==len(loader):
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if step==0 or (step+1)%max(1,len(loader)//10)==0:print(f'epoch{epoch}/30 step{step+1}/{len(loader)} loss{losses[-1]:.5f}',flush=True)
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),lr=float(optimizer.param_groups[0]['lr']),microbatches=len(loader),optimizer_updates=math.ceil(len(loader)/cfg['accumulate']),elapsed_seconds=time.monotonic()-started)
        scheduler.step()
        if epoch==1 or epoch%cfg['validate_every']==0 or epoch==cfg['epochs']:
            best,grid,predictions=validate_native(model,store,val_names,device,amp_dtype,cfg)
            row.update(validation=best,validation_grid=grid)
            if best['pq']>best_pq:
                best_pq,best_epoch=best['pq'],epoch
                save_state(output/'best.pt',dict(model_state_dict=model.state_dict(),config=cfg,epoch=epoch,validation=best))
                for (threshold,area),predicted in predictions.items():write_csv(output/f'best_t{threshold:g}_a{area}.csv',predicted)
                write_csv(output/'best_validation.csv',predictions[best['threshold'],best['min_area']])
                (output/'best_epoch.json').write_text(json.dumps(dict(epoch=epoch,best=best,grid=grid),indent=2),encoding='utf-8')
        history.append(row)
        save_state(output/'last.pt',dict(model_state_dict=model.state_dict(),config=cfg,epoch=epoch,optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict(),scaler_state_dict=scaler.state_dict()))
        (output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8');print(json.dumps(row),flush=True)
        if 'validation' in row:
            running=dict(status='running',best_development_pq=best_pq,best_epoch=best_epoch,epochs_completed=epoch,target_pq=.5,target_exceeded=best_pq>.5,checkpoint_directory=str(output),config=cfg,scope='Same development validation; not independent CV or Kaggle LB')
            print('자동 저장 ZIP:',package_results(output,running),flush=True)
    summary=dict(status='complete',scope='Same development validation selects checkpoint/decoder; not independent CV or Kaggle LB',best_development_pq=best_pq,best_epoch=best_epoch,target_pq=.5,target_exceeded=best_pq>.5,epochs_completed=len(history),elapsed_seconds=time.monotonic()-started,checkpoint_directory=str(output),initialization=cfg['initialization'],config=cfg)
    archive=package_results(output,summary)
    Path(sys.argv[1]).write_text(str(archive),encoding='utf-8');print('결과 ZIP:',archive,flush=True)
'''

def build_worker(source):
    tree=ast.parse(source);names={'DATA','RUN','IMAGES','ANN','BEST','TRAIN','VAL','yy','xx','rr','disk','radii','bins'}
    nodes=[];seen=set()
    for node in tree.body:
        if isinstance(node,ast.Assign):
            targets={n.id for t in node.targets for n in ast.walk(t) if isinstance(n,ast.Name)}
            if targets and targets<=names:nodes.append(node);seen|=targets
        elif isinstance(node,ast.FunctionDef) and node.name=='limb':nodes.append(node);seen.add('limb')
    assert (names|{'limb'})<=seen,'SCRIPT 복구 셀을 먼저 실행하세요.'
    worker=PREFIX+ast.unparse(ast.Module(body=nodes,type_ignores=[]))+'\npreprocess_limb_only=limb\n'+CORE+'\n'+TRAINING+'\nif __name__ == "__main__": train_native()\n'
    compile(worker,'native-unet-training','exec');return worker

assert isinstance(globals().get('SCRIPT'),str),'SCRIPT 복구 셀을 먼저 실행하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','segmentation-models-pytorch==0.5.0','timm==1.0.20','huggingface-hub==0.34.4','pycocotools==2.0.11','scipy==1.17.0','opencv-python==5.0.0.93'],check=True)
from google.colab import files
worker=build_worker(SCRIPT);work=Path(tempfile.mkdtemp(prefix='solar_native_'))
script=work/'train.py';script.write_text(worker,encoding='utf-8');result=work/'archive.txt'
subprocess.run([sys.executable,str(script),str(result)],check=True)
files.download(result.read_text(encoding='utf-8'))
