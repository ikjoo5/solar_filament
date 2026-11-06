"""Deterministic train-only v2/v3 YOLO data; generated pixels stay in local cache.

Rater annotations remain separate records. Training polygons are audited against
authoritative binary masks; their measured raster error is never hidden.
"""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import csv
import hashlib
import json
import math
import os

import cv2
import numpy as np
from pycocotools import mask as mu

EXPECTED = {
    'annotations': '5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1',
    'train_split': 'aa0733424815731c5c108ada91e45525eb17ca3a42b377bf19e3089d86aede7e',
    'val_split': '54c6a9aac5a3b70b3ecc6721109a8b0fe8ec020e90cb56e3848fccc92eadc61f',
    'synthetic_annotations': '5e7ad5f20fa12f0c820a82a315ba60e802480d54b4b3ecebd9fba25ae8fc6aea',
    'synthetic_split': 'e8c9f3848c7b1fe1283167d973e459a5df7cd6336cc1d871268a34024d1ea881',
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    temporary.replace(path)


def atomic_png(path, pixels):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode('.png', pixels)
    if not ok: raise OSError('PNG encoding failed')
    temporary = path.with_suffix('.pending')
    temporary.write_bytes(encoded.tobytes()); temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def split_roles(data, train_path, val_path, expected=(616, 91)):
    by_id = {str(row['id']): row for row in data['images']}
    if len(by_id) != len(data['images']): raise ValueError('Duplicate annotation image ID')
    roles = {}
    for role, path in [('train', train_path), ('validation', val_path)]:
        with Path(path).open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        ids = [row['image_id'] for row in rows]
        if not ids or len(set(ids)) != len(ids) or set(ids) - set(by_id):
            raise ValueError('Invalid split IDs')
        if any(row['file_name'] != by_id[row['image_id']]['file_name'] for row in rows):
            raise ValueError('Split filename mismatch')
        names = {row['file_name'] for row in rows}
        if any(Path(n).name != n for n in names): raise ValueError('Expected image basenames')
        if set(ids) != {key for key, row in by_id.items() if row['file_name'] in names}:
            raise ValueError('Annotator siblings split across roles')
        roles[role] = dict(ids=set(ids), names=names)
    if roles['train']['names'] & roles['validation']['names']:
        raise ValueError('Physical train/validation overlap')
    if roles['train']['ids'] | roles['validation']['ids'] != set(by_id):
        raise ValueError('Split union does not cover annotation groups')
    if expected and tuple(len(roles[r]['names']) for r in ('train', 'validation')) != expected:
        raise ValueError('Expected original fixed616/91 physical split')
    return roles


def annotation_mask(ann, shape=(2048, 2048), legacy=False):
    seg = ann['segmentation']
    if legacy:
        out = np.zeros(shape, np.uint8)
        for polygon in seg:
            cv2.fillPoly(out, [np.rint(np.asarray(polygon)).astype(np.int32).reshape(-1, 2)], 1)
        return out
    if isinstance(seg, list): value = mu.merge(mu.frPyObjects(seg, *shape))
    elif isinstance(seg['counts'], list): value = mu.frPyObjects(seg, *shape)
    else:
        value = dict(seg)
        if isinstance(value['counts'], str): value['counts'] = value['counts'].encode('ascii')
    out = np.ascontiguousarray(mu.decode(value), dtype=np.uint8)
    if out.shape != shape or not out.any(): raise ValueError('Empty/malformed annotation mask')
    return out


def rle(mask):
    value = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    value['counts'] = value['counts'].decode('ascii')
    return value


def mask_to_yolo(mask, min_iou=.95):
    """One instance stays one polygon; zero-area bridges join disconnected parts.

    Coordinates retain exact float32 pixel roundtrip. YOLO fills polygon holes,
    hence full-resolution polygon IoU is measured and checked, not called exact.
    """
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c.reshape(-1, 2) for c in contours if len(c) >= 3]
    if not contours: raise ValueError('Instance cannot be represented by a YOLO polygon')
    points = max(contours, key=len).copy()
    remaining = [c for c in contours if c is not max(contours, key=len)]
    for other in remaining:
        distance = ((points[:, None].astype(np.int64) - other[None].astype(np.int64))**2).sum(axis=2)
        i, j = np.unravel_index(distance.argmin(), distance.shape)
        cycle = np.concatenate([other[j:], other[:j+1]])
        points = np.concatenate([points[:i+1], cycle, points[i:]])
    height, width = mask.shape
    normalized = points.astype(np.float64) / np.asarray([width, height])
    line = '0 ' + ' '.join(f'{number:.12f}' for number in normalized.ravel())
    parsed = np.asarray(line.split()[1:], np.float32).reshape(-1, 2) * [width, height]
    recovered = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(recovered, [parsed.astype(np.int32)], 1)
    score = float(np.count_nonzero(mask & recovered) / np.count_nonzero(mask | recovered))
    if score < min_iou: raise ValueError(f'YOLO polygon raster IoU {score:.5f} < {min_iou}; preserve source and inspect instance')
    return line, dict(iou=score, source_area=int(mask.sum()), polygon_area=int(recovered.sum()),
                      connected_contours=len(contours), polygon_vertices=len(points))


def make_donor_patch(image, mask, rng):
    """Copy measured dark residual, preserving target texture instead of a box."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 300: raise ValueError('Donor too small')
    x0, x1 = max(0, int(xs.min())-12), min(image.shape[1], int(xs.max())+13)
    y0, y1 = max(0, int(ys.min())-12), min(image.shape[0], int(ys.max())+13)
    patch = image[y0:y1, x0:x1].copy(); support = mask[y0:y1, x0:x1].copy()
    if cv2.connectedComponents(support)[0] != 2: raise ValueError('Disconnected donor')
    repaired = cv2.inpaint(patch, support*255, 3., cv2.INPAINT_TELEA)
    smooth = cv2.GaussianBlur(repaired.astype(np.float32), (0, 0), 3.)
    residual = np.minimum(np.log((patch.astype(np.float32)+1)/(smooth+1)), 0)
    if float(np.median(-residual[support > 0])) < .008:
        raise ValueError('No reliable dark donor contrast')
    residual *= support
    turns = int(rng.integers(4)); flip = bool(rng.integers(2)); scale = float(rng.uniform(.85, 1.15))
    residual = np.rot90(residual, turns).copy(); support = np.rot90(support, turns).copy()
    if flip: residual = np.fliplr(residual).copy(); support = np.fliplr(support).copy()
    target_size = (max(3, round(support.shape[1]*scale)), max(3, round(support.shape[0]*scale)))
    support = cv2.resize(support, target_size, interpolation=cv2.INTER_NEAREST)
    residual = cv2.resize(residual, target_size, interpolation=cv2.INTER_LINEAR) * support
    if support.sum() < 300 or support.sum() > 5000: raise ValueError('Transformed donor outside area budget')
    taper = np.minimum(cv2.distanceTransform(support, cv2.DIST_L2, 3)/2, 1)
    residual *= taper
    return residual, support, dict(rotation_quarter_turns=turns, horizontal_flip=flip, scale=scale,
        original_crop=[x0, y0, x1, y1], source_area=int(mask.sum()), transformed_area=int(support.sum()))


def paste_residual(background, residual, support, forbidden, x, y, strength=1.):
    """Return exact changed support, preserving all forbidden/background pixels."""
    h, w = support.shape
    if x < 0 or y < 0 or x+w > background.shape[1] or y+h > background.shape[0]:
        raise ValueError('Donor outside image')
    ys, xs = np.nonzero(support)
    radius = np.sqrt((xs+x-1024)**2+(ys+y-1024)**2)/900
    if np.any(radius > .94) or np.any(forbidden[y:y+h, x:x+w] & support.astype(bool)):
        raise ValueError('Donor overlaps known foreground or solar rim')
    target = background[y:y+h, x:x+w]
    changed = np.clip((target.astype(np.float32)+1)*np.exp(strength*residual)-1, 0, 255).astype(np.uint8)
    decrement = target[support > 0].astype(float)-changed[support > 0].astype(float)
    if float(np.median(decrement)) < 1. or float(np.mean(decrement)) > 55:
        raise ValueError('Invisible or implausibly dark pasted instance')
    out = background.copy(); region = out[y:y+h, x:x+w]
    region[support > 0] = changed[support > 0]
    placed = np.zeros(background.shape, np.uint8); placed[y:y+h, x:x+w] = support
    return out, placed, dict(median_darkening_DN=float(np.median(decrement)), mean_darkening_DN=float(np.mean(decrement)))


def _cached_record(image_path, label_path, meta_path, expected, build):
    """Checks both pixels and labels, including after an interrupted write."""
    if meta_path.exists():
        meta = read_json(meta_path)
        if meta['expected'] != expected: raise ValueError('Cached source identity changed')
        if image_path.exists() and label_path.exists() and digest(image_path) == meta['image_sha256'] and digest(label_path) == meta['label_sha256']:
            return meta
    pixels, masks, details = build()
    converted = [mask_to_yolo(mask) for mask in masks]
    label_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = label_path.with_suffix('.pending')
    temporary.write_text('\n'.join(line for line, _ in converted) + ('\n' if converted else ''))
    atomic_png(image_path, pixels); temporary.replace(label_path)
    meta = dict(expected=expected, image_sha256=digest(image_path), label_sha256=digest(label_path),
                annotation_instances=len(masks), polygon_quality=[v for _, v in converted], details=details)
    atomic_json(meta_path, meta)
    return meta


def prepare_dataset(cfg):
    """Build paired datasets. GPU is neither used nor required by this phase."""
    from submission_inference import limb
    cv2.setNumThreads(1)
    hashes = {key:digest(cfg[key]) for key in EXPECTED}
    if hashes != EXPECTED: raise ValueError('Use reviewed original annotation, fixed splits, and existing v2 metadata')
    data = read_json(cfg['annotations']); roles = split_roles(data, cfg['train_split'], cfg['val_split'])
    by_id = {str(row['id']):row for row in data['images']}
    train_ids = sorted(roles['train']['ids']); names = roles['train']['names']
    anns = defaultdict(list); by_ann = {}
    for ann in data['annotations']:
        sid = str(ann['image_id'])
        if sid in roles['train']['ids']:
            anns[sid].append(ann); by_ann[str(ann['id'])] = ann
    source_split = read_json(cfg['synthetic_split'])
    old_train = {str(v) for v in source_split['train_image_ids']}; old_val = {str(v) for v in source_split['val_image_ids']}
    if old_train & old_val or old_train | old_val != set(by_id): raise ValueError('v2 source split invalid')
    if {by_id[sid]['file_name'] for sid in old_train} - names: raise ValueError('v2 donor pool outside fixed train')
    count = int(cfg.get('synthetic_samples', 263)); seed = int(cfg.get('seed', 42))
    if not 1 <= count <= 462: raise ValueError('synthetic_samples must be1..462 without repeated v2 rows')
    hard_path = Path(cfg.get('hard_donors', Path(__file__).with_name('targeted_synthetic_hard_donors.json')))
    hard_info = read_json(hard_path); hard_ids = list(map(str, hard_info['annotation_ids']))
    if set(hard_ids)-set(by_ann): raise ValueError('Hard donor annotations outside fixed train')
    if hard_info.get('annotations_sha256') != hashes['annotations'] or hard_info.get('train_split_sha256') != hashes['train_split']:
        raise ValueError('Hard donor source fingerprint changed')
    settings = dict(version='targeted_synthetic_v3_1', sources=hashes, hard_donors_sha256=digest(hard_path),
        source_sha256=digest(__file__), preprocessing_sha256=digest(Path(__file__).with_name('submission_inference.py')),
        seed=seed, synthetic_samples=count, real_annotation_records=len(train_ids),
        hard_donor_probability=.7, pasted_objects_per_image=2, generator_min_area=300,
        raster_min_iou=.95, numpy=np.__version__, opencv=cv2.__version__, preprocessing='raw synthesis then limb once')
    key = identity(settings); local = Path(cfg['cache_root'])/'targeted_data'/key[:16]
    saved = Path(cfg['output_root'])/'data'/key[:16]; saved.mkdir(parents=True, exist_ok=True)
    atomic_json(saved/'settings.json', settings)
    source_registry_path = saved/'source_images.json'
    previous_sources = read_json(source_registry_path) if source_registry_path.exists() else {}
    source_hashes = {}; records = []; physical = local/'physical'
    physical.mkdir(parents=True, exist_ok=True)
    @lru_cache(maxsize=4)
    def image(name, synthetic=False):
        directory = cfg['synthetic_images'] if synthetic else cfg['images']
        path = Path(directory)/name
        payload = path.read_bytes(); stamp = hashlib.sha256(payload).hexdigest(); source_key = ('v2/' if synthetic else 'real/')+name
        if source_key in previous_sources and previous_sources[source_key] != stamp:
            raise ValueError('Source image changed since data preparation: '+name)
        source_hashes[source_key] = stamp
        value = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
        if value is None or value.shape != (2048, 2048): raise ValueError('Expected2048-square image: '+name)
        return value
    all_groups = defaultdict(list)
    for sid in train_ids: all_groups[by_id[sid]['file_name']].append(sid)
    @lru_cache(maxsize=4)
    def occupied(name):
        union = np.zeros((2048, 2048), np.uint8)
        for sid in all_groups[name]:
            for ann in anns[sid]: union |= annotation_mask(ann)
        return cv2.dilate(union, np.ones((17, 17), np.uint8)).astype(bool)
    @lru_cache(maxsize=24)
    def donor(aid):
        ann = by_ann[aid]; name = by_id[str(ann['image_id'])]['file_name']
        return name, annotation_mask(ann)
    eligible=[]
    for aid, ann in by_ann.items():
        seg=ann['segmentation']
        encoded=mu.merge(mu.frPyObjects(seg,2048,2048)) if isinstance(seg,list) else dict(seg)
        if isinstance(encoded.get('counts'),str):encoded['counts']=encoded['counts'].encode('ascii')
        area=int(mu.area(encoded))
        if 300<=area<=5000:eligible.append(aid)
    hard_ids=sorted(set(hard_ids)&set(eligible));eligible=sorted(eligible)
    if not hard_ids or not eligible: raise ValueError('No eligible real donor pool')
    def append_record(domain, alias, expected, build):
        ip=local/'images'/domain/(alias+'.png'); lp=local/'labels'/domain/(alias+'.txt'); mp=local/'metadata'/domain/(alias+'.json')
        meta=_cached_record(ip,lp,mp,expected,build)
        records.append(dict(domain=domain,alias=alias,path=str(ip.resolve()),**meta))
        return str(ip.resolve())
    real_paths=[]
    for index,sid in enumerate(train_ids,1):
        name=by_id[sid]['file_name'];pixels=image(name);stamp=source_hashes['real/'+name]
        alias='real_'+identity(sid)[:24]
        real_paths.append(append_record('real',alias,dict(dataset=key,annotation_group=sid,source_sha256=stamp),
            lambda sid=sid,name=name,pixels=pixels:(limb(pixels),[annotation_mask(a) for a in anns[sid]],dict(file_name=name,annotation_group=sid))))
        if index%100==0 or index==len(train_ids):
            atomic_json(source_registry_path,source_hashes);print(f'Real training records {index}/{len(train_ids)}',flush=True)
    synthetic=read_json(cfg['synthetic_annotations']);sanns=defaultdict(list)
    for ann in synthetic['annotations']:sanns[str(ann['image_id'])].append(ann)
    selected=np.random.default_rng(seed).choice(len(synthetic['images']),count,replace=False)
    paths={'control_v2':[],'targeted_v3':[]}
    for index,position in enumerate(selected):
        row=synthetic['images'][int(position)];sid=str(row['id']);name=Path(row['file_name']).name;pixels=image(name,True)
        paths['control_v2'].append(append_record('v2',f'v2_{index:04}',dict(dataset=key,v2_image_id=sid,source_sha256=source_hashes['v2/'+name]),
            lambda sid=sid,name=name,pixels=pixels:(limb(pixels),[annotation_mask(a,legacy=True) for a in sanns[sid]],dict(file_name=name,original_v2_image_id=sid))))
        if (index+1)%25==0:print(f'Existing synthetic control {index+1}/{count}',flush=True)
    for index in range(count):
        rng=np.random.default_rng(np.random.SeedSequence([seed,20261101,index]));sid=train_ids[int(rng.integers(len(train_ids)))]
        name=by_id[sid]['file_name'];base=image(name);bg_hash=source_hashes['real/'+name]
        def build_v3(sid=sid,name=name,base=base,index=index):
            rng=np.random.default_rng(np.random.SeedSequence([seed,20261101,index]))
            assert train_ids[int(rng.integers(len(train_ids)))]==sid
            composite=base.copy();forbidden=occupied(name).copy();added=[];provenance=[];failures=defaultdict(int)
            for trial in range(240):
                pool=hard_ids if rng.random()<.7 else eligible;aid=pool[int(rng.integers(len(pool)))];donor_name,mask=donor(aid)
                try:
                    residual,support,trans=make_donor_patch(image(donor_name),mask,rng)
                    h,w=support.shape;x=int(rng.integers(0,2048-w+1));y=int(rng.integers(0,2048-h+1))
                    strength=float(rng.uniform(.85,1.15))
                    proposal,placed,contrast=paste_residual(composite,residual,support,forbidden,x,y,strength)
                    mask_to_yolo(placed)
                except ValueError as error:
                    failures[str(error)]+=1;continue
                composite=proposal;added.append(placed)
                forbidden |= cv2.dilate(placed,np.ones((17,17),np.uint8)).astype(bool)
                provenance.append(dict(donor_annotation_id=aid,donor_file=donor_name,
                    donor_source_sha256=source_hashes['real/'+donor_name],donor_annotation_group=str(by_ann[aid]['image_id']),
                    hard_pool_member=aid in hard_ids,placement_xy=[x,y],strength=strength,transform=trans,contrast=contrast,mask_rle=rle(placed)))
                if len(added)==2:break
            if len(added)!=2:raise RuntimeError(f'Could not create2 valid donors for sample{index}; source preserved; rejected={dict(failures)}')
            background_masks=[annotation_mask(a) for a in anns[sid]]
            original_known=np.zeros((2048,2048),bool)
            for mask in background_masks:original_known |= mask.astype(bool)
            if not np.array_equal(base[original_known],composite[original_known]):raise AssertionError('Background foreground was modified')
            return limb(composite),background_masks+added,dict(background_file=name,background_annotation_group=sid,
                background_source_sha256=bg_hash,seed_components=[seed,20261101,index],added_instances=provenance,
                background_instances_preserved=len(background_masks),rejected_attempts=dict(failures))
        paths['targeted_v3'].append(append_record('v3',f'v3_{index:04}',dict(dataset=key,background_group=sid,background_sha256=bg_hash,sample=index),build_v3))
        if (index+1)%10==0 or index+1==count:
            atomic_json(source_registry_path,source_hashes);print(f'Targeted synthetic v3 {index+1}/{count}',flush=True)
    # Source registry checks every used original image on rerun, including cached donors.
    for source_key,stamp in previous_sources.items():
        domain,name=source_key.split('/',1);image(name,domain=='v2')
        if source_hashes[source_key]!=stamp:raise ValueError('Input image changed')
    atomic_json(source_registry_path,source_hashes)
    manifests = {record['alias']:record for record in records}
    final_identity=identity(dict(settings=settings,source_images=source_hashes,
        outputs={alias:{k:r[k] for k in ('image_sha256','label_sha256')} for alias,r in manifests.items()}))
    arms={}
    for arm in ('control_v2','targeted_v3','real_only'):
        selected_paths=real_paths+(paths[arm] if arm in paths else [])
        listing=local/(arm+'.txt');listing.write_text('\n'.join(selected_paths)+'\n')
        yaml=saved/(arm+'.yaml');yaml.write_text(json.dumps(dict(path=str(local.resolve()),train=str(listing.resolve()),val=str((local/'real_only.txt').resolve()),names={0:'filament'})))
        arms[arm]=dict(yaml=str(yaml),training_records=len(selected_paths),real_records=len(real_paths),synthetic_records=len(selected_paths)-len(real_paths))
    manifest=dict(identity=final_identity,settings=settings,training_physical_images=len(names),validation_physical_images=len(roles['validation']['names']),
        train_annotation_ids=train_ids,validation_used_for_training=False,synthetic_fraction=count/(count+len(train_ids)),
        all_rater_foreground_union_used_only_for_placement_exclusion=True,records=records,source_images=source_hashes,
        cache_disposable=True,persistent_generated_pixels=False,arms=arms)
    old_path=saved/'dataset_manifest.json'
    if old_path.exists() and read_json(old_path)['identity']!=final_identity:raise ValueError('Rebuilt dataset differs from saved training dataset')
    atomic_json(old_path,manifest)
    previews=[]; native_previews=[]
    for record in [r for r in records if r['domain']=='v3'][:6]:
        pixels=cv2.imread(record['path'],cv2.IMREAD_GRAYSCALE)
        color=cv2.cvtColor(pixels,cv2.COLOR_GRAY2BGR)
        for added in record['details']['added_instances']:
            mask=annotation_mask(dict(segmentation=added['mask_rle']))
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(color,contours,-1,(0,0,255),2)
            if len(native_previews)<6:
                yy,xx=np.nonzero(mask);cx=int(np.median(xx));cy=int(np.median(yy))
                left=max(0,min(1792,cx-128));top=max(0,min(1792,cy-128))
                before=limb(image(record['details']['background_file']))[top:top+256,left:left+256]
                after=pixels[top:top+256,left:left+256]
                overlay=cv2.cvtColor(after,cv2.COLOR_GRAY2BGR)
                outlines,_=cv2.findContours(np.ascontiguousarray(mask[top:top+256,left:left+256]),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay,outlines,-1,(0,0,255),1)
                tiles=[cv2.cvtColor(before,cv2.COLOR_GRAY2BGR),cv2.cvtColor(after,cv2.COLOR_GRAY2BGR),overlay]
                for tile,title in zip(tiles,['Before (limb)','After (limb)','Added GT contour']):
                    cv2.putText(tile,title,(5,15),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
                native_previews.append(np.hstack(tiles))
        previews.append(cv2.resize(color,(512,512)))
    if previews:
        while len(previews)%3:previews.append(np.zeros_like(previews[0]))
        montage=np.vstack([np.hstack(previews[i:i+3]) for i in range(0,len(previews),3)])
        atomic_png(saved/'synthetic_preview.png',montage)
    if native_previews:atomic_png(saved/'synthetic_native_preview.png',np.vstack(native_previews))
    return dict(identity=final_identity,arms={k:arms[k] for k in ('control_v2','targeted_v3')},
        control_yaml=arms['control_v2']['yaml'],targeted_yaml=arms['targeted_v3']['yaml'],
        real_yaml=arms['real_only']['yaml'],manifest=str(old_path),real_samples=len(real_paths),
        synthetic_samples=count,synthetic_fraction=count/(count+len(train_ids)),preview=str(saved/'synthetic_preview.png'),
        native_preview=str(saved/'synthetic_native_preview.png'))
