"""CHASE mask geometry rendered on user-supplied clean solar backgrounds.

Appearance calibration uses only original best84 training observations. Source
GONG and clean-background files remain read-only. Raw generated PNGs are local
cache; compact masks, provenance and previews persist across runtime resets.
Validation/test images, when present, are read only for exact-duplicate guards.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import fcntl
import hashlib
import json
import math

import cv2
import numpy as np
from pycocotools import mask as mu

from targeted_synthetic_data import annotation_mask, atomic_json, atomic_png, split_roles

VERSION = 'chase_shape_clean_background_best84_v1'
EXPECTED = {
    'annotations': '5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1',
    'train_split': 'aa0733424815731c5c108ada91e45525eb17ca3a42b377bf19e3089d86aede7e',
    'val_split': '54c6a9aac5a3b70b3ecc6721109a8b0fe8ec020e90cb56e3848fccc92eadc61f',
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def encode(mask):
    value = mu.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
    value['counts'] = value['counts'].decode('ascii')
    return value


def decode(value):
    value = dict(value)
    if isinstance(value['counts'], str):
        value['counts'] = value['counts'].encode('ascii')
    return mu.decode(value).astype(np.uint8)


def save_envelope(path, identity, payload):
    atomic_json(path, dict(identity=identity, payload_sha256=fingerprint(payload), payload=payload))


def load_envelope(path, identity):
    value = json.loads(Path(path).read_text())
    if value.get('identity') != identity or fingerprint(value.get('payload')) != value.get('payload_sha256'):
        raise ValueError('Saved synthesis metadata changed: '+str(path))
    return value['payload']


def read_gray(path, size, expected_sha=None):
    payload = Path(path).read_bytes()
    if expected_sha and hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError('Source image changed: '+str(path))
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is not None and image.dtype==np.uint8 and image.ndim==3 and image.shape[2]==3:
        # The archived best84 image loader asks OpenCV for grayscale; RGB JPEG
        # storage is common even when a solar frame is visually monochrome.
        image=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (size, size) or image.dtype != np.uint8:
        raise ValueError('Expected square native grayscale uint8 image: '+str(path))
    return image


def _ellipse_geometry(image):
    from chase_transfer_data import estimate_disk
    return estimate_disk(image)


def read_clean(path, size, expected_sha=None):
    """Grayscale conversion and similarity alignment only; never rewrite source."""
    payload = Path(path).read_bytes()
    if expected_sha and hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError('Clean background source changed: '+str(path))
    raw = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.uint8 or raw.ndim not in (2,3):
        raise ValueError('Clean background must be a uint8 grayscale/RGB image')
    if raw.ndim == 3:
        if raw.shape[2] not in (3,4):
            raise ValueError('Unsupported clean-background channel count')
        if raw.shape[2] == 4 and np.any(raw[:,:,3] != 255):
            raise ValueError('Clean background contains transparent/unobserved pixels')
        gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY if raw.shape[2]==4 else cv2.COLOR_BGR2GRAY)
    else:
        gray = raw
    source_geometry = _ellipse_geometry(gray)
    radius = size*900/2048
    scale = radius/source_geometry['radius']
    matrix = np.array([[scale,0,size/2-scale*source_geometry['cx']],
                       [0,scale,size/2-scale*source_geometry['cy']]],np.float64)
    image = cv2.warpAffine(gray,matrix,(size,size),flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    metadata = dict(source_shape=list(raw.shape), source_gray_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),
        source_gray_shape=list(gray.shape), source_geometry=source_geometry, affine=matrix.tolist(),
        target_size=size,target_radius=radius,alignment_only=True,
        photometric_processing='grayscale conversion only; no brightness or contrast normalization')
    return image,dict(cx=size/2,cy=size/2,radius=radius),metadata


def clean_quality(image, geometry, profiles):
    """Reject obvious defects; these diagnostics cannot certify an empty Sun."""
    yy,xx=np.indices(image.shape)
    disk=np.hypot(xx-geometry['cx'],yy-geometry['cy']) < .85*geometry['radius']
    values=image[disk].astype(float)
    if np.mean((values<=3)|(values>=253)) > .01:
        raise ValueError('Clean background has substantial clipped disk pixels')
    blur=cv2.GaussianBlur(image.astype(np.float32),(0,0),max(2.,image.shape[0]/128))
    depth=np.log((blur+1)/(image.astype(np.float32)+1))
    # Only an extreme contamination veto. Moderate features remain an explicit
    # user-declared label assumption and must be inspected in the preview.
    threshold=max(.6,2*float(np.quantile([p['center_tau'] for p in profiles],.99)))
    severe=(depth>threshold)&disk
    _,_,stats,_=cv2.connectedComponentsWithStats(severe.astype(np.uint8),connectivity=8)
    largest=int(stats[1:,cv2.CC_STAT_AREA].max()) if len(stats)>1 else 0
    limit=max(8,int(np.quantile([p['area'] for p in profiles],.1)))
    if largest>=limit:
        raise ValueError('Clean background has a strong unlabelled dark feature; supply a cleaner background or its labels')
    return dict(disk_brightness_quantiles=np.quantile(values,[.01,.5,.99]).tolist(),
        severe_dark_threshold_tau=threshold,largest_severe_dark_component=largest,
        severe_component_rejection_area=limit,
        empty_background_status='user_declared_not_independently_verified',
        label_assumption='supplied_clean_background_assumed_empty_not_verified_by_filename')


def protected_foreground(data, groups, size):
    union = np.zeros((size, size), np.uint8)
    for group in groups:
        for ann in data.get(group, []):
            union |= annotation_mask(ann, shape=(size, size))
    return union


def measure_appearance(image, mask, known, geometry):
    """Measure a joint local attenuation/edge tuple from one real train object.

    A robust plane fitted to a surrounding ring removes local illumination
    gradients. All annotators' known foreground is excluded from that ring.
    """
    yy, xx = np.nonzero(mask)
    if len(xx) < 8:
        raise ValueError('Too few pixels for stable appearance measurement')
    size = image.shape[0]
    pad = max(5, int(round(16*size/2048)))
    y0, y1 = max(0, int(yy.min())-pad), min(size, int(yy.max())+pad+1)
    x0, x1 = max(0, int(xx.min())-pad), min(size, int(xx.max())+pad+1)
    local = mask[y0:y1, x0:x1].astype(np.uint8)
    pixels = image[y0:y1, x0:x1].astype(np.float32)
    ring = cv2.dilate(local, np.ones((2*pad+1, 2*pad+1), np.uint8)).astype(bool)
    ring &= ~cv2.dilate(local, np.ones((3, 3), np.uint8)).astype(bool)
    ring &= ~known[y0:y1, x0:x1].astype(bool)
    ring &= (pixels > 3) & (pixels < 252)
    ry, rx = np.nonzero(ring)
    if len(rx) < 24:
        raise ValueError('No trustworthy foreground-free surrounding ring')
    stride = max(1, len(rx)//2048)
    ry, rx = ry[::stride], rx[::stride]
    cx, cy = float(local.shape[1]/2), float(local.shape[0]/2)
    scale = max(local.shape)
    design = np.column_stack([np.ones(len(rx)), (rx-cx)/scale, (ry-cy)/scale])
    values = pixels[ry, rx]
    keep = np.ones(len(rx), bool)
    for _ in range(3):
        if keep.sum() < 24:
            raise ValueError('Unstable surrounding-plane estimate')
        coefficients = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]
        errors = values-design@coefficients
        mad = float(np.median(np.abs(errors[keep]-np.median(errors[keep]))))
        keep = np.abs(errors-np.median(errors[keep])) <= max(2., 3*1.4826*mad)
    grid_y, grid_x = np.indices(local.shape)
    background = coefficients[0]+coefficients[1]*(grid_x-cx)/scale+coefficients[2]*(grid_y-cy)/scale
    inside = local.astype(bool)
    if np.any(background[inside] <= 5):
        raise ValueError('Invalid local background estimate')
    depth = np.log((background+1)/(pixels+1))
    distance = cv2.distanceTransform(local, cv2.DIST_L2, 5)
    core = inside & (distance >= np.quantile(distance[inside], .6))
    edge = inside & (distance <= np.quantile(distance[inside], .25))
    center = float(np.median(depth[core]))
    positive_fraction = float(np.mean(depth[inside] > 0))
    if not .002 <= center <= 1.2 or positive_fraction < .6:
        raise ValueError('Real target has no reliable dark absorption profile')
    edge_ratio = float(np.clip(np.median(depth[edge])/center, .08, 1.))
    variation = float(np.clip(1.4826*np.median(np.abs(depth[core]-center))/center, 0., .35))
    radius = math.hypot(float(xx.mean())-geometry['cx'], float(yy.mean())-geometry['cy'])/geometry['radius']
    return dict(area=int(len(xx)), width=float(2*np.quantile(distance[inside], .9)),
                radius=float(radius), background=float(np.median(background[inside])),
                center_tau=center, edge_ratio=edge_ratio, variation=variation,
                ring_noise=float(1.4826*np.median(np.abs(errors[keep]-np.median(errors[keep])))),
                positive_fraction=positive_fraction)


def choose_appearance(profiles, brightness, radius, area, rng):
    """Sample a real joint tuple conditioned on location, brightness and scale."""
    if not profiles:
        raise ValueError('No measured training appearance tuples')
    distance = np.array([((p['background']-brightness)/40.)**2+
        ((p['radius']-radius)/.25)**2+(math.log(max(1, area)/p['area'])/2.)**2 for p in profiles])
    nearest = np.argsort(distance, kind='stable')[:min(12, len(profiles))]
    weights = np.exp(-.5*(distance[nearest]-distance[nearest].min()))
    index = int(rng.choice(nearest, p=weights/weights.sum()))
    return dict(profiles[index]), index, float(distance[index])


def transform_shape(mask, rng):
    turns, flip = int(rng.integers(4)), bool(rng.integers(2))
    scale = float(rng.uniform(.8, 1.25))
    transformed = np.rot90(mask, turns).copy()
    if flip:
        transformed = np.fliplr(transformed).copy()
    width = max(1, round(transformed.shape[1]*scale))
    height = max(1, round(transformed.shape[0]*scale))
    transformed = cv2.resize(transformed, (width, height), interpolation=cv2.INTER_NEAREST)
    if not transformed.any():
        raise ValueError('Shape vanished during geometric resampling')
    yy, xx = np.nonzero(transformed)
    transformed = transformed[yy.min():yy.max()+1, xx.min():xx.max()+1]
    return transformed, dict(rotation_quarter_turns=turns, horizontal_flip=flip, isotropic_scale=scale)


def render_at(background, shape, forbidden, x, y, geometry, profile, rng):
    """Attenuate existing texture only within a declared synthetic instance."""
    height, width = shape.shape
    if x < 0 or y < 0 or x+width > background.shape[1] or y+height > background.shape[0]:
        raise ValueError('Placement outside image')
    support = shape.astype(bool)
    sy, sx = np.nonzero(support)
    radius = np.hypot(sx+x-geometry['cx'], sy+y-geometry['cy'])/geometry['radius']
    if np.any(radius > .94):
        raise ValueError('Placement reaches the excluded solar rim')
    if np.any(forbidden[y:y+height, x:x+width] & support):
        raise ValueError('Placement intersects protected foreground')
    patch = background[y:y+height, x:x+width].astype(np.float32)
    if np.median(patch[support]) <= 8:
        raise ValueError('Placement has insufficient background signal')
    distance = cv2.distanceTransform(shape.astype(np.uint8), cv2.DIST_L2, 5)
    normalized = np.clip(distance/max(1., float(np.quantile(distance[support], .9))), 0, 1)
    profile_field = profile['edge_ratio']+(1-profile['edge_ratio'])*normalized**.7
    noise = rng.normal(size=shape.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), max(.7, profile['width']/4))
    std = float(noise[support].std())
    if std > 1e-8:
        noise = np.clip((noise-float(noise[support].mean()))/std, -2, 2)
    else:
        noise.fill(0)
    tau = profile['center_tau']*profile_field*np.maximum(.2, 1+profile['variation']*noise)
    tau *= support
    changed = np.clip(np.rint((patch+1)*np.exp(-tau)-1), 0, 255).astype(np.uint8)
    decrement = patch[support]-changed[support]
    visible = float(np.mean(decrement >= 1))
    if visible < .9 or float(np.median(decrement)) < 1 or np.any(changed[support] == 0):
        raise ValueError('Synthetic target is invisible, incomplete, or clipped after uint8 conversion')
    original = background[y:y+height, x:x+width]
    # A strong unexplained compact feature is a location veto, never a negative label.
    if float(np.quantile(original[support], .05)) < .45*float(np.median(original[support])):
        raise ValueError('Unexplained strong dark structure at insertion location')
    output = background.copy()
    output[y:y+height, x:x+width][support] = changed[support]
    placed = np.zeros_like(background, np.uint8)
    placed[y:y+height, x:x+width] = shape
    if not np.array_equal(output[placed == 0], background[placed == 0]):
        raise AssertionError('Renderer modified pixels outside its mask')
    return output, placed, dict(visible_fraction=visible, median_darkening_dn=float(np.median(decrement)),
        mean_darkening_dn=float(np.mean(decrement)), source_center_tau=profile['center_tau'])


def _annotations(masks, image_id):
    output = []
    for index, mask in enumerate(masks):
        encoded = encode(mask)
        output.append(dict(id=f'{image_id}_{index}', image_id=image_id, segmentation=encoded,
                           area=int(mu.area(encoded)), bbox=list(map(float, mu.toBbox(encoded))),
                           category_id=1, iscrowd=0))
    return output


def _preview(path, original, composite, placed, masks):
    yy, xx = np.nonzero(placed)
    margin = max(12, int(max(xx.max()-xx.min(), yy.max()-yy.min())*.35))
    y0, y1 = max(0, int(yy.min())-margin), min(original.shape[0], int(yy.max())+margin+1)
    x0, x1 = max(0, int(xx.min())-margin), min(original.shape[1], int(xx.max())+margin+1)
    panels = []
    overlay = cv2.cvtColor(composite, cv2.COLOR_GRAY2BGR)
    union = np.zeros_like(placed)
    for mask in masks:
        union |= mask
    overlay[union > 0] = np.rint(.6*overlay[union > 0]+.4*np.array([0, 210, 255])).astype(np.uint8)
    for label, pixels in [('Original', original), ('Inserted shape', placed*255),
                          ('Composite', composite), ('All labels', overlay)]:
        crop = pixels[y0:y1, x0:x1]
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_NEAREST if label=='Inserted shape' else cv2.INTER_AREA)
        crop = cv2.copyMakeBorder(crop, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(crop, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
        panels.append(crop)
    atomic_png(path, np.concatenate(panels, axis=1))


def _prepare_synthesis(cfg, *, expected_hashes=EXPECTED, expected_counts=(616, 91),
                       size=2048, chase_provider=None, aligned_provider=None, expected_chase=695):
    """Internal dependency-injected core; public entry point has no split bypass."""
    from chase_transfer_data import prepare_chase, ChaseDataset, load_manual_exclusions
    source_paths = {key: Path(cfg[key]) for key in ('annotations','train_split','val_split')}
    hashes = {key: digest(path) for key, path in source_paths.items()}
    if hashes != expected_hashes:
        raise ValueError('GONG annotation/split hashes differ from the original best84 experiment')
    data = json.loads(source_paths['annotations'].read_text(encoding='utf-8-sig'))
    roles = split_roles(data, source_paths['train_split'], source_paths['val_split'], expected=expected_counts)
    rows = {str(row['id']): row for row in data['images']}
    annotations = defaultdict(list)
    for ann in data['annotations']:
        if str(ann['image_id']) in roles['train']['ids']:
            annotations[str(ann['image_id'])].append(ann)
    groups_by_name = defaultdict(list)
    for group in sorted(roles['train']['ids']):
        groups_by_name[rows[group]['file_name']].append(group)
    names = sorted(groups_by_name)
    cache = Path(cfg['cache_root']).resolve()
    persistent = Path(cfg['output_root'])/'synthesis'
    if cache.is_relative_to(Path('/content/drive')) or cache.is_relative_to(persistent.resolve()):
        raise ValueError('Synthetic image cache must stay on local disk outside persistent Drive output')
    persistent.mkdir(parents=True, exist_ok=True)
    provenance=cfg.get('clean_background_provenance')
    if provenance not in ('external_user_supplied','train_only'):
        raise ValueError('Set clean_background_provenance to external_user_supplied or train_only with its source note')
    clean_paths=cfg.get('clean_backgrounds') or [cfg.get('clean_background')]
    if not isinstance(clean_paths,(list,tuple)) or not clean_paths or any(not p for p in clean_paths):
        raise ValueError('Supply clean_background or a nonempty clean_backgrounds list; no GONG fallback is used')
    clean_paths=[Path(p) for p in clean_paths]
    clean_sources=[]
    clean_arrays={}
    for path in clean_paths:
        sha=digest(path)
        image,geom,metadata=read_clean(path,size,sha)
        key=sha[:20]
        if any(item['source_gray_sha256']==metadata['source_gray_sha256'] for item in clean_sources):
            raise ValueError('Repeated clean-background pixels do not count as distinct backgrounds')
        clean_sources.append(dict(id=key,file_name=path.name,sha256=sha,**metadata))
        clean_arrays[key]=(path,geom)
    clean_file_hashes={item['sha256'] for item in clean_sources}
    clean_pixel_hashes={(tuple(item['source_gray_shape']),item['source_gray_sha256']) for item in clean_sources}
    real_hashes, geometry = {}, {}
    for number, name in enumerate(names, 1):
        path = Path(cfg['images'])/name
        real_hashes[name] = digest(path)
        image = read_gray(path, size, real_hashes[name])
        if real_hashes[name] in clean_file_hashes or (image.shape,hashlib.sha256(image.tobytes()).hexdigest()) in clean_pixel_hashes:
            raise ValueError('Supplied clean background is an annotated GONG training image; its foreground cannot be dropped')
        geometry[name] = _ellipse_geometry(image)
        if number % 50 == 0 or number == len(names):
            print(f'GONG training background audit {number}/{len(names)}', flush=True)
    duplicate_audit=dict(validation_checked=0,validation_unavailable=0,test_checked=0,
        test_directory_available=False,scope='Exact source-file/grayscale-pixel duplicates only; not a perceptual leakage proof')
    audit_paths=[('validation',Path(cfg['images'])/name) for name in sorted(roles['validation']['names'])]
    test_root=Path(cfg['test_images']) if cfg.get('test_images') else None
    if test_root is not None and test_root.is_dir():
        duplicate_audit['test_directory_available']=True
        audit_paths.extend(('test',path) for path in sorted(test_root.rglob('*'))
                           if path.is_file() and path.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff','.bmp'))
    for role,path in audit_paths:
        if not path.is_file():
            if role=='validation':duplicate_audit['validation_unavailable']+=1
            continue
        payload=path.read_bytes()
        if hashlib.sha256(payload).hexdigest() in clean_file_hashes:
            raise ValueError(f'Clean background duplicates a held-out {role} image: {path.name}')
        pixels=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_GRAYSCALE)
        if pixels is None:
            raise ValueError('Cannot decode held-out image for duplicate audit: '+str(path))
        if (pixels.shape,hashlib.sha256(pixels.tobytes()).hexdigest()) in clean_pixel_hashes:
            raise ValueError(f'Clean background duplicates held-out {role} pixels: {path.name}')
        duplicate_audit[role+'_checked']+=1
    manual_path = cfg.get('manual_exclusions')
    manual = load_manual_exclusions(manual_path) if manual_path else {}
    records, chase_report = (chase_provider or prepare_chase)(
        cfg['chase_root'], str(cache/'chase_download'), cfg['chase_manifest'])
    if len(records) != expected_chase or len({r['id'] for r in records}) != len(records):
        raise ValueError('Unexpected CHASE donor observation inventory')
    if not set(manual) <= {r['id'] for r in records}:
        raise ValueError('Explicit exclusions refer to unknown CHASE observations')
    donor_sources = {r['id']: {key:r[key] for key in ('image_sha256','mask_sha256')} for r in records}
    for record in records:
        for kind in ('image','mask'):
            if digest(record[kind+'_path']) != record[kind+'_sha256']:
                raise ValueError('CHASE source changed: '+record['id'])
    source_files = [Path(__file__), Path(__file__).with_name('targeted_synthetic_data.py'),
                    Path(__file__).with_name('chase_transfer_data.py')]
    settings = dict(version=VERSION, seed=int(cfg.get('seed', 42)), size=size,
        real_training_sources=real_hashes, clean_backgrounds=clean_sources,
        clean_background_provenance=provenance, clean_background_source_note=cfg.get('clean_background_source_note',''),
        label_assumption='supplied_clean_background_assumed_empty_not_verified_by_filename',
        visibility_scope='Raw uint8 synthetic masks checked before shared limb preprocessing; random crop inclusion is not forced',
        metadata_sha256=hashes, chase_sources=donor_sources,
        chase_manifest_sha256=digest(cfg['chase_manifest']), explicit_exclusions=manual,
        source_sha256={p.name:digest(p) for p in source_files},
        numpy=np.__version__, opencv=cv2.__version__,
        appearance='robust_local_ring_plane;conditional_joint_tuples_v1',
        instances_per_composite=3, synthetic_images=len(names),
        synthesis_space='raw_grayscale_then_shared_limb_once',
        scale_range=[.8,1.25], margin_pixels=max(1,round(8*size/2048)),
        max_radius=.94, max_attempts=300, measurement_cap_per_background=6)
    identity = fingerprint(settings)
    identity_path = persistent/'source_identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != dict(identity=identity, settings=settings):
            raise ValueError('Synthesis inputs/code changed; use a new experiment output directory')
    else:
        atomic_json(identity_path, dict(identity=identity, settings=settings))
    local = cache/'chase_shape'/identity[:16]
    images_dir = local/'images'; images_dir.mkdir(parents=True, exist_ok=True)
    stats_path, bank_path = persistent/'appearance_statistics.json', persistent/'shape_inventory.json'
    if stats_path.exists():
        measured = load_envelope(stats_path, identity)
    else:
        profiles, rejected = [], Counter()
        for number, name in enumerate(names, 1):
            partial = persistent/'calibration'/(hashlib.sha256(name.encode()).hexdigest()[:20]+'.json')
            if partial.exists():
                result = load_envelope(partial, identity)
            else:
                image = read_gray(Path(cfg['images'])/name, size, real_hashes[name])
                known = protected_foreground(annotations, groups_by_name[name], size)
                # Select one deterministic rater per physical image; all raters protect the ring.
                rng = np.random.default_rng(np.random.SeedSequence([settings['seed'], number, 11]))
                group = groups_by_name[name][int(rng.integers(len(groups_by_name[name])))]
                pool = annotations[group]
                order = rng.permutation(len(pool))[:6]
                values, failures = [], Counter()
                for position in order:
                    ann = pool[int(position)]
                    try:
                        measured_profile = measure_appearance(image, annotation_mask(ann, (size,size)), known, geometry[name])
                        measured_profile.update(source_file=name, source_group=group, source_annotation_id=str(ann['id']))
                        values.append(measured_profile)
                    except ValueError as error:
                        failures[str(error)] += 1
                result = dict(profiles=values, rejected=dict(failures))
                save_envelope(partial, identity, result)
            profiles.extend(result['profiles']); rejected.update(result['rejected'])
            if number % 25 == 0 or number == len(names):
                print(f'GONG appearance calibration {number}/{len(names)}; tuples={len(profiles)}', flush=True)
        if len(profiles) < min(32, len(names)):
            raise ValueError('Too few reliable training-only appearance measurements; inspect calibration records')
        measured = dict(profiles=profiles, rejected=dict(rejected), training_observations=len(names),
                        measurement_count=len(profiles), validation_images_read=0)
        save_envelope(stats_path, identity, measured)
    profiles = measured['profiles']
    clean_diagnostics=[]
    for source in clean_sources:
        path,geom=clean_arrays[source['id']]
        image,_,_=read_clean(path,size,source['sha256'])
        diagnostic=clean_quality(image,geom,profiles)
        clean_diagnostics.append(dict(source_id=source['id'],**diagnostic))
        # One full-disk panel per actual source makes the empty-background
        # assumption inspectable without disguising repeated texture as data.
        panel=cv2.cvtColor(cv2.resize(image,(512,512),interpolation=cv2.INTER_AREA),cv2.COLOR_GRAY2BGR)
        panel=cv2.copyMakeBorder(panel,42,0,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
        cv2.putText(panel,'Supplied clean source; empty by user report',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1)
        cv2.putText(panel,'Grayscale + disk alignment only',(8,36),cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1)
        atomic_png(persistent/'preview'/('clean_source_'+source['id']+'.png'),panel)
    atomic_json(persistent/'clean_background_qa.json',dict(identity=identity,
        unique_backgrounds=len(clean_sources),diagnostics=clean_diagnostics,duplicate_audit=duplicate_audit,
        limitation='Variants reuse supplied background texture; they are not independent solar observations'))
    if bank_path.exists():
        inventory = load_envelope(bank_path, identity)
    else:
        if aligned_provider is None:
            chase = ChaseDataset(records, crop=1024, length=len(records), seed=settings['seed'],
                                 cache_dir=local/'aligned_chase', manual_exclusions=manual)
            aligned_provider = chase.aligned
        inventory = []
        for number, record in enumerate(records, 1):
            partial = persistent/'donors'/(record['id']+'.json')
            if partial.exists():
                item = load_envelope(partial, identity)
            else:
                _, foreground, valid = aligned_provider(record)
                if foreground.shape != (size,size) or valid.shape != (size,size):
                    raise ValueError('Aligned donor coordinate grid differs')
                count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
                shapes, excluded = [], 0
                for label in range(1, count):
                    x,y,w,h,area = map(int, stats[label])
                    mask = (labels[y:y+h,x:x+w] == label).astype(np.uint8)
                    if np.any(mask.astype(bool) & ~(valid[y:y+h,x:x+w]>0)):
                        excluded += 1
                        continue  # Only explicitly ignored/unobserved support, never a branch heuristic.
                    shapes.append(dict(component=label, bbox=[x,y,w,h], area=area, mask=encode(mask)))
                item = dict(source_id=record['id'], shapes=shapes, explicit_or_unobserved_exclusions=excluded,
                            source_components=count-1, geometry_is_not_physical_instance_labels=True)
                save_envelope(partial, identity, item)
            inventory.append(item)
            if number % 50 == 0 or number == len(records):
                print(f'CHASE shape inventory {number}/{len(records)}', flush=True)
        save_envelope(bank_path, identity, inventory)
    usable = [record for record in inventory if record['shapes']]
    if not usable:
        raise ValueError('No usable CHASE geometry after explicit/observed-support exclusions')
    generated, output_annotations, details = [], [], []
    previews=[str(persistent/'preview'/('clean_source_'+item['id']+'.png')) for item in clean_sources]
    preview_indices = set(np.random.default_rng(settings['seed']).choice(len(names), min(96,len(names)), replace=False).tolist())
    area_min, area_max = np.quantile([p['area'] for p in profiles], [.01,.99])
    for index in range(len(names)):
        alias = f'chase_shape_{index:06d}'
        image_path, record_path = images_dir/(alias+'.png'), persistent/'records'/(alias+'.json')
        saved = load_envelope(record_path, identity) if record_path.exists() else None
        preview_path = persistent/'preview'/(alias+'.png')
        ready = saved is not None and image_path.exists() and digest(image_path)==saved['png_sha256']
        if ready and (index not in preview_indices or preview_path.exists()):
            result = saved
        else:
            rng = np.random.default_rng(np.random.SeedSequence([settings['seed'], index, 20261106]))
            source=clean_sources[index%len(clean_sources)]
            source_path,geometry_row=clean_arrays[source['id']]
            background,_,_=read_clean(source_path,size,source['sha256'])
            margin = settings['margin_pixels']
            forbidden=np.zeros_like(background,bool)
            composite=background.copy()
            masks,insertions=[],[]
            failures = Counter()
            for insertion in range(settings['instances_per_composite']):
                for attempt in range(settings['max_attempts']):
                    donor = usable[int(rng.integers(len(usable)))]  # Observation first, not component-count sampling.
                    reference = profiles[int(rng.integers(len(profiles)))]
                    distances = np.array([abs(math.log(max(1,s['area'])/reference['area'])) for s in donor['shapes']])
                    probabilities = np.exp(-distances); probabilities /= probabilities.sum()
                    component = donor['shapes'][int(rng.choice(len(donor['shapes']),p=probabilities))]
                    try:
                        shape, transform = transform_shape(decode(component['mask']),rng)
                        area = int(shape.sum())
                        if not max(1,area_min*.5) <= area <= area_max*2:
                            raise ValueError('Transformed shape outside measured training area support')
                        radius = float(np.clip(reference['radius']+rng.normal(0,.04),0,.90))
                        angle = float(rng.uniform(0,2*np.pi))
                        x = round(geometry_row['cx']+radius*geometry_row['radius']*math.cos(angle)-shape.shape[1]/2)
                        y = round(geometry_row['cy']+radius*geometry_row['radius']*math.sin(angle)-shape.shape[0]/2)
                        if min(x,y)<0 or x+shape.shape[1]>size or y+shape.shape[0]>size:
                            raise ValueError('Placement outside image')
                        brightness = float(np.median(background[y:y+shape.shape[0],x:x+shape.shape[1]][shape>0]))
                        profile, profile_index, profile_distance = choose_appearance(profiles,brightness,radius,area,rng)
                        composite, placed, quality = render_at(composite,shape,forbidden,x,y,geometry_row,profile,rng)
                    except ValueError as error:
                        failures[str(error)] += 1
                        continue
                    break
                else:
                    raise RuntimeError(f'Could not render valid composite {index}, object {insertion}; no sample silently omitted: {dict(failures)}')
                masks.append(placed)
                forbidden |= cv2.dilate(placed,np.ones((2*margin+1,2*margin+1),np.uint8)).astype(bool)
                insertions.append(dict(donor_source=donor['source_id'], donor_component=component['component'],
                    transform=transform, placement=[x,y],appearance_index=profile_index,
                    appearance_distance=profile_distance,appearance=profile,quality=quality))
            union=np.zeros_like(background,np.uint8)
            for mask in masks:
                if np.any(union&mask):raise AssertionError('Generated labels overlap')
                union |= mask
            if not np.array_equal(background[union==0],composite[union==0]):
                raise AssertionError('Clean background pixels changed outside inserted masks')
            atomic_png(image_path,composite)
            result = dict(image=dict(id=alias,file_name=image_path.name,width=size,height=size),
                annotations=_annotations(masks,alias), png_sha256=digest(image_path),
                background_file=source['file_name'], background_sha256=source['sha256'],
                original_instances=0, added_instances=len(masks),insertions=insertions,
                rejected_attempts=dict(failures),synthetic_spine_supervision=False,
                all_background_pixels_outside_insertions_preserved=True,
                seed_components=[settings['seed'],index,20261106])
            if saved is not None and result != saved:
                raise ValueError('Reconstructed synthetic cache differs from its saved provenance')
            save_envelope(record_path,identity,result)
            if index in preview_indices:
                _preview(preview_path,background,composite,union,masks)
        generated.append(result['image']); output_annotations.extend(result['annotations'])
        details.append({key:value for key,value in result.items() if key not in ('image','annotations')})
        if index in preview_indices:
            previews.append(str(preview_path))
        if (index+1)%10==0 or index+1==len(names):
            print(f'Clean-background synthetic variants {index+1}/{len(names)}',flush=True)
    annotation_path = persistent/'synthetic_annotations.json'
    annotation_document = dict(images=generated,annotations=output_annotations,
                               categories=[dict(id=1,name='filament')],
                               info=dict(generator=VERSION,identity=identity,canonical_mask_format='compressed_COCO_RLE'))
    if annotation_path.exists() and json.loads(annotation_path.read_text()) != annotation_document:
        raise ValueError('Saved synthetic annotations differ from verified records')
    atomic_json(annotation_path,annotation_document)
    manifest = dict(identity=identity,settings=settings,records=details,
        unique_physical_backgrounds=len(clean_sources),synthetic_variants=len(generated),
        synthetic_instances=len(output_annotations),
        observed_donor_sources=len({entry['donor_source'] for r in details for entry in r['insertions']}),available_donor_sources=len(usable),
        statistical_profiles=len(profiles),annotations_sha256=digest(annotation_path),
        statistics_sha256=digest(stats_path),inventory_sha256=digest(bank_path),
        preview_count=len(previews),preview_policy='all clean source full disks plus deterministic sample of96 composites',
        validation_pixels_used_for_calibration_or_training=False,
        duplicate_guard='Exact duplicates checked separately in clean_background_qa.json when held-out source files are available',
        limitation='Repeated clean background texture is not equivalent to additional independent observations',
        chase_source_content_fingerprint=fingerprint(donor_sources),
        role='static synthetic domain; sampling probability/schedule controlled by trainer')
    manifest_path=persistent/'manifest.json'
    if manifest_path.exists() and load_envelope(manifest_path,identity)!=manifest:
        raise ValueError('Synthesis manifest differs after reconstruction')
    save_envelope(manifest_path,identity,manifest)
    return dict(identity=identity,images_dir=str(images_dir),annotations_path=str(annotation_path),
                manifest_path=str(manifest_path),preview_paths=previews,statistics_path=str(stats_path),
                source_sha256=hashes,physical_images=len(generated))


def prepare_synthesis(cfg):
    """Prepare clean-background CHASE shapes using fixed GONG train calibration."""
    destination=Path(cfg['output_root'])/'synthesis';destination.mkdir(parents=True,exist_ok=True)
    with (destination/'prepare.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another process is preparing this synthesis directory') from error
        return _prepare_synthesis(cfg)
