"""Fixed616/91 development experiment: frozen YOLO proposals, learned mask quality.

Run --phase prepare and --phase train in separate Python processes. Only the
quality model is optimized; validation labels select its development checkpoint.
All defaults form a new, real20-epoch schedule, independent of fivefold exports.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import csv
from functools import lru_cache
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time
import zipfile

import cv2
import numpy as np
from pycocotools import mask as mu
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from evaluate_grouped_oof import checked_rle, truth_for
from grouped_cv_training import atomic_json, digest, metric

BASE = Path('/content/drive/MyDrive/filament-segmentation-2026')
YOLO_RUN = Path('/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes')
EXPECTED = dict(annotations='5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1',
    train_split='aa0733424815731c5c108ada91e45525eb17ca3a42b377bf19e3089d86aede7e',
    val_split='54c6a9aac5a3b70b3ecc6721109a8b0fe8ec020e90cb56e3848fccc92eadc61f',
    yolo='81c2a27d95821b5ede283169c0da8c08d360ecc4c2a60a1b9c6baf96d76bcccc')
THRESHOLDS = [.1, .15, .2, .25, .3]
PROTOCOL = dict(version=1, views=['base2048', 'scale2560', 'hflip2048', 'vflip2048'],
    raw_confidence=.05, roi_size=256, context_scale=1.25, context_pad=16,
    regression='huber_beta0.1', match_bce_weight=.5, nms_iou=.5, min_area=300,
    thresholds=THRESHOLDS, baseline_confidence=.25, pretrained='timm/resnet18.a1_in1k',
    targets='mean_over_raters(max_IoU_if_strictly_above_0.5_else_0)',
    scope='original fixed616/91 development only; no fivefold training or OOF claim')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_splits(data, train_path, val_path, expected_counts=(616, 91)):
    """Read actual annotation-ID CSVs; never infer one rater's split by stem."""
    by_id = {str(row['id']): row for row in data['images']}
    if len(by_id) != len(data['images']):
        raise ValueError('Duplicate annotation image IDs')
    roles, ids = {}, {}
    for role, path in [('train', train_path), ('validation', val_path)]:
        with Path(path).open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        selected = [row['image_id'] for row in rows]
        if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(by_id):
            raise ValueError('Empty, duplicate, or unknown split annotation IDs')
        if any(row['file_name'] != by_id[row['image_id']]['file_name'] for row in rows):
            raise ValueError('Split filename differs from annotation metadata')
        names = {row['file_name'] for row in rows}
        if any(Path(name).name != name or Path(name).suffix.lower() != '.jpeg' for name in names):
            raise ValueError('Expected original JPEG basenames')
        if set(selected) != {key for key, row in by_id.items() if row['file_name'] in names}:
            raise ValueError('Every annotator sibling must stay in its physical image split')
        ids[role], roles[role] = set(selected), sorted(names)
    if ids['train'] & ids['validation'] or set(roles['train']) & set(roles['validation']):
        raise ValueError('Train/validation leakage')
    if ids['train'] | ids['validation'] != set(by_id):
        raise ValueError('Split union must cover every original annotation group')
    if tuple(map(lambda role: len(roles[role]), roles)) != expected_counts:
        raise ValueError('Expected fixed616 train and91 validation physical images')
    return roles


def soft_targets(proposals, rater_groups):
    """Each annotator contributes one vote, including an empty annotation group."""
    result = np.zeros((len(proposals), 2), np.float32)
    if not rater_groups:
        raise ValueError('A physical image must have at least one annotation group')
    if not proposals:
        return result
    for gt in rater_groups:
        if gt:
            best = np.asarray(mu.iou(gt, proposals, [0]*len(proposals))).max(axis=0)
            matched = best > .5
            result[:, 0] += np.where(matched, best, 0)
            result[:, 1] += matched
    return result / len(rater_groups)


def roi_box(mask):
    yy, xx = np.nonzero(mask)
    if not len(yy):
        raise ValueError('Empty proposal mask')
    h, w = int(yy.max()-yy.min()+1), int(xx.max()-xx.min()+1)
    cy, cx = (yy.min()+yy.max()+1)/2, (xx.min()+xx.max()+1)/2
    hh, hw = h*1.25/2+16, w*1.25/2+16
    return (max(0, int(np.floor(cx-hw))), max(0, int(np.floor(cy-hh))),
            min(mask.shape[1], int(np.ceil(cx+hw))), min(mask.shape[0], int(np.ceil(cy+hh))))


def roi_array(corrected, value, size=256):
    mask = mu.decode(value).astype(np.uint8)
    x1, y1, x2, y2 = roi_box(mask)
    h, w = y2-y1, x2-x1
    scale = size/max(h, w)
    nh, nw = max(1, round(h*scale)), max(1, round(w*scale))
    top, left = (size-nh)//2, (size-nw)//2
    result = np.zeros((2, size, size), np.uint8)
    result[0, top:top+nh, left:left+nw] = cv2.resize(corrected[y1:y2, x1:x2], (nw, nh))
    result[1, top:top+nh, left:left+nw] = cv2.resize(mask[y1:y2, x1:x2], (nw, nh), interpolation=cv2.INTER_NEAREST)*255
    return result


def select_training(instances, cap=48):
    """Train-only balancing: positives plus highest-confidence hard negatives."""
    order = sorted(range(len(instances)), key=lambda i: (-instances[i]['confidence'], i))
    positive = [i for i in order if instances[i]['target'][1] > 0]
    negative = [i for i in order if instances[i]['target'][1] == 0]
    selected = positive[:cap//2]+negative[:cap-cap//2]
    return sorted((selected+[i for i in order if i not in selected])[:cap])


def read_raw(path):
    with gzip.open(path, 'rt') as stream:
        value = json.load(stream)
    checksum = value.pop('checksum', None)
    if checksum != fingerprint(value):
        raise ValueError('Proposal cache checksum mismatch: '+str(path))
    return value


def atomic_raw(path, value):
    pending = path.with_suffix('.pending')
    with gzip.open(pending, 'wt') as stream:
        json.dump(dict(value, checksum=fingerprint(value)), stream)
    pending.replace(path)


def source_identity(args):
    shas = {key: digest(getattr(args, key)) for key in EXPECTED}
    if shas != EXPECTED:
        raise ValueError('Use the exact reviewed annotations,616/91 CSVs, and original solar YOLO best.pt')
    versions = {name: importlib.metadata.version(name) for name in
                ['torch', 'timm', 'ultralytics', 'numpy', 'pycocotools', 'scipy']}
    versions.update(opencv=cv2.__version__, python=__import__('platform').python_version())
    code = {name: digest(Path(__file__).with_name(name)) for name in
            ['fixed_split_mask_quality.py', 'submission_inference.py', 'evaluate_grouped_oof.py', 'grouped_cv_training.py']}
    return dict(inputs=shas, code=code, runtime=versions, protocol=PROTOCOL, seed=args.seed, train_cap=args.train_cap)


def prepare(args):
    from submission_inference import InferenceProvider, limb
    identity = source_identity(args)
    output, cache = Path(args.output), Path(args.cache)
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    identity_path = output/'corpus_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Cached proposal identity changed; use a new experiment folder')
    atomic_json(identity_path, identity)
    corpus_id = fingerprint(identity)
    data = json.loads(Path(args.annotations).read_text())
    roles = read_splits(data, args.train_split, args.val_split)
    truth, mapping = truth_for(data, set(roles['train']+roles['validation']))
    groups = defaultdict(list)
    for sid, stem in mapping.items():
        groups[stem].append(truth[sid])
    provider = InferenceProvider(args.yolo, None, device=device_for(args.device), verify_weights=True)
    empty_counts = mu.encode(np.zeros((2048, 2048), np.uint8, order='F'))['counts'].decode('ascii')
    (output/'proposals').mkdir(exist_ok=True)
    records = []
    for role, names in roles.items():
        for number, name in enumerate(names, 1):
            started = time.monotonic()
            path, stem = Path(args.images)/name, Path(name).stem
            image_sha = digest(path)
            raw_path = output/'proposals'/f'{stem}.json.gz'
            expected = dict(corpus=corpus_id, image_sha=image_sha, name=name, role=role)
            raw = read_raw(raw_path) if raw_path.exists() else None
            if raw is not None and raw['identity'] != expected:
                raise ValueError('Cached image/model/split fingerprint changed: '+name)
            pixels = None
            if raw is None:
                pixels = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                views = provider.geometric(stem, pixels)
                unique, empty_skipped = {}, 0
                for view in views:
                    for item in view['instances']:
                        if item['rle'] == empty_counts:
                            empty_skipped += 1
                            continue  # Exact canonical empty YOLO mask; no GT-dependent filtering.
                        if item['rle'] not in unique or item['confidence'] > unique[item['rle']]['confidence']:
                            unique[item['rle']] = dict(item)
                instances = list(unique.values())
                encoded = [checked_rle(dict(size=[2048, 2048], counts=item['rle'])) for item in instances]
                targets = soft_targets(encoded, groups[stem])
                for item, target in zip(instances, targets):
                    item['target'] = target.tolist()
                raw = dict(identity=expected, instances=instances, empty_skipped=empty_skipped,
                    views=[dict(view=v['view'], max_det_reached=v['max_det_reached']) for v in views])
                atomic_raw(raw_path, raw)
            selected = select_training(raw['instances'], args.train_cap) if role == 'train' else list(range(len(raw['instances'])))
            raw_sha = digest(raw_path)
            local = cache/f'{stem}.npy'
            meta_path = cache/f'{stem}.json'
            roi_identity = dict(raw_sha=raw_sha, selected=selected, corpus=corpus_id)
            valid = local.exists() and meta_path.exists()
            if valid:
                meta = json.loads(meta_path.read_text())
                valid = meta['identity'] == roi_identity and meta['array_sha'] == digest(local)
            if not valid:
                if pixels is None:
                    pixels = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                corrected = limb(pixels)
                values = [roi_array(corrected, checked_rle(dict(size=[2048, 2048], counts=raw['instances'][i]['rle']))) for i in selected]
                array = np.stack(values) if values else np.empty((0, 2, 256, 256), np.uint8)
                pending = local.with_suffix('.pending')
                with pending.open('wb') as stream:
                    np.save(stream, array, allow_pickle=False)
                pending.replace(local)
                atomic_json(meta_path, dict(identity=roi_identity, array_sha=digest(local)))
            records.append(dict(name=name, role=role, selected=selected, raw_sha=raw_sha,
                                proposal_count=len(raw['instances']), roi_count=len(selected)))
            print(f'prepare {role} [{number}/{len(names)}] {name}: raw={len(raw["instances"])} roi={len(selected)} {time.monotonic()-started:.1f}s', flush=True)
    manifest = dict(identity=corpus_id, records=records, training_images=616, validation_images=91)
    atomic_json(output/'corpus.json', manifest)
    print('Prepared. ROI arrays are local only; launch --phase train in a NEW Python process.', flush=True)


class QualityDataset(Dataset):
    def __init__(self, records, output, cache, role):
        self.rows = []
        self.targets = []
        self.records = [r for r in records if r['role'] == role]
        for record in self.records:
            stem = Path(record['name']).stem
            raw_path = Path(output)/'proposals'/f'{stem}.json.gz'
            if digest(raw_path) != record['raw_sha']:
                raise ValueError('Proposal archive changed: '+stem)
            raw = read_raw(raw_path)
            local = Path(cache)/f'{stem}.npy'
            meta = json.loads(local.with_suffix('.json').read_text())
            if meta['identity']['raw_sha'] != record['raw_sha'] or meta['identity']['selected'] != record['selected'] or meta['array_sha'] != digest(local):
                raise ValueError('Local ROI cache differs; rerun prepare')
            if role == 'validation' and record['selected'] != list(range(record['proposal_count'])):
                raise ValueError('Validation must include every proposal')
            for j, i in enumerate(record['selected']):
                self.rows.append((str(local), j))
                self.targets.append(raw['instances'][i]['target'])

    @staticmethod
    @lru_cache(maxsize=3)
    def array(path):
        return np.load(path, mmap_mode='r', allow_pickle=False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, j = self.rows[index]
        value = torch.from_numpy(self.array(path)[j].copy()).float()/255
        rgb = value[0:1].repeat(3, 1, 1)
        rgb = (rgb-torch.tensor([.485, .456, .406])[:, None, None])/torch.tensor([.229, .224, .225])[:, None, None]
        return torch.cat([rgb, value[1:2]], 0), torch.tensor(self.targets[index], dtype=torch.float32)


def expand_conv(backbone):
    old = backbone.conv1
    new = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight[:, :3].copy_(old.weight)
        new.weight[:, 3:].zero_()
        if old.bias is not None:
            new.bias.copy_(old.bias)
    backbone.conv1 = new


class QualityModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model('resnet18.a1_in1k', pretrained=pretrained, num_classes=0)
        expand_conv(self.backbone)
        self.head = nn.Linear(self.backbone.num_features, 2)

    def forward(self, value):
        return self.head(self.backbone(value))


def device_for(requested):
    device = torch.device('cuda' if requested == 'auto' and torch.cuda.is_available() else 'cpu' if requested == 'auto' else requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; choose --device auto or cpu')
    return device


def autocast_for(device):
    return torch.autocast(device_type='cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()


def decode(instances, scores, threshold, min_area=300, return_scores=False):
    """Mask NMS, then score-ordered ownership clipping: no output mask overlaps."""
    if len(scores) != len(instances) or not np.isfinite(scores).all():
        raise ValueError('Invalid proposal scores')
    selected, selected_scores = [], []
    for i in sorted(range(len(scores)), key=lambda j: (-float(scores[j]), j)):
        if scores[i] < threshold:
            continue
        value = checked_rle(dict(size=[2048, 2048], counts=instances[i]['rle']))
        if int(mu.area(value)) < min_area:
            continue
        if selected and np.asarray(mu.iou([value], selected, [0]*len(selected))).max() > .5:
            continue
        selected.append(value)
        selected_scores.append(float(scores[i]))
    owned = np.zeros((2048, 2048), bool)
    result = []
    for value, score in zip(selected, selected_scores):
        remaining = mu.decode(value).astype(bool) & ~owned
        if remaining.sum() >= min_area:
            owned |= remaining
            encoded = mu.encode(np.asfortranarray(remaining.astype(np.uint8)))
            result.append((encoded, score) if return_scores else encoded)
    return result


def threshold_reports(pools, scores, truth, mapping):
    # Every threshold takes a prefix of the same score order: lower-scored masks
    # cannot change higher-scored ownership. Decode once rather than five times.
    candidates = {stem: decode(items, scores[stem], min(THRESHOLDS), return_scores=True) for stem, items in pools.items()}
    reports = []
    for threshold in THRESHOLDS:
        predicted = {stem: [value for value, score in items if score >= threshold] for stem, items in candidates.items()}
        report = metric(truth, predicted, mapping)
        report.pop('per_group')
        reports.append(dict(threshold=threshold, **report))
    return reports


def validation(model, dataset, args, device, truth, mapping, epoch):
    model.eval()
    scores = []
    with torch.inference_mode():
        for batch, (inputs, _) in enumerate(DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0), 1):
            with autocast_for(device):
                prediction = model(inputs.to(device)).sigmoid()
            scores.extend(prediction[:, 0].float().cpu().tolist())
            if batch % 100 == 0:
                print(f'validation epoch{epoch}: scored {len(scores)}/{len(dataset)} proposals', flush=True)
    all_scores = {}
    offset = 0
    pools = {}
    for record in dataset.records:
        stem, count = Path(record['name']).stem, record['proposal_count']
        pools[stem] = read_raw(Path(args.output)/'proposals'/f'{stem}.json.gz')['instances']
        all_scores[stem] = scores[offset:offset+count]
        offset += count
    if offset != len(scores):
        raise ValueError('Validation image/proposal order drift')
    output = Path(args.output)
    reports = threshold_reports(pools, all_scores, truth, mapping)
    best = max(reports, key=lambda row: row['pq'])
    predictions = {stem: decode(items, all_scores[stem], best['threshold']) for stem, items in pools.items()}
    path = output/'validation_predictions.csv'
    pending = path.with_suffix('.pending')
    with pending.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['filament_id', 'segmentation_rle'])
        for stem, values in predictions.items():
            for i, value in enumerate(values):
                writer.writerow([f'{stem}_{i}', value['counts'].decode('ascii')])
    pending.replace(path)
    atomic_json(output/'latest_validation_scores.json', dict(epoch=epoch, scores=all_scores))
    return dict(epoch=epoch, thresholds=reports, selected=best)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_torch(path, value):
    pending = path.with_suffix('.pending')
    torch.save(value, pending)
    pending.replace(path)


def commit_best(output, checkpoint):
    """last.pt is authoritative; best files can be rebuilt after an interruption."""
    for name, value in checkpoint['artifacts'].items():
        pending = output/(name+'.pending')
        pending.write_bytes(value)
        pending.replace(output/name)
    atomic_torch(output/'best.pt', {key: value for key, value in checkpoint.items() if key != 'artifacts'})


def fit(model, dataset, args, device, identity, evaluator):
    """Completed-epoch resume; the evaluator never feeds an optimizer step."""
    output = Path(args.output)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*.05)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    model.to(device)
    history, start, best_pq, best_checkpoint = [], 0, -1., None
    last = output/'last.pt'
    if last.exists():
        checkpoint = torch.load(last, map_location='cpu', weights_only=False)  # Only this experiment's own checkpoint.
        if checkpoint['identity'] != identity:
            raise ValueError('Resume settings/source changed; keep settings or choose a new output folder')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        history, start, best_pq = checkpoint['history'], checkpoint['epoch'], checkpoint['best_pq']
        atomic_json(output/'history.json', history)
        best_checkpoint = checkpoint['best_checkpoint']
        if best_checkpoint is not None:
            commit_best(output, best_checkpoint)
        restore_rng(checkpoint['rng'])
        print(f'Resume completed epoch{start}/{args.epochs}', flush=True)
    if len(dataset) == 0:
        raise ValueError('No training proposals')
    for epoch in range(start+1, min(args.run_until, args.epochs)+1):
        frozen = epoch <= args.freeze_epochs
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(not frozen)
        model.train()
        if frozen:
            model.backbone.eval()
        generator = torch.Generator().manual_seed(args.seed+epoch)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
        total_loss, seen = 0., 0
        started = time.monotonic()
        for batch, (inputs, targets) in enumerate(loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_for(device):
                logits = model(inputs)
                loss = nn.functional.smooth_l1_loss(logits[:, 0].float().sigmoid(), targets[:, 0], beta=.1)
                loss = loss+.5*nn.functional.binary_cross_entropy_with_logits(logits[:, 1].float(), targets[:, 1])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())*len(inputs)
            seen += len(inputs)
            if batch % 100 == 0:
                print(f'train epoch{epoch}/{args.epochs} [{batch}/{len(loader)}] seen={seen} loss={total_loss/seen:.5f}', flush=True)
        scheduler.step()
        report = dict(epoch=epoch, loss=total_loss/seen, proposals_seen=seen, steps=len(loader),
                      seconds=time.monotonic()-started, backbone_frozen=frozen, lr=optimizer.param_groups[0]['lr'])
        improved = False
        if epoch == 1 or epoch % args.validate_every == 0 or epoch == args.epochs:
            report['validation'] = evaluator(model, epoch)
            pq = report['validation']['selected']['pq']
            if pq > best_pq:
                best_pq, improved = pq, True
                best_checkpoint = dict(model={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    epoch=epoch, identity=identity, selected=report['validation']['selected'],
                    architecture='resnet18.a1_in1k_4ch_2head', artifacts={})
                for source, target in [('validation_predictions.csv', 'best_validation_predictions.csv'),
                                       ('latest_validation_scores.json', 'best_validation_scores.json')]:
                    if (output/source).exists():
                        best_checkpoint['artifacts'][target] = (output/source).read_bytes()
        history.append(report)
        atomic_torch(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state(), history=history,
            epoch=epoch, identity=identity, best_pq=best_pq, best_checkpoint=best_checkpoint))
        if improved:
            commit_best(output, best_checkpoint)
        atomic_json(output/'history.json', history)
        print(f'epoch{epoch}/{args.epochs} loss={report["loss"]:.5f} seen={seen} steps={len(loader)} best_dev_PQ={best_pq:.6f} time={report["seconds"]:.1f}s', flush=True)
    return history, best_pq


def train(args):
    identity = source_identity(args)
    output = Path(args.output)
    if json.loads((output/'corpus_identity.json').read_text()) != identity:
        raise ValueError('Run prepare with this exact code/runtime/input before training')
    corpus = json.loads((output/'corpus.json').read_text())
    if corpus['identity'] != fingerprint(identity):
        raise ValueError('Corpus manifest identity mismatch')
    data = json.loads(Path(args.annotations).read_text())
    roles = read_splits(data, args.train_split, args.val_split)
    if {(row['role'], row['name']) for row in corpus['records']} != {(role, name) for role, names in roles.items() for name in names}:
        raise ValueError('Prepared corpus does not match fixed train/validation roles')
    trainset = QualityDataset(corpus['records'], output, args.cache, 'train')
    valset = QualityDataset(corpus['records'], output, args.cache, 'validation')
    truth, mapping = truth_for(data, set(roles['validation']))
    device = device_for(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
    training_identity = dict(corpus=corpus['identity'], epochs=args.epochs, batch_size=args.batch_size,
        freeze_epochs=args.freeze_epochs, lr=args.lr, seed=args.seed, validate_every=args.validate_every,
        precision='cuda_fp16' if device.type == 'cuda' else 'cpu_float32')
    config_path = output/'training_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != training_identity:
        raise ValueError('Training settings changed; resume only permits changing --run-until')
    atomic_json(config_path, training_identity)
    baseline_path = output/'baseline.json'
    if not baseline_path.exists():
        pools = {Path(row['name']).stem: read_raw(output/'proposals'/f'{Path(row["name"]).stem}.json.gz')['instances']
                 for row in corpus['records'] if row['role'] == 'validation'}
        reports = threshold_reports(pools, {stem: [item['confidence'] for item in items] for stem, items in pools.items()}, truth, mapping)
        baseline = max(reports, key=lambda row: row['pq'])
        atomic_json(baseline_path, dict(**baseline, thresholds=reports,
            reference_confidence025=next(row for row in reports if row['threshold'] == .25),
            protocol='same_raw_pool_same5thresholds_maskNMS0.5_ownership_minarea300'))
    model = QualityModel(pretrained=not (output/'last.pt').exists())
    print(f'Quality model only: {len(trainset)} train proposals / {len(valset)} all validation proposals; device={device}; epochs={args.epochs}', flush=True)
    history, best_pq = fit(model, trainset, args, device, training_identity,
                         lambda current, epoch: validation(current, valset, args, device, truth, mapping, epoch))
    result = dict(status='complete' if history and history[-1]['epoch'] == args.epochs else 'paused',
        epochs_completed=history[-1]['epoch'] if history else 0, final_epoch=args.epochs,
        best_development_pq=best_pq, baseline=json.loads(baseline_path.read_text()), scope=PROTOCOL['scope'],
        training_images=616, validation_images=91, training_proposals=len(trainset), validation_proposals=len(valset),
        selection='best model and one of five fixed thresholds on original91 development images; not independent OOF',
        baseline_caveat='YOLO-only same-pool comparator; not the historic YOLO+native ensemble')
    atomic_json(output/'summary.json', result)
    archive = output/'mask_quality_results.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name in ['summary.json', 'history.json', 'training_config.json', 'corpus_identity.json', 'corpus.json',
                     'baseline.json', 'best_validation_predictions.csv', 'best_validation_scores.json']:
            if (output/name).exists():
                bundle.write(output/name, name)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print('Result ZIP:', archive, flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--phase', choices=['prepare', 'train'], required=True)
    for name, value in dict(images=BASE/'MAGFiLO_1.0_Kaggle_2026/train/train_images',
        annotations=BASE/'MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json',
        train_split=YOLO_RUN/'train_split.csv', val_split=YOLO_RUN/'val_split.csv', yolo=YOLO_RUN/'best.pt',
        output=BASE/'fixed_split_experiments/mask_quality_v1', cache='/content/solar_mask_quality_cache').items():
        result.add_argument('--'+name.replace('_', '-'), default=str(value))
    for name, value in dict(epochs=20, run_until=20, batch_size=16, freeze_epochs=2, train_cap=48, validate_every=2, seed=42).items():
        result.add_argument('--'+name.replace('_', '-'), type=int, default=value)
    result.add_argument('--lr', type=float, default=1e-4)
    result.add_argument('--device', default='auto')
    return result


def main():
    args = parser().parse_args()
    if min(args.epochs, args.run_until, args.batch_size, args.train_cap, args.validate_every) <= 0 or not 0 <= args.freeze_epochs <= args.epochs or args.lr <= 0:
        raise ValueError('Invalid positive training settings')
    if args.run_until > args.epochs:
        raise ValueError('--run-until cannot exceed the fixed --epochs schedule')
    if str(Path(args.cache).resolve()).startswith(str(Path('/content/drive').resolve())):
        raise ValueError('ROI cache must be local, never on Drive')
    if 'fivefold' in str(args.output).lower() or Path(args.output).resolve() == Path(args.yolo).parent.resolve():
        raise ValueError('Use a new fixed_split_experiments output folder')
    (prepare if args.phase == 'prepare' else train)(args)


if __name__ == '__main__':
    main()
