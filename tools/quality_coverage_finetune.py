"""Fixed-split quality-model fine-tuning with rotating, coverage-balanced proposals.

Only the auxiliary ResNet18 is updated. YOLO/native predictions, split, limb
correction, ROI transform, loss, and the Q50 append rule remain frozen. Epochs
1..30 are ADDITIONAL epochs after the old quality checkpoint at epoch18.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import csv
from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import zipfile

import cv2
import numpy as np
from pycocotools import mask as mu
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from quality_coverage_sampling import select_training
from quality_append_api import append_q50
from submission_inference import limb

BASE = Path('/content/drive/MyDrive/filament-segmentation-2026')
ARCHITECTURE = 'resnet18.a1_in1k_4ch_2head'
CORPUS = 'fb2ecb8ce04cbc194b487618747e79f4a968d8145dcb01999f2b28d7c0b4f818'
ASSETS = {
    'old_corpus.json': '92ed7aa77262767f4af3bd8f5d6d96e15c12419a43071fd015bf88864ba9d3ae',
    'old_corpus_identity.json': '25c7f1303b0f1e718731250c623d47bf3b9d06e1576f2b95272034979161f366',
    'old_training_config.json': '4fc99bc271aedd38519caac9801f87ab2b26feccb88190de523eebf3587dd211',
    'old_best_validation_scores.json': '6858ec7a3c98c1225c21146e3617e1767820de1b8bbf434d82b28e7426f42a97',
    'full_baseline.csv': '9c95ced21b915d9e29f9d83d50153fb619b60bf34f7a483f8cf183d1c15b1d56',
    'validation_gt.csv': 'f1851c7b0e741443996fb416a6776a50f9294873d0e9840272c8de16616b6dea',
    'raw_views.jsonl': '86ce8a9f105216f3564eb39379f4f8e39f8533d35226db80e37d5ddc931113dc',
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    pending.replace(path)


def atomic_torch(path, value):
    path = Path(path)
    pending = path.with_name(path.name+'.pending')
    torch.save(value, pending)
    pending.replace(path)


def rle(value):
    return {'size': [2048, 2048], 'counts': value.encode('ascii')}


def read_raw(path, expected_sha):
    if digest(path) != expected_sha:
        raise ValueError(f'Raw proposal file changed: {path}')
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        value = json.load(stream)
    checksum = value.pop('checksum')
    if checksum != fingerprint(value):
        raise ValueError(f'Proposal checksum mismatch: {path}')
    return value


def read_baseline(path):
    csv.field_size_limit(2**31-1)
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['filament_id'].rsplit('_', 1)[0]].append(row)
    return dict(grouped)


def load_contract(args):
    """Pin the original split and raw candidate identities before reading targets."""
    root, old, assets = Path(args.root), Path(args.old_root), Path(args.assets)
    if root.resolve() == old.resolve() or old.resolve() in root.resolve().parents:
        raise ValueError('Use a NEW sibling experiment folder, never the original experiment')
    for name, expected in ASSETS.items():
        if digest(assets/name) != expected:
            raise ValueError(f'Bundled asset changed: {name}')
    for source, target in [('corpus.json', 'old_corpus.json'),
                           ('corpus_identity.json', 'old_corpus_identity.json'),
                           ('training_config.json', 'old_training_config.json')]:
        if digest(old/source) != ASSETS[target]:
            raise ValueError(f'Original quality experiment identity changed: {old/source}')
    corpus = json.loads((assets/'old_corpus.json').read_text())
    corpus_identity = json.loads((assets/'old_corpus_identity.json').read_text())
    if corpus['identity'] != CORPUS or fingerprint(corpus_identity) != CORPUS:
        raise ValueError('Original corpus identity mismatch')
    entries, seen = [], set()
    for index, row in enumerate(corpus['records'], 1):
        name, role = row['name'], row['role']
        if Path(name).name != name or name in seen or role not in {'train', 'validation'}:
            raise ValueError('Invalid or overlapping train/validation image manifest')
        seen.add(name)
        raw = read_raw(old/'proposals'/f'{Path(name).stem}.json.gz', row['raw_sha'])
        identity = raw['identity']
        if identity['name'] != name or identity['role'] != role or identity['corpus'] != CORPUS:
            raise ValueError(f'Raw proposal role/image identity mismatch: {name}')
        if len(raw['instances']) != row['proposal_count']:
            raise ValueError(f'Raw candidate count mismatch: {name}')
        instances = []
        for candidate in raw['instances']:
            item = {key: candidate[key] for key in ['rle', 'confidence']}
            if not np.isfinite(item['confidence']) or not 0 <= item['confidence'] <= 1:
                raise ValueError(f'Invalid confidence in {name}')
            if role == 'train':
                target = np.asarray(candidate['target'], dtype=np.float32)
                if target.shape != (2,) or not np.isfinite(target).all() or np.any((target < 0) | (target > 1)):
                    raise ValueError(f'Invalid training target in {name}')
                item['target'] = target.tolist()
            # Validation targets are intentionally not propagated to any dataset.
            instances.append(item)
        entries.append(dict(name=name, role=role, raw_sha=row['raw_sha'],
                            image_sha=identity['image_sha'], instances=instances))
        if index % 50 == 0 or index == len(corpus['records']):
            print(f'Verify cached proposals {index}/{len(corpus["records"])}', flush=True)
    counts = Counter(row['role'] for row in entries)
    totals = {role: sum(len(row['instances']) for row in entries if row['role'] == role)
              for role in ['train', 'validation']}
    if counts != {'train': 616, 'validation': 91} or totals != {'train': 53535, 'validation': 7172}:
        raise ValueError(f'Unexpected split/candidate counts: {counts} / {totals}')
    baseline = read_baseline(assets/'full_baseline.csv')
    validation_names = {Path(row['name']).stem for row in entries if row['role'] == 'validation'}
    if set(baseline) - validation_names or sum(map(len, baseline.values())) != 626:
        raise ValueError('Full baseline images/count differ from the fixed validation split')
    # Three validation images have zero baseline masks and therefore no CSV rows.
    baseline = {stem: baseline.get(stem, []) for stem in sorted(validation_names)}
    code_names = ['quality_coverage_finetune.py', 'quality_coverage_sampling.py',
                  'quality_append_api.py', 'submission_inference.py', 'filament_eval.py']
    identity = dict(protocol='quality_coverage_fixed_split_v1', corpus=CORPUS,
        assets=ASSETS, old_checkpoint_sha=digest(old/'best.pt'),
        source={name: digest(Path(__file__).parent/name) for name in code_names},
        architecture=ARCHITECTURE, additional_epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed, cap_per_image=48, validate_every=5,
        loss='Huber(sigmoid(head0),IoU,beta=.1)+.5*BCEWithLogits(head1,match)',
        preprocessing='original_limb_ROI256_pad16_scale1.25_RGB_ImageNet_mask4ch',
        selection='best full baseline plus frozen Q50 PQ, epoch0 included; no threshold search')
    return entries, identity, baseline


def roi_box(mask):
    yy, xx = np.nonzero(mask)
    if not len(yy):
        raise ValueError('Empty ROI mask')
    h, w = int(yy.max()-yy.min()+1), int(xx.max()-xx.min()+1)
    cy, cx = (yy.min()+yy.max()+1)/2, (xx.min()+xx.max()+1)/2
    hh, hw = h*1.25/2+16, w*1.25/2+16
    return (max(0, int(np.floor(cx-hw))), max(0, int(np.floor(cy-hh))),
            min(mask.shape[1], int(np.ceil(cx+hw))), min(mask.shape[0], int(np.ceil(cy+hh))))


def roi_array(corrected, value, size=256):
    """Pixel-for-pixel copy of the old quality-model input transform."""
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


def normalize(array):
    value = torch.from_numpy(array.copy()).float()/255
    rgb = value[0:1].repeat(3, 1, 1)
    rgb = (rgb-torch.tensor([.485, .456, .406])[:, None, None])/torch.tensor([.229, .224, .225])[:, None, None]
    return torch.cat([rgb, value[1:2]], dim=0)


def cache_identity(entry, values, kind):
    return dict(raw_sha=entry['raw_sha'], image_sha=entry['image_sha'],
                role=entry['role'], kind=kind, transform='old_exact_roi256_v1',
                ordered_rle_sha=fingerprint(values), count=len(values))


def cache_valid(path, identity):
    meta = path.with_suffix('.json')
    if not path.is_file() or not meta.is_file():
        return False
    saved = json.loads(meta.read_text())
    return saved.get('identity') == identity and saved.get('array_sha') == digest(path)


def prepare_cache(entries, args, baseline):
    """Build all ROIs locally; safe to interrupt at any image and rerun."""
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    jobs = []
    for entry in entries:
        stem = Path(entry['name']).stem
        jobs.append((entry, cache/f'{stem}.npy', [v['rle'] for v in entry['instances']], 'proposals'))
        if entry['role'] == 'validation':
            values = [v['segmentation_rle'] for v in baseline[stem] if v['segmentation_rle']]
            jobs.append((entry, cache/f'{stem}_incumbents.npy', values, 'incumbents'))
    pending = []
    for entry, path, values, kind in jobs:
        identity = cache_identity(entry, values, kind)
        if not cache_valid(path, identity):
            pending.append((entry, path, values, identity))
    needed = sum(len(values)*2*256*256+1024 for _, _, values, _ in pending)
    if shutil.disk_usage(cache).free < needed+256*1024**2:
        raise OSError(f'Local ROI cache needs {needed/1024**3:.2f} GiB free; cache={cache}')
    print(f'ROI prepare: {len(jobs)-len(pending)}/{len(jobs)} verified; build {len(pending)} arrays ({needed/1024**3:.2f} GiB)', flush=True)
    last_name, corrected = None, None
    for index, (entry, path, values, identity) in enumerate(pending, 1):
        if last_name != entry['name']:
            image_path = Path(args.images)/entry['name']
            if digest(image_path) != entry['image_sha']:
                raise ValueError(f'Original image hash differs from frozen proposal image: {image_path}')
            corrected = limb(cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE))
            last_name = entry['name']
        temporary = path.with_name(path.name+'.pending')
        array = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8, shape=(len(values), 2, 256, 256))
        for i, counts in enumerate(values):
            array[i] = roi_array(corrected, rle(counts))
        array.flush()
        del array
        temporary.replace(path)
        atomic_json(path.with_suffix('.json'), dict(identity=identity, array_sha=digest(path)))
        print(f'ROI [{index}/{len(pending)}] {entry["role"]} {entry["name"]} {identity["kind"]} n={len(values)}', flush=True)


class QualityDataset(Dataset):
    def __init__(self, entries, cache, role, seed=20261103):
        if role not in {'train', 'validation'}:
            raise ValueError('Dataset role must be train or validation')
        self.entries = [row for row in entries if row['role'] == role]
        self.cache, self.role, self.seed = Path(cache), role, seed
        self.rows = []
        self.set_epoch(1)

    def set_epoch(self, epoch):
        self.rows = []
        for i, entry in enumerate(self.entries):
            if self.role == 'train':
                selected = select_training(entry['instances'], epoch=epoch, seed=self.seed,
                                           image=entry['name'], role=self.role, cap=48)
            else:
                selected = range(len(entry['instances']))
            self.rows.extend((i, j) for j in selected)

    @lru_cache(maxsize=3)
    def array(self, index):
        return np.load(self.cache/f'{Path(self.entries[index]["name"]).stem}.npy', mmap_mode='r', allow_pickle=False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        i, j = self.rows[index]
        inputs = normalize(self.array(i)[j])
        if self.role == 'validation':
            return inputs, index
        return inputs, torch.tensor(self.entries[i]['instances'][j]['target'], dtype=torch.float32)


class QualityModel(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model('resnet18.a1_in1k', pretrained=False, num_classes=0)
        old = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding,
                                        bias=old.bias is not None)
        self.head = nn.Linear(self.backbone.num_features, 2)

    def forward(self, value):
        return self.head(self.backbone(value))


def autocast(device):
    return torch.amp.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()


def rng_state():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=(state[0], state[1].tolist(), *state[2:]),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def commit_best(root, checkpoint):
    for name, value in checkpoint['artifacts'].items():
        pending = root/(name+'.pending')
        pending.write_bytes(value)
        pending.replace(root/name)
    atomic_torch(root/'best.pt', {k: v for k, v in checkpoint.items() if k != 'artifacts'})


def snapshot(model, epoch, identity, report, root):
    artifacts = {}
    for filename in ['validation_scores.json', 'validation_predictions.csv', 'incumbent_scores.json', 'validation_report.json']:
        if (root/filename).exists():
            artifacts['best_'+filename] = (root/filename).read_bytes()
    return dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                epoch=epoch, identity=identity, architecture=ARCHITECTURE,
                selected=report, artifacts=artifacts)


def fit(model, dataset, args, device, identity, evaluator):
    """Injectable training core; last.pt atomically commits a completed epoch."""
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*.05)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    history, start, best, best_pq = [], 0, None, -1.
    path = root/'last.pt'
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['identity'] != identity:
            raise ValueError('Resume source/data/settings changed; only --run-until may change')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        history, start = saved['history'], saved['epoch']
        best, best_pq = saved['best_checkpoint'], saved['best_pq']
        restore_rng(saved['rng'])
        commit_best(root, best)
        atomic_json(root/'history.json', history)
        print(f'Resume additional epoch {start}/{args.epochs}', flush=True)

    def save(epoch):
        atomic_torch(path, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng=rng_state(),
            identity=identity, epoch=epoch, history=history, best_pq=best_pq, best_checkpoint=best))
        commit_best(root, best)
        atomic_json(root/'history.json', history)

    if best is None:
        print('Evaluate epoch0: unchanged old quality epoch18, full baseline + frozen Q50', flush=True)
        control = evaluator(model, 0)
        best_pq = float(control['pq'])
        best = snapshot(model, 0, identity, control, root)
        history.append(dict(epoch=0, validation=control, role='warmstart_control'))
        for name in ['validation_scores.json', 'validation_predictions.csv', 'incumbent_scores.json', 'validation_report.json']:
            if (root/name).exists():
                shutil.copyfile(root/name, root/('control_'+name))
        save(0)
    for epoch in range(start+1, min(args.run_until, args.epochs)+1):
        dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(args.seed+epoch)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
                            num_workers=0, pin_memory=device.type == 'cuda')
        model.train()
        total, seen, started = 0., 0, time.monotonic()
        for batch, (inputs, targets) in enumerate(loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                logits = model(inputs)
                loss = nn.functional.smooth_l1_loss(logits[:, 0].float().sigmoid(), targets[:, 0], beta=.1)
                loss = loss+.5*nn.functional.binary_cross_entropy_with_logits(logits[:, 1].float(), targets[:, 1])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite training loss at epoch{epoch}, batch{batch}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            seen += len(inputs)
            total += float(loss.detach())*len(inputs)
            if batch == 1 or batch % 100 == 0 or batch == len(loader):
                print(f'Train additional epoch {epoch}/{args.epochs} [{batch}/{len(loader)}] loss={total/seen:.6f}', flush=True)
        if not seen:
            raise ValueError('No training examples selected')
        scheduler.step()
        report = dict(epoch=epoch, loss=total/seen, proposals_seen=seen, steps=len(loader),
                      lr=optimizer.param_groups[0]['lr'], seconds=time.monotonic()-started)
        if epoch % 5 == 0 or epoch == args.epochs:
            report['validation'] = evaluator(model, epoch)
            pq = float(report['validation']['pq'])
            if pq > best_pq:
                best_pq = pq
                best = snapshot(model, epoch, identity, report['validation'], root)
        history.append(report)
        save(epoch)
        print(f'Committed completed epoch {epoch}/{args.epochs}; best fixed-Q50 PQ={best_pq:.6f} at epoch{best["epoch"]}', flush=True)
    return history, best


def check_cache(entries, args, baseline):
    for entry in entries:
        stem = Path(entry['name']).stem
        values = [v['rle'] for v in entry['instances']]
        if not cache_valid(Path(args.cache)/f'{stem}.npy', cache_identity(entry, values, 'proposals')):
            raise ValueError(f'Local ROI cache missing/changed: {stem}; run --phase prepare')
        if entry['role'] == 'validation':
            values = [v['segmentation_rle'] for v in baseline[stem] if v['segmentation_rle']]
            if not cache_valid(Path(args.cache)/f'{stem}_incumbents.npy', cache_identity(entry, values, 'incumbents')):
                raise ValueError(f'Incumbent cache missing/changed: {stem}; run --phase prepare')


def evaluate_model(model, entries, baseline, geometry, args, device, epoch):
    """Compute predictions first. Only the evaluator then reads validation GT."""
    from filament_eval import read_csv, evaluate
    model.eval()
    root, all_scores, incumbent_scores, outputs, audits = Path(args.root), {}, {}, {}, {}
    selected = [row for row in entries if row['role'] == 'validation']
    with torch.inference_mode():
        for index, entry in enumerate(selected, 1):
            stem = Path(entry['name']).stem
            def score(path):
                data = np.load(path, mmap_mode='r', allow_pickle=False)
                values = []
                for offset in range(0, len(data), args.batch_size):
                    inputs = torch.stack([normalize(x) for x in data[offset:offset+args.batch_size]]).to(device)
                    with autocast(device):
                        prediction = model(inputs).sigmoid()
                    values.extend(prediction[:, 0].float().cpu().tolist())
                return values
            qualities = score(Path(args.cache)/f'{stem}.npy')
            all_scores[stem] = qualities
            incumbent_scores[stem] = score(Path(args.cache)/f'{stem}_incumbents.npy')
            originals = [rle(row['segmentation_rle']) for row in baseline[stem] if row['segmentation_rle']]
            output, audit, decisions = append_q50(originals, entry['instances'], qualities, geometry[stem])
            if output[:len(originals)] != originals:
                raise ValueError('Q50 changed an existing baseline mask')
            outputs[stem] = output
            audits[stem] = dict(stats=audit, decisions=decisions)
            print(f'Validation epoch{epoch} [{index}/{len(selected)}] {stem}: {len(originals)} -> {len(output)} masks', flush=True)
    csv_path = root/'validation_predictions.csv'
    temporary = csv_path.with_name(csv_path.name+'.pending')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['filament_id', 'segmentation_rle'])
        for stem, output in outputs.items():
            original_rows = baseline[stem]
            nonempty = [row for row in original_rows if row['segmentation_rle']]
            for row in original_rows:
                writer.writerow([row['filament_id'], row['segmentation_rle']])
            suffix = max((int(row['filament_id'].rsplit('_', 1)[1]) for row in original_rows), default=-1)+1
            for i, value in enumerate(output[len(nonempty):]):
                counts = value['counts']
                writer.writerow([f'{stem}_{suffix+i}', counts.decode('ascii') if isinstance(counts, bytes) else counts])
    temporary.replace(csv_path)
    atomic_json(root/'validation_scores.json', dict(epoch=epoch, scores=all_scores))
    atomic_json(root/'incumbent_scores.json', dict(epoch=epoch, scores=incumbent_scores,
        use='diagnostics only; never used for mask selection or training'))
    truth = read_csv(Path(args.assets)/'validation_gt.csv', ground_truth=True)
    metrics = evaluate(truth, read_csv(csv_path), 2048, 2048)
    baseline_metrics = evaluate(truth, read_csv(Path(args.assets)/'full_baseline.csv'), 2048, 2048)
    if abs(baseline_metrics['pq']-.4722735674653581) > 1e-10:
        raise ValueError('Full frozen baseline PQ does not reproduce')
    report = dict(epoch=epoch, **metrics, baseline_pq=baseline_metrics['pq'],
                  delta_baseline=metrics['pq']-baseline_metrics['pq'], audits=audits,
                  rule='fixed_Q50_append_only', split='fixed91 development; not independent OOF')
    if epoch == 0:
        previous = json.loads((Path(args.assets)/'old_best_validation_scores.json').read_text())['scores']
        if set(previous) != set(all_scores) or any(len(previous[k]) != len(v) for k, v in all_scores.items()):
            raise ValueError('Old/new quality validation candidate ordering/count differs')
        delta = np.concatenate([np.abs(np.asarray(all_scores[k])-np.asarray(previous[k])) for k in sorted(all_scores)])
        report['warmstart_replay'] = dict(candidates=len(delta), score_mean_absolute_drift=float(delta.mean()),
            score_max_absolute_drift=float(delta.max()), previous_fixed_Q50_pq=.4725866370653959,
            pq_delta_previous=metrics['pq']-.4725866370653959,
            note='Runtime/AMP/batch-kernel differences are reported; current epoch0 remains the control')
    atomic_json(root/'validation_report.json', report)
    print(f'Validation epoch{epoch}: PQ={metrics["pq"]:.6f} TP={metrics["tp_pairs"]} FP={metrics["fp"]} FN={metrics["fn"]}', flush=True)
    return {key: report[key] for key in ['epoch', 'pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn', 'baseline_pq', 'delta_baseline']}


def prepare(args):
    entries, identity, baseline = load_contract(args)
    prepare_cache(entries, args, baseline)
    atomic_json(Path(args.root)/'prepared_manifest.json', dict(identity=identity,
        images={row['name']: dict(role=row['role'], raw_sha=row['raw_sha'], image_sha=row['image_sha'],
                                  count=len(row['instances'])) for row in entries}))


def train(args):
    entries, identity, baseline = load_contract(args)
    check_cache(entries, args, baseline)
    prepared = json.loads((Path(args.root)/'prepared_manifest.json').read_text())
    if prepared['identity'] != identity:
        raise ValueError('Preparation/training settings differ; rerun the matching prepare stage')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; select a T4 Colab runtime')
    identity = dict(identity, runtime=dict(torch=str(torch.__version__), numpy=np.__version__,
                                          cv2=cv2.__version__, device=device.type))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
    model = QualityModel()
    old = torch.load(Path(args.old_root)/'best.pt', map_location='cpu', weights_only=True)
    expected = json.loads((Path(args.assets)/'old_training_config.json').read_text())
    if old['epoch'] != 18 or old['architecture'] != ARCHITECTURE or old['identity'] != expected:
        raise ValueError('Warm start must be the original quality epoch18 checkpoint')
    model.load_state_dict(old['model'], strict=True)
    geometry = defaultdict(list)
    with (Path(args.assets)/'raw_views.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            geometry[row['image']].append(row)
    for entry in entries:
        if entry['role'] != 'validation':
            continue
        stem = Path(entry['name']).stem
        joined = {}
        for view in geometry[stem]:
            for value in view['instances']:
                joined[value['rle']] = max(joined.get(value['rle'], -1), value['confidence'])
        if any(value['rle'] not in joined or abs(value['confidence']-joined[value['rle']]) > 1e-7 for value in entry['instances']):
            raise ValueError(f'Validation geometry/raw proposal ordering join failed: {stem}')
    dataset = QualityDataset(entries, args.cache, 'train', args.seed)
    config_path = Path(args.root)/'training_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != identity:
        raise ValueError('Saved training configuration changed; only --run-until may change')
    atomic_json(config_path, identity)
    history, best = fit(model, dataset, args, device, identity,
        lambda current, epoch: evaluate_model(current, entries, baseline, geometry, args, device, epoch))
    root = Path(args.root)
    result = dict(status='complete' if history[-1]['epoch'] == args.epochs else 'paused',
        epochs_completed=history[-1]['epoch'], additional_epochs=args.epochs, best_epoch=best['epoch'],
        best_pq=best['selected']['pq'], control_pq=history[0]['validation']['pq'],
        baseline_pq=.4722735674653581, archive=str(root/'quality_coverage_results.zip'),
        scope='One auxiliary quality model; YOLO and best84 frozen; original fixed train616/valid91',
        selection='fixed Q50 pipeline PQ; includes unchanged warmstart epoch0; no threshold tuning',
        caveat='Development PQ, not test/LB or independent cross-validation; no public test reference used',
        attribution='Single fine-tuning arm: gains cannot isolate sampler change from additional training',
        reference_tracking={name: dict(measured=False, matched_count=None, scope='validation-only experiment')
                            for name in ['reference216', 'reference102']},
        identity=fingerprint(identity))
    atomic_json(root/'result.json', result)
    archive = Path(result['archive'])
    pending = archive.with_suffix('.pending')
    with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix in {'.json', '.csv'}:
                bundle.write(path, path.name)
    pending.replace(archive)
    print(json.dumps(result, indent=2), flush=True)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--phase', choices=['prepare', 'train', 'all'], default='all')
    result.add_argument('--root', type=Path, default=BASE/'fixed_split_experiments/quality_coverage_t4_v1')
    result.add_argument('--old-root', type=Path, default=BASE/'fixed_split_experiments/mask_quality_v1')
    result.add_argument('--images', type=Path, default=BASE/'MAGFiLO_1.0_Kaggle_2026/train/train_images')
    result.add_argument('--cache', type=Path, default=Path('/content/solar_quality_coverage_v1/roi'))
    result.add_argument('--assets', type=Path, required=True)
    result.add_argument('--run-until', type=int, default=30)
    result.add_argument('--epochs', type=int, default=30)
    result.add_argument('--batch-size', type=int, default=16)
    result.add_argument('--lr', type=float, default=3e-5)
    result.add_argument('--seed', type=int, default=20261103)
    result.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    return result


def main():
    args = parser().parse_args()
    if not 0 <= args.run_until <= args.epochs or args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError('Require 0 <= run-until <= additional epochs; positive batch/lr/epochs')
    if args.phase == 'all':
        # Separate processes release all CPU ROI preparation state before CUDA starts.
        tail = sys.argv[1:]
        if '--phase' in tail:
            position = tail.index('--phase')
            tail = tail[:position]+tail[position+2:]
        for phase in ['prepare', 'train']:
            subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), *tail, '--phase', phase], check=True)
    elif args.phase == 'prepare':
        prepare(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
