"""Audit and re-score fixed-final-epoch OOF exports, including blank predictions.

Reports partial folds honestly; complete OOF PQ pools matched IoU and counts.
This reads prediction exports, never trains a model or selects a checkpoint.
"""
import argparse
from collections import Counter
import csv
from datetime import date
import hashlib
import io
import json
import math
from pathlib import Path
import re

from pycocotools import mask as mu
from grouped_cv_training import IDENTITY_KEYS, digest, metric


def checked_rle(value):
    """Bound compressed runs before calling the native COCO decoder."""
    if not isinstance(value, dict) or value.get('size') != [2048, 2048]:
        raise ValueError('Expected2048-square RLE')
    counts = value.get('counts')
    if not isinstance(counts, str) or not counts:
        raise ValueError('Expected nonempty compressed RLE string')
    previous, total, area, index, position = [], 0, 0, 0, 0
    while position < len(counts):
        number = shift = 0
        while True:
            if position >= len(counts):
                raise ValueError('Truncated compressed RLE')
            digit = ord(counts[position])-48
            position += 1
            if not 0 <= digit <= 63:
                raise ValueError('Invalid compressed RLE character')
            number |= (digit & 31) << shift
            shift += 5
            if shift > 65:
                raise ValueError('Compressed RLE integer overflow')
            if not digit & 32:
                if digit & 16:
                    number |= -1 << shift
                break
        if index > 2:
            number += previous[-2]
        if number < 0 or total+number > 2048**2:
            raise ValueError('Compressed RLE run outside image')
        total += number
        area += number if index % 2 else 0
        previous.append(number)
        previous = previous[-2:]
        index += 1
    if total != 2048**2 or area <= 0:
        raise ValueError('Invalid/empty instance RLE')
    result = dict(size=[2048, 2048], counts=counts.encode('ascii'))
    if int(mu.area(result)) != area:
        raise ValueError('Compressed RLE area mismatch')
    return result


def split_sha(plan, fold_info, role):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['image_id', 'file_name'])
    writer.writerows((sid, row['file_name']) for row in plan['observations']
                     if fold_info['roles'][row['physical_identifier']] == role
                     for sid in row['annotation_ids'])
    return hashlib.sha256(stream.getvalue().encode()).hexdigest()


def aggregate(rows):
    result = {key: sum(row[key] for row in rows)
              for key in ('tp_pairs', 'fp', 'fn', 'matched_iou_sum')}
    den = result['tp_pairs']+.5*(result['fp']+result['fn'])
    result.update(pq=result['matched_iou_sum']/den if den else 0.,
        sq=result['matched_iou_sum']/result['tp_pairs'] if result['tp_pairs'] else 0.,
        rq=result['tp_pairs']/den if den else 0., annotation_groups=len(rows),
        physical_images=len({row['image'] for row in rows}))
    return result


def truth_for(data, allowed):
    truth = {str(row['id']): [] for row in data['images'] if row['file_name'] in allowed}
    mapping = {str(row['id']): Path(row['file_name']).stem for row in data['images'] if row['file_name'] in allowed}
    for ann in data['annotations']:
        if str(ann['image_id']) not in truth:
            continue
        seg = ann['segmentation']
        value = (mu.merge(mu.frPyObjects(seg, 2048, 2048)) if isinstance(seg, list)
                 else mu.frPyObjects(seg, 2048, 2048) if isinstance(seg['counts'], list)
                 else dict(seg))
        if isinstance(value['counts'], str):
            value['counts'] = value['counts'].encode()
        if value['size'] != [2048, 2048] or int(mu.area(value)) <= 0:
            raise ValueError('Invalid validation ground truth')
        truth[str(ann['image_id'])].append(value)
    return truth, mapping


def audit_fold(folder, data, plan, source_shas, fold):
    folder = Path(folder)
    summary = json.loads((folder/'oof_summary.json').read_text())
    cfg = json.loads((folder/'config.json').read_text())
    payload = cfg.get('cv_identity_payload')
    if not isinstance(payload, dict) or hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != cfg['cv_identity']:
        raise ValueError('Unverifiable CV identity payload')
    if (payload.get('settings') != {key: cfg[key] for key in IDENTITY_KEYS}
            or payload.get('fold') != fold or payload.get('sources') != cfg['source_sha256']
            or payload.get('precision') != cfg['amp_dtype'] or payload.get('final_selection') != 'fixed_epoch'):
        raise ValueError('CV identity disagrees with configuration')
    if any(cfg.get(key) for key in ('initial_checkpoint', 'warm_start', 'checkpoint', 'resume_dir')) or cfg['synthetic_ratio'] != 0:
        raise ValueError('Unverified warm start or synthetic data in outer-fold model')
    if (cfg['fold'] != fold or summary['fold'] != fold or summary['status'] != 'complete'
            or summary['selected_epoch'] != cfg['epochs'] or summary['epochs_completed'] != cfg['epochs']
            or summary['fixed_final_epoch'] != cfg['epochs'] or not summary['oof_score_available']
            or summary.get('diagnostic_scores_select_checkpoint') is not False
            or summary['cv_identity'] != cfg['cv_identity'] or summary['source_sha256'] != cfg['source_sha256']):
        raise ValueError('Only the fixed-final-epoch checkpoint qualifies as OOF')
    fold_info = next(row for row in plan['fold_summaries'] if row['fold'] == fold)
    for key, expected in dict(source_shas, train_split=split_sha(plan, fold_info, 'train'),
                             val_split=split_sha(plan, fold_info, 'validation')).items():
        if cfg['source_sha256'].get(key) != expected:
            raise ValueError('OOF source/split provenance differs: '+key)
    train = sorted(key for key, role in fold_info['roles'].items() if role == 'train')
    val = {row['file_name'] for row in plan['observations'] if fold_info['roles'][row['physical_identifier']] == 'validation'}
    grouped = {role: [row for row in plan['observations'] if fold_info['roles'][row['physical_identifier']] == role]
               for role in ('train', 'validation')}
    if not all(grouped.values()) or min(abs((date.fromisoformat(a['capture_day'])-date.fromisoformat(b['capture_day'])).days)
            for a in grouped['train'] for b in grouped['validation']) <= 3:
        raise ValueError('Training/validation date embargo was violated')
    if {r['block'] for r in grouped['train']} & {r['block'] for r in grouped['validation']}:
        raise ValueError('A temporal block crosses training/validation')
    if train != summary['train_physical_identifiers'] or len(summary['image_names']) != len(val) or set(summary['image_names']) != val:
        raise ValueError('Training membership or validation coverage changed')
    if summary['threshold'] != cfg['threshold'] or summary['min_area'] != cfg['min_area']:
        raise ValueError('Prediction decoding settings changed')
    checkpoint_sha = summary['checkpoint_sha256']
    if not re.fullmatch('[0-9a-f]{64}', checkpoint_sha):
        raise ValueError('Invalid checkpoint fingerprint')
    if (folder/'last.pt').exists() and digest(folder/'last.pt') != checkpoint_sha:
        raise ValueError('Checkpoint bytes differ from exported predictions')
    for name, field in [('oof_predictions.csv', 'oof_csv_sha256'), ('oof_records.jsonl', 'oof_records_sha256')]:
        if digest(folder/name) != summary[field]:
            raise ValueError('Export file checksum differs: '+name)
    records = [json.loads(line) for line in (folder/'oof_records.jsonl').read_text().splitlines()]
    if len(records) != len(val) or summary['image_coverage'] != len(val) or {r['file_name'] for r in records} != val:
        raise ValueError('Every validation image, including empty predictions, must have one record')
    predictions = {}
    for record in records:
        if (record['fold'] != fold or record['cv_identity'] != cfg['cv_identity']
                or record['checkpoint_sha256'] != checkpoint_sha or record['selected_epoch'] != cfg['epochs']
                or not re.fullmatch('[0-9a-f]{64}', record['image_sha256'])):
            raise ValueError('Per-image checkpoint/source provenance changed')
        decoded = {}
        for threshold, values in record['components'].items():
            rles = [checked_rle(row['rle']) for row in values]
            if any(int(mu.area(rle)) != row['area'] or row['area'] < record.get('component_min_area', cfg['min_area'])
                   for row, rle in zip(values, rles)):
                raise ValueError('Invalid component area')
            if rles and int(mu.area(mu.merge(rles))) != sum(int(mu.area(rle)) for rle in rles):
                raise ValueError('Overlapping/duplicate predictions are not accepted')
            decoded[threshold] = [rle for row, rle in zip(values, rles) if row['area'] >= cfg['min_area']]
        predictions[Path(record['file_name']).stem] = decoded[f"{cfg['threshold']:g}"]
    expected_rows = Counter((f'{image}_{index}', rle['counts'].decode()) for image, values in predictions.items()
                            for index, rle in enumerate(values, 1))
    with (folder/'oof_predictions.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['filament_id', 'segmentation_rle']:
            raise ValueError('Unexpected prediction CSV schema')
        actual_rows = Counter((row['filament_id'], row['segmentation_rle']) for row in reader)
    if actual_rows != expected_rows:
        raise ValueError('CSV masks differ from the full-coverage prediction records')
    truth, mapping = truth_for(data, val)
    measured = metric(truth, predictions, mapping)
    for field in ('tp_pairs', 'fp', 'fn', 'matched_iou_sum', 'pq', 'sq', 'rq'):
        if not math.isclose(measured[field], summary['metric'][field], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError('Saved metric does not reproduce from CSV/RLE: '+field)
    if measured['per_group'] != summary['metric']['per_group']:
        raise ValueError('Per-annotator metrics do not reproduce')
    return measured, cfg, dict(fold=fold, images=len(val), checkpoint_bytes_checked=(folder/'last.pt').is_file(),
                              csv_sha256=summary['oof_csv_sha256'], **aggregate(measured['per_group']))


def evaluate_oof(annotations, fold_plan, runs_root):
    annotations, fold_plan, runs_root = map(Path, (annotations, fold_plan, runs_root))
    data, plan = json.loads(annotations.read_text()), json.loads(fold_plan.read_text())
    source_shas = dict(annotations=digest(annotations), fold_plan=digest(fold_plan))
    if plan['sources_sha256']['annotations'] != source_shas['annotations']:
        raise ValueError('Fold plan annotation source differs')
    by_name = {row['file_name']: row for row in plan['observations']}
    by_stem = {Path(name).stem: row for name, row in by_name.items()}
    metadata = {sid: row['file_name'] for row in plan['observations'] for sid in row['annotation_ids']}
    if metadata != {str(row['id']): row['file_name'] for row in data['images']} or len(metadata) != len(data['images']):
        raise ValueError('Fold metadata does not cover every annotation variant')
    all_rows, fold_reports, blanks, blank_fp, regime = [], [], 0, 0, None
    for fold in range(plan['folds']):
        folder = runs_root/f'fold{fold}'
        if not (folder/'oof_summary.json').is_file():
            continue
        measured, cfg, report = audit_fold(folder, data, plan, source_shas, fold)
        current = ({key: cfg[key] for key in IDENTITY_KEYS if key != 'seed'},
                   {key: cfg['source_sha256'][key] for key in ('base_runtime', 'cv_runner')})
        if regime is not None and current != regime:
            raise ValueError('Fold model/training regimes differ; do not pool different experiments')
        regime = current
        all_rows.extend(measured['per_group']); fold_reports.append(report)
        blanks += measured['blank_gt_groups']; blank_fp += measured['fp_on_blank_gt_groups']
    complete = len(fold_reports) == plan['folds']
    if len({row['group_id'] for row in all_rows}) != len(all_rows):
        raise ValueError('An annotation group was evaluated in multiple folds')
    completed_images = sum(row['images'] for row in fold_reports)
    if complete and completed_images != len(by_name):
        raise ValueError('Complete OOF prediction coverage differs from the plan')
    pooled = aggregate(all_rows) if fold_reports else None
    site = {row['image']: by_stem[row['image']]['site'] for row in all_rows}
    def period(image):
        year = int(by_stem[image]['capture_day'][:4])
        return next((f'{lo}-{hi}' for lo, hi in [(2011, 2014), (2015, 2018), (2019, 2022)]
                     if lo <= year <= hi), 'outside_predeclared_periods')
    return dict(status='complete_oof' if complete else 'partial_folds' if fold_reports else 'awaiting_completed_folds',
        complete_oof=complete, completed_folds=[row['fold'] for row in fold_reports],
        missing_folds=sorted(set(range(plan['folds']))-{row['fold'] for row in fold_reports}),
        expected_physical_images=len(by_name), completed_physical_images=completed_images,
        pooled_oof=pooled if complete else None, pooled_completed_folds=pooled,
        by_fold=fold_reports, by_site={value: aggregate([r for r in all_rows if site[r['image']] == value])
                                     for value in sorted(set(site.values()))},
        by_period={value: aggregate([r for r in all_rows if period(r['image']) == value])
                   for value in sorted({period(r['image']) for r in all_rows})},
        blank_gt_groups=blanks, fp_on_blank_gt_groups=blank_fp, source_sha256=source_shas,
        aggregation='Sum matched IoU, TP, FP and FN before computing PQ; not mean fold PQ.',
        limitation='Internal grouped validation; the previous91 images influenced recipe development. No independent LB claim.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('annotations', 'fold-plan', 'runs-root', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    report = evaluate_oof(args.annotations, args.fold_plan, args.runs_root)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
