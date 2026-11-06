"""Six frozen native OOF methods at one common fivefold learning budget.

All prediction CSVs are generated before the annotation JSON is decoded. A
stage with fewer than five completed folds remains a partial-stage report.
"""
import argparse
from collections import Counter
import csv
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re

from pycocotools import mask as mu

from evaluate_grouped_oof import aggregate, checked_rle, split_sha, truth_for
from grouped_cv_training import IDENTITY_KEYS, digest, metric
from submission_native_recipe import _final_admission_pruning, _refine_boundaries

METHODS = {
    'base': dict(branch='base', threshold=.5, minimum_area=300),
    'tta': dict(branch='four_flip_mean', threshold=.5, minimum_area=300),
    'clahe': dict(branch='limb_then_clahe', clip_limit=2., grid=[8, 8], threshold=.5, minimum_area=300),
    'guarded_boundary': dict(original='base0.5_area300', peers=['base0.65', 'tta0.65'],
        minimum_peer_iou=.3, minimum_peer_mean=.8, majority_votes=2,
        minimum_original_iou=.85, area_ratio=[.5, 1.5]),
    'guarded_consensus': dict(boundary='guarded_boundary', admission_threshold=.35,
        minimum_area=300, minimum_mean=.9, minimum_support_views=3,
        per_view_coverage=.5, maximum_original_iou=.1, maximum_union_overlap=.25,
        pruning_tta_threshold=.2, maximum_tta_coverage=.4, maximum_vote2_coverage=.25),
    'small_object_recall': dict(branch='base', threshold=.5, minimum_area=100,
        sub300_minimum_mean=.8, preserve_base=True),
}


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def union_rle(value):
    # This is the canonical COCO encoding emitted by pycocotools for empty2048².
    if value == dict(size=[2048, 2048], counts='PPPP4'):
        return dict(size=[2048, 2048], counts=b'PPPP4')
    return checked_rle(value)


def disjoint(values):
    if values and int(mu.area(mu.merge(values))) != sum(int(mu.area(r)) for r in values):
        raise ValueError('Overlapping or duplicate instance masks')


def check_branch(branch):
    if not all(key in branch['components'] for key in ('0.35', '0.5', '0.65')):
        raise ValueError('Missing portable native thresholds')
    for values in branch['components'].values():
        rles = [checked_rle(row['rle']) for row in values]
        for row, rle in zip(values, rles):
            mean = row['stats']['foreground']['mean']
            if (row['area'] != int(mu.area(rle)) or row['area'] < 100
                    or not math.isfinite(mean) or not 0 <= mean <= 1
                    or abs(row.get('fg_mean', mean)-mean) > 1e-7
                    or list(mu.toBbox(rle).astype(int)) != row['bbox']):
                raise ValueError('Component metadata differs from native RLE')
        disjoint(rles)
    for value in branch.get('unions', {}).get('foreground', {}).values():
        union_rle(value)


def metadata_fold(folder, study, plan, fold, stage, annotation_sha, plan_sha):
    summary = json.loads((folder/'stage_summary.json').read_text())
    if (summary['status'] != 'complete' or summary['fold'] != fold or summary['stage_epoch'] != stage
            or summary['study_identity'] != study['study_identity']):
        raise ValueError('Mixed/incomplete stage or fold identity')
    cfg = summary['training_config']
    payload = cfg['cv_identity_payload']
    if (identity(payload) != cfg['cv_identity'] or payload['fold'] != fold
            or payload['settings'] != {key: cfg[key] for key in IDENTITY_KEYS}
            or payload['sources'] != cfg['source_sha256'] or payload['final_selection'] != 'fixed_epoch'
            or payload['precision'] != cfg['amp_dtype']
            or summary['training_cv_identity'] != cfg['cv_identity']):
        raise ValueError('Training identity is not reproducible')
    if (cfg['fold'] != fold or cfg['epochs'] != study['identity_payload']['training_settings']['epochs']
            or stage > cfg['epochs'] or stage < 1 or cfg['synthetic_ratio'] != 0
            or any(cfg.get(key) for key in ('initial_checkpoint', 'warm_start', 'checkpoint', 'resume_dir'))):
        raise ValueError('Unverified warm start, training budget or synthetic data')
    for key in IDENTITY_KEYS:
        expected = study['identity_payload']['training_settings'][key]
        if key == 'seed':
            expected += fold
        if cfg[key] != expected:
            raise ValueError('Stage training setting differs from study: '+key)
    info = next(row for row in plan['fold_summaries'] if row['fold'] == fold)
    for key, expected in dict(annotations=annotation_sha, fold_plan=plan_sha,
            base_runtime=study['identity_payload']['training_source_sha256']['runtime'],
            cv_runner=study['identity_payload']['training_source_sha256']['training'],
            train_split=split_sha(plan, info, 'train'), val_split=split_sha(plan, info, 'validation')).items():
        if cfg['source_sha256'].get(key) != expected:
            raise ValueError('Stage training source differs: '+key)
    for key, value in study['identity_payload']['training_source_sha256'].items():
        if summary['source_sha256'].get(key) != value:
            raise ValueError('Stage training code sources differ from study identity')
    rows = {role: [r for r in plan['observations'] if info['roles'][r['physical_identifier']] == role]
            for role in ('train', 'validation')}
    train_ids = sorted(r['physical_identifier'] for r in rows['train'])
    names = {r['file_name'] for r in rows['validation']}
    if summary['train_physical_identifiers'] != train_ids or set(summary['image_names']) != names:
        raise ValueError('Stage training/validation membership changed')
    if (not all(rows.values()) or {r['block'] for r in rows['train']} & {r['block'] for r in rows['validation']}
            or min(abs((date.fromisoformat(t['capture_day'])-date.fromisoformat(v['capture_day'])).days)
                   for t in rows['train'] for v in rows['validation']) <= 3):
        raise ValueError('Training/validation temporal isolation violated')
    model_sha = summary['model_sha256']
    if not re.fullmatch('[0-9a-f]{64}', model_sha):
        raise ValueError('Invalid model checkpoint hash')
    if (folder/'model.pt').exists() and digest(folder/'model.pt') != model_sha:
        raise ValueError('Stage checkpoint bytes differ')
    path = folder/'rich_records.jsonl'
    if not path.is_file():
        path = folder/'rich_predictions.jsonl'  # Compatible with the initial exporter filename.
    if digest(path) != summary['rich_predictions_sha256']:
        raise ValueError('Rich prediction export checksum changed')
    if digest(folder/'export_summary.json') != summary['export_summary_sha256']:
        raise ValueError('Inference export summary checksum changed')
    exported = json.loads((folder/'export_summary.json').read_text())
    if (exported['stage_epoch'] != stage or exported['fold'] != fold
            or exported['provenance']['study_identity'] != study['study_identity']
            or exported['provenance']['model_sha256'] != model_sha
            or exported['rich_predictions_sha256'] != digest(path)):
        raise ValueError('Inference summary stage/model provenance changed')
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if (len(records) != len(names) or len({r['file_name'] for r in records}) != len(names)
            or {r['file_name'] for r in records} != names or summary['image_coverage'] != len(names)):
        raise ValueError('Missing/duplicate prediction coverage, including blank images')
    for record in records:
        if (record['study_identity'] != study['study_identity'] or record['stage_epoch'] != stage
                or record['fold'] != fold or record['checkpoint_sha256'] != model_sha
                or not re.fullmatch('[0-9a-f]{64}', record['image_sha256'])):
            raise ValueError('Per-image source/model/stage identity changed')
        unsigned = {key: value for key, value in record.items() if key != 'record_sha256'}
        expected_record_sha = hashlib.sha256(json.dumps(unsigned, ensure_ascii=False, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        if (record.get('record_sha256') != expected_record_sha
                or exported['records_sha256'].get(record['file_name']) != expected_record_sha
                or record['inference_identity'] != exported['inference_identity']):
            raise ValueError('Cached record content/source checksum changed')
        check_branch(record)
        check_branch(record['tta'])
        if len(record['view_unions']) != 4 or '0.2' not in record['tta']['unions']['foreground']:
            raise ValueError('Four views and weak TTA support are required')
        for value in record['view_unions'].values():
            union_rle(value)
        if 'clahe' in record.get('preprocess_views', {}):
            check_branch(record['preprocess_views']['clahe'])
    return records, cfg, dict(settings=exported['settings'],
                              source=summary['source_sha256'].get('rich_inference'))


def methods_for(record):
    base = [checked_rle(row['rle']) for row in record['components']['0.5'] if row['area'] >= 300]
    refined, _ = _refine_boundaries(base, record)
    consensus, _ = _final_admission_pruning(refined, record)
    result = dict(base=base,
        tta=[checked_rle(row['rle']) for row in record['tta']['components']['0.5'] if row['area'] >= 300],
        guarded_boundary=refined, guarded_consensus=consensus,
        small_object_recall=[checked_rle(row['rle']) for row in record['components']['0.5']
                             if row['area'] >= 300 or row['area'] >= 100 and row['stats']['foreground']['mean'] >= .8])
    if 'clahe' in record.get('preprocess_views', {}):
        result['clahe'] = [checked_rle(row['rle']) for row in record['preprocess_views']['clahe']['components']['0.5']
                           if row['area'] >= 300]
    for values in result.values():
        disjoint(values)
    return result


def save_csv(path, predicted):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['filament_id', 'segmentation_rle'])
        writer.writerows((f'{image}_{i}', r['counts'].decode() if isinstance(r['counts'], bytes) else r['counts'])
                         for image, values in sorted(predicted.items()) for i, r in enumerate(values, 1))


def read_csv(path, names):
    result = {Path(name).stem: [] for name in names}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            image = row['filament_id'].rsplit('_', 1)[0]
            if image not in result:
                raise ValueError('Prediction references a non-validation image')
            result[image].append(checked_rle(dict(size=[2048, 2048], counts=row['segmentation_rle'])))
    for values in result.values():
        disjoint(values)
    return result


def selected_stage_folders(root, stage, study_identity, folds):
    """Follow the latest declared inference run, including prediction-only variants."""
    progress_path = root/f'stage{stage:03d}_progress.json'
    if not progress_path.is_file():
        return {fold: (root/f'fold{fold}'/f'stage{stage:03d}', None) for fold in range(folds)}
    progress = json.loads(progress_path.read_text())
    items = progress['summaries']
    numbers = [row['fold'] for row in items]
    if (progress['study_identity'] != study_identity or progress['stage_epoch'] != stage
            or len(numbers) != len(set(numbers)) or set(numbers) != set(progress['completed_folds'])
            or any(fold not in range(folds) for fold in numbers)):
        raise ValueError('Selected stage progress changed fold/stage identity')
    result = {}
    for item in items:
        fold = item['fold']; canonical = root/f'fold{fold}'/f'stage{stage:03d}'
        parts = Path(item['export']['rich_predictions']).parts
        matches = [index for index in range(len(parts)-1)
                   if parts[index:index+2] == (f'fold{fold}', f'stage{stage:03d}')]
        if len(matches) != 1:
            raise ValueError('Selected export is outside the expected fold/stage')
        suffix = parts[matches[0]+2:-1]
        if suffix and (len(suffix) != 2 or suffix[0] != 'variants'
                       or not re.fullmatch('[A-Za-z0-9_-]+', suffix[1])):
            raise ValueError('Unsafe inference variant path')
        result[fold] = (canonical.joinpath(*suffix), item)
    return result


def run(annotations, fold_plan, study_root, stage, output, report):
    annotations, fold_plan, study_root, output, report = map(Path,
        (annotations, fold_plan, study_root, output, report))
    stage = int(stage)
    study = json.loads((study_root/'study_config.json').read_text())
    plan = json.loads(fold_plan.read_text())
    if identity(study['identity_payload']) != study['study_identity']:
        raise ValueError('Study identity payload checksum changed')
    annotation_sha, plan_sha = digest(annotations), digest(fold_plan)
    if (study['identity_payload']['annotations_sha256'] != annotation_sha
            or study['identity_payload']['fold_plan_sha256'] != plan_sha
            or plan['sources_sha256']['annotations'] != annotation_sha):
        raise ValueError('Official annotation/fold plan source changed')
    stage_plan = json.loads((study_root/f'stage{stage:03d}_plan.json').read_text())
    if (stage_plan['stage_epoch'] != stage or stage_plan['training_epochs'] != study['identity_payload']['training_settings']['epochs']
            or stage_plan['expected_folds'] != list(range(plan['folds']))):
        raise ValueError('Stage budget does not match all five folds')
    output.mkdir(parents=True, exist_ok=False)
    prereg = dict(methods=METHODS, native_only=True, fixed_before_GT=True,
        excludes='YOLO-dependent reference/admission requires separately trained fold-local YOLO models; no old solar weights are used.')
    (output/'predeclared_methods.json').write_text(json.dumps(prereg, indent=2)+'\n')
    selected, regimes = {}, []
    folders = selected_stage_folders(study_root, stage, study['study_identity'], plan['folds'])
    for fold, (folder, expected_summary) in sorted(folders.items()):
        if not (folder/'stage_summary.json').is_file():
            if expected_summary is not None:
                raise FileNotFoundError('Selected inference stage is missing: '+str(folder))
            continue
        if expected_summary is not None:
            actual = json.loads((folder/'stage_summary.json').read_text())
            if any(actual.get(key) != expected_summary.get(key) for key in
                   ('fold','stage_epoch','study_identity','model_sha256','rich_predictions_sha256','source_sha256')):
                raise ValueError('Selected inference progress differs from stage files')
        records, cfg, inference_regime = metadata_fold(folder, study, plan, fold, stage, annotation_sha, plan_sha)
        regimes.append(dict(training={key: cfg[key] for key in IDENTITY_KEYS if key != 'seed'}, inference=inference_regime))
        selected[fold] = records
    if regimes and any(value != regimes[0] for value in regimes):
        raise ValueError('Cannot pool different model/training regimes')
    clahe_available = bool(selected) and all('clahe' in r.get('preprocess_views', {})
        for records in selected.values() for r in records)
    available = [name for name in METHODS if name != 'clahe' or clahe_available]
    for fold, records in selected.items():
        folder = output/f'fold{fold}'; folder.mkdir()
        predicted = {name: {} for name in available}
        for record in records:
            for name, values in methods_for(record).items():
                if name in predicted:
                    predicted[name][Path(record['file_name']).stem] = values
        for name, values in predicted.items():
            save_csv(folder/(name+'.csv'), values)
        expected_base = {Path(record['file_name']).stem: [row['rle']['counts']
            for row in record['components']['0.5'] if row['area'] >= 300] for record in records}
        actual_base = {image: [row['counts'].decode() for row in values]
            for image, values in read_csv(folder/'base.csv', expected_base).items()}
        if actual_base != expected_base:
            raise ValueError('Base masks changed during native method transfer')
    # Every candidate has now been generated. This is the first annotation decode.
    data = json.loads(annotations.read_text())
    metadata = {sid: row['file_name'] for row in plan['observations'] for sid in row['annotation_ids']}
    if metadata != {str(row['id']): row['file_name'] for row in data['images']}:
        raise ValueError('Fold plan omits or changes annotation variants')
    complete = len(selected) == plan['folds']
    by_image = {Path(row['file_name']).stem: row for row in plan['observations']}
    results = {}
    for method in available:
        rows, by_fold = [], {}
        for fold, records in selected.items():
            names = {row['file_name'] for row in records}
            truth, mapping = truth_for(data, names)
            measured = metric(truth, read_csv(output/f'fold{fold}'/(method+'.csv'), names), mapping)
            rows.extend(measured['per_group']); by_fold[str(fold)] = aggregate(measured['per_group'])
        if len({row['group_id'] for row in rows}) != len(rows):
            raise ValueError('One annotation group appears in multiple folds')
        pooled = aggregate(rows) if rows else None
        sites = {by_image[row['image']]['site'] for row in rows}
        def period(image):
            year = int(by_image[image]['capture_day'][:4])
            return next((f'{lo}-{hi}' for lo, hi in ((2011,2014),(2015,2018),(2019,2022)) if lo <= year <= hi), 'other')
        periods = {period(row['image']) for row in rows}
        results[method] = dict(pooled_oof=pooled if complete else None, pooled_completed_folds=pooled,
            by_fold=by_fold,
            by_site={site: aggregate([r for r in rows if by_image[r['image']]['site'] == site]) for site in sorted(sites)},
            by_period={p: aggregate([r for r in rows if period(r['image']) == p]) for p in sorted(periods)})
    result = dict(schema_version=2, status='complete_stage' if complete else 'partial_stage' if selected else 'awaiting_stage_predictions',
        stage_epoch=stage, training_schedule_epochs=stage_plan['training_epochs'],
        evaluation_kind=stage_plan.get('evaluation_kind'), complete_stage=complete,
        completed_folds=sorted(selected), missing_folds=sorted(set(range(plan['folds']))-set(selected)),
        selected_inference_paths={str(fold):str(folders[fold][0].relative_to(study_root)) for fold in selected},
        expected_physical_images=len(plan['observations']),
        completed_physical_images=sum(len(records) for records in selected.values()),
        results=results, methods=METHODS,
        skipped_methods={} if clahe_available else {'clahe': 'CLAHE inference was not exported for every included image.'},
        deltas_from_base={method: {key: values['pooled_completed_folds'][key]-results['base']['pooled_completed_folds'][key]
            for key in ('pq','tp_pairs','fp','fn')} for method, values in results.items()
            if method != 'base' and values['pooled_completed_folds'] is not None},
        all_csvs_generated_before_GT_decode=True, selection_performed=False, adopted=False,
        base_rle_bytes_preserved=True if selected else None,
        model_training_performed=False, neural_inference_performed=False,
        source_sha256=dict(annotations=annotation_sha, fold_plan=plan_sha, evaluator=digest(__file__)),
        limitation='Internal native-only grouped CV at equal budget; the old91 informed recipe design. Not full YOLO+native reproduction or an independent leaderboard score.')
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('annotations', 'fold-plan', 'study-root', 'output', 'report'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--stage', required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.annotations, args.fold_plan, args.study_root, args.stage, args.output, args.report), indent=2))
