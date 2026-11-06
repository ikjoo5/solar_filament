"""Deterministic, purged fivefold metadata manifests for official training images.

No model is loaded. All annotators of an observation stay together. The output
is a split plan, not a claim that the fixed616/91 legacy trainer accepts it.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import re


def physical_key(value):
    match = re.search(r'(\d{14}[a-z]h)', str(value), re.I)
    if not match:
        raise ValueError(f'Unknown observation identifier: {value}')
    return match.group(1).lower()


def capture_day(key):
    return date(int(key[:4]), int(key[4:6]), int(key[6:8]))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def observations(data, block_days=27):
    if block_days < 1:
        raise ValueError('block_days must be positive')
    image_rows = data['images']
    ids = [str(row['id']) for row in image_rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate annotation image ID')
    counts = Counter(str(row['image_id']) for row in data['annotations'])
    if set(counts) - set(ids):
        raise ValueError('Unknown annotation image ID')
    groups = defaultdict(list)
    for row in image_rows:
        groups[physical_key(row['file_name'])].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        names = {row['file_name'] for row in rows}
        if len(names) != 1:
            raise ValueError('Multiple filenames for one physical observation')
        day = capture_day(key)
        result.append(dict(physical_identifier=key, file_name=next(iter(names)),
            capture_day=day.isoformat(), site=key[-2:], year=day.year,
            block=(day-date(2000, 1, 1)).days//block_days,
            annotation_ids=sorted(str(row['id']) for row in rows),
            annotation_instances=sum(counts[str(row['id'])] for row in rows)))
    return result


def assign_folds(records, folds=5, seed=42):
    """Greedy group balance: images, raters, instance counts, sites and years."""
    groups = defaultdict(list)
    for row in records:
        groups[row['block']].append(row)
    if folds < 2 or len(groups) < folds:
        raise ValueError('Need at least one temporal group per fold')
    sites = sorted({r['site'] for r in records})
    years = sorted({r['year'] for r in records})
    weights = [4., .5, .5] + [1/len(sites)]*len(sites) + [1/len(years)]*len(years)

    def vector(rows):
        return ([len(rows), sum(len(r['annotation_ids']) for r in rows),
                 sum(r['annotation_instances'] for r in rows)]
                + [sum(r['site'] == site for r in rows) for site in sites]
                + [sum(r['year'] == year for r in rows) for year in years])

    total = vector(records)
    target = [max(value/folds, 1.) for value in total]
    loads = [[0.]*len(total) for _ in range(folds)]
    ordered = sorted(groups, key=lambda block: (
        -len(groups[block]), -sum(r['annotation_instances'] for r in groups[block]),
        digest(f'{seed}:{block}'.encode())))
    assignment = {}
    for block in ordered:
        additions = vector(groups[block])
        def cost(fold):
            change = sum(weight*((old+extra)**2-old**2)/(scale**2)
                         for old, extra, scale, weight in zip(loads[fold], additions, target, weights))
            return change, digest(f'{seed}:{block}:{fold}'.encode())
        fold = min(range(folds), key=cost)
        assignment[block] = fold
        loads[fold] = [old+extra for old, extra in zip(loads[fold], additions)]
    return {row['physical_identifier']: assignment[row['block']] for row in records}


def build_plan(data, synthetic_split=None, synthetic_annotations=None,
               folds=5, seed=42, block_days=27, embargo_days=3):
    if embargo_days < 0:
        raise ValueError('embargo_days cannot be negative')
    records = observations(data, block_days)
    assignment = assign_folds(records, folds, seed)
    all_keys = {row['physical_identifier'] for row in records}
    donors = ({physical_key(value) for value in synthetic_split['train_image_ids']}
              if synthetic_split else set())
    if donors - all_keys:
        raise ValueError('Synthetic donor pool references observations outside official train')
    image_fields = sorted({key for row in (synthetic_annotations or {}).get('images', []) for key in row})
    annotation_fields = sorted({key for row in (synthetic_annotations or {}).get('annotations', []) for key in row})
    fold_records = []
    for fold in range(folds):
        validation = [row for row in records if assignment[row['physical_identifier']] == fold]
        val_days = {date.fromisoformat(row['capture_day']) for row in validation}
        train, embargo = [], []
        for row in records:
            if assignment[row['physical_identifier']] == fold:
                continue
            distance = min(abs((date.fromisoformat(row['capture_day'])-day).days) for day in val_days)
            (embargo if distance <= embargo_days else train).append(row)
        if not train or not validation:
            raise ValueError('An empty training/validation fold cannot be used')
        train_keys = {row['physical_identifier'] for row in train}
        val_keys = {row['physical_identifier'] for row in validation}
        embargo_keys = {row['physical_identifier'] for row in embargo}
        assert train_keys | val_keys | embargo_keys == all_keys
        assert not (train_keys & val_keys or train_keys & embargo_keys or val_keys & embargo_keys)
        distance = min(abs((date.fromisoformat(t['capture_day'])-v).days)
                       for t in train for v in val_days)
        fold_records.append(dict(fold=fold, train_physical=len(train), validation_physical=len(validation),
            embargo_physical=len(embargo), train_annotation_groups=sum(len(r['annotation_ids']) for r in train),
            validation_annotation_groups=sum(len(r['annotation_ids']) for r in validation),
            validation_annotation_instances=sum(r['annotation_instances'] for r in validation),
            validation_sites=dict(sorted(Counter(r['site'] for r in validation).items())),
            validation_temporal_blocks=len({r['block'] for r in validation}),
            minimum_train_validation_day_distance=distance,
            synthetic_enabled=False,
            old_synthetic_possible_donors_in_validation=len(donors & val_keys),
            old_synthetic_possible_donors_in_embargo=len(donors & embargo_keys),
            roles={row['physical_identifier']: role for role, rows in
                   [('train', train), ('validation', validation), ('embargo', embargo)] for row in rows}))
    return dict(schema_version=1, status='metadata_splits_prepared_training_not_started',
        target_pq=.7, folds=folds, seed=seed, block_days=block_days, embargo_days=embargo_days,
        physical_images=len(records), annotation_groups=len(data['images']),
        annotation_instances=len(data['annotations']),
        temporal_blocks=len({r['block'] for r in records}),
        assignments=assignment, observations=records, fold_summaries=fold_records,
        synthetic_policy=dict(enabled=False, existing_synthetic_images=len((synthetic_annotations or {}).get('images', [])),
            possible_donor_physical=len(donors), metadata_image_fields=image_fields,
            metadata_annotation_fields=annotation_fields,
            per_image_donor_and_background_provenance_verified=False,
            explanation='The supplied synthetic data is valid for the old split, but source mappings for individual synthetic images/backgrounds are not provided. Do not reuse it across new folds; regenerate within each training fold or supply full provenance.'),
        initialization_policy=dict(allow_old_best84=False, allow_context45_or60=False,
            allow_old_yolo_checkpoint=False,
            allowed='Generic non-solar pretrained initialization allowed by competition rules, or random initialization. Resume only the same fold and identical training manifest.'),
        evaluation_policy=dict(outer_validation='Every physical image is held out exactly once; all its annotators remain together.',
            epoch_and_postprocess_selection='Freeze epochs and recipe before outer evaluation, or choose them using an inner grouped split from outer training only.',
            aggregate='Sum matched IoU, TP, FP, FN from out-of-fold predictions, then compute global PQ; also show individual folds.',
            previously_tuned91='These images have been repeatedly used for recipe selection; fivefold is an internal robustness check, not a pristine new test set.',
            learned_component_leakage='All learned branches and any learned ranking model must be trained without their outer validation observations.'),
        runtime_compatibility='The original training runtime hard-codes616train/91validation. These manifests require a separate CV-aware training adapter; do not pass them unchanged to the old launcher.',
        training_performed=False, new_network_inference_performed=False, external_data_used=False)


def write_outputs(plan, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = plan['observations']
    for fold in plan['fold_summaries']:
        folder = output/f"fold{fold['fold']}"
        folder.mkdir()
        for role, name in [('train', 'train_split.csv'), ('validation', 'val_split.csv'), ('embargo', 'excluded_embargo.csv')]:
            with (folder/name).open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['image_id', 'file_name'])
                writer.writerows((identifier, row['file_name']) for row in records
                                 if fold['roles'][row['physical_identifier']] == role
                                 for identifier in row['annotation_ids'])
    with (output/'image_fold_manifest.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['physical_identifier', 'file_name', 'capture_day', 'temporal_block', 'validation_fold'])
        writer.writerows((row['physical_identifier'], row['file_name'], row['capture_day'], row['block'],
                          plan['assignments'][row['physical_identifier']]) for row in records)
    raw = json.dumps(plan, ensure_ascii=False, indent=2)+'\n'
    (output/'plan.json').write_text(raw)
    return dict(output=str(output.resolve()), files_sha256={str(path.relative_to(output)): digest(path.read_bytes())
                for path in sorted(output.rglob('*')) if path.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--synthetic-split', type=Path)
    parser.add_argument('--synthetic-annotations', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text()) if path else None
    plan = build_plan(read(args.annotations), read(args.synthetic_split), read(args.synthetic_annotations))
    plan['sources_sha256'] = {name: digest(path.read_bytes()) for name, path in
        [('annotations', args.annotations), ('synthetic_split', args.synthetic_split),
         ('synthetic_annotations', args.synthetic_annotations)] if path}
    outputs = write_outputs(plan, args.output)
    brief = {key: value for key, value in plan.items() if key not in ('observations', 'assignments')}
    brief['fold_summaries'] = [{key: value for key, value in row.items() if key != 'roles'}
                               for row in plan['fold_summaries']]
    brief.update(outputs)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(brief, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(brief['fold_summaries'], indent=2))


if __name__ == '__main__':
    main()
