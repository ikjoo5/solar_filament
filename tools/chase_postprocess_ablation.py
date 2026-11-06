"""CPU-only, preregistered CHASE postprocessing diagnostics from existing caches.

No model inference, training, arbitrary pixel threshold search, or test labels.
The existing 91-image development set remains exploratory even after date splits.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from pycocotools import mask as mu
from scipy import ndimage

VERSION = '057b2b49a0cdf3e0'
MANIFEST_SHA = '31145ebed1889291a129f8ae3ed6c40833080e233a65ae094ba02245f4c615e2'
POLICIES = {
    'frozen': 'Original finalizer, unchanged.',
    'yolo_only': 'Original YOLO reference before any native stage.',
    'before_native_veto': 'Native reference + boundary refinement; no final admission or pruning.',
    'no_native_veto': 'Original final admission retained; native pruning disabled.',
    'relaxed_native_veto': 'Only coverage cutoffs .4/.25 become .2/.1; pixel thresholds unchanged.',
    'looser_native_admission': 'Only final admission foreground mean .9 becomes .8.',
    'native_only035': 'TTA foreground .35 connected components, cached minimum area100; diagnostic.',
    'native_only050': 'TTA foreground .5 connected components, cached minimum area100; diagnostic.',
}
STAGES = ('yolo_consensus', 'yolo_preprocessing_rescue', 'yolo_smooth',
          'yolo_tiled', 'native_reference', 'rich_refined', 'final')
RAW_MODE = 'raw_original_gray_v1'
MINIMAL_NATIVE = ('native_base050_area1', 'native_base050_area100')
MINIMAL_YOLO = ('yolo_base_conf005', 'yolo_base_conf020', 'yolo_base_conf050')


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sources(code):
    """Import only the released immutable code; helper remains outside it."""
    path = code/'source_manifest.json'
    if sha(path) != MANIFEST_SHA:
        raise ValueError('Frozen training source manifest differs: '+str(path))
    manifest = json.loads(path.read_text())
    for name, expected in manifest['sources'].items():
        if sha(code/name) != expected:
            raise ValueError('Frozen source changed: '+name)
    if {p.name for p in code.glob('*.py')} != set(manifest['sources']):
        raise ValueError('Unexpected Python file in immutable source folder')
    sys.path.insert(0, str(code))
    modules = [importlib.import_module(name) for name in
               ('submission_driver', 'submission_native_recipe',
                'submission_yolo_recipe', 'chase_best84_evaluation')]
    if any(Path(module.__file__).resolve().parent != code.resolve() for module in modules):
        raise ValueError('A module was imported from a different experiment')
    return modules


def canonical(rles):
    return sorted(r['counts'].decode('ascii') if isinstance(r['counts'], bytes)
                  else r['counts'] for r in rles)


def read_csv(path, names, driver):
    result = {Path(name).stem: [] for name in names}
    seen = set()
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['filament_id', 'segmentation_rle']:
            raise ValueError('Unexpected prediction CSV header')
        for row in reader:
            identifier = row['filament_id']
            stem = identifier.rsplit('_', 1)[0]
            if identifier in seen or stem not in result:
                raise ValueError('Duplicate/unknown prediction identifier: '+identifier)
            seen.add(identifier)
            result[stem].append(driver.checked_rle(dict(size=[2048, 2048],
                                                       counts=row['segmentation_rle'])))
    return {key: driver.validate_predictions(value) for key, value in result.items()}


def final_variant(originals, rich, native, *, veto=True, coverage=.4,
                  stable_coverage=.25, admission_mean=.9):
    """Same original order/overlap policy; change only the declared switches."""
    covered = mu.merge(originals) if originals else None
    views = [native._rle(value) for value in rich['view_unions'].values()]
    count = sum((mu.decode(value) for value in views), np.zeros((2048, 2048), np.uint8))
    stable = mu.encode(np.asfortranarray((count >= 2).astype(np.uint8)))
    support = native._rle(rich['tta']['unions']['foreground']['0.2'])
    components = rich['tta']['components']['0.35']
    proposals = [native._rle(c['rle']) for c in components]
    matrix = native._iou(proposals, originals)
    admitted = []
    for index, component in enumerate(components):
        area = component['area']
        if area < 300 or component['stats']['foreground']['mean'] < admission_mean:
            continue
        if originals and matrix[index].max() > .1:
            continue
        if covered is not None and native._intersection(proposals[index], covered)/area > .25:
            continue
        if sum(native._intersection(proposals[index], view)/area >= .5 for view in views) < 3:
            continue
        candidate = native._trim(proposals[index], covered)
        if candidate is not None and native._area(candidate) >= 300:
            admitted.append(candidate)
    kept = []
    for original in originals:
        area = native._area(original)
        remove = (native._intersection(original, support)/area < coverage
                  and native._intersection(original, stable)/area < stable_coverage)
        if not veto or not remove:
            kept.append(original)
    return kept+admitted


def policy_outputs(stem, frame, rich, yolo_stages, driver, native):
    stages = native.build_native_submission(stem, yolo_stages['tiled'],
        frame['geometric'], frame['preprocessing'], frame['tiled'], rich,
        return_stages=True)
    original = stages['rich_refined']
    # This equivalence guard catches implementation drift in the parameterized copy.
    if canonical(final_variant(original, rich, native)) != canonical(stages['final']):
        raise ValueError('Parameterized finalizer failed unchanged-control equivalence')
    policies = dict(frozen=stages['final'], yolo_only=yolo_stages['tiled'],
        before_native_veto=original,
        no_native_veto=final_variant(original, rich, native, veto=False),
        relaxed_native_veto=final_variant(original, rich, native, coverage=.2, stable_coverage=.1),
        looser_native_admission=final_variant(original, rich, native, admission_mean=.8),
        native_only035=[native._rle(c['rle']) for c in rich['tta']['components']['0.35']],
        native_only050=[native._rle(c['rle']) for c in rich['tta']['components']['0.5']])
    all_stages = {'yolo_'+name: value for name, value in yolo_stages.items()}
    all_stages.update({name: stages[name] for name in ('native_reference', 'rich_refined', 'final')})
    return dict(policies={key: driver.validate_predictions(value) for key, value in policies.items()},
                stages={key: driver.validate_predictions(all_stages[key]) for key in STAGES},
                decisions=stages['decisions'])


def native_minimal(rich, native, min_area=1):
    """Single base-view foreground union, not the already-filtered component list."""
    binary = mu.decode(native._rle(rich['unions']['foreground']['0.5'])).astype(bool)
    labels, _ = ndimage.label(binary, structure=np.ones((3, 3), np.uint8))
    areas = np.bincount(labels.ravel())
    result = []
    for label, region in enumerate(ndimage.find_objects(labels), 1):
        if region is not None and areas[label] >= min_area:
            result.append(native._encode_roi(labels[region] == label, region[0].start, region[1].start))
    return result


def yolo_minimal(frame, native, confidence=.05):
    """Single base view, confidence-first pixel ownership; no fusion or cleanup."""
    bases = [record for record in frame['geometric'] if record['view'] == 'base']
    if len(bases) != 1:
        raise ValueError('Expected exactly one YOLO geometric base view')
    indexed = [(index, item) for index, item in enumerate(bases[0]['instances'])
               if item['confidence'] >= confidence]
    indexed.sort(key=lambda pair: (-pair[1]['confidence'], pair[0]))
    raw = [dict(size=[2048, 2048], counts=item['rle'].encode('ascii')
                if isinstance(item['rle'], str) else item['rle']) for _, item in indexed]
    occupied = np.zeros((2048, 2048), bool)
    legal = []
    for rle in raw:
        mask = mu.decode(rle).astype(bool) & ~occupied
        if mask.any():
            occupied |= mask
            yy, xx = np.nonzero(mask)
            top, left = int(yy.min()), int(xx.min())
            legal.append(native._encode_roi(mask[top:int(yy.max())+1, left:int(xx.max())+1], top, left))
    return legal, raw


def candidate_coverage(groups, raw):
    """Coverage only for potentially overlapping proposals; never call this PQ."""
    covered = 0
    for truth in groups:
        if truth and raw:
            covered += int((np.asarray(mu.iou(truth, raw, [0]*len(raw))) > .5).any(axis=1).sum())
    return dict(raw_candidate_count=len(raw), gt_with_any_raw_candidate_iou_gt05=covered)


def counts_for(groups, predicted):
    """Official all-pairs >.5 convention. Inputs must already be nonoverlapping."""
    result = dict(tp_pairs=0, fp=0, fn=0, matched_iou_sum=0.)
    for truth in groups:
        if not truth:
            result['fp'] += len(predicted)
            continue
        if not predicted:
            result['fn'] += len(truth)
            continue
        values = np.asarray(mu.iou(truth, predicted, [0]*len(predicted)))
        hits = values > .5
        result['tp_pairs'] += int(hits.sum())
        result['fp'] += int((~hits.any(axis=0)).sum())
        result['fn'] += int((~hits.any(axis=1)).sum())
        result['matched_iou_sum'] += float(values[hits].sum())
    return result


def aggregate(rows):
    counts = {key: sum(row[key] for row in rows)
              for key in ('tp_pairs', 'fp', 'fn', 'matched_iou_sum')}
    tp, fp, fn, iou = (counts[key] for key in ('tp_pairs', 'fp', 'fn', 'matched_iou_sum'))
    denominator = tp+.5*fp+.5*fn
    return dict(counts, pq=iou/denominator if denominator else 0.,
                rq=tp/denominator if denominator else 0., sq=iou/tp if tp else 0.)


def transition(groups, before, after):
    """Count covered/lost GT once per annotator group, not one per candidate."""
    lost = recovered = 0
    for truth in groups:
        def covered(predicted):
            if not truth or not predicted:
                return np.zeros(len(truth), bool)
            return (np.asarray(mu.iou(truth, predicted, [0]*len(predicted))) > .5).any(axis=1)
        first, second = covered(before), covered(after)
        lost += int((first & ~second).sum())
        recovered += int((~first & second).sum())
    return dict(gt_lost=lost, gt_recovered=recovered,
                predictions_before=len(before), predictions_after=len(after))


def fold_plan(names):
    dates = {Path(name).stem[:8] for name in names}
    if any(len(date) != 8 or not date.isdigit() for date in dates):
        raise ValueError('Cannot group the official filenames by observation date')
    dates = sorted(dates, key=lambda date: hashlib.sha256(('chase-ablation-date-v1:'+date).encode()).hexdigest())
    assignment = {date: index % 5 for index, date in enumerate(dates)}
    return {Path(name).stem: assignment[Path(name).stem[:8]] for name in names}


def crossfit(per_image, folds):
    """Policy choice uses other dates only; this is not new model OOF testing."""
    choices, held_out = [], []
    methods = list(per_image)
    for fold in range(5):
        training = {method: aggregate([row for image, row in per_image[method].items()
                                      if folds[image] != fold]) for method in methods}
        chosen = max(methods, key=lambda method: training[method]['pq'])
        test = [row for image, row in per_image[chosen].items() if folds[image] == fold]
        held_out.extend(test)
        choices.append(dict(fold=fold, selected_using_other_dates=chosen,
                            selection_pq=training[chosen]['pq'], held_out=aggregate(test)))
    return dict(scope='Exploratory policy cross-fit within the repeatedly used development91 set; '
                      'NOT five newly trained fold models, NOT an independent holdout or leaderboard estimate.',
                folds=choices, pooled_held_out=aggregate(held_out))


def native_spec(root, metric, evaluation, driver):
    identity = metric['native_identity']
    directory = root/'native_validation_cache'/identity
    saved = json.loads((directory/'identity.json').read_text())
    if saved['identity'] != identity or evaluation.fingerprint(saved['protocol']) != identity:
        raise ValueError('Native cache identity mismatch')
    if saved['protocol']['model_sha256'] != metric['model_sha256']:
        raise ValueError('Native cache uses different model tensors')
    return dict(identity=identity, directory=directory, protocol=saved['protocol'])


def raw_spec(path, expected_new, evaluation):
    summary = json.loads(Path(path).read_text())
    if (summary.get('schema_version') != 1 or summary.get('mode') != RAW_MODE
            or summary.get('original_native_identity') != expected_new['identity']
            or summary.get('model_sha256') != expected_new['protocol']['model_sha256']):
        raise ValueError('Raw input cache is not the same CHASE60 native model')
    directory = Path(summary['native_cache_root'])
    saved = json.loads((directory/'identity.json').read_text())
    identity = summary['native_identity']
    protocol = saved['protocol']
    if (saved['identity'] != identity or evaluation.fingerprint(protocol) != identity
            or protocol.get('input_mode') != RAW_MODE
            or protocol.get('model_sha256') != summary['model_sha256']
            or summary['source_images'] != expected_new['protocol']['source_images']
            or protocol['source_images'] != summary['source_images']):
        raise ValueError('Raw input cache provenance differs')
    return dict(identity=identity, directory=directory, protocol=protocol)


def run(root, raw_summary=None, *, code_dir=None, annotations=None, train_split=None,
        val_split=None, old_yolo_cache=None):
    root = Path(root).resolve()
    code = Path(code_dir).resolve() if code_dir else root/'code'/VERSION
    driver, native, yolo_recipe, evaluation = sources(code)
    config = json.loads((root/'configs/comparison.json').read_text())
    for key, value in dict(annotations=annotations, train_split=train_split, val_split=val_split).items():
        if value is not None:
            config[key] = str(Path(value).resolve())
    if old_yolo_cache is not None:
        config['yolo_cache'] = str(Path(old_yolo_cache).resolve())
    elif (root/'assets'/VERSION/'frozen_yolo_validation.json.gz').is_file():
        config['yolo_cache'] = str(root/'assets'/VERSION/'frozen_yolo_validation.json.gz')
    cohort = evaluation.load_cohort(config)
    names = cohort['names']; stems = [Path(name).stem for name in names]
    groups = defaultdict(list)
    for value in cohort['groups'].values():
        groups[value['image']].append(value['masks'])
    comparison_path = root/'comparison/epoch060_comparison.json'
    comparison = json.loads(comparison_path.read_text())
    if comparison.get('epoch_per_model') != 60:
        raise ValueError('Expected completed epoch60 comparison')
    controls = {row['role']: row for row in comparison['results']}
    expected_roles = {y+'_'+n for y in ('old_yolo','new_yolo') for n in ('old_native','new_native')}
    if set(controls) != expected_roles:
        raise ValueError('Missing four unchanged comparison controls')
    yolo = {
        'old_yolo': evaluation.load_yolo_cache(config['yolo_cache'], names),
        'new_yolo': evaluation.load_yolo_cache(root/'comparison/yolo_epoch060.json.gz', names),
    }
    images = yolo['old_yolo']['image_sha256']
    if yolo['new_yolo']['image_sha256'] != images:
        raise ValueError('YOLO caches belong to different input images')
    specs = {name: native_spec(root, controls['old_yolo_'+name], evaluation, driver)
             for name in ('old_native', 'new_native')}
    if raw_summary:
        specs['new_native_raw'] = raw_spec(raw_summary, specs['new_native'], evaluation)
    for spec in specs.values():
        if spec['protocol']['source_images'] != images:
            raise ValueError('Native/YOLO source image hashes differ')
    references = {}
    for role, metric in controls.items():
        reference_dir = root/'comparison/validation'/('epoch060_'+role)
        saved = json.loads((reference_dir/'identity.json').read_text())
        source_protocol = saved['protocol']
        yn = 'old_yolo' if role.startswith('old_yolo_') else 'new_yolo'
        nn = 'old_native' if role.endswith('_old_native') else 'new_native'
        if (saved['identity'] != metric['identity']
                or evaluation.fingerprint(source_protocol) != saved['identity']
                or source_protocol['yolo_cache_sha256'] != yolo[yn]['sha256']
                or source_protocol['source_images'] != images
                or source_protocol['model_sha256'] != specs[nn]['protocol']['model_sha256']):
            raise ValueError('Original comparison cache/model identity differs: '+role)
        path = reference_dir/'predictions.csv'
        if sha(path) != metric['csv_sha256']:
            raise ValueError('Reference CSV changed: '+role)
        references[role] = read_csv(path, names, driver)
    protocol = dict(schema_version=1, script_sha256=sha(__file__), policies=POLICIES,
        minimal_model_policies=dict(native_policies=MINIMAL_NATIVE, yolo_policies=MINIMAL_YOLO,
            input='Existing cached limb-corrected images; preprocessing is NOT removed.',
            native='Base view foreground>=.5; 8-connected components; area1 or100.',
            yolo='Base2048 view; confidences .05/.2/.5; descending-confidence pixel ownership; area>=1.',
            retained_upstream='YOLO original inference conf floor .05, box NMS IoU.7 and max_det100; native tiled blending.'),
        reference_comparison_sha256=sha(comparison_path), sources_manifest_sha256=MANIFEST_SHA,
        source_images=images, yolo={key: value['sha256'] for key, value in yolo.items()},
        native={key: value['identity'] for key, value in specs.items()},
        ground_truth_sources=cohort['source_sha256'],
        scope='Cache replay only; no retraining; no arbitrary dense pixel probability access.')
    identity = evaluation.fingerprint(protocol)
    parent = root/'postprocess_ablation_v1'; output = parent/identity[:16]
    (output/'frames').mkdir(parents=True, exist_ok=True)
    driver.atomic_json(output/'identity.json', dict(identity=identity, protocol=protocol))
    roles = [(yn+'_'+nn, yn, nn) for yn in yolo for nn in specs]
    per_image = {role+'/'+policy: {} for role, _, _ in roles for policy in POLICIES}
    minimal_per_image = {name+'/'+policy: {} for name in specs for policy in MINIMAL_NATIVE}
    minimal_per_image.update({name+'/'+policy: {} for name in yolo for policy in MINIMAL_YOLO})
    raw_coverage = {name+'/'+policy: {} for name in yolo for policy in MINIMAL_YOLO}
    stage_metrics = {role+'/'+stage: {} for role, _, _ in roles for stage in STAGES}
    transitions = {role: {} for role, _, _ in roles}
    for number, filename in enumerate(names, 1):
        stem = Path(filename).stem; image_sha = images[filename]
        frame_path = output/'frames'/(stem+'.json.gz')
        result = driver.read_envelope(frame_path, identity, image_sha)
        if result is None:
            rich = {}
            for name, spec in specs.items():
                value = driver.read_envelope(spec['directory']/(stem+'.json.gz'), spec['identity'], image_sha)
                if not evaluation._native_valid(value, stem, filename, image_sha, spec['identity']):
                    raise ValueError('Missing/invalid native rich cache; no inference was launched: '
                                     +str(spec['directory']/(stem+'.json.gz')))
                rich[name] = value
            ys = {name: yolo_recipe.build_yolo_reference(stem, frame['geometric'],
                    frame['preprocessing'], frame['tiled'], return_stages=True)
                  for name, data in yolo.items() for frame in [data['frames'][stem]]}
            pipeline = {role: policy_outputs(stem, yolo[yn]['frames'][stem], rich[nn],
                                            ys[yn], driver, native) for role, yn, nn in roles}
            minimal = {name+'/'+policy: dict(legal=native_minimal(value, native, minimum))
                       for name, value in rich.items()
                       for policy, minimum in zip(MINIMAL_NATIVE, (1, 100))}
            for name, value in yolo.items():
                for policy, confidence in zip(MINIMAL_YOLO, (.05, .2, .5)):
                    legal, raw = yolo_minimal(value['frames'][stem], native, confidence)
                    minimal[name+'/'+policy] = dict(legal=legal, raw=raw)
            result = dict(pipeline=pipeline, minimal=minimal)
            driver.write_envelope(frame_path, result, identity, image_sha)
        if set(result.get('pipeline', {})) != {role for role, _, _ in roles}:
            raise ValueError('Unexpected cached role set')
        if set(result.get('minimal', {})) != set(minimal_per_image):
            raise ValueError('Unexpected cached minimal-model set')
        for method, record in result['minimal'].items():
            masks = driver.validate_predictions(record['legal'])
            minimal_per_image[method][stem] = counts_for(groups[stem], masks)
            if 'raw' in record:
                raw = [driver.checked_rle(value) for value in record['raw']]
                raw_coverage[method][stem] = candidate_coverage(groups[stem], raw)
        for role, _, _ in roles:
            row = result['pipeline'][role]
            if set(row['policies']) != set(POLICIES) or set(row['stages']) != set(STAGES):
                raise ValueError('Unexpected cached policy/stage set')
            policies = {key: driver.validate_predictions(value) for key, value in row['policies'].items()}
            stages = {key: driver.validate_predictions(value) for key, value in row['stages'].items()}
            if role in references and canonical(policies['frozen']) != canonical(references[role][stem]):
                raise ValueError('Frozen CSV control mismatch before interpreting changes: '+role+'/'+stem)
            for key, value in policies.items():
                per_image[role+'/'+key][stem] = counts_for(groups[stem], value)
            for key, value in stages.items():
                stage_metrics[role+'/'+key][stem] = counts_for(groups[stem], value)
            changes = {first+' -> '+second: transition(groups[stem], stages[first], stages[second])
                       for first, second in zip(STAGES[:-1], STAGES[1:])}
            changes['native_veto_only'] = transition(groups[stem], policies['no_native_veto'], policies['frozen'])
            transitions[role][stem] = dict(changes=changes, decisions=row['decisions'])
        print(f'CPU cache replay [{number}/{len(names)}] {filename}', flush=True)
    summary = {key: aggregate(list(value.values())) for key, value in per_image.items()}
    for role, expected in controls.items():
        actual = summary[role+'/frozen']
        for key in ('pq', 'sq', 'rq', 'matched_iou_sum', 'tp_pairs', 'fp', 'fn'):
            if not np.isclose(actual[key], expected[key], rtol=0, atol=1e-9):
                raise ValueError('Frozen aggregate score mismatch: '+role+'/'+key)
    folds = fold_plan(names)
    fold_results = {method: {str(fold): aggregate([row for image, row in rows.items() if folds[image] == fold])
                            for fold in range(5)} for method, rows in per_image.items()}
    stage_summary = {key: aggregate(list(value.values())) for key, value in stage_metrics.items()}
    baseline = summary['old_yolo_old_native/frozen']
    table = []
    for method, value in summary.items():
        role, policy = method.split('/')
        frozen = summary[role+'/frozen']
        table.append(dict(method=method, **value, delta_pq_vs_same_weights_frozen=value['pq']-frozen['pq'],
                          delta_pq_vs_original_baseline=value['pq']-baseline['pq']))
    report = dict(schema_version=1, identity=identity, protocol=protocol,
        controls_verified_exact_csv_and_metrics=True, physical_images=len(names),
        annotation_groups=len(cohort['groups']), gt_instances=708,
        results=table, stage_results=stage_summary, per_date_fold_results=fold_results,
        minimal_model_results={method: aggregate(list(rows.values())) for method, rows in minimal_per_image.items()},
        raw_yolo_candidate_coverage={method: {key: sum(row[key] for row in rows.values())
            for key in ('raw_candidate_count', 'gt_with_any_raw_candidate_iou_gt05')}
            for method, rows in raw_coverage.items()},
        exploratory_policy_crossfit=crossfit(per_image, folds),
        automatic_production_selection=False,
        limitations=['Threshold caches contain RLEs and component statistics, not dense probabilities.',
                     'All results reuse the fixed development91 set; this is not independent five-fold model CV.',
                     'Policies are declared before this replay; repeated project use of this split still permits overfitting.',
                     'No leaderboard score, test prediction, or public318 reference coverage is measured.'])
    driver.atomic_json(output/'report.json', report)
    driver.atomic_json(output/'per_image_metrics.json', per_image)
    driver.atomic_json(output/'minimal_model_per_image.json', dict(metrics=minimal_per_image,
                                                               raw_yolo_candidate_coverage=raw_coverage))
    driver.atomic_json(output/'stage_transitions.json', transitions)
    driver.atomic_json(output/'fold_manifest.json', dict(image_to_fold=folds, group='observation YYYYMMDD',
        scope='Diagnostic partitions of existing validation; no model retraining'))
    archive = output/'diagnostic_results.zip'
    with zipfile.ZipFile(archive.with_suffix('.pending.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ('identity.json','report.json','per_image_metrics.json','minimal_model_per_image.json',
                     'stage_transitions.json','fold_manifest.json'):
            z.write(output/name, name)
        z.write(__file__, 'chase_postprocess_ablation.py')
    archive.with_suffix('.pending.zip').replace(archive)
    latest = dict(status='complete', run_identity=identity, report=str(output/'report.json'), archive=str(archive))
    driver.atomic_json(parent/'latest_result.json', latest)
    print(json.dumps(latest, ensure_ascii=False, indent=2), flush=True)
    return latest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--raw-summary', type=Path)
    parser.add_argument('--code-dir', type=Path)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--train-split', type=Path)
    parser.add_argument('--val-split', type=Path)
    parser.add_argument('--old-yolo-cache', type=Path)
    args = parser.parse_args()
    run(args.root, args.raw_summary, code_dir=args.code_dir, annotations=args.annotations,
        train_split=args.train_split, val_split=args.val_split, old_yolo_cache=args.old_yolo_cache)
