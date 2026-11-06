"""CPU audit of an external prediction reference against our masks and cached stages.

This measures prediction agreement, not ground-truth accuracy. It never edits
submissions, runs a neural model, trains on test predictions, or calculates PQ.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from pycocotools import mask as mu
from filament_eval import _check_compressed_counts, read_csv

SIZE = 2048
STAGES = ('geometric', 'preprocessing', 'tiled', 'native')
FALLBACK_IMAGES = {'20160513133714Th', '20180507140210Th'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_mask(value, *, allow_empty=False):
    if not isinstance(value, dict) or value.get('size') != [SIZE, SIZE]:
        raise ValueError('Expected original-coordinate 2048x2048 mask')
    counts = value.get('counts')
    if isinstance(counts, bytes):
        counts = counts.decode('ascii')
    area = _check_compressed_counts(counts, SIZE * SIZE)
    if area == 0 and not allow_empty:
        raise ValueError('Empty instance mask')
    result = dict(size=[SIZE, SIZE], counts=counts.encode('ascii'))
    if int(mu.area(result)) != area:
        raise ValueError('Mask area mismatch')
    return result


def ious(left, right):
    if not left or not right:
        return np.zeros((len(left), len(right)), dtype=float)
    return np.asarray(mu.iou(left, right, [0] * len(right)), dtype=float)


def known_fallback(stem, mask):
    return (stem in FALLBACK_IMAGES and int(mu.area(mask)) == 2304
            and np.array_equal(mu.toBbox(mask), [1000, 1000, 48, 48]))


def row_masks(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row.image_id].append((row.filament_id, row.rle(SIZE, SIZE)))
    return groups


def names_from(value):
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text())
    if isinstance(value, dict):
        value = value['test_images']
    if not isinstance(value, list) or not value or any(not isinstance(n, str) or Path(n).name != n for n in value):
        raise ValueError('Expected a nonempty official image-name list')
    stems = [Path(name).stem for name in value]
    if len(stems) != len(set(stems)):
        raise ValueError('Duplicate image names/stems')
    return sorted(stems)


def load_context(root, snapshot):
    def read(path):
        snapshot[path] = file_sha(path)
        return json.loads(path.read_text())
    envelope = read(root/'protocol.json')
    protocol, identity = envelope['protocol'], envelope['identity']
    if identity != fingerprint(protocol):
        raise ValueError('Protocol identity mismatch')
    runtime = read(root/'runtime_device.json')
    if runtime['sha256'] != fingerprint(runtime['payload']) or runtime['payload'].get('identity') != identity:
        raise ValueError('Runtime identity/checksum mismatch')
    manifest = read(root/'run_manifest.json')
    images = protocol['images']
    if (manifest.get('status') != 'complete' or manifest.get('identity') != identity
            or manifest.get('official_images') != len(images)):
        raise ValueError('Inference must finish before this audit: incomplete/mismatched run manifest')
    stems = names_from([image['name'] for image in images])
    if any(image['stem'] != Path(image['name']).stem for image in images):
        raise ValueError('Invalid protocol image stem')
    for name in ('submission_pipeline_baseline.csv', 'submission_pipeline_R1.csv'):
        path = root/name
        snapshot[path] = file_sha(path)
        if snapshot[path] != manifest['output_sha256'].get(name):
            raise ValueError('Completed submission changed: '+name)
    return protocol, identity, runtime['sha256'], stems


def load_image(root, image, identity, runtime_signature, snapshot):
    records = {}
    for stage in (*STAGES, 'baseline', 'final'):
        path = root/'cache'/image['stem']/(stage+'.json.gz')
        if not path.is_file():
            raise ValueError('Incomplete cache; no absence inference is possible: '+str(path))
        raw = path.read_bytes(); snapshot[path] = hashlib.sha256(raw).hexdigest()
        envelope = json.loads(gzip.decompress(raw)); payload = envelope['payload']
        if envelope['sha256'] != fingerprint(payload):
            raise ValueError('Cache checksum mismatch: '+str(path))
        expected = dict(identity=identity, image=image['stem'], image_sha256=image['sha256'],
                        stage=stage, runtime_signature=runtime_signature)
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError('Cache identity/runtime mismatch: '+str(path))
        records[stage] = envelope
    supports = {stage: records[stage]['sha256'] for stage in STAGES}
    baseline = records['baseline']['payload']['data']
    final = records['final']['payload']['data']
    if baseline['support_sha256'] != supports or final['support_sha256'] != supports:
        raise ValueError('Downstream support dependency mismatch')
    if final['baseline_sha256'] != records['baseline']['sha256']:
        raise ValueError('Final baseline dependency mismatch')
    stages = {stage: records[stage]['payload']['data'] for stage in STAGES}
    for stage, count in (('geometric', 4), ('preprocessing', 8), ('tiled', 18)):
        views = stages[stage]
        if (not isinstance(views, list) or len(views) != count
                or len({view.get('view') for view in views}) != count):
            raise ValueError('Incomplete saved view inventory: '+stage)
    native = stages['native']
    expected_native = dict(identity=identity, file_name=image['name'], image_sha256=image['sha256'], size=[SIZE, SIZE])
    if any(native.get(key) != value for key, value in expected_native.items()):
        raise ValueError('Native record identity mismatch')
    expected_thresholds = {'0.2', '0.35', '0.5', '0.65', '0.8'}
    if any(set(branch['components']) != expected_thresholds for branch in (native, native['tta'])):
        raise ValueError('Incomplete stored native component thresholds')
    return stages, [checked_mask(m) for m in baseline['masks']], [checked_mask(m) for m in final['masks']]


def candidates(stem, stages):
    result = []
    for stage in STAGES[:3]:
        for view in stages[stage]:
            if (view.get('image') != stem or view.get('shape') != [SIZE, SIZE]
                    or view.get('coordinate_space') != 'original'):
                raise ValueError('Raw view coordinate/image mismatch')
            for index, item in enumerate(view['instances']):
                confidence = float(item['confidence'])
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError('Invalid confidence')
                result.append(dict(source=stage, view=str(view['view']), index=index,
                    confidence=confidence, threshold=None,
                    mask=checked_mask(dict(size=[SIZE, SIZE], counts=item['rle']))))
    native = stages['native']
    if Path(native['file_name']).stem != stem or native.get('size') != [SIZE, SIZE]:
        raise ValueError('Native coordinate/image mismatch')
    # Every stored component threshold is inventoried, but thresholds are NOT views.
    for branch, record in (('base', native), ('tta', native['tta'])):
        for thresholds in record['unions'].values():
            for union in thresholds.values():
                checked_mask(union, allow_empty=True)
        for threshold, components in record['components'].items():
            for index, item in enumerate(components):
                mask = checked_mask(item['rle'])
                if int(mu.area(mask)) != item['area']:
                    raise ValueError('Native component area mismatch')
                result.append(dict(source='native', view=branch, index=index,
                    confidence=None, threshold=float(threshold), mask=mask))
    for union in native.get('view_unions', {}).values():
        checked_mask(union, allow_empty=True)
    unions = {branch: checked_mask(record['unions']['foreground']['0.35'], allow_empty=True)
              for branch, record in (('base', native), ('tta', native['tta']))}
    return result, unions


def coverage(mask, other):
    return int(mu.area(mu.merge([mask, other], intersect=True))) / int(mu.area(mask))


def distinct_view_support(rows):
    """Maximum bipartite matching of YOLO view labels to distinct exact masks.

    A repeated mask may support only one view. Native threshold variants and the
    aggregate TTA output are intentionally excluded from this view count.
    """
    edges = defaultdict(set)
    for row in rows:
        if row['source'] == 'native':
            continue
        edges[row['source']+'/'+row['view']].add(row['mask']['counts'])
    assigned = {}
    def visit(view, seen):
        for key in sorted(edges[view]):
            if key in seen:
                continue
            seen.add(key)
            if key not in assigned or visit(assigned[key], seen):
                assigned[key] = view
                return True
        return False
    return sum(visit(view, set()) for view in sorted(edges))


def analyze_image(stem, reference, baseline, final=None, raw=None, unions=None, replay=None):
    reference_masks = [mask for _, mask in reference]
    baseline_masks = [mask for _, mask in baseline]
    bmatrix = ious(reference_masks, baseline_masks)
    fmatrix = ious(reference_masks, final) if final is not None else None
    raw_masks = [item['mask'] for item in raw] if raw is not None else []
    rmatrix = ious(reference_masks, raw_masks) if raw is not None else None
    stage_scores = {name: ious(reference_masks, masks) for name, masks in (replay or {}).items()}
    rows, ours = [], []
    for i, (identifier, mask) in enumerate(reference):
        baseline_iou = float(bmatrix[i].max()) if len(baseline) else 0.
        best_b = int(bmatrix[i].argmax()) if len(baseline) else None
        final_iou = (float(fmatrix[i].max()) if final else 0.) if final is not None else None
        best_raw = int(rmatrix[i].argmax()) if raw else None
        raw_iou = (float(rmatrix[i].max()) if raw else 0.) if raw is not None else None
        matching_raw = [item for j, item in enumerate(raw or []) if rmatrix[i, j] > .5]
        fallback = known_fallback(stem, mask)
        if fallback:
            category = 'known_placeholder_excluded'
        elif baseline_iou > .5:
            category = 'similar_final_instance'
        elif baseline_iou >= .1:
            category = 'shape_or_grouping_difference'
        elif raw is None:
            category = 'weak_final_overlap_raw_unavailable'
        elif raw_iou > .5:
            category = 'raw_similar_final_different'
        else:
            category = 'no_similar_raw_instance_in_saved_pool'
        item = raw[best_raw] if best_raw is not None else None
        row = dict(image=stem, reference_id=identifier, reference_area=int(mu.area(mask)),
            excluded_placeholder=fallback, category=category,
            baseline_max_iou=baseline_iou,
            baseline_best_id=baseline[best_b][0] if best_b is not None else '',
            baseline_to_reference_area_ratio=(int(mu.area(baseline_masks[best_b])) / int(mu.area(mask)))
                if best_b is not None and baseline_iou > 0 else None,
            r1_max_iou=final_iou,
            r1_agreement_delta=final_iou-baseline_iou if final_iou is not None else None,
            raw_max_iou=raw_iou, raw_best_source=item['source'] if item else '',
            raw_similar_but_final_not_similar=(not fallback and baseline_iou <= .5 and raw_iou > .5)
                if raw_iou is not None else None,
            raw_best_view=item['view'] if item else '', raw_best_confidence=item['confidence'] if item else None,
            raw_best_native_threshold=item['threshold'] if item else None,
            matching_yolo_views_deduplicated=distinct_view_support(matching_raw) if raw is not None else None,
            matching_raw_sources='|'.join(sorted({r['source'] for r in matching_raw})),
            raw_distinct_matching_masks=len({r['mask']['counts'] for r in matching_raw}) if raw is not None else None,
            native_base035_reference_pixel_coverage=coverage(mask, unions['base']) if unions else None,
            native_tta035_reference_pixel_coverage=coverage(mask, unions['tta']) if unions else None)
        for source in STAGES:
            indices = [j for j, candidate in enumerate(raw or []) if candidate['source'] == source]
            row['raw_'+source+'_max_iou'] = (float(rmatrix[i, indices].max()) if indices else 0.) if raw is not None else None
        for branch in ('base', 'tta'):
            for threshold in (.2, .35, .5, .65, .8):
                indices = [j for j, candidate in enumerate(raw or []) if candidate['source'] == 'native'
                           and candidate['view'] == branch and candidate['threshold'] == threshold]
                key = f'native_{branch}_{str(threshold).replace(".", "_")}_instance_max_iou'
                row[key] = (float(rmatrix[i, indices].max()) if indices else 0.) if raw is not None else None
        # The aggregate raw pool includes rescue sources introduced later. It is
        # not an earlier pipeline stage and cannot establish an initial loss.
        trace = []
        for name, matrix in stage_scores.items():
            score = float(matrix[i].max()) if matrix.shape[1] else 0.
            row[name+'_max_iou'] = score
            trace.append((name, score))
        if not replay:
            trace.append(('baseline', baseline_iou))
        if final_iou is not None:
            trace.append(('R1', final_iou))
        losses = [f'{a}->{b}' for (a, av), (b, bv) in zip(trace, trace[1:]) if av > .5 and bv <= .5]
        gains = [f'{a}->{b}' for (a, av), (b, bv) in zip(trace, trace[1:]) if av <= .5 and bv > .5]
        row['similarity_loss_transitions'] = '|'.join(losses)
        row['similarity_gain_transitions'] = '|'.join(gains)
        rows.append(row)
    usable = [i for i, (_, mask) in enumerate(reference) if not known_fallback(stem, mask)]
    placeholder_only = bool(reference) and not usable
    for j, (identifier, mask) in enumerate(baseline):
        score = float(bmatrix[usable, j].max()) if usable else 0.
        ours.append(dict(image=stem, baseline_id=identifier, baseline_area=int(mu.area(mask)),
            reference_comparison_available=not placeholder_only,
            reference_max_iou=score if not placeholder_only else None,
            category='reference_placeholder_only_unassessed' if placeholder_only else
                ('weak_reference_overlap' if score < .1 else 'reference_overlap')))
    counts = Counter(row['category'] for row in rows)
    summary = dict(image=stem, reference_instances=len(reference), baseline_instances=len(baseline),
        raw_instances=len(raw) if raw is not None else None, raw_trace_available=raw is not None,
        similar_final_instances=counts['similar_final_instance'],
        weak_final_overlap=sum(row['baseline_max_iou'] < .1 and not row['excluded_placeholder'] for row in rows),
        raw_similar_final_different=counts['raw_similar_final_different'] if raw is not None else None,
        all_raw_similar_but_final_not_similar=sum(row['raw_similar_but_final_not_similar'] is True for row in rows)
            if raw is not None else None,
        no_similar_raw_instance=counts['no_similar_raw_instance_in_saved_pool'] if raw is not None else None,
        our_weak_reference_overlap=sum(row['category'] == 'weak_reference_overlap' for row in ours))
    return rows, ours, summary


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ['image'])
        writer.writeheader(); writer.writerows(rows)


def run(reference_csv, output_root, *, baseline_csv=None, cache_root=None, expected_names=None):
    reference_csv, output = Path(reference_csv).resolve(), Path(output_root).resolve()
    if output.exists():
        raise ValueError('Use a new audit output folder: '+str(output))
    if (baseline_csv is None) == (cache_root is None):
        raise ValueError('Choose exactly one: baseline_csv or cache_root')
    snapshot = {reference_csv: file_sha(reference_csv)}
    reference = row_masks(read_csv(reference_csv))
    protocol = None
    if cache_root is not None:
        root = Path(cache_root).resolve()
        if output == root or root in output.parents or output in root.parents:
            raise ValueError('Audit output must be separate from the inference cache folder')
        protocol, identity, runtime_signature, names = load_context(root, snapshot)
        baseline_csv = root/'submission_pipeline_baseline.csv'
        baseline_groups = row_masks(read_csv(baseline_csv))
        final_groups = row_masks(read_csv(root/'submission_pipeline_R1.csv'))
        if expected_names is not None and names_from(expected_names) != names:
            raise ValueError('Official image names disagree with cache protocol')
        from public_reference_stage_trace import replay_stages
    else:
        baseline_csv = Path(baseline_csv).resolve()
        snapshot[baseline_csv] = file_sha(baseline_csv)
        baseline_groups = row_masks(read_csv(baseline_csv))
        if expected_names is None:
            raise ValueError('CSV-only mode requires the official image-name manifest')
        names = names_from(expected_names)
    if isinstance(expected_names, (str, Path)):
        snapshot[Path(expected_names).resolve()] = file_sha(expected_names)
    if (set(reference) | set(baseline_groups)) - set(names):
        raise ValueError('CSV contains images outside the image-name manifest')
    if protocol and set(final_groups) - set(names):
        raise ValueError('R1 CSV contains images outside the cache protocol')
    all_reference, all_ours, image_rows = [], [], []
    images = {image['stem']: image for image in protocol['images']} if protocol else {}
    for number, stem in enumerate(names, 1):
        baseline = baseline_groups.get(stem, [])
        raw = unions = replay = final = None
        if protocol:
            stages, masks, final = load_image(root, images[stem], identity, runtime_signature, snapshot)
            expected_ids = [f'{stem}_{index}' for index in range(1, len(masks)+1)]
            if [identifier for identifier, _ in baseline] != expected_ids or [identifier for identifier, _ in final_groups.get(stem, [])] != expected_ids:
                raise ValueError('CSV instance IDs disagree with same-run baseline ordering')
            if [r['counts'] for r in masks] != [m['counts'] for _, m in baseline]:
                raise ValueError('Baseline CSV disagrees with cached mask order/content')
            if [r['counts'] for r in final] != [m['counts'] for _, m in final_groups.get(stem, [])]:
                raise ValueError('R1 CSV disagrees with cached mask order/content')
            raw, unions = candidates(stem, stages)
            replay = replay_stages(stem, stages, masks, expected_source_sha256=protocol['sources'])
        rows, ours, image = analyze_image(stem, reference.get(stem, []), baseline, final, raw, unions, replay)
        all_reference.extend(rows); all_ours.extend(ours); image_rows.append(image)
        if number % 20 == 0 or number == len(names):
            print(f'Reference audit {number}/{len(names)}', flush=True)
    for path, expected in snapshot.items():
        if file_sha(path) != expected:
            raise ValueError('Input changed during audit: '+str(path))
    counts = Counter(row['category'] for row in all_reference)
    summary = dict(schema_version=1, mode='completed_inference_cache' if protocol else 'CSV_geometry_only',
        scope='Prediction agreement against external reference; no ground-truth accuracy, PQ, TP, FP or FN inferred.',
        reference_role='Higher-public-score prediction reference, not per-instance ground truth',
        images=len(names), reference_instances=len(all_reference), baseline_instances=len(all_ours),
        excluded_known_placeholders=counts['known_placeholder_excluded'], categories=dict(counts),
        our_weak_reference_overlap=sum(row['category'] == 'weak_reference_overlap' for row in all_ours),
        raw_trace_available=bool(protocol),
        all_raw_similar_but_final_not_similar=sum(row['raw_similar_but_final_not_similar'] is True
            for row in all_reference) if protocol else None,
        candidate_inventory='All saved YOLO views and native base/TTA components at all stored thresholds; stored native components omit areas below100 pixels; not all possible neural predictions' if protocol else None,
        similarity_rule='Strict IoU > 0.5; weak final overlap < 0.1; native union coverage is pixel support only',
        view_support='Maximum distinct YOLO stage/view-to-exact-mask matching; threshold variants and native TTA are not independent views',
        transition_scope='Changes between actual pipeline output stages in maximum reference similarity; candidate identity may switch, not proof of specific confidence/NMS decisions. Aggregate raw pool is separate and excluded from transitions.',
        cache_identity=identity if protocol else None,
        input_sha256={str(path): value for path, value in snapshot.items()},
        source_sha256=file_sha(Path(__file__)), output_root=str(output))
    gaps = [row for row in all_reference if not row['excluded_placeholder'] and row['baseline_max_iou'] <= .5]
    gaps.sort(key=lambda r: (r['raw_similar_but_final_not_similar'] is not True, r['baseline_max_iou'],
                             -(r['raw_max_iou'] or 0.), r['reference_id']))
    output.mkdir(parents=True)
    for name, rows in (('reference_instances.csv', all_reference), ('our_instances.csv', all_ours),
                       ('per_image.csv', image_rows), ('prioritized_reference_gaps.csv', gaps)):
        write_csv(output/name, rows)
    summary['artifacts'] = {name: str(output/name) for name in ('reference_instances.csv', 'our_instances.csv',
        'per_image.csv', 'prioritized_reference_gaps.csv', 'summary.json')}
    (output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-csv', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--baseline-csv', type=Path)
    source.add_argument('--cache-root', type=Path)
    parser.add_argument('--expected-names', type=Path)
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps({key: result[key] for key in ('mode', 'images', 'categories', 'output_root')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
