"""Fixed-weight, resumable validation ablations for reviewed SWEFil epoch100.

The same 91 development images were used for checkpoint selection: scores here
are development comparisons, not fresh cross-validation or leaderboard scores.
No training, public predictions, or test images are used. Probability maps stay
in the experiment cache; the small review ZIP contains masks and scores only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
import uuid
import zipfile

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage
import torch

import swefil_test_inference as reviewed
from chase_independent_decode import _encode, validate_masks

VERSION = 'swefil_epoch100_validation_ablation_v1'
SOURCE_FILES = ('swefil_inference_ablation.py', 'swefil_test_inference.py',
                'solar_a100_train_standalone.py', 'chase_best84_training.py',
                'chase_independent_decode.py')
RECIPES = {
    'baseline': dict(preprocessing='limb_only', overlap=.25, views=['base']),
    'tta4': dict(preprocessing='limb_only', overlap=.25, views=['base', 'h', 'v', 'hv']),
    # Native2048/tile1024 still has 3x3 tiles; this changes their placement.
    'dense': dict(preprocessing='limb_only', overlap=.5, views=['base']),
    'raw': dict(preprocessing='raw', overlap=.25, views=['base']),
}
THRESHOLDS = (.35, .5, .65)
MIN_AREAS = (100, 300, 500)
BASELINE_TOLERANCE = .003


def select_recipes(names):
    names = tuple(names.split(',') if isinstance(names, str) else names)
    if not names or len(set(names)) != len(names) or any(n not in RECIPES for n in names):
        raise ValueError('Choose distinct recipes from baseline,tta4,dense,raw')
    # Always reproduce the reviewed operating point before trying alternatives.
    return ('baseline',) + tuple(n for n in names if n != 'baseline')


def transform(array, view):
    """Flip is its own inverse; use identically on input and foreground output."""
    if view not in ('base', 'h', 'v', 'hv'):
        raise ValueError('Unknown geometric view')
    if 'v' in view:
        array = np.flip(array, axis=0)
    if 'h' in view:
        array = np.flip(array, axis=1)
    return np.ascontiguousarray(array)


def checked_probability(value, shape=(2048, 2048)):
    if (not isinstance(value, np.ndarray) or value.shape != shape
            or value.dtype != np.float32 or not np.isfinite(value).all()
            or value.min() < 0 or value.max() > 1):
        raise ValueError('Expected finite native float32 probabilities in [0,1]')
    return value


def cached_probability(path, cache_identity, producer, shape=(2048, 2048)):
    """A committed sidecar verifies identity and lossless probability bytes.

    Interrupted writes leave no committed sidecar; rerunning recomputes that
    view. An existing committed result with corrupt bytes is rejected.
    """
    path = Path(path)
    sidecar = path.with_suffix('.json')
    if path.is_symlink() or sidecar.is_symlink():
        raise ValueError('Unsafe probability cache link')
    path.parent.mkdir(parents=True, exist_ok=True)
    if sidecar.exists():
        record = reviewed.read_envelope(sidecar, cache_identity)
        if (not path.is_file() or record.get('npz_sha256') != reviewed.digest(path)
                or record.get('shape') != list(shape)):
            raise ValueError('Committed probability cache is missing or changed: ' + str(path))
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != ['probability']:
                raise ValueError('Unexpected probability cache contents')
            value = checked_probability(archive['probability'], shape)
        return value, True
    value = checked_probability(producer(), shape)
    pending = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    try:
        with pending.open('wb') as stream:
            np.savez_compressed(stream, probability=value)
        pending.replace(path)
        reviewed.envelope(sidecar, cache_identity,
                          dict(npz_sha256=reviewed.digest(path), shape=list(shape), dtype='float32'))
    finally:
        pending.unlink(missing_ok=True)
    return value, False


def masks_at_threshold(probability, threshold, minimum=100):
    """Eight-connected native masks, encoded locally without full-mask copies."""
    checked_probability(probability)
    labels, _ = ndimage.label(probability >= threshold, structure=np.ones((3, 3), np.uint8))
    areas = np.bincount(labels.ravel())
    masks = [_encode(labels[box] == number, box[0].start, box[1].start)
             for number, box in enumerate(ndimage.find_objects(labels), 1)
             if box is not None and areas[number] >= minimum]
    return validate_masks(masks)


def score_predictions(groups, predictions):
    """Global official strict IoU>.5 matching, including empty predictions."""
    if set(predictions) != {group['image'] for group in groups.values()}:
        raise ValueError('Every validation image, including empty images, must be present')
    validated = {name: validate_masks(masks) for name, masks in predictions.items()}
    rows = []
    for identifier, group in sorted(groups.items()):
        truth, masks = group['masks'], validated[group['image']]
        matrix = (np.asarray(mu.iou(truth, masks, [0] * len(masks))) if truth and masks
                  else np.zeros((len(truth), len(masks))))
        hit = matrix > .5
        rows.append(dict(group=identifier, image=group['image'], tp_pairs=int(hit.sum()),
                         fp=int((~hit.any(0)).sum()), fn=int((~hit.any(1)).sum()),
                         matched_iou_sum=float(matrix[hit].sum())))
    tp = sum(r['tp_pairs'] for r in rows)
    fp = sum(r['fp'] for r in rows)
    fn = sum(r['fn'] for r in rows)
    total = sum(r['matched_iou_sum'] for r in rows)
    denominator = tp + .5 * (fp + fn)
    return dict(pq=total / denominator if denominator else 0.,
                sq=total / tp if tp else 0., rq=tp / denominator if denominator else 0.,
                tp_pairs=tp, fp=fp, fn=fn, matched_iou_sum=total,
                precision=tp / (tp + fp) if tp + fp else 0.,
                recall=tp / (tp + fn) if tp + fn else 0.,
                physical_images=len(predictions), annotation_groups=len(groups),
                gt_instances=sum(len(g['masks']) for g in groups.values()),
                prediction_instances=sum(map(len, predictions.values())), per_group=rows)


def write_masks_csv(path, predicted):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    try:
        with pending.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['filament_id', 'segmentation_rle'])
            for stem, masks in sorted(predicted.items()):
                for index, mask in enumerate(masks, 1):
                    writer.writerow([f'{stem}_{index}', mask['counts'].decode('ascii')])
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def validation_inputs(training_root, cfg, *, images=None, annotations=None, val_split=None):
    identity = reviewed.read_envelope(Path(training_root) / 'train/identity.json',
                                     reviewed.TRAINING_IDENTITY)
    source_paths = dict(annotations=Path(annotations or cfg['annotations']),
                        train_split=Path(cfg['train_split']), val_split=Path(val_split or cfg['val_split']))
    for key, path in source_paths.items():
        if not path.is_file() or path.is_symlink() or reviewed.digest(path) != identity['sources'][key]:
            raise ValueError('Reviewed validation source changed or missing: ' + key + ' at ' + str(path))
    data = json.loads(source_paths['annotations'].read_text())
    by_id = {row['id']: row for row in data['images']}
    if len(by_id) != len(data['images']):
        raise ValueError('Duplicate annotation groups')

    def split(path):
        with path.open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        ids = {row['image_id'] for row in rows}
        if (len(ids) != len(rows) or not ids <= set(by_id)
                or any(by_id[r['image_id']]['file_name'] != r['file_name'] for r in rows)):
            raise ValueError('Split annotation groups or filenames differ')
        return ids, {row['file_name'] for row in rows}

    train_ids, train_names = split(source_paths['train_split'])
    val_ids, names = split(source_paths['val_split'])
    if (train_ids & val_ids or train_names & names or len(train_names) != 616
            or len(names) != 91 or len(val_ids) != 101
            or val_ids != {r['id'] for r in data['images'] if r['file_name'] in names}):
        raise ValueError('Expected the original disjoint 616/91 physical-image split and all101 validation groups')
    groups = {key: dict(image=Path(by_id[key]['file_name']).stem, masks=[]) for key in sorted(val_ids)}
    for ann in data['annotations']:
        if ann['image_id'] in groups:
            groups[ann['image_id']]['masks'].append(mu.merge(mu.frPyObjects(ann['segmentation'], 2048, 2048)))
    if sum(len(g['masks']) for g in groups.values()) != 708 or any(not g['masks'] for g in groups.values()):
        raise ValueError('Expected all 708 original validation instances')
    images = Path(images or cfg['images']).resolve()
    sources = {}
    for index, name in enumerate(sorted(names), 1):
        if Path(name).name != name or name not in identity['source_images']:
            raise ValueError('Unexpected validation filename')
        path = images / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError('Validation image missing: ' + str(path))
        sources[name] = reviewed.digest(path)
        if sources[name] != identity['source_images'][name]:
            raise ValueError('Validation image bytes differ from training review: ' + name)
        if index % 25 == 0 or index == len(names):
            print(f'Validation source verification {index}/{len(names)}', flush=True)
    return images, sources, groups, {key: reviewed.digest(path) for key, path in source_paths.items()}


def probability_for_recipe(raw, name, sources, recipe_name, output, identity, predictor):
    recipe = RECIPES[recipe_name]
    image = None
    accumulator = None
    reused = 0
    for view in recipe['views']:
        settings = dict(preprocessing=recipe['preprocessing'], overlap=recipe['overlap'], view=view, tile=1024)
        key = reviewed.fingerprint(settings)[:16]
        cache_identity = reviewed.fingerprint(dict(experiment=identity, file_name=name,
            image_sha256=sources[name], settings=settings))

        def produce():
            nonlocal image
            if image is None:
                image = reviewed.base.limb(raw) if recipe['preprocessing'] == 'limb_only' else raw
            probabilities = predictor(transform(image, view), recipe['overlap'])
            return transform(probabilities, view)

        probability, cached = cached_probability(Path(output) / 'probabilities' / key / (name + '.npz'),
                                                 cache_identity, produce)
        reused += int(cached)
        if accumulator is None:
            accumulator = probability
        else:
            accumulator += probability
    accumulator /= len(recipe['views'])
    return checked_probability(accumulator), reused


def summarize_recipe(output, recipe_name, groups, predictions):
    rows = []
    for threshold in THRESHOLDS:
        for area in MIN_AREAS:
            chosen = {stem: [m for m in masks if int(mu.area(m)) >= area]
                      for stem, masks in predictions[threshold].items()}
            metric = score_predictions(groups, chosen)
            filename = f'{recipe_name}_t{threshold:g}_a{area}.csv'
            path = Path(output) / 'masks' / filename
            write_masks_csv(path, chosen)
            rows.append(dict(recipe=recipe_name, threshold=threshold, min_area=area,
                             csv='masks/' + filename, csv_sha256=reviewed.digest(path), **metric))
    reviewed.atomic_json(Path(output) / (recipe_name + '_scores.json'), rows)
    return rows


def export_review(output, protocol, scores, baseline, status, elapsed):
    output = Path(output)
    ranking = sorted(scores, key=lambda r: (-r['pq'], r['recipe'], r['threshold'], r['min_area']))
    top = []
    seen = set()
    for row in ranking:
        if row['csv_sha256'] not in seen:
            top.append({k: v for k, v in row.items() if k != 'per_group'})
            seen.add(row['csv_sha256'])
        if len(top) == 3:
            break
    result = dict(status=status, identity=reviewed.fingerprint(protocol), output_root=str(output),
        checkpoint_sha256=reviewed.CHECKPOINT_SHA, checkpoint_epoch=100,
        validation_images=len(protocol['source_images']), annotation_groups=protocol['annotation_groups'],
        gt_instances=protocol['gt_instances'], recipes=protocol['recipe_order'],
        completed_recipes=sorted({r['recipe'] for r in scores}),
        configurations_scored=len(scores), best=top[0], top_three=top,
        baseline={k: v for k, v in baseline.items() if k != 'per_group'},
        baseline_expected_pq=reviewed.REVIEWED_PQ, baseline_tolerance=BASELINE_TOLERANCE,
        baseline_pq_difference=baseline['pq'] - reviewed.REVIEWED_PQ,
        elapsed_seconds=elapsed,
        scope='Repeated 91-image development validation; no retraining or test/reference predictions. '
              'These images already selected the checkpoint, so this is not an unbiased holdout estimate.')
    reviewed.atomic_json(output / 'scores.json', scores)
    reviewed.atomic_json(output / 'review_manifest.json', result)
    archive = output / 'swefil_inference_ablation_results.zip'
    pending = archive.with_suffix('.pending.zip')
    try:
        with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED) as z:
            for name in ('identity.json', 'checkpoint_selection.json', 'validation_gt.csv',
                         'scores.json', 'review_manifest.json', 'progress.json'):
                z.write(output / name, name)
            for row in scores:
                z.write(output / row['csv'], row['csv'])
            for name in SOURCE_FILES:
                z.write(Path(__file__).with_name(name), 'code/' + name)
        pending.replace(archive)
    finally:
        pending.unlink(missing_ok=True)
    result.update(archive=str(archive), archive_sha256=reviewed.digest(archive))
    reviewed.atomic_json(output / 'latest_result.json', result)
    return result


def run(training_root, checkpoint, output, images=None, annotations=None, val_split=None,
        recipes=('baseline', 'tta4', 'dense', 'raw')):
    recipes = select_recipes(recipes)
    training_root, output = Path(training_root).resolve(), Path(output).resolve()
    if output == training_root or output in training_root.parents or training_root in output.parents:
        raise ValueError('Ablation output must be separate from training')
    metadata = reviewed.training_metadata(training_root, checkpoint)
    images, sources, groups, source_hashes = validation_inputs(training_root, metadata['config'],
        images=images, annotations=annotations, val_split=val_split)
    if (output == images or output in images.parents or images in output.parents
            or Path(checkpoint).resolve().is_relative_to(output)):
        raise ValueError('Ablation output must be separate from input images and checkpoint')
    if not torch.cuda.is_available():
        raise RuntimeError('Select a T4 GPU runtime; inference is not training')
    reviewed.configure_cuda()
    cv2.setNumThreads(2)
    protocol = dict(version=VERSION, checkpoint_sha256=reviewed.CHECKPOINT_SHA,
        training_identity=reviewed.TRAINING_IDENTITY, source_images=sources, validation_sources=source_hashes,
        annotation_groups=len(groups), gt_instances=sum(len(g['masks']) for g in groups.values()),
        recipes={name: RECIPES[name] for name in recipes}, recipe_order=list(recipes),
        thresholds=THRESHOLDS, min_areas=MIN_AREAS,
        tile=1024, infer_batch=1, precision='float16', channel=0, connectivity=8,
        score='Global all-pairs IoU>0.5, all annotation groups',
        source_code={name: reviewed.digest(Path(__file__).with_name(name)) for name in SOURCE_FILES},
        runtime=reviewed.runtime_identity())
    # Canonical JSON roundtrip gives stable tuples/lists equality when resuming.
    protocol = json.loads(reviewed.canonical(protocol))
    identity = reviewed.fingerprint(protocol)
    output.mkdir(parents=True, exist_ok=True)
    with reviewed.locked(output):
        identity_path = output / 'identity.json'
        if identity_path.exists():
            if reviewed.read_envelope(identity_path, identity) != protocol:
                raise ValueError('Ablation protocol changed; use a different output folder')
        else:
            if any((output / p).exists() for p in ('probabilities', 'masks', 'latest_result.json')):
                raise ValueError('Existing ablation cache has no identity; use another output folder')
            reviewed.envelope(identity_path, identity, protocol)
        reviewed.atomic_json(output / 'checkpoint_selection.json', metadata['checkpoint_selection'])
        write_masks_csv(output / 'validation_gt.csv', {key: group['masks'] for key, group in groups.items()})
        # Invalidate a prior completion marker before any resumed work or checks.
        reviewed.atomic_json(output / 'latest_result.json', dict(status='running', identity=identity,
                                                               output_root=str(output)))
        model = None
        device = torch.device('cuda')

        def predictor(image, overlap):
            nonlocal model
            if model is None:
                payload = reviewed.load_reviewed_checkpoint(checkpoint)
                model = reviewed.base.build_model('tu-convnext_tiny', pretrained=False)
                model.load_state_dict(payload['model_state_dict'], strict=True)
                del payload
                model = model.to(device).eval()
            return reviewed.base.predict_foreground(model, image, device, tile=1024, overlap=overlap,
                                                     batch=1, amp_dtype=torch.float16)

        started = time.monotonic()
        scores = []
        baseline = None
        status = 'complete'
        print('Validation only: frozen epoch100, selected inference families, no optimization updates.', flush=True)
        if 'dense' in recipes:
            print('dense = uniform_overlap: 1024 tiles at overlap0.5; still nine tiles per 2048 image.',
                  flush=True)
        for recipe_name in recipes:
            predictions = {threshold: {} for threshold in THRESHOLDS}
            for index, name in enumerate(sorted(sources), 1):
                raw = reviewed.read_image(images / name, sources[name])
                probability, reused = probability_for_recipe(raw, name, sources, recipe_name, output,
                                                              identity, predictor)
                for threshold in THRESHOLDS:
                    predictions[threshold][Path(name).stem] = masks_at_threshold(probability, threshold)
                del raw, probability
                reviewed.atomic_json(output / 'progress.json', dict(status='running', recipe=recipe_name,
                    images_completed=index, validation_images=len(sources), identity=identity))
                print(f'{recipe_name} [{index}/{len(sources)}] {name}: '
                      f'{reused}/{len(RECIPES[recipe_name]["views"])} views cached', flush=True)
            rows = summarize_recipe(output, recipe_name, groups, predictions)
            scores.extend(rows)
            best = max(rows, key=lambda row: row['pq'])
            print(f'{recipe_name}: best development PQ={best["pq"]:.6f}, '
                  f'TP={best["tp_pairs"]} FP={best["fp"]} FN={best["fn"]}', flush=True)
            if recipe_name == 'baseline':
                baseline = next(r for r in rows if r['threshold'] == .65 and r['min_area'] == 300)
                if abs(baseline['pq'] - reviewed.REVIEWED_PQ) > BASELINE_TOLERANCE:
                    status = 'baseline_mismatch'
                    print('STOP: baseline does not reproduce the reviewed PQ within .003; '
                          'inspect the review ZIP before interpreting alternatives.', flush=True)
                    break
        reviewed.atomic_json(output / 'progress.json', dict(status=status, identity=identity,
            completed_recipes=sorted({r['recipe'] for r in scores}), validation_images=len(sources)))
        return export_review(output, protocol, scores, baseline, status, time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training-root', 'checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('images', 'annotations', 'val-split'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--recipes', default='baseline,tta4,dense,raw')
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps(result, indent=2), flush=True)
    if result['status'] != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
