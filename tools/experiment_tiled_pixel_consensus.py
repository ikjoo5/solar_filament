"""Fixed global tile/whole-frame foreground votes, with native spatial coverage.

One patch contributes at most one foreground vote. Interior crop margins are
excluded from both vote numerator and denominator. No GT object is selected.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pycocotools import mask as mu

from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from experiment_view_combinations import GT, VAL, SIZE, ROOT, component_rles
from filament_eval import Filament, evaluate

TILES = Path('/workspace/artifacts/tiled_results_20261027T205426Z_eba01cfb/raw_tiled_views.jsonl')
OLD = Path('/workspace/artifacts/validation_results_20261027T120958Z_444da7f4/raw_views.jsonl')
BASE = Path('/workspace/artifacts/boundary_smoothing_candidates_v2/gaussian_s1.2_t0.6.csv')
CONFIDENCES = (.1, .2, .3)


def tile_interior(top, left, margin=128, crop=1024, size=2048):
    """Global valid region; physical image borders are never trimmed."""
    return np.s_[top+(margin if top else 0):top+crop-(margin if top+crop<size else 0),
                 left+(margin if left else 0):left+crop-(margin if left+crop<size else 0)]


def plan():
    settings = [dict(name='baseline', kind='identity'), dict(name='old_BSV_close5', kind='old')]
    for c in CONFIDENCES:
        for family in (1536, 2048, 'both'):
            for fraction in (.25, .5, .75):
                settings.append(dict(name=f'tile{family}_c{c}_f{fraction}', kind='tile',
                                     confidence=c, family=family, fraction=fraction))
        settings += [dict(name=f'base_tile2of3_c{c}', kind='base_three', confidence=c),
                     dict(name=f'base_or_tile_agreement_c{c}', kind='base_union', confidence=c)]
        for required in (3, 4):
            settings.append(dict(name=f'BSV_tile5of{required}_c{c}', kind='five',
                                 confidence=c, required=required))
    assert len(settings) == 41
    return settings


def coverage_fixture():
    cover = np.zeros((SIZE, SIZE), np.uint8)
    for top in (0, 512, 1024):
        for left in (0, 512, 1024):
            cover[tile_interior(top, left)] += 1
    assert cover.min() == 1 and cover.max() == 4
    assert cover[0, 0] == 1 and cover[700, 700] == 4 and cover[1024, 1024] == 1
    assert tile_interior(0, 0) == np.s_[0:896, 0:896]
    assert tile_interior(1024, 1024) == np.s_[1152:2048, 1152:2048]
    return cover


def main():
    started = time.monotonic(); cv2.setNumThreads(1)
    settings = plan(); denominator = coverage_fixture()
    with VAL.open(newline='') as f:
        selection = list(csv.DictReader(f))
    groups = {r['image_id'] for r in selection}
    images = {Path(r['file_name']).stem for r in selection}
    baseline = read_rows(BASE); base_by_image = group_rows(baseline)
    raw = defaultdict(list); old = defaultdict(list)
    for line in TILES.open():
        r = json.loads(line); assert r['image'] in images and r['shape'] == [SIZE, SIZE]
        assert r['coordinate_space'] == 'original'
        raw[r['image']].append(r)
    assert set(raw) == images and all(len(v) == 18 for v in raw.values())
    for line in OLD.open():
        r = json.loads(line)
        if r['view'] in {'base', 'scale', 'vflip'}:
            old[r['image']].append(r)
    predictions = {s['name']: [] for s in settings}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for number, image in enumerate(sorted(images), 1):
        votes = {(size, c): np.zeros((SIZE, SIZE), np.uint8)
                 for size in (1536, 2048) for c in CONFIDENCES}
        for record in raw[image]:
            left, top, width, height = record['crop_bbox']
            assert width == height == 1024
            size = record['input_size']; region = tile_interior(top, left)
            unions = {c: np.zeros((SIZE, SIZE), bool) for c in CONFIDENCES}
            for proposal in record['instances']:
                if proposal['edge_clipped'] or proposal['confidence'] < min(CONFIDENCES):
                    continue
                mask = mu.decode(dict(size=[SIZE, SIZE], counts=proposal['rle'].encode('ascii'))).astype(bool)
                for c in CONFIDENCES:
                    if proposal['confidence'] >= c:
                        unions[c][region] |= mask[region]
            for c in CONFIDENCES:
                votes[size, c][region] += unions[c][region]
        densities = {(size, c): value.astype(np.float32)/denominator for (size, c), value in votes.items()}
        source_old = np.zeros((SIZE, SIZE), np.uint8)
        for record in old[image]:
            rles = [dict(size=[SIZE, SIZE], counts=p['rle'].encode('ascii'))
                    for p in record['instances'] if p['confidence'] >= .2]
            if rles:
                source_old += mu.decode(mu.merge(rles)).astype(np.uint8)
        base_members = base_by_image.get(image, [])
        base_rles = [r.rle(SIZE, SIZE) for r in base_members]
        base_union = mu.decode(mu.merge(base_rles)).astype(bool) if base_rles else np.zeros((SIZE, SIZE), bool)
        half = {(size, c): densities[size, c] >= .5 for size in (1536, 2048) for c in CONFIDENCES}
        for spec in settings:
            rows = predictions[spec['name']]
            if spec['kind'] == 'identity':
                rows.extend(base_members); continue
            kind = spec['kind']; c = spec.get('confidence')
            if kind == 'old':
                foreground = source_old >= 2
            elif kind == 'tile':
                density = ((densities[1536, c]+densities[2048, c])*.5
                           if spec['family'] == 'both' else densities[spec['family'], c])
                foreground = density >= spec['fraction']
            elif kind == 'base_three':
                count = base_union.astype(np.uint8) + half[1536, c] + half[2048, c]
                foreground = count >= 2
            elif kind == 'base_union':
                foreground = base_union | (half[1536, c] & half[2048, c])
            else:
                count = source_old + half[1536, c] + half[2048, c]
                foreground = count >= spec['required']
            closed = cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            for index, (rle, area) in enumerate(component_rles(closed, minimum=300), 1):
                rows.append(Filament(f'{image}_{index}', image, image, rle['counts'].decode('ascii'), area))
        if number % 10 == 0 or number == len(images):
            print(f'Tiled pixel consensus {number}/{len(images)}', flush=True)
    truth = read_rows(GT, gt=True, selected=groups)
    assert len(truth) == 708 and len(groups) == 101 and len(images) == 91
    reports = []
    for spec in settings:
        rows = predictions[spec['name']]; result = evaluate(truth, rows, SIZE, SIZE)
        assert result['tp_pairs'] == len(truth)-result['fn']
        reports.append(dict(**spec, **{k: result[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},
                            prediction_rows=len(rows), overlap_images=0, extra_matching_pairs=0))
    ref = reports[0]
    assert abs(ref['pq']-.4607591314345197) < 1e-12
    assert abs(reports[1]['pq']-.4571518474011121) < 1e-12
    best = max(reports, key=lambda r: r['pq']); directory = Path('/workspace/artifacts/tiled_pixel_consensus_candidates_v1')
    directory.mkdir(exist_ok=False)
    for record in (best, max(reports, key=lambda r: (r['pq'] if r['fp'] <= ref['fp']+25 else -1))):
        path = directory/(record['name']+'.csv')
        if path.exists():
            continue
        rows = predictions[record['name']]
        with path.open('x', newline='') as f:
            writer = csv.writer(f); writer.writerow(['filament_id','segmentation_rle'])
            writer.writerows((r.filament_id, r.counts) for r in rows)
        assert overlap_stats(read_rows(path))['overlap_images'] == 0
        assert abs(evaluate(truth, read_rows(path), SIZE, SIZE)['pq']-record['pq']) < 1e-12
        record['saved_csv'] = str(path)
    report = dict(scope='41 fixed same-development global foreground comparisons; not independent CV/LB',
        physical_images=len(images), annotation_groups=len(groups), gt_instances=len(truth),
        native_tile_interior_margin=128, normalization='per-pixel actual valid patch coverage, one vote per patch',
        edge_clipped_instances='excluded', tile_raw_sha256=hashlib.sha256(TILES.read_bytes()).hexdigest(),
        baseline_reproduced=True, old_BSV_reference_reproduced=True, experiments=reports,
        best=best, target_exceeded=best['pq']>.5, elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/tiled_pixel_consensus_experiments.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(best=best, elapsed_seconds=report['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
