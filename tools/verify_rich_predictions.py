"""Verify the user-run best84 rich export without running attached source code."""
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from filament_eval import Filament, _check_compressed_counts, evaluate
from verify_native_training_results import GT, ROOT, VAL, hits_by_gt

DATA = Path('/workspace/artifacts/best84_rich_predictions_tta')
PREVIOUS = Path('/workspace/artifacts/a100_results_20261028_bedf3e21')
INCUMBENT = Path('/workspace/artifacts/native_high_confidence_candidates_v1/native_t0.5_a300_weak0.6_families1.csv')
FIELDS = ('pq', 'sq', 'rq', 'tp_pairs', 'fp', 'fn')
THRESHOLDS = [.2, .35, .5, .65, .8]


def rows_from(records, mode, threshold, minimum=300):
    rows = []
    for record in records:
        source = record if mode == 'base' else record['tta']
        image = Path(record['file_name']).stem
        for index, component in enumerate(source['components'][f'{threshold:g}'], 1):
            if component['area'] >= minimum:
                rows.append(Filament(f'{image}_{index}', image, image,
                    component['rle']['counts'], component['area']))
    return rows


def main():
    manifest = json.loads((DATA/'manifest.json').read_text())
    summary = json.loads((DATA/'summary.json').read_text())
    provenance = manifest['provenance']
    compact = lambda value: json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    assert hashlib.sha256(compact(provenance).encode()).hexdigest() == manifest['identity']
    assert summary['physical_images'] == 91 and summary['best_epoch'] == 84
    assert summary['checkpoint_sha256'] == provenance['checkpoint_sha256']
    assert provenance['source_sha256'] == '6d60b5705fb0fbde03a6fd2922c8b83839fa49b5a78cc3434e7dcf4a1b4a08b4'
    # Exact worker text pasted in the preceding user-facing response. The local
    # copy has an extra explanatory docstring/comment; numerical code is equal.
    assert provenance['exporter_sha256'] == 'ada352addf19217168b777da631c14b191291a418456000d0ff25c0586e1f633'
    assert provenance['tta']['views'] == ['base', 'hflip', 'vflip', 'hvflip']
    assert provenance['crop'] == 1024 and provenance['overlap'] == .25
    assert provenance['foreground_thresholds'] == THRESHOLDS
    assert provenance['spine_thresholds'] == [.3, .5, .7]
    assert provenance['min_area'] == 100
    with VAL.open(newline='') as f:
        split = list(csv.DictReader(f))
    groups = {r['image_id'] for r in split}
    expected_names = {r['file_name'] for r in split}
    assert len(expected_names) == 91 and len(groups) == 101
    source_cfg = json.loads((PREVIOUS/'config.json').read_text())
    for key, digest in provenance['source_hashes'].items():
        assert digest == source_cfg['source_sha256'][key]
    records = []
    checked_rles = 0
    for line in (DATA/'predictions.jsonl').read_bytes().splitlines(keepends=True):
        record = json.loads(line)
        name = record['file_name']
        entry = manifest['completed'][name]
        assert hashlib.sha256(line).hexdigest() == entry['record_sha256']
        assert entry['image_sha256'] == record['image_sha256']
        assert len(record['image_sha256']) == 64
        assert name in expected_names and record['size'] == [2048, 2048]
        assert record['identity'] == manifest['identity']
        assert set(record['view_unions']) == {'base', 'hflip', 'vflip', 'hvflip'}
        for source in (record, record['tta']):
            assert source['file_name'] == name and source['size'] == [2048, 2048]
            assert source['identity'] == record['identity'] and source['image_sha256'] == record['image_sha256']
            assert set(source['components']) == {f'{t:g}' for t in THRESHOLDS}
            for head, thresholds in [('foreground', THRESHOLDS), ('spine', [.3, .5, .7])]:
                assert set(source['unions'][head]) == {f'{t:g}' for t in thresholds}
                previous = None
                for t in thresholds:
                    encoded = source['unions'][head][f'{t:g}']
                    assert encoded['size'] == [2048,2048]
                    _check_compressed_counts(encoded['counts'], 2048**2)
                    checked_rles += 1
                    if previous is not None:
                        assert int(mu.area(mu.merge([previous, encoded], intersect=True))) == int(mu.area(encoded))
                    previous = encoded
            for key, components in source['components'].items():
                encoded = [c['rle'] for c in components]
                areas = []
                for component in components:
                    value = component['rle']
                    assert value['size'] == [2048,2048]
                    area = _check_compressed_counts(value['counts'], 2048**2)
                    assert area == component['area'] == int(mu.area(value)) and area >= 100
                    assert list(map(int, mu.toBbox(value))) == component['bbox']
                    checked_rles += 1
                    areas.append(area)
                    for head in ('foreground', 'spine'):
                        stats = component['stats'][head]
                        assert set(stats) == {'mean','max','p90'}
                        assert all(np.isfinite(v) and 0 <= v <= 1 for v in stats.values())
                        assert stats['mean'] <= stats['max']+1e-6 and stats['p90'] <= stats['max']+1e-6
                    assert component['stats']['foreground']['mean'] >= float(key)-1e-6
                if encoded:
                    merged = mu.merge(encoded)
                    assert int(mu.area(merged)) == sum(areas), 'Components overlap'
                    assert int(mu.area(mu.merge([merged, source['unions']['foreground'][key]], intersect=True))) == sum(areas)
        for encoded in record['view_unions'].values():
            assert encoded['size'] == [2048,2048]
            _check_compressed_counts(encoded['counts'], 2048**2)
            checked_rles += 1
        assert record['view_unions']['base'] == record['unions']['foreground']['0.5']
        records.append(record)
    assert len(records) == 91 and {r['file_name'] for r in records} == expected_names
    assert set(manifest['completed']) == expected_names
    # Verify the newly re-inferred base exactly reproduces every prior saved grid.
    parity = []
    for threshold in (.35,.5,.65):
        for minimum in (100,200,300):
            rows = rows_from(records, 'base', threshold, minimum)
            prior = read_rows(PREVIOUS/f'best_t{threshold:g}_a{minimum}.csv')
            before = Counter((r.image_id,r.counts) for r in prior)
            after = Counter((r.image_id,r.counts) for r in rows)
            assert before == after, (threshold,minimum)
            parity.append(dict(threshold=threshold,min_area=minimum,identical_masks=len(rows)))
    truth = read_rows(GT, gt=True, selected=groups)
    assert len(truth) == 708
    incumbent = read_rows(INCUMBENT)
    incumbent_metric = evaluate(truth, incumbent,2048,2048)
    before = hits_by_gt(truth, incumbent)
    grid = []
    for mode in ('base','tta'):
        for threshold in THRESHOLDS:
            rows = rows_from(records,mode,threshold)
            metric = evaluate(truth,rows,2048,2048)
            assert metric['tp_pairs'] == 708-metric['fn']
            grid.append(dict(mode=mode,threshold=threshold,min_area=300,
                **{k:metric[k] for k in FIELDS},physical_masks=len(rows)))
    all_tta = [r for t in THRESHOLDS for r in rows_from(records,'tta',t,100)]
    best_iou = hits_by_gt(truth,all_tta)
    status = Counter()
    for fid, value in before.items():
        b,a=value>.5,best_iou[fid]>.5
        status['retained' if b and a else 'tta_only' if a else 'incumbent_only' if b else 'both_miss']+=1
    report = dict(scope='Archive/data integrity and exact base reproduction; fixed development set, not Kaggle LB',
        archive_sha256='7077c7b7f8f46608b45dd7697bf3008b2ca50440b41be3fd268534c0176c1e1b',
        records=91,annotation_groups=101,gt_instances=708,checked_rles=checked_rles,
        manifest_record_sha256_all_match=True,source_hashes_all_match=True,
        checkpoint_sha256=provenance['checkpoint_sha256'],exporter_sha256=provenance['exporter_sha256'],
        exporter_matches_exact_delivered_cell=True,
        actual_image_bytes_available_locally=False,image_sha_note='Hashes agree within export; original Drive image bytes are not present locally for independent hashing.',
        probability_stats_note='Finite/range-consistent metadata; continuous maps and checkpoint are not in this ZIP, so individual numeric probabilities cannot be independently recomputed here.',
        base_parity_with_all9_training_grids=parity,
        incumbent={k:incumbent_metric[k] for k in FIELDS},grid_area300=grid,
        all_tta_thresholds_oracle_availability=dict(status),
        oracle_note='GT-aware error diagnosis across all five thresholds and area>=100. Not a realizable selector or achieved ensemble score.')
    destination=ROOT/'docs/evidence/rich_predictions_verification.json'
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='base_parity_with_all9_training_grids'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
