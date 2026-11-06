"""Recompute uploaded scores from compressed RLEs and bootstrap physical images."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from filament_eval import Filament, _check_compressed_counts, evaluate, parse_filament_id


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path('/workspace/artifacts/validation_results_20261027T112905Z_947c4b1f')
VAL = Path('/workspace/attachments/bcb64220-b208-4cd7-9c3f-3f2be0f04167/val_split.csv')
GT = Path('/workspace/artifacts/solar_filament_gt_v1/gt.csv')


def load(path, ground_truth=False, groups=None):
    rows = []
    seen = set()
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ['filament_id', 'segmentation_rle']
        for row in reader:
            image_id, group_id = parse_filament_id(row['filament_id'], ground_truth)
            if groups is not None and group_id not in groups:
                continue
            assert row['filament_id'] not in seen
            seen.add(row['filament_id'])
            area = _check_compressed_counts(row['segmentation_rle'], 2048**2)
            assert area > 0
            rows.append(Filament(row['filament_id'], image_id, group_id,
                                 row['segmentation_rle'], area))
    return rows


def image_statistics(result):
    stats = defaultdict(lambda: np.zeros(4))
    for row in result['per_group']:
        stats[row['image_id']] += [row['matched_iou_sum'], row['tp_pairs'], row['fp'], row['fn']]
    return stats


def pq(totals):
    denominator = totals[..., 1] + .5*(totals[..., 2] + totals[..., 3])
    return np.divide(totals[..., 0], denominator, out=np.zeros_like(denominator), where=denominator > 0)


def main():
    with VAL.open(newline='') as handle:
        selection = list(csv.DictReader(handle))
    groups = {r['image_id'] for r in selection}
    truth = load(GT, True, groups)
    assert len(truth) == 708 and len(groups) == 101
    summary = json.loads((RESULTS/'summary.json').read_text())
    recomputed, table = {}, []
    for row in summary['comparisons']:
        prediction = load(RESULTS/row['prediction_csv'])
        result = evaluate(truth, prediction, 2048, 2048)
        for given, actual in [('PQ','pq'), ('SQ','sq'), ('RQ','rq'), ('TP','tp_pairs'), ('FP','fp'), ('FN','fn')]:
            assert abs(row[given] - result[actual]) < 1e-12, (row['prediction_csv'], given)
        recomputed[row['prediction_csv']] = result
        table.append({**row, 'prediction_rows': len(prediction),
                      'unique_matched_gt': 708-result['fn'],
                      'extra_matching_pairs': result['tp_pairs']-(708-result['fn']),
                      'images_without_predictions': len(result['images_without_predictions'])})
    reference = image_statistics(recomputed['union_cc_conf0.25.csv'])
    images = sorted(reference)
    assert len(images) == 91
    # All annotator variants of a physical image remain in the same bootstrap unit.
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(images), size=(5000, len(images)))
    base = np.asarray([reference[i] for i in images])
    base_scores = pq(base[draws].sum(axis=1))
    comparisons = []
    for name in ['preserve_conf0.15.csv', 'preserve_conf0.25.csv']:
        other_stats = image_statistics(recomputed[name])
        other = np.asarray([other_stats[i] for i in images])
        delta = pq(other[draws].sum(axis=1)) - base_scores
        comparisons.append({'candidate': name, 'baseline': 'union_cc_conf0.25.csv',
                            'development_delta_pq': float(pq(other.sum(axis=0))-pq(base.sum(axis=0))),
                            'paired_physical_image_bootstrap_95pct_interval': np.quantile(delta,[.025,.975]).tolist()})
    report = {'source_summary_sha256': hashlib.sha256((RESULTS/'summary.json').read_bytes()).hexdigest(),
              'physical_images':91, 'annotation_groups':101, 'gt_instances':708,
              'all_10_reported_scores_recomputed':True, 'comparisons':table,
              'bootstrap': {'draws':5000, 'seed':42, 'unit':'physical image, all annotators kept together',
                            'comparisons':comparisons,
                            'limitations':'Development-set resampling only; does not adjust for model/threshold selection, training leakage, cross-image temporal correlation, or predict leaderboard gains.'}}
    out = ROOT/'docs/evidence/validation_score_verification.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'recomputed_scores':10,'bootstrap':comparisons},indent=2))


if __name__ == '__main__':
    main()
