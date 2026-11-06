"""Cached-cohort parity test for the portable native submission recipe."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from filament_eval import Filament, evaluate
from submission_native_recipe import build_native_submission
from verify_native_training_results import GT, ROOT, VAL

BASE = Path('/workspace/artifacts/tiled_object_fusion_candidates_v2/c0.35_iou0.3_majority_guard0.85.csv')
OLD = Path('/workspace/artifacts/validation_results_20261027T120958Z_444da7f4/raw_views.jsonl')
PREP = Path('/workspace/artifacts/validation_results_20261027T125130Z_2312e759/raw_preprocess_views.jsonl')
TILES = Path('/workspace/artifacts/tiled_results_20261027T205426Z_eba01cfb/raw_tiled_views.jsonl')
RICH = Path('/workspace/artifacts/best84_rich_predictions_tta/predictions.jsonl')
EXPECTED = {
    'native_reference': Path('/workspace/artifacts/native_high_confidence_candidates_v1/native_t0.5_a300_weak0.6_families1.csv'),
    'rich_refined': Path('/workspace/artifacts/rich_boundary_refinement_v1/t0.65_fgmean0.8_guard0.85.csv'),
    'final': Path('/workspace/artifacts/rich_boundary_composition_v1/tta_precision.csv'),
}


def load_raw(path):
    result = defaultdict(list)
    for record in map(json.loads, path.open()):
        result[record['image']].append(record)
    return dict(result)


def main():
    base = group_rows(read_rows(BASE))
    old, prep, tiles = load_raw(OLD), load_raw(PREP), load_raw(TILES)
    expected = {stage: group_rows(read_rows(path)) for stage, path in EXPECTED.items()}
    final = []
    totals = Counter()
    decisions = Counter()
    records = list(map(json.loads, RICH.open()))
    assert len(records) == 91
    for number, record in enumerate(records, 1):
        image = Path(record['file_name']).stem
        result = build_native_submission(image, [r.rle(2048,2048) for r in base.get(image,[])],
            old[image],prep[image],tiles[image],record,return_stages=True)
        for stage in EXPECTED:
            counts = Counter(r['counts'].decode('ascii') for r in result[stage])
            target = Counter(r.counts for r in expected[stage].get(image,[]))
            assert counts == target, (image,stage,'mask mismatch')
            totals[stage] += len(result[stage])
        for index, encoded in enumerate(result['final'],1):
            from pycocotools import mask as mu
            final.append(Filament(f'{image}_{index}',image,image,
                encoded['counts'].decode('ascii'),int(mu.area(encoded))))
        decisions.update(result['decisions'])
        if number % 20 == 0:
            print('Native replay',number,'/91',flush=True)
    assert totals == dict(native_reference=630,rich_refined=630,final=625)
    with VAL.open(newline='') as f:
        groups = {r['image_id'] for r in csv.DictReader(f)}
    truth = read_rows(GT,gt=True,selected=groups)
    score = evaluate(truth,final,2048,2048)
    assert abs(score['pq']-.47259304445299505) < 1e-12
    audit = overlap_stats(final)
    assert not any(audit.values()) and score['tp_pairs'] == 708-score['fn']
    report = dict(scope='Portable submission native stages exactly replay91 cached frames; GPU test inference not executed',
        images=91,all_stage_masks_byte_identical=True,stages=dict(totals),decisions=dict(decisions),
        **{k:score[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},**audit,
        extra_matching_pairs=0)
    (ROOT/'docs/evidence/submission_native_recipe_parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
