"""Re-score frozen candidates across predeclared metadata strata of the old91.

This is an internal heterogeneity diagnostic, not new validation observations.
No model or postprocessing parameters are fitted here.
"""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

from experiment_mask_postprocess import read_rows
from filament_eval import evaluate
from verify_native_training_results import GT, VAL

ROOT = Path(__file__).resolve().parents[1]
TRAIN = Path('/workspace/attachments/1cca1607-f94f-4334-9586-c1fb5ab292d7/train_split.csv')
METHODS = {
    'selected_submission': Path('/workspace/artifacts/rich_boundary_composition_v1/tta_precision.csv'),
    'round9_context_final': Path('/workspace/artifacts/context_pipeline_v1/context_final_only.csv'),
    'round10_signed_distance': Path('/workspace/artifacts/round10_joint_boundary/tta0.65_signed_distance.csv'),
}


def day(name):
    return datetime.strptime(name[:8], '%Y%m%d').date()


def pooled(rows):
    values = {key: sum(row[key] for row in rows) for key in ('tp_pairs','fp','fn','matched_iou_sum')}
    den = values['tp_pairs']+.5*(values['fp']+values['fn'])
    return dict(**values, pq=values['matched_iou_sum']/den if den else None,
        physical_images=len({row['image_id'] for row in rows}), annotation_groups=len(rows))


def main():
    selected = list(csv.DictReader(VAL.open()))
    groups = {r['image_id'] for r in selected}
    names = {Path(r['file_name']).stem for r in selected}
    train_days = {day(r['file_name']) for r in csv.DictReader(TRAIN.open())}
    distance = {name: min(abs((day(name)-d).days) for d in train_days) for name in names}
    strata = {'all91': names,
        'within3days_of_training': {name for name in names if distance[name] <= 3},
        'more_than3days_from_training': {name for name in names if distance[name] > 3}}
    for site in sorted({name[-2:] for name in names}):
        strata['observatory_'+site] = {name for name in names if name[-2:] == site}
    for lo, hi in ((2011,2014),(2015,2018),(2019,2022)):
        strata[f'period_{lo}_{hi}'] = {name for name in names if lo <= int(name[:4]) <= hi}
    truth = read_rows(GT, gt=True, selected=groups)
    results = {}
    for method, path in METHODS.items():
        score = evaluate(truth, read_rows(path), 2048, 2048)
        results[method] = {name: pooled([r for r in score['per_group'] if r['image_id'] in images])
                           for name, images in strata.items()}
    deltas = {}
    for method in list(METHODS)[1:]:
        deltas[method] = {name: {key: results[method][name][key]-results['selected_submission'][name][key]
                               for key in ('pq','tp_pairs','fp','fn')} for name in strata}
    report = dict(scope='Same91 observations regrouped by metadata; no new held-out images or new training',
        predeclared_strata='Training date distance<=3 versus>3 days; sixobservatories; fixed2011-14/2015-18/2019-22periods',
        methods=results, deltas_from_selected_submission=deltas,
        prediction_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name,path in METHODS.items()},
        limitations=['Methods were already selected on all91 images; no subgroup is an untouched confirmation set.',
                     'Small per-site/period groups have high uncertainty; do not fit site-specific thresholds from this table.'])
    (ROOT/'docs/evidence/validation_strata_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    for method in results:
        print(method, {name: results[method][name] for name in ('all91','within3days_of_training','more_than3days_from_training')})


if __name__ == '__main__':
    main()
