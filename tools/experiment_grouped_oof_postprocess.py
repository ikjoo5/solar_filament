"""Four fixed OOF decoding rules; generate every CSV before reading GT labels."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from pycocotools import mask as mu
from evaluate_grouped_oof import aggregate, checked_rle, evaluate_oof, split_sha, truth_for
from grouped_cv_training import IDENTITY_KEYS, digest, metric

PRESETS = ('baseline', 'small_object_recall', 'precision', 'hysteresis')
PLAN = dict(baseline=dict(threshold=.5, min_area=300),
    small_object_recall=dict(threshold=.5, min_area=100, sub300_fg_mean=.8),
    precision=dict(threshold=.65, min_area=300),
    hysteresis=dict(threshold=.35, min_area=300, seed_threshold=.65, min_seed_overlap=100))


def metadata_records(folder, plan, fold, annotation_sha, plan_sha):
    """No annotation JSON or stored GT metric is used to choose predictions."""
    cfg = json.loads((folder/'config.json').read_text())
    summary = json.loads((folder/'oof_summary.json').read_text())
    payload = cfg.get('cv_identity_payload', {})
    if (hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != cfg['cv_identity']
            or payload.get('settings') != {k: cfg[k] for k in IDENTITY_KEYS}
            or payload.get('sources') != cfg['source_sha256'] or payload.get('fold') != fold
            or payload.get('final_selection') != 'fixed_epoch' or cfg['fold'] != fold
            or cfg['threshold'] != .5 or cfg['min_area'] != 300):
        raise ValueError('Expected verified fixed-final fold exports with baseline0.5/300')
    if (summary['status'] != 'complete' or not summary['oof_score_available']
            or summary['selected_epoch'] != cfg['epochs'] or summary['fixed_final_epoch'] != cfg['epochs']
            or summary['cv_identity'] != cfg['cv_identity']):
        raise ValueError('Only completed fixed-final models can enter OOF experiments')
    info = next(r for r in plan['fold_summaries'] if r['fold'] == fold)
    for key, value in dict(annotations=annotation_sha, fold_plan=plan_sha,
            train_split=split_sha(plan, info, 'train'), val_split=split_sha(plan, info, 'validation')).items():
        if cfg['source_sha256'].get(key) != value:
            raise ValueError('Fold/source metadata changed: '+key)
    for name, field in [('oof_predictions.csv','oof_csv_sha256'), ('oof_records.jsonl','oof_records_sha256')]:
        if digest(folder/name) != summary[field]:
            raise ValueError('Prediction export checksum changed')
    records = [json.loads(line) for line in (folder/'oof_records.jsonl').read_text().splitlines()]
    names = {r['file_name'] for r in plan['observations'] if info['roles'][r['physical_identifier']] == 'validation'}
    if len(records) != len(names) or {r['file_name'] for r in records} != names:
        raise ValueError('Missing/duplicate records, including empty prediction images')
    expected_csv = Counter()
    for record in records:
        if (record['fold'] != fold or record['cv_identity'] != cfg['cv_identity']
                or record['selected_epoch'] != cfg['epochs']
                or record['checkpoint_sha256'] != summary['checkpoint_sha256']
                or record.get('component_min_area') != 100):
            raise ValueError('Per-image model provenance changed')
        for threshold in ('0.35','0.5','0.65'):
            values = record['components'][threshold]
            rles = [checked_rle(r['rle']) for r in values]
            if any(int(mu.area(rle)) != row['area'] or not math.isfinite(row['fg_mean'])
                   or not 0 <= row['fg_mean'] <= 1 for row,rle in zip(values,rles)):
                raise ValueError('Invalid component metadata')
            if rles and int(mu.area(mu.merge(rles))) != sum(int(mu.area(rle)) for rle in rles):
                raise ValueError('Overlapping/duplicate components')
        baseline = [r['rle'] for r in record['components']['0.5'] if r['area'] >= 300]
        expected_csv.update((f"{Path(record['file_name']).stem}_{i}", r['counts'])
                            for i,r in enumerate(baseline,1))
    with (folder/'oof_predictions.csv').open(newline='') as stream:
        actual = Counter((r['filament_id'],r['segmentation_rle']) for r in csv.DictReader(stream))
    if actual != expected_csv:
        raise ValueError('Baseline CSV differs from full-coverage records')
    return records


def preset_masks(record):
    mid, high, low = [record['components'][t] for t in ('0.5','0.65','0.35')]
    seed = mu.merge([checked_rle(r['rle']) for r in high]) if high else None
    hysteresis = [r['rle'] for r in low if r['area'] >= 300 and seed is not None
        and int(mu.area(mu.merge([checked_rle(r['rle']),seed], intersect=True))) >= 100]
    return dict(baseline=[r['rle'] for r in mid if r['area'] >= 300],
        small_object_recall=[r['rle'] for r in mid if r['area'] >= 300
                             or r['area'] >= 100 and r['fg_mean'] >= .8],
        precision=[r['rle'] for r in high if r['area'] >= 300], hysteresis=hysteresis)


def save_csv(path, predicted):
    with path.open('w',newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['filament_id','segmentation_rle'])
        writer.writerows((f'{image}_{i}',r['counts']) for image,values in sorted(predicted.items())
                         for i,r in enumerate(values,1))


def read_csv(path, images):
    predicted = {image:[] for image in images}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            image = row['filament_id'].rsplit('_',1)[0]
            predicted[image].append(checked_rle(dict(size=[2048,2048],counts=row['segmentation_rle'])))
    return predicted


def run(annotations, fold_plan, runs_root, output_dir, report):
    annotations, fold_plan, runs_root, output_dir, report = map(Path,
        (annotations,fold_plan,runs_root,output_dir,report))
    plan = json.loads(fold_plan.read_text())
    output_dir.mkdir(parents=True,exist_ok=False)
    (output_dir/'predeclared_plan.json').write_text(json.dumps(PLAN,indent=2)+'\n')
    selected = {}
    for fold in range(plan['folds']):
        source = runs_root/f'fold{fold}'
        if not (source/'oof_summary.json').is_file():
            continue
        records = metadata_records(source,plan,fold,digest(annotations),digest(fold_plan))
        folder = output_dir/f'fold{fold}'; folder.mkdir()
        predicted = {name:{} for name in PRESETS}
        for record in records:
            for name,values in preset_masks(record).items():
                predicted[name][Path(record['file_name']).stem] = values
        for name,values in predicted.items():
            save_csv(folder/f'{name}.csv',values)
        if (folder/'baseline.csv').read_bytes() != (source/'oof_predictions.csv').read_bytes():
            raise ValueError('Generated baseline CSV is not byte-identical')
        selected[fold] = records
    # Every candidate CSV now exists. Only this second phase opens GT JSON.
    audited = evaluate_oof(annotations,fold_plan,runs_root)
    data = json.loads(annotations.read_text())
    by_stem = {Path(r['file_name']).stem:r for r in plan['observations']}
    results = {}
    for name in PRESETS:
        rows,folds = [],{}
        for fold,records in selected.items():
            truth,mapping = truth_for(data,{r['file_name'] for r in records})
            predicted = read_csv(output_dir/f'fold{fold}/{name}.csv',
                                 {Path(r['file_name']).stem for r in records})
            measured = metric(truth,predicted,mapping)
            rows.extend(measured['per_group']); folds[str(fold)] = aggregate(measured['per_group'])
        pooled = aggregate(rows) if rows else None
        def period(image):
            year = int(by_stem[image]['capture_day'][:4])
            return next((f'{lo}-{hi}' for lo,hi in [(2011,2014),(2015,2018),(2019,2022)] if lo<=year<=hi),'other')
        sites = {by_stem[r['image']]['site'] for r in rows}
        periods = {period(r['image']) for r in rows}
        results[name] = dict(pooled_oof=pooled if audited['complete_oof'] else None,
            pooled_completed_folds=pooled,by_fold=folds,
            by_site={site:aggregate([r for r in rows if by_stem[r['image']]['site']==site]) for site in sorted(sites)},
            by_period={p:aggregate([r for r in rows if period(r['image'])==p]) for p in sorted(periods)})
    deltas = {name:{key:values['pooled_completed_folds'][key]-results['baseline']['pooled_completed_folds'][key]
                   for key in ('pq','tp_pairs','fp','fn')} for name,values in results.items()
              if name!='baseline' and values['pooled_completed_folds'] is not None}
    result = dict(status=audited['status'],complete_oof=audited['complete_oof'],
        completed_folds=sorted(selected),expected_physical_images=audited['expected_physical_images'],
        completed_physical_images=audited['completed_physical_images'],presets=PLAN,results=results,
        deltas_from_baseline=deltas,all_csvs_generated_before_GT_read=True,baseline_byte_identical=True if selected else None,
        selection_performed=False,adopted=False,training_performed=False,new_network_inference=False,
        limitation='Fixed internal OOF comparisons; no automatic threshold selection or independent LB claim.')
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('annotations','fold-plan','runs-root','output-dir','report'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    result=run(args.annotations,args.fold_plan,args.runs_root,args.output_dir,args.report)
    print(json.dumps(result,indent=2))
