"""Ninety-six global additions from probability/TTA-confirmed U-Net proposals.

All admission metadata and all candidate predictions are constructed before GT
is loaded. Original masks are preserved byte-for-byte apart from row IDs.
"""
from collections import Counter
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mu

from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from experiment_native_cross_model import matched_ids, trim
from experiment_native_weak_support import load_sources
from experiment_view_combinations import GT, ROOT, SIZE, VAL
from filament_eval import Filament, _check_compressed_counts, evaluate

SOURCE = Path('/workspace/artifacts/best84_rich_predictions_tta/predictions.jsonl')
BASELINE = Path('/workspace/artifacts/native_high_confidence_candidates_v1/native_t0.5_a300_weak0.6_families1.csv')
OUT = Path('/workspace/artifacts/rich_cross_model_candidates_v1')


def plan(refine_confirmation=False):
    if refine_confirmation:
        result = [dict(name=f'{domain}_t0.35_a300_fg{fg:g}_view3_weak{weak:g}',
            domain=domain, threshold=.35, min_area=300,
            max_original_iou=.1,max_original_union_fraction=.25,
            minimum_support_iou=.3,per_view_coverage=.5,
            policy='joint_view_probability_weak_support',weak_confidence=weak,
            fg_mean=fg,minimum_views=3)
            for domain in ('base','tta') for fg in (.9,.95) for weak in (.1,.3)]
        assert len(result) == 8
        return result
    policies = [
        dict(policy='weak0.6', weak_confidence=.6, fg_mean=0., minimum_views=0),
        dict(policy='weak0.3_fg0.9', weak_confidence=.3, fg_mean=.9, minimum_views=0),
        dict(policy='weak0.3_fg0.95', weak_confidence=.3, fg_mean=.95, minimum_views=0),
        dict(policy='fg0.9_view3', weak_confidence=0., fg_mean=.9, minimum_views=3),
        dict(policy='fg0.95_view4', weak_confidence=0., fg_mean=.95, minimum_views=4),
        dict(policy='weak0.6_fg0.9_view2', weak_confidence=.6, fg_mean=.9, minimum_views=2),
    ]
    result = [dict(name=f'{domain}_t{threshold:g}_a{area}_{p["policy"]}',
        domain=domain, threshold=threshold, min_area=area,
        max_original_iou=.1, max_original_union_fraction=.25,
        minimum_support_iou=.3, per_view_coverage=.5, **p)
        for domain in ('base', 'tta') for threshold in (.2,.35,.5,.65)
        for area in (300,600) for p in policies]
    assert len(result) == 96
    return result


def json_rle(value):
    assert value['size'] == [SIZE,SIZE]
    area = _check_compressed_counts(value['counts'], SIZE*SIZE)
    return dict(size=[SIZE,SIZE], counts=value['counts'].encode('ascii')), area


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refine-confirmation',action='store_true')
    args = parser.parse_args()
    output = OUT.with_name('rich_cross_model_confirmation_v1') if args.refine_confirmation else OUT
    started = time.monotonic()
    settings = plan(args.refine_confirmation)
    output.mkdir(exist_ok=False)
    (output/'predeclared_plan.json').write_text(json.dumps(settings,indent=2)+'\n')
    with VAL.open(newline='') as f:
        selection = list(csv.DictReader(f))
    groups = {r['image_id'] for r in selection}
    images = {Path(r['file_name']).stem for r in selection}
    assert len(groups) == 101 and len(images) == 91
    baseline = read_rows(BASELINE)
    assert len(baseline) == 630 and not overlap_stats(baseline)['overlap_images']
    by_image = group_rows(baseline)
    records = {}
    for line in SOURCE.open():
        row = json.loads(line)
        image = Path(row['file_name']).stem
        assert image in images and image not in records and row['size'] == [SIZE,SIZE]
        records[image] = row
    assert set(records) == images
    weak, weak_meta = load_sources(images)
    predictions = {s['name']:[] for s in settings}
    counters = {s['name']:Counter() for s in settings}
    metadata_counts = Counter()
    for image in sorted(images):
        original = by_image.get(image, [])
        br = [r.rle(SIZE,SIZE) for r in original]
        covered = mu.merge(br) if br else None
        record = records[image]
        proposals = weak[image]
        wr = [p['rle'] for p in proposals]
        views = [json_rle(value)[0] for value in record['view_unions'].values()]
        assert len(views) == 4
        for domain in ('base','tta'):
            data = record if domain == 'base' else record['tta']
            for threshold in (.2,.35,.5,.65):
                native = data['components'][f'{threshold:g}']
                nr = [json_rle(c['rle'])[0] for c in native]
                bm = (np.asarray(mu.iou(nr,br,[0]*len(br))) if nr and br
                      else np.zeros((len(nr),len(br))))
                wm = (np.asarray(mu.iou(nr,wr,[0]*len(wr))) if nr and wr
                      else np.zeros((len(nr),len(wr))))
                eligible = []
                for i, component in enumerate(native):
                    area = component['area']
                    assert int(mu.area(nr[i])) == area
                    if area < 300 or (br and bm[i].max() > .1):
                        continue
                    fraction = (int(mu.area(mu.merge([nr[i],covered],intersect=True)))/area
                                if covered else 0.)
                    if fraction > .25:
                        continue
                    row = Filament(f'{image}_{i+1}',image,image,component['rle']['counts'],area)
                    clipped = trim(row,covered)
                    if clipped is None or clipped.area < 300:
                        continue
                    confidence = max((p['confidence'] for j,p in enumerate(proposals)
                                      if wm[i,j] >= .3),default=0.)
                    view_coverage = [int(mu.area(mu.merge([nr[i],r],intersect=True)))/area
                                     for r in views]
                    n_views = sum(value >= .5 for value in view_coverage)
                    eligible.append(dict(row=clipped,original_area=area,
                        weak_confidence=confidence,fg_mean=component['stats']['foreground']['mean'],
                        minimum_views=n_views))
                metadata_counts[f'{domain}_{threshold:g}'] += len(eligible)
                for setting in (s for s in settings if s['domain']==domain and s['threshold']==threshold):
                    result = list(original)
                    for p in eligible:
                        if min(p['row'].area,p['original_area']) < setting['min_area']:
                            continue
                        if (p['weak_confidence'] < setting['weak_confidence']
                                or p['fg_mean'] < setting['fg_mean']
                                or p['minimum_views'] < setting['minimum_views']):
                            continue
                        result.append(p['row'])
                        counters[setting['name']]['physical_additions'] += 1
                    predictions[setting['name']].extend(Filament(f'{image}_{j}',image,image,r.counts,r.area)
                                                       for j,r in enumerate(result,1))
    # The complete predictions above use only images/model outputs; GT starts here.
    truth = read_rows(GT,gt=True,selected=groups)
    assert len(truth) == 708
    reference = evaluate(truth,baseline,SIZE,SIZE)
    assert abs(reference['pq']-.46556248016002527) < 1e-12
    refmatches = matched_ids(truth,baseline)
    fields = ('pq','sq','rq','tp_pairs','fp','fn')
    reports = [dict(name='incumbent',**{k:reference[k] for k in fields},prediction_rows=len(baseline))]
    for setting in settings:
        rows = predictions[setting['name']]
        metric = evaluate(truth,rows,SIZE,SIZE)
        assert metric['tp_pairs'] == len(truth)-metric['fn']
        audit = overlap_stats(rows)
        assert not any(audit.values()), (setting,audit)
        reports.append(dict(**setting,**{k:metric[k] for k in fields},
            prediction_rows=len(rows),**dict(counters[setting['name']])))
    best = max(reports,key=lambda r:r['pq'])
    actual = max(reports[1:],key=lambda r:r['pq'])
    recall = min(reports,key=lambda r:(r['fn'],-r['pq']))
    nontrivial = [r for r in reports[1:] if r.get('physical_additions',0)]
    best_changed = max(nontrivial,key=lambda r:r['pq']) if nontrivial else None
    targets = {r['name']:r for r in (best,recall,best_changed) if r and r['name']!='incumbent'}
    saved = []
    for name, setting in targets.items():
        path = output/(name+'.csv')
        with path.open('w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filament_id','segmentation_rle'])
            writer.writerows((r.filament_id,r.counts) for r in predictions[name])
        rows = read_rows(path)
        metric = evaluate(truth,rows,SIZE,SIZE)
        assert abs(metric['pq']-setting['pq']) < 1e-12
        matches = matched_ids(truth,rows)
        assert not refmatches-matches
        saved.append(dict(name=name,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            **{k:metric[k] for k in fields},recovered_gt_ids=sorted(matches-refmatches),
            lost_gt_ids=[],preserved_gt=len(refmatches),score_recomputed_after_save=True))
    report = dict(scope=f'{len(settings)} globally fixed rich U-Net admissions; same-development tuning, not CV/LB',
        prediction_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        baseline_sha256=hashlib.sha256(BASELINE.read_bytes()).hexdigest(),weak_sources=weak_meta,
        physical_images=91,annotation_groups=101,gt_instances=708,
        metadata_counts=dict(metadata_counts),experiments=reports,best=best,best_actual=actual,
        best_changed=best_changed,highest_recall=recall,saved_candidates=saved,
        target_exceeded=best['pq']>.5,elapsed_seconds=time.monotonic()-started)
    filename = 'rich_cross_model_confirmation.json' if args.refine_confirmation else 'rich_cross_model.json'
    (ROOT/'docs/evidence'/filename).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(best=best,best_changed=best_changed,highest_recall=recall,
                         saved_candidates=saved,elapsed_seconds=report['elapsed_seconds']),indent=2))


if __name__ == '__main__':
    main()
