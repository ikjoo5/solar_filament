"""Nine fixed admission/pruning compositions on a refined disjoint reference.

Support, proposal novelty and clipping are recomputed from the actual refined
masks. No masks are copied as previously clipped rescue tails. GT is loaded
only after all nine complete prediction sets exist.
"""
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
from experiment_rich_cross_model import SOURCE, BASELINE as ORIGINAL, json_rle
from experiment_view_combinations import GT, ROOT, SIZE, VAL
from filament_eval import Filament, evaluate


def plan():
    controls = [dict(prune='none',tta_coverage=0.,stable_coverage=0.,maximum_area=0),
        dict(prune='preserve',tta_coverage=.25,stable_coverage=.05,maximum_area=600),
        dict(prune='precision',tta_coverage=.4,stable_coverage=.25,maximum_area=10**10)]
    return [dict(name=f'{domain}_{p["prune"]}',domain=domain,threshold=.35,
        minimum_area=300,minimum_foreground_mean=.9,minimum_views=3,per_view_coverage=.5,
        maximum_original_iou=.1,maximum_original_union_fraction=.25,**p)
        for domain in ('none','tta','base') for p in controls]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline',type=Path,required=True)
    args = parser.parse_args()
    output = Path('/workspace/artifacts/rich_boundary_composition_v1')
    started = time.monotonic()
    grid = plan()
    assert len(grid) == 9
    output.mkdir(exist_ok=False)
    (output/'predeclared_plan.json').write_text(json.dumps(grid,indent=2)+'\n')
    with VAL.open(newline='') as f:
        split = list(csv.DictReader(f))
    groups = {r['image_id'] for r in split}
    images = {Path(r['file_name']).stem for r in split}
    base = read_rows(args.baseline)
    assert len(base) == 630 and not any(overlap_stats(base).values())
    originals = group_rows(base)
    outputs = {s['name']:[] for s in grid}
    counts = {s['name']:dict(removed=0,added=0) for s in grid}
    seen = set()
    for line in SOURCE.open():
        record = json.loads(line)
        image = Path(record['file_name']).stem
        assert image in images and image not in seen
        seen.add(image)
        old = originals.get(image,[])
        br = [r.rle(SIZE,SIZE) for r in old]
        covered = mu.merge(br) if br else None
        views = [json_rle(value)[0] for value in record['view_unions'].values()]
        view_masks = [mu.decode(r) for r in views]
        stable = mu.encode(np.asfortranarray((sum(view_masks,np.zeros((SIZE,SIZE),np.uint8))>=2).astype(np.uint8)))
        tta_support = json_rle(record['tta']['unions']['foreground']['0.2'])[0]
        support = {}
        for row in old:
            rle = row.rle(SIZE,SIZE)
            support[row.counts] = (int(mu.area(mu.merge([rle,tta_support],intersect=True)))/row.area,
                int(mu.area(mu.merge([rle,stable],intersect=True)))/row.area)
        proposals = dict(none=[])
        for domain in ('base','tta'):
            data = record if domain == 'base' else record['tta']
            source = data['components']['0.35']
            nr = [json_rle(c['rle'])[0] for c in source]
            matrix = (np.asarray(mu.iou(nr,br,[0]*len(br))) if nr and br
                      else np.zeros((len(nr),len(br))))
            admitted = []
            for i,c in enumerate(source):
                area = c['area']
                assert int(mu.area(nr[i])) == area
                if area<300 or c['stats']['foreground']['mean']<.9:
                    continue
                if br and matrix[i].max()>.1:
                    continue
                fraction = int(mu.area(mu.merge([nr[i],covered],intersect=True)))/area if covered else 0.
                if fraction>.25:
                    continue
                agreement = sum(int(mu.area(mu.merge([nr[i],v],intersect=True)))/area>=.5 for v in views)
                if agreement<3:
                    continue
                candidate = Filament(f'{image}_{i+1}',image,image,c['rle']['counts'],area)
                clipped = trim(candidate,covered)
                if clipped is not None and clipped.area>=300:
                    admitted.append(clipped)
            proposals[domain] = admitted
        for setting in grid:
            rows = []
            for row in old:
                coverage,agreement = support[row.counts]
                if (row.area<=setting['maximum_area'] and coverage<setting['tta_coverage']
                        and agreement<setting['stable_coverage']):
                    counts[setting['name']]['removed'] += 1
                else:
                    rows.append(row)
            rows.extend(proposals[setting['domain']])
            counts[setting['name']]['added'] += len(proposals[setting['domain']])
            outputs[setting['name']].extend(Filament(f'{image}_{j}',image,image,r.counts,r.area)
                                           for j,r in enumerate(rows,1))
    assert seen == images
    # Every decision above depends only on fixed model outputs.
    truth = read_rows(GT,gt=True,selected=groups)
    assert len(truth) == 708
    original_matches = matched_ids(truth,read_rows(ORIGINAL))
    refined_matches = matched_ids(truth,base)
    assert len(original_matches) == 483 and not original_matches-refined_matches
    fields = ('pq','sq','rq','tp_pairs','fp','fn')
    reports = []
    for setting in grid:
        rows = outputs[setting['name']]
        metric = evaluate(truth,rows,SIZE,SIZE)
        assert not any(overlap_stats(rows).values())
        matches = matched_ids(truth,rows)
        assert metric['tp_pairs'] == len(matches)
        reports.append(dict(**setting,**{k:metric[k] for k in fields},
            physical_masks=len(rows),**counts[setting['name']],
            recovered_original_gt=len(matches-original_matches),lost_original_gt=len(original_matches-matches),
            recovered_original_gt_ids=sorted(matches-original_matches),lost_original_gt_ids=sorted(original_matches-matches),
            recovered_refined_gt=len(matches-refined_matches),lost_refined_gt=len(refined_matches-matches),
            lost_refined_gt_ids=sorted(refined_matches-matches)))
    highest = max(reports,key=lambda r:r['pq'])
    preserving = max((r for r in reports if not r['lost_original_gt']),key=lambda r:r['pq'])
    preserving_refined = max((r for r in reports if not r['lost_refined_gt']),key=lambda r:r['pq'])
    recall = min(reports,key=lambda r:(r['fn'],-r['pq']))
    saved = []
    for result in {r['name']:r for r in (highest,preserving,preserving_refined,recall)}.values():
        path = output/(result['name']+'.csv')
        with path.open('w',newline='') as f:
            writer = csv.writer(f);writer.writerow(['filament_id','segmentation_rle'])
            writer.writerows((r.filament_id,r.counts) for r in outputs[result['name']])
        metric = evaluate(truth,read_rows(path),SIZE,SIZE)
        assert abs(metric['pq']-result['pq'])<1e-12
        saved.append(dict(name=result['name'],path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            **{k:result[k] for k in fields},recovered_original_gt_ids=result['recovered_original_gt_ids'],
            lost_original_gt_ids=result['lost_original_gt_ids'],lost_refined_gt_ids=result['lost_refined_gt_ids'],
            score_recomputed_after_save=True))
    report = dict(scope='Nine adaptive global boundary/admission/pruning compositions; same development set, not CV/LB',
        refined_baseline=str(args.baseline),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.baseline,SOURCE,ORIGINAL)},physical_images=91,annotation_groups=101,gt_instances=708,
        original_matches=len(original_matches),refined_matches=len(refined_matches),experiments=reports,
        highest_pq=highest,highest_pq_preserving_original=preserving,
        highest_pq_preserving_refined=preserving_refined,highest_recall=recall,
        saved_candidates=saved,target_exceeded=highest['pq']>.5,elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/rich_boundary_composition.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(best=highest,preserving_original=preserving,preserving_refined=preserving_refined,
                         highest_recall=recall,saved_candidates=saved),indent=2))


if __name__ == '__main__':
    main()
