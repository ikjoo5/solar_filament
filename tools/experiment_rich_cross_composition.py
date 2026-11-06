"""Twelve adaptive global compositions of previously evaluated rich rules.

Earlier global development winners motivate these controls. There is no
per-image/instance selection from GT. Addition masks remain trimmed against
all original630 masks even when a support filter removes some originals.
"""
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mu

from experiment_mask_postprocess import group_rows, overlap_stats, read_rows
from experiment_native_cross_model import matched_ids
from experiment_rich_support_filter import BASELINE, SOURCE, byte_rle
from experiment_view_combinations import ROOT, GT, VAL, SIZE
from filament_eval import Filament, evaluate

OUT = Path('/workspace/artifacts/rich_cross_composition_v1')
ADMISSIONS = dict(
    none=None,
    tta=Path('/workspace/artifacts/rich_cross_model_candidates_v1/tta_t0.35_a300_fg0.9_view3.csv'),
    base=Path('/workspace/artifacts/rich_cross_model_candidates_v1/base_t0.35_a300_fg0.9_view3.csv'))


def plan():
    controls = [
        dict(prune='none',tta_coverage=0.,stable_coverage=0.,maximum_area=0),
        dict(prune='conservative',tta_coverage=.1,stable_coverage=.05,maximum_area=600),
        dict(prune='preserve_control',tta_coverage=.25,stable_coverage=.05,maximum_area=600),
        dict(prune='precision_control',tta_coverage=.4,stable_coverage=.25,maximum_area=10**10)]
    return [dict(name=f'{a}_{c["prune"]}',admissions=a,**c) for a in ADMISSIONS for c in controls]


def main():
    started = time.monotonic()
    grid = plan()
    assert len(grid) == 12
    OUT.mkdir(exist_ok=False)
    (OUT/'predeclared_plan.json').write_text(json.dumps(grid,indent=2)+'\n')
    with VAL.open(newline='') as f:
        selection = list(csv.DictReader(f))
    groups = {r['image_id'] for r in selection}
    images = {Path(r['file_name']).stem for r in selection}
    base = read_rows(BASELINE)
    grouped = group_rows(base)
    assert len(base) == 630
    extras = {name:{} for name in ADMISSIONS}
    for name,path in ADMISSIONS.items():
        if path is None:
            continue
        rescue = group_rows(read_rows(path))
        for image in images:
            old = {r.counts for r in grouped.get(image,[])}
            all_rows = rescue.get(image,[])
            assert old <= {r.counts for r in all_rows}
            extras[name][image] = [r for r in all_rows if r.counts not in old]
    metadata = {}
    for line in SOURCE.open():
        record = json.loads(line)
        image = Path(record['file_name']).stem
        assert image in images and image not in metadata
        views = [mu.decode(byte_rle(v)) for v in record['view_unions'].values()]
        stable = mu.encode(np.asfortranarray((sum(views,np.zeros((SIZE,SIZE),np.uint8))>=2).astype(np.uint8)))
        tta = byte_rle(record['tta']['unions']['foreground']['0.2'])
        metadata[image] = {}
        for row in grouped.get(image,[]):
            rle = row.rle(SIZE,SIZE)
            metadata[image][row.counts] = (
                int(mu.area(mu.merge([rle,tta],intersect=True)))/row.area,
                int(mu.area(mu.merge([rle,stable],intersect=True)))/row.area)
    outputs = {}
    for s in grid:
        result = []
        for image in sorted(images):
            members = []
            for row in grouped.get(image,[]):
                coverage,stable = metadata[image][row.counts]
                remove = (row.area<=s['maximum_area'] and coverage<s['tta_coverage']
                          and stable<s['stable_coverage'])
                if not remove:
                    members.append(row)
            members.extend(extras[s['admissions']].get(image,[]))
            result.extend(Filament(f'{image}_{j}',image,image,r.counts,r.area)
                          for j,r in enumerate(members,1))
        outputs[s['name']] = result
    truth = read_rows(GT,gt=True,selected=groups)
    assert len(truth) == 708
    original_matches = matched_ids(truth,base)
    fields = ('pq','sq','rq','tp_pairs','fp','fn')
    reports = []
    for s in grid:
        rows = outputs[s['name']]
        score = evaluate(truth,rows,SIZE,SIZE)
        assert not any(overlap_stats(rows).values())
        matches = matched_ids(truth,rows)
        assert score['tp_pairs'] == len(matches)
        reports.append(dict(**s,**{k:score[k] for k in fields},physical_masks=len(rows),
            recovered_gt=len(matches-original_matches),lost_gt=len(original_matches-matches),
            recovered_gt_ids=sorted(matches-original_matches),lost_gt_ids=sorted(original_matches-matches)))
    best = max(reports,key=lambda r:r['pq'])
    preserving = max((r for r in reports if not r['lost_gt']),key=lambda r:r['pq'])
    recall = min(reports,key=lambda r:(r['fn'],-r['pq']))
    saved = []
    for s in {r['name']:r for r in (best,preserving,recall)}.values():
        path = OUT/(s['name']+'.csv')
        with path.open('w',newline='') as f:
            writer = csv.writer(f);writer.writerow(['filament_id','segmentation_rle'])
            writer.writerows((r.filament_id,r.counts) for r in outputs[s['name']])
        score = evaluate(truth,read_rows(path),SIZE,SIZE)
        assert abs(score['pq']-s['pq']) < 1e-12
        saved.append(dict(name=s['name'],path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            **{k:s[k] for k in fields},recovered_gt_ids=s['recovered_gt_ids'],lost_gt_ids=s['lost_gt_ids'],
            score_recomputed_after_save=True))
    report = dict(scope='12 adaptive global compositions of earlier development rules; not CV/LB',
        sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASELINE,SOURCE,*[v for v in ADMISSIONS.values() if v]]},
        physical_images=91,annotation_groups=101,gt_instances=708,experiments=reports,
        highest_pq=best,highest_pq_preserving_all_matches=preserving,highest_recall=recall,
        saved_candidates=saved,target_exceeded=best['pq']>.5,elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/rich_cross_composition.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(best=best,preserving=preserving,highest_recall=recall,saved_candidates=saved),indent=2))


if __name__ == '__main__':
    main()
