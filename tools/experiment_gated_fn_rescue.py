"""Thirty-two fixed GT-free low-recall rescue comparisons on cached RLEs.

Keep the established consensus prediction and admit only novel preprocessing
proposals corroborated by an old geometric view. GT is used for scoring only.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mu

from experiment_view_combinations import ROOT, GT, VAL, SIZE, encode_roi
from experiment_mask_postprocess import read_rows, group_rows
from filament_eval import Filament, _check_compressed_counts, evaluate

OLD = Path('/workspace/artifacts/validation_results_20261027T120958Z_444da7f4/raw_views.jsonl')
NEW = Path('/workspace/artifacts/validation_results_20261027T125130Z_2312e759/raw_preprocess_views.jsonl')
BASELINE = Path('/workspace/artifacts/morph_order_candidates_v1/before0_close5_after300.csv')
EXPECTED = .4571518474011121
NEW_VIEWS = {'limb','clahe16','clahe32','percentile','unsharp','gaussian','bilateral','blackhat'}
OLD_VIEWS = {'base','scale','hflip','vflip'}


def load(path, images, expected_views):
    result=defaultdict(dict);records=masks=0
    for line in path.open():
        row=json.loads(line);image,view=row['image'],row['view']
        assert image in images and view in expected_views
        assert row['shape']==[SIZE,SIZE] and row['coordinate_space']=='original'
        assert view not in result[image]
        for proposal in row['instances']:
            c=proposal['confidence']
            assert np.isfinite(c) and .05<=c<=1
            area=_check_compressed_counts(proposal['rle'],SIZE*SIZE)
            assert area>0
            proposal['area']=area
            proposal['native_rle']={'size':[SIZE,SIZE],'counts':proposal['rle'].encode('ascii')}
            masks+=1
        result[image][view]=row['instances'];records+=1
    assert set(result)==images and all(set(rows)==expected_views for rows in result.values())
    return result,dict(records=records,masks=masks,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fixture():
    canvas=np.zeros((40,50),bool);canvas[8:18,10:25]=True
    existing=canvas.copy();existing[:,18:]=False
    roi=canvas[8:18,10:25]&~existing[8:18,10:25]
    rle=encode_roi(roi,8,10,40,50)
    expected=canvas&~existing
    np.testing.assert_array_equal(mu.decode(rle),expected)
    assert int(mu.area(rle))==int(expected.sum())


def main():
    started=time.monotonic();fixture()
    with VAL.open(newline='') as f:selection=list(csv.DictReader(f))
    ids={r['image_id'] for r in selection};images={Path(r['file_name']).stem for r in selection}
    truth=read_rows(GT,gt=True,selected=ids);baseline=read_rows(BASELINE)
    reference=evaluate(truth,baseline,SIZE,SIZE)
    assert abs(reference['pq']-EXPECTED)<1e-12
    assert len(truth)==708 and len(images)==91
    base_by_image=group_rows(baseline)
    old,old_meta=load(OLD,images,OLD_VIEWS);new,new_meta=load(NEW,images,NEW_VIEWS)
    settings=[dict(name=f'c{c}_old{i}_novel{n}_area{a}_prep{s}',confidence=c,old_iou=i,novelty=n,minimum_area=a,support=s)
              for c in [.2,.3] for i in [.3,.5] for n in [.1,.25] for a in [200,300] for s in [1,2]]
    assert len(settings)==32
    output={s['name']:[] for s in settings};additions={s['name']:0 for s in settings}
    for number,image in enumerate(sorted(images),1):
        base=base_by_image.get(image,[]);base_rles=[r.rle(SIZE,SIZE) for r in base]
        if base_rles:
            union=mu.merge(base_rles)
            assert int(mu.area(union))==sum(r.area for r in base)
            base_covered=mu.decode(union).astype(bool)
        else:
            base_covered=np.zeros((SIZE,SIZE),bool)
        candidates=[]
        for view,rows in new[image].items():
            if view=='limb':continue
            candidates.extend(dict(**p,view=view) for p in rows if p['confidence']>=.2)
        candidates.sort(key=lambda p:-p['confidence'])
        cr=[p['native_rle'] for p in candidates]
        old_rles=[p['native_rle'] for rows in old[image].values() for p in rows if p['confidence']>=.1]
        matrix=np.asarray(mu.iou(cr,cr,[0]*len(cr))) if cr else np.zeros((0,0))
        old_iou=np.asarray(mu.iou(cr,old_rles,[0]*len(old_rles))).max(axis=1) if cr and old_rles else np.zeros(len(cr))
        supports={}
        for confidence in [.2,.3]:
            supports[confidence]=[len({p['view'] for j,p in enumerate(candidates)
                if p['confidence']>=confidence and matrix[i,j]>=.3}) for i in range(len(candidates))]
        decoded={}
        for setting in settings:
            covered=base_covered.copy();rows=output[setting['name']]
            for index,row in enumerate(base,1):
                rows.append(Filament(f'{image}_{index}',image,image,row.counts,row.area))
            index=len(base)
            for i,proposal in enumerate(candidates):
                if (proposal['confidence']<setting['confidence'] or proposal['area']<setting['minimum_area']
                    or old_iou[i]<setting['old_iou'] or supports[setting['confidence']][i]<setting['support']):
                    continue
                if i not in decoded:
                    x,y,w,h=map(int,mu.toBbox(proposal['native_rle']))
                    decoded[i]=(x,y,w,h,mu.decode(proposal['native_rle'])[y:y+h,x:x+w].astype(bool))
                x,y,w,h,mask=decoded[i]
                occupied=covered[y:y+h,x:x+w]
                fraction=int((mask&occupied).sum())/proposal['area']
                if fraction>=setting['novelty']:continue
                remaining=mask&~occupied;area=int(remaining.sum())
                if area<setting['minimum_area']:continue
                occupied |= remaining
                rle=encode_roi(remaining,y,x)
                assert int(mu.area(rle))==area
                index+=1;additions[setting['name']]+=1
                rows.append(Filament(f'{image}_{index}',image,image,rle['counts'].decode('ascii'),area))
        if number%20==0:print(number,'/',len(images),'images',flush=True)
    results=[]
    for setting in settings:
        predicted=output[setting['name']];score=evaluate(truth,predicted,SIZE,SIZE)
        assert score['tp_pairs']-(708-score['fn'])==0
        results.append(dict(**setting,**{k:score[k] for k in ['pq','sq','rq','tp_pairs','fp','fn']},
                            added_instances=additions[setting['name']],overlap_images=0,
                            extra_matching_pairs=0,disjoint_by_pixel_assignment=True))
    best=max(results,key=lambda r:r['pq']);saved=None
    if best['pq']>reference['pq']:
        out=Path('/workspace/artifacts/gated_rescue_candidates_v1');out.mkdir(exist_ok=False)
        path=out/(best['name']+'.csv')
        with path.open('x',newline='') as f:
            writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle'])
            writer.writerows((r.filament_id,r.counts) for r in output[best['name']])
        saved=str(path)
    report=dict(scope='32 fixed same-development GT-free candidate rescue comparisons; not independent CV/LB',
        physical_images=91,annotation_groups=101,gt_instances=708,source_cache={'old':old_meta,'new':new_meta},
        rules='Preserve baseline; actual-confidence sorted novel new-preprocessing candidates; old-view corroboration; distinct-new-view support; assign overlap to existing masks; no GT selection',
        baseline={k:reference[k] for k in ['pq','sq','rq','tp_pairs','fp','fn']},
        baseline_reproduced=True,experiments=results,best=best,saved_candidate=saved,
        target_exceeded=best['pq']>.5,elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/gated_fn_rescue_experiments.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(best,indent=2));print('seconds',report['elapsed_seconds'])


if __name__=='__main__':main()
