"""Replace102 isolated partial matches in the scored0.42 hybrid; keep216 fixed.

One new hypothesis, no threshold search: exact public masks replace the old
ones only when both have exactly one positive-overlap counterpart and their
mask IoU is in[0.1,0.5]. Final masks must be disjoint before publication.
"""
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from pycocotools import mask as mu
from track_reference216 import read_masks, agreement, load_targets, DEFAULT_TARGETS
from build_public_addition_submission import require_disjoint, known_placeholder

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts'
HYBRID=ART/'downloads/submission_own039_plus_public216_no_overlap.csv'
ORIGINAL=ROOT/'attachments/39d1023a-a635-4f96-90ec-d273fa62693e/submission_pq0472593.csv'
PUBLIC=ROOT/'attachments/d03f8d87-a646-4f04-89dd-4ad6ab534cec/submission.csv'
OUT=ART/'downloads/submission_hybrid042_replace102_no_overlap.csv'
REPORT=ART/'public055_replace102_v1'
SHA={HYBRID:'65d98408a49636396a8e076afffdbb342100291b868951bf46eb27a63ed38d38',
     ORIGINAL:'13dd9cc48bf2e444792254611a86e4e8a1c7412717de073f61310ba98cb54db0',
     PUBLIC:'0ab3932122a563ba96f4d243934d5c00680dd49b9d1165912a4f180907d2c603'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, content):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_bytes()!=content:raise ValueError('Existing artifact differs: '+str(path))
    path.write_bytes(content)


def encode_csv(rows, fields):
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
    return stream.getvalue().encode()


def main():
    for path,expected in SHA.items():assert sha(path)==expected
    original, public, hybrid=read_masks(ORIGINAL),read_masks(PUBLIC),read_masks(HYBRID)
    assert len(original)==1347 and len(public)==1344 and len(hybrid)==1563
    require_disjoint(hybrid);require_disjoint(public)
    old_by_image=defaultdict(list);public_by_image=defaultdict(list)
    for row in original:old_by_image[row.image_id].append(row)
    for row in public:
        if not known_placeholder(row):public_by_image[row.image_id].append(row)
    replacements={};mapping=[]
    for image,refs in sorted(public_by_image.items()):
        old=old_by_image[image]
        if not old:continue
        matrix=np.asarray(mu.iou([r.rle(2048,2048) for r in refs],
            [r.rle(2048,2048) for r in old],[0]*len(old)),float)
        overlaps=matrix>0
        for i,reference in enumerate(refs):
            j=int(matrix[i].argmax());value=float(matrix[i,j])
            if not .1<=value<=.5:continue
            if int(matrix[:,j].argmax())!=i:continue
            if int(overlaps[i].sum())!=1 or int(overlaps[:,j].sum())!=1:continue
            item=old[j]
            assert item.filament_id not in replacements
            replacements[item.filament_id]=reference
            mapping.append(dict(image=image,old_id=item.filament_id,public_id=reference.filament_id,
                prediction_iou=value,old_area=item.area,public_area=reference.area,
                new_area_ratio=reference.area/item.area,other_old_masks_overlapped=0))
    assert len(replacements)==102
    with (ART/'public055_remaining_v1/isolated_partial_replacement_candidates.csv').open(newline='') as stream:
        independent={(r['old_id'],r['public_id']) for r in csv.DictReader(stream)}
    assert {(m['old_id'],m['public_id']) for m in mapping}==independent
    fixed=load_targets(DEFAULT_TARGETS)
    public_by_id={r.filament_id:r for r in public}
    with (ART/'public055_addition_to039_v2/added_instances.csv').open(newline='') as stream:
        added=list(csv.DictReader(stream))
    added_ids={m['added_id'] for m in added}
    assert len(added_ids)==216 and not added_ids.intersection(replacements)
    data=[dict(filament_id=r.filament_id,segmentation_rle=replacements[r.filament_id].counts
        if r.filament_id in replacements else r.counts) for r in hybrid]
    content=encode_csv(data,['filament_id','segmentation_rle'])
    REPORT.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=REPORT,suffix='.pending.csv') as temporary:
        temporary.write(content);temporary.flush();final=read_masks(temporary.name)
    require_disjoint(final)
    assert len(final)==1563 and [r.filament_id for r in final]==[r.filament_id for r in hybrid]
    assert len({r.image_id for r in final})==180
    assert sum(a.counts!=b.counts for a,b in zip(hybrid,final))==102
    final_by_id={r.filament_id:r for r in final}
    assert all(final_by_id[m['added_id']].counts==public_by_id[m['reference_id']].counts for m in added)
    assert all(final_by_id[m['old_id']].counts==public_by_id[m['public_id']].counts for m in mapping)
    reference_matches=agreement([public_by_id[r['reference_id']] for r in fixed],final)
    assert len(reference_matches)==216 and all(r['max_iou']==1 for r in reference_matches)
    for path,expected in SHA.items():assert sha(path)==expected
    save(OUT,content)
    manifest=dict(experiment='hybrid042_replace102_isolated_partial',
        status='prepared_not_submitted',control_public_lb_user_reported=.42,new_public_lb=None,
        selection='Natural public masks with IoU in[0.1,0.5] against mutual-best oldmask, each mask overlaps exactlyone counterpart across the complete old/public sets',
        action='Replace only102 selected original mask RLEs with exactpublic RLEs, retainingtheirIDs; no addition/deletion, no clipping',
        control=str(HYBRID),output=str(OUT),output_sha256=sha(OUT),
        input_sha256={str(k):v for k,v in SHA.items()},code_sha256=sha(__file__),
        original_rows=1347,hybrid_rows=1563,output_rows=1563,changed_original_masks=102,
        changed_images=len({r['image'] for r in mapping}),unchanged_hybrid_masks=1461,
        fixed216_preserved_pixel_exact=True,fixed216_reference_match_count=216,
        fixed216_reference_mean_max_iou=1,origin='hybrid',reference_copy_is_not_model_recovery=True,
        unique_ids=True,all_control_ids_and_order_preserved=True,overlap_pixels=0,
        neural_inference=False,training=False,ground_truth_used=False,
        interpretation='Compare with the scored0.42 hybrid to measure net effect of this fixed102-shape replacement. No predictedPQgain or per-mask correctness inferred.',
        limitations='Isolated correspondence is geometric only. Extent and semantic object identity may differ; replacingwithpublic masks can worsen individual cases. This is a CSV hybrid experiment.')
    save(REPORT/'manifest.json',(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n').encode())
    save(REPORT/'replaced102.csv',encode_csv(mapping,list(mapping[0])))
    save(REPORT/'build_public_partial_replacement_submission.py',Path(__file__).read_bytes())
    note='''# 0.42 혼합 제출: 독립적인 부분 대응102개 교체

- 비교 기준은 실제publicLB0.42인 `submission_own039_plus_public216_no_overlap.csv`다.
- 기존0.39모델에서 온 마스크102개를 대응하는 공개 마스크로 교체했다. 새ID나 객체를 추가하지 않았고 전체1,563개/180장이다.
- 공개/기존 양쪽 모두 다른 객체와 픽셀 겹침이 없는1:1부분대응만 선택했다. 대응IoU는0.1~0.5다.102개 전부 정답이라는 의미가 아니다.
- 효과를 확인한216개는ID·픽셀 그대로 보존했다. 이216개참조대응률100%는복사한예측의보존검사이며자체모델의새검출능력이아니다.
- 기존나머지1,461개는변경하지않았다. 전체겹침0픽셀·ID중복없음을검사했다.
- 제출점수는아직모른다.0.42와비교해선택/형상교체묶음의순효과를본다. 자체모델의개선과별도로기록한다.
- `replaced102.csv`는교체된기존ID↔공개ID와변경면적기록이다.
'''
    save(REPORT/'README.ko.md',note.encode())
    package=ART/'downloads/SolarFilament_Hybrid042_Replace102_Analysis.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(OUT,OUT.name)
        for path in REPORT.iterdir():
            if path.is_file():z.write(path,path.name)
    print(json.dumps({k:manifest[k] for k in ('output','output_rows','changed_original_masks','changed_images',
        'fixed216_preserved_pixel_exact','overlap_pixels','output_sha256')},indent=2))


if __name__=='__main__':main()
