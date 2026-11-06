import sys
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
from pycocotools import mask as mu
import candidate_gate_trace as trace
import submission_yolo_recipe as yolo


def box(x,y,w,h):
    a=np.zeros((2048,2048),np.uint8);a[y:y+h,x:x+w]=1
    return mu.encode(np.asfortranarray(a))


def proposal(mask,confidence):
    return dict(rle=mask['counts'].decode(),confidence=confidence)


def test_preprocessing_gate_order_and_exact_output():
    masks=[box(100+i*100,100,20,20) for i in range(6)]
    small=box(800,100,10,20)
    partial=box(1000,100,10,31)
    small_overlap=box(1000,100,10,2)
    geo={'base':dict(instances=[proposal(m,1.) for m in masks+[small,partial]])}
    candidates=[proposal(masks[0],.19),proposal(masks[1],.25),proposal(small,.8),
                proposal(box(1200,100,20,20),.8),proposal(masks[2],.8),
                proposal(masks[3],.8),proposal(partial,.8)]
    pre={'limb':dict(instances=[proposal(masks[4],.9)]),'clahe16':dict(instances=candidates)}
    originals=[masks[2],small_overlap]
    result,rows=trace.trace_preprocessing(originals,geo,pre)
    reasons={(r['view'],r['index']):r['route_reason'] for r in rows}
    assert reasons[('limb',0)]=='limb_view_excluded_from_rescue'
    assert reasons[('clahe16',0)]=='below_rescue_pool_confidence_0_2'
    assert reasons[('clahe16',1)]=='confidence_below_0_3'
    assert reasons[('clahe16',2)]=='area_below_300'
    assert reasons[('clahe16',3)]=='geometric_agreement_below_0_3'
    assert reasons[('clahe16',4)]=='occupied_fraction_at_least_0_1'
    assert reasons[('clahe16',5)]=='admitted_to_preprocessing_output'
    assert reasons[('clahe16',6)]=='remaining_area_below_300'
    trace.parity(result,yolo._preprocessing_rescue(originals,geo,pre),'test')
    assert len(result)==3


def test_exact_matching_confidence_uses_all_matching_raw_candidates():
    m=box(50,50,20,10); other=box(100,100,20,20)
    raw=[dict(source='geometric',view='base',index=0,confidence=.1,mask=trace.portable(m),route_reason='x'),
         dict(source='preprocessing',view='clahe16',index=0,confidence=.8,mask=trace.portable(m),route_reason='y'),
         dict(source='tiled',view='tile',index=0,confidence=.95,mask=trace.portable(other),route_reason='z')]
    value=dict(raw_yolo=raw,stages=dict(yolo_consensus=[],baseline_final=[]),
        geometry_components_before_area=[trace.portable(m)],final_pruning=[])
    rows,candidates,_=trace.join_reference('sample',value,[('sample_1',m)])
    assert rows[0]['max_confidence_among_matching_yolo']==.8
    assert rows[0]['matching_yolo_conf_ge_0_2']==1
    assert rows[0]['matching_yolo_conf_ge_0_1']==2
    assert rows[0]['consensus_similarity_lost_only_by_area300']
    assert len(candidates)==2


def test_full_run_orchestration_freezes_all_own_traces_before_reference():
    """Run all180 orchestration iterations with lightweight dependency fixtures.

    Tests the real run() control flow, namespace bindings, serialization, joins,
    input snapshot verification, and output writes. Pixel math is tested above
    and by the real cached-image replay.
    """
    with tempfile.TemporaryDirectory() as temporary:
        base=Path(temporary);root=base/'inference';root.mkdir();out=base/'trace'
        names=[f'frame{i:03d}' for i in range(180)]
        names_file=base/'names.json';names_file.write_text(json.dumps([n+'.jpeg' for n in names]))
        public=base/'public.csv';public.write_text('fixture public content')
        p216=base/'panel216.json';p216.write_text('{}')
        p102=base/'panel102.json';p102.write_text('{}')
        all_rows=[dict(image=names[i%180],reference_id=f'{names[i%180]}_{i//180+1}') for i in range(318)]
        reference={name:[] for name in names}
        for row in all_rows:reference[row['image']].append((row['reference_id'],None))
        calls=dict(validated=0,traced=0,reference_opened=False)
        actual_sha=trace.file_sha
        def digest(path):
            if Path(path)==public:
                assert calls['traced']==180
                return trace.REFERENCE_SHA
            return actual_sha(path)
        def checked_candidates(stem,stages):
            calls['validated']+=1
            return [],{}
        def own(stem,stages,baseline,sources):
            assert not calls['reference_opened']
            calls['traced']+=1
            return dict(image=stem,fixture=True)
        def read_reference(path):
            assert calls['validated']==calls['traced']==180
            assert len(list((out/'own_trace').glob('*.json.gz')))==180
            calls['reference_opened']=True
            return reference
        def join(stem,own_trace,refs):
            assert calls['reference_opened']
            return ([dict(image=stem,reference_id=r[0]) for r in refs],[],[])
        protocol=dict(images=[dict(stem=n,name=n+'.jpeg',sha256='0'*64) for n in names],sources={})
        replacements=dict(
            load_context=lambda root,snapshot:(protocol,'fixture_identity','fixture_runtime',names),
            load_image=lambda *args:({},[],[]),candidates=checked_candidates,
            trace_own_image=own,read_csv=read_reference,row_masks=lambda value:value,
            load_panel=lambda path,count:all_rows[:216] if count==216 else all_rows[216:],
            join_reference=join,file_sha=digest)
        with ExitStack() as stack:
            for name,value in replacements.items():stack.enter_context(patch.object(trace,name,value))
            summary=trace.run(root,public,names_file,p216,p102,out)
        assert summary['status']=='complete'
        assert summary['images']==180 and len(summary['own_trace_sha256'])==180
        assert summary['own_trace_frozen_before_reference_content_loaded']
        assert (out/'reference216_gate_summary.csv').is_file()
        assert (out/'reference102_gate_summary.csv').is_file()
