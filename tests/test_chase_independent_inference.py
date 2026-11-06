from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pycocotools import mask as mu

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
import chase_independent_inference as worker
import chase_best84_evaluation as evaluation
import chase_independent_decode as decoder
import submission_inference as inference
from submission_driver import write_envelope


def protocol(role='old_native',mode='raw',**changes):
    fields=dict(role=role,mode=mode,checkpoint_sha='a'*64,model_sha='b'*64,
        source_images={'one.jpeg':'c'*64},sources={'frozen':'d'*64},data_sha={'GT':'e'*64},
        runtime={'torch':'test','gpu':'test'},helper_sha='f'*64,decoder_sha='1'*64,batch=1)
    fields.update(changes)
    return worker.make_protocol(**fields)


def test_all_roles_modes_and_dependencies_separate_identity():
    keys={evaluation.fingerprint(protocol(role,mode))for role in worker.ROLES for mode in worker.MODES}
    assert len(keys)==8
    original=evaluation.fingerprint(protocol())
    for field,value in [('checkpoint_sha','2'*64),('model_sha','3'*64),
        ('helper_sha','4'*64),('decoder_sha','5'*64),('runtime',{'gpu':'other'}),
        ('source_images',{'one.jpeg':'6'*64}),('sources',{'frozen':'7'*64})]:
        assert original!=evaluation.fingerprint(protocol(**{field:value}))
    assert not protocol()['fusion'] and not protocol()['tta'] and not protocol()['training']


def test_native_single_base_raw_and_limb_do_not_call_tta(monkeypatch):
    raw=np.zeros((2048,2048),np.uint8);corrected=np.ones_like(raw)
    model=torch.nn.Linear(1,1);before={k:v.clone()for k,v in model.state_dict().items()}
    probabilities=np.zeros((2,2048,2048),np.float32)
    seen=[]
    monkeypatch.setattr(inference,'limb',lambda x:corrected)
    def heads(selected,pixels,device,**kwargs):
        assert selected is model and not model.training and not torch.is_grad_enabled()
        assert kwargs==dict(tile=1024,overlap=.25,batch=1,amp_dtype=None)
        seen.append(pixels)
        return probabilities
    monkeypatch.setattr(inference,'predict_heads',heads)
    monkeypatch.setattr(inference,'predict_views',lambda *a,**k:pytest.fail('No TTA permitted'))
    monkeypatch.setattr(inference,'prediction_record',lambda *a:{'heads':'base only'})
    for mode in worker.MODES:
        record=worker.infer_record(model,raw,'one.jpeg','sha','id',role='new_native',mode=mode,
            device=torch.device('cpu'))
        assert record['heads']=='base only'
        assert set(record['foreground_low_threshold_unions'])=={'0.05','0.1'}
        assert record['foreground_low_threshold_unions']['0.05']['size']==[2048,2048]
    assert seen[0] is raw and seen[1] is corrected
    assert all(torch.equal(v,before[k])for k,v in model.state_dict().items())


def test_yolo_single_base_never_uses_native_or_geometric(monkeypatch):
    raw=np.zeros((2048,2048),np.uint8);corrected=np.ones_like(raw);seen=[]
    monkeypatch.setattr(inference,'limb',lambda x:corrected)
    monkeypatch.setattr(inference,'predict_heads',lambda *a,**k:pytest.fail('No native permitted'))
    class Provider:
        def _result(self,pixels,size):
            assert size==2048 and not torch.is_grad_enabled();seen.append(pixels)
            return SimpleNamespace(boxes=[1,2])
        def _whole_record(self,result,stem,view,size):
            assert (stem,view,size)==('one','base',2048)
            return {'instances':[],'view':view}
        def geometric(self,*args):pytest.fail('No geometric multi-view permitted')
    for mode in worker.MODES:
        result=worker.infer_record(Provider(),raw,'one.jpeg','sha','id',role='old_yolo',mode=mode,
            device=torch.device('cpu'))
        assert result['raw_box_count']==2 and not result['max_det_reached']
    assert seen[0] is raw and seen[1] is corrected


def small_mask():
    pixels=np.zeros((2048,2048),np.uint8);pixels[10:30,10:30]=1
    value=mu.encode(np.asfortranarray(pixels));value['counts']=value['counts'].decode()
    return value


def test_partial_resume_reuses_only_completed_mode(tmp_path):
    record=small_mask();names=['one.jpeg'];sha='a'*64
    conditions={mode:(evaluation.fingerprint(protocol(mode=mode)),tmp_path/mode)for mode in worker.MODES}
    for _,folder in conditions.values():(folder/'frames').mkdir(parents=True)
    calls=[]
    def predict(image,name,image_sha,identity,mode):
        calls.append(mode)
        if mode=='limb':raise RuntimeError('simulated interruption')
        return {'unions':{'foreground':{'0.5':record}}}
    def simple_decode(raw,role):return decoder.validate_masks([raw['unions']['foreground']['0.5']])
    kwargs=dict(role='new_native',names=names,image_hashes={'one.jpeg':sha},images_root=tmp_path,
        conditions=conditions,read_image=lambda *a:None,decode=simple_decode)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        worker.run_frames(**kwargs,predict=predict)
    calls.clear()
    def resumed(image,name,image_sha,identity,mode):
        calls.append(mode);return {'unions':{'foreground':{'0.5':record}}}
    output,reused=worker.run_frames(**kwargs,predict=resumed)
    assert calls==['limb'] and reused=={'raw':1,'limb':0}
    calls.clear()
    output,reused=worker.run_frames(**kwargs,predict=resumed)
    assert calls==[] and reused=={'raw':1,'limb':1}
    assert set(output)=={'raw','limb'}


def test_checkpoint_selection_never_chooses_best_epoch0(tmp_path):
    (tmp_path/'native').mkdir();(tmp_path/'yolo').mkdir()
    result=dict(status='complete',epochs_completed=60,identity='run',checkpoint_sha256='sha')
    for family in ('native','yolo'):(tmp_path/family/'result.json').write_text(json.dumps(result))
    cfg=dict(initial_checkpoint='/original/best84.pt',yolo_checkpoint='/original/yolo.pt')
    rows={'old_yolo_old_native':{'model_sha256':'old'},
          'old_yolo_new_native':{'model_sha256':'new','native_checkpoint_sha256':'new-file'}}
    path,sha,model,identity=worker.select_checkpoint(tmp_path,'new_native',cfg,rows)
    assert path==tmp_path/'native/last.pt' and (sha,model,identity)==('new-file','new','run')
    assert worker.select_checkpoint(tmp_path,'new_yolo',cfg,rows)[0]==tmp_path/'yolo/epoch060.pt'
    result['epochs_completed']=59
    (tmp_path/'native/result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='completed epoch60'):
        worker.select_checkpoint(tmp_path,'new_native',cfg,rows)


def test_mock_full_worker_saves_two_metrics_and_resumes_without_inference(tmp_path,monkeypatch):
    import chase_best84_training as training
    import solar_a100_train_standalone as legacy
    root=tmp_path/worker.ROOT_NAME;root.mkdir()
    (root/'configs').mkdir();(root/'comparison/validation/epoch060_old_yolo_old_native').mkdir(parents=True)
    images=root/'source_images';images.mkdir()
    names=[f'{index:03}.jpeg'for index in range(91)]
    for name in names:(images/name).write_bytes(b'fake-image')
    image_sha={name:hashlib.sha256(b'fake-image').hexdigest()for name in names}
    data_sha={'annotations':'fake','train_split':'fake','val_split':'fake'}
    original_protocol=dict(source_images=image_sha,data_sha256=data_sha)
    original_id=evaluation.fingerprint(original_protocol)
    (root/'comparison/validation/epoch060_old_yolo_old_native/identity.json').write_text(
        json.dumps(dict(identity=original_id,protocol=original_protocol)))
    class Tiny(torch.nn.Linear):
        def __init__(self):super().__init__(1,1)
        def to(self,*args,**kwargs):return self
    model=Tiny();model_sha=evaluation.model_digest(model)
    checkpoint=root/'original.pt'
    torch.save(dict(epoch=84,config={'encoder':'tu-convnext_tiny'},model_state_dict=model.state_dict()),checkpoint)
    before=(checkpoint.read_bytes(),checkpoint.stat().st_mtime_ns)
    (root/'configs/comparison.json').write_text(json.dumps({'images':str(images)}))
    (root/'comparison/epoch060_comparison.json').write_text(json.dumps({'results':[
        {'role':'old_yolo_old_native','identity':original_id,'model_sha256':model_sha}]}))
    monkeypatch.setattr(worker,'verify_sources',lambda r:(TOOLS,{'sources':{'frozen':'sha'}}))
    monkeypatch.setattr(worker,'select_checkpoint',lambda *a:(checkpoint,worker.digest(checkpoint),model_sha,None))
    monkeypatch.setattr(evaluation,'load_cohort',lambda cfg:dict(names=names,source_sha256=data_sha,groups={}))
    monkeypatch.setattr(evaluation,'runtime_identity',lambda device:{'device':device,'gpu':'mock'})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(training,'configure_cuda',lambda:None)
    monkeypatch.setattr(legacy,'build_model',lambda **kw:Tiny())
    mask=small_mask();calls=[]
    def mock_infer(predictor,image,filename,sha,identity,**kwargs):
        calls.append((filename,kwargs['mode']))
        return {'unions':{'foreground':{'0.5':mask}}}
    monkeypatch.setattr(worker,'infer_record',mock_infer)
    monkeypatch.setattr(decoder,'native_masks',lambda value,min_area=100:[value])
    monkeypatch.setattr(decoder,'score_groups',lambda *a:dict(pq=.5,sq=.8,rq=.625,tp_pairs=5,
        fp=1,fn=2,physical_images=91,annotation_groups=101,gt_instances=708,prediction_instances=91))
    result=worker.run(root,'old_native')
    assert len(calls)==182 and result['status']=='complete'
    assert {r['input_mode']for r in result['results']}=={'raw','limb'}
    for row in result['results']:
        assert row['pq']==.5 and row['physical_images']==91
        assert Path(row['metrics_path']).is_file()and Path(row['csv_path']).is_file()
        assert len(list((Path(row['cache_root'])/'frames').glob('*.json.gz')))==91
    calls.clear();again=worker.run(root,'old_native')
    assert calls==[] and all(row['reused_images']==91 for row in again['results'])
    assert before==(checkpoint.read_bytes(),checkpoint.stat().st_mtime_ns)
