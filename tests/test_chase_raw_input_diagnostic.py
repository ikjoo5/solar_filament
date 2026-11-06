from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import zipfile

import numpy as np
import torch

TOOLS = Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
import chase_raw_input_diagnostic as raw
import chase_best84_evaluation as evaluation
import submission_inference as inference
from submission_driver import read_envelope, write_envelope


def protocol(**overrides):
    fields = dict(checkpoint_sha='a'*64,model_sha='b'*64,image_hashes={'one.jpeg':'c'*64},
        frozen_sources={'submission_inference.py':'d'*64},runtime={'torch':'test'},
        batch=1,helper_sha='e'*64)
    fields.update(overrides)
    return raw.input_protocol(**fields)


def test_input_mode_and_code_are_part_of_cache_identity(tmp_path):
    original = evaluation.fingerprint(protocol())
    assert original != evaluation.fingerprint(protocol(mode='limb_v1'))
    assert original != evaluation.fingerprint(protocol(helper_sha='f'*64))
    assert original != evaluation.fingerprint(protocol(checkpoint_sha='1'*64))
    assert original != evaluation.fingerprint(protocol(image_hashes={'one.jpeg':'2'*64}))
    path = tmp_path/'one.json.gz'
    write_envelope(path,{'example':True},original,'c'*64)
    assert read_envelope(path,original,'c'*64) == {'example':True}
    assert read_envelope(path,evaluation.fingerprint(protocol(mode='limb_v1')),'c'*64) is None


def test_raw_provider_bypasses_limb_preserves_inference_contract(monkeypatch):
    image = np.full((2048,2048),3,np.uint8)
    model = torch.nn.Linear(1,1)
    before = {k:v.clone()for k,v in model.state_dict().items()}
    calls = []
    def unexpected(_):
        raise AssertionError('raw mode must not call limb')
    def predict(selected,pixels,device,**kwargs):
        assert selected is model and pixels is image
        assert kwargs == dict(tile=1024,overlap=.25,batch=1,amp_dtype=None)
        assert not selected.training and not torch.is_grad_enabled()
        calls.append(True)
        return 'base','average',{'base':'mask'}
    monkeypatch.setattr(inference,'limb',unexpected)
    monkeypatch.setattr(inference,'predict_views',predict)
    monkeypatch.setattr(inference,'prediction_record',lambda heads,name,sha,identity:
        dict(kind=heads,file_name=name,image_sha256=sha,identity=identity))
    result = raw.predict_native_record(model,image,'one.jpeg','image-sha','raw-id',
                                      device=torch.device('cpu'))
    assert result['kind']=='base' and result['tta']['kind']=='average'
    assert result['view_unions']=={'base':'mask'} and len(calls)==1
    for key,value in model.state_dict().items(): assert torch.equal(value,before[key])


def test_control_limb_mode_applies_transform(monkeypatch):
    image=np.zeros((2048,2048),np.uint8)
    corrected=np.full_like(image,9)
    seen=[]
    monkeypatch.setattr(inference,'limb',lambda pixels:corrected)
    def predict(model,pixels,device,**kw):
        assert pixels is corrected
        seen.append(True)
        return 1,2,{}
    monkeypatch.setattr(inference,'predict_views',predict)
    monkeypatch.setattr(inference,'prediction_record',lambda *args:{})
    raw.predict_native_record(torch.nn.Identity(),image,'one.jpeg','sha','id',
                              device=torch.device('cpu'),mode='limb_v1')
    assert seen==[True]


def test_actual_frozen_source_assets_verified_readonly(tmp_path):
    archive=Path('/workspace/artifacts/downloads/SolarFilament_CHASE_Only_BothModels_60Epoch_FullPass_FixPNG_Package.zip')
    metadata=json.loads(archive.with_name(archive.name.replace('_Package.zip','_manifest.json')).read_text())
    source_manifest=dict(sources=metadata['source_sha256'],assets=metadata['asset_sha256'])
    with zipfile.ZipFile(archive) as z:
        for folder,key in [('code','sources'),('assets','assets')]:
            for name in source_manifest[key]:
                target=tmp_path/folder/raw.VERSION/name
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(z.read(folder+'/'+name))
    p=tmp_path/'code'/raw.VERSION/'source_manifest.json'
    p.write_text(json.dumps(source_manifest,sort_keys=True,indent=2))
    checkpoint=tmp_path/'native'/'last.pt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'untouched')
    original={str(x):(x.read_bytes(),x.stat().st_mtime_ns)for x in tmp_path.rglob('*')if x.is_file()}
    code,returned=raw.verify_sources(tmp_path)
    assert code==p.parent and returned==source_manifest
    assert original=={str(x):(x.read_bytes(),x.stat().st_mtime_ns)for x in tmp_path.rglob('*')if x.is_file()}
    bad=code/'submission_inference.py';bad.write_bytes(b'changed')
    import pytest
    with pytest.raises(ValueError,match='Frozen code/asset changed'):
        raw.verify_sources(tmp_path)
