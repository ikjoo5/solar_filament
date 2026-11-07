"""CPU-only inference contracts; no test truth, model weights or GPU needed."""
import csv
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
from pycocotools import mask as mu
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import swefil_test_inference as worker


def probability(rectangles=()):
    value=np.zeros((2048,2048),np.float32)
    for y,x,h,w,p in rectangles:value[y:y+h,x:x+w]=p
    return value


def encode(rectangle):
    return worker.decode_probability(probability([(*rectangle,.8)]))[0]


@pytest.fixture
def scenario(tmp_path):
    images=tmp_path/'images';images.mkdir()
    for name in ('a.jpeg','b.jpeg','c.jpeg'):(images/name).write_bytes(name.encode())
    sources={p.name:worker.digest(p) for p in sorted(images.iterdir())}
    protocol=dict(version='CPU_fixture',recipe=worker.RECIPE.copy(),source_images=sources,
                  checkpoint_sha256='fixture_weight_sha',runtime=dict(device='cpu'))
    output=tmp_path/'output'
    def reader(path,checksum):
        assert worker.digest(path)==checksum
        return path.name
    def predictor(name):
        return probability({'a.jpeg':[(10,10,20,20,.8)],'b.jpeg':[],
                            'c.jpeg':[(50,50,20,20,.8),(100,100,20,20,.7)]}[name])
    return images,output,protocol,reader,predictor


def test_validation_recipe_uses_exact_threshold_area_and_eight_connectivity():
    p=probability([(10,10,20,20,.65),(100,100,10,20,.9),(400,400,20,20,.649)])
    masks=worker.decode_probability(p)
    assert len(masks)==1 and int(mu.area(worker.validate_masks(masks)[0]))==400
    # Two diagonal 150px components are one eight-connected 300px instance.
    p=probability([(10,10,10,15,.8),(20,25,10,15,.8)])
    assert len(worker.decode_probability(p))==1
    p[0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):worker.decode_probability(p)


def test_partial_run_resumes_and_blank_image_is_complete_without_dummy_row(scenario):
    images,output,protocol,reader,predictor=scenario
    calls=[]
    def interrupted(name):
        calls.append(name)
        if name=='b.jpeg':raise RuntimeError('fixture interrupt')
        return predictor(name)
    with pytest.raises(RuntimeError,match='fixture interrupt'):
        worker._run_images(images,output,protocol,interrupted,reader)
    assert json.loads((output/'progress.json').read_text())['images_completed']==1
    assert not (output/'submission_swefil_epoch100_native.csv').exists()
    assert not (output/'latest_result.json').exists()
    calls=[]
    def resumed(name):calls.append(name);return predictor(name)
    result=worker._run_images(images,output,protocol,resumed,reader)
    assert calls==['b.jpeg','c.jpeg']
    assert result['images_completed']==3 and result['empty_images']==['b.jpeg']
    assert result['prediction_instances']==3 and result['overlap_pixels']==0
    with Path(result['csv']).open(newline='') as f:rows=list(csv.DictReader(f))
    assert [r['filament_id'] for r in rows]==['a_1','c_1','c_2']
    with zipfile.ZipFile(result['archive']) as archive:
        assert 'predictions/b.jpeg.json' in archive.namelist()
        assert not any(n.endswith(('.pt','.png','.jpeg')) for n in archive.namelist())
    assert worker.digest(result['csv'])==result['csv_sha256']
    again=worker._run_images(images,output,protocol,lambda *_:pytest.fail('cached predictions reran'),reader)
    assert again['csv_sha256']==result['csv_sha256']


@pytest.mark.parametrize('field,changed_value',[('runtime',dict(device='another-runtime')),
    ('recipe',dict(worker.RECIPE,threshold=.5)),('checkpoint_sha256','different-weights'),
    ('source_images',{'other.jpeg':'different-image'})])
def test_cache_corruption_and_changed_identity_do_not_silently_resume(scenario,field,changed_value):
    images,output,protocol,reader,predictor=scenario
    result=worker._run_images(images,output,protocol,predictor,reader)
    previous=Path(result['csv']).read_bytes()
    changed=dict(protocol,**{field:changed_value})
    with pytest.raises(ValueError,match='identity/checksum'):
        worker._run_images(images,output,changed,predictor,reader)
    target=output/'predictions/a.jpeg.json'
    record=json.loads(target.read_text());record['payload']['instances']=999
    target.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='identity/checksum'):
        worker._run_images(images,output,protocol,predictor,reader)
    assert Path(result['csv']).read_bytes()==previous


def test_resume_keeps_completed_predictions_when_drive_state_observation_changes(scenario):
    images,output,protocol,reader,predictor=scenario
    first=dict(selected_epoch=100,selected_sha256=protocol['checkpoint_sha256'],
               selected_path='/drive/backup/best_epoch100.pt',training_state=dict(completed_epoch=51,best_epoch=42))
    second=dict(first,selected_path='/drive/restored/best_epoch100.pt',
                training_state=dict(completed_epoch=100,best_epoch=100))
    def interrupted(name):
        if name=='b.jpeg':raise RuntimeError('fixture interrupt')
        return predictor(name)
    with pytest.raises(RuntimeError,match='fixture interrupt'):
        worker._run_images(images,output,protocol,interrupted,reader,provenance=first)
    original_identity=(output/'identity.json').read_bytes()
    original_prediction=(output/'predictions/a.jpeg.json').read_bytes()
    calls=[]
    def resumed(name):calls.append(name);return predictor(name)
    result=worker._run_images(images,output,protocol,resumed,reader,provenance=second)
    assert calls==['b.jpeg','c.jpeg']
    assert (output/'identity.json').read_bytes()==original_identity
    assert (output/'predictions/a.jpeg.json').read_bytes()==original_prediction
    assert result['checkpoint_selection']==second
    assert json.loads((output/'checkpoint_selection.json').read_text())==second
    with zipfile.ZipFile(result['archive']) as archive:
        assert json.loads(archive.read('checkpoint_selection.json'))==second


def test_rechecks_overlap_even_if_modified_cache_envelope_is_consistent(scenario):
    images,output,protocol,reader,predictor=scenario
    worker._run_images(images,output,protocol,predictor,reader)
    identity=worker.fingerprint(protocol);target=output/'predictions/a.jpeg.json'
    record=worker.read_envelope(target,identity)
    record['masks']=[encode((10,10,20,20)),encode((15,15,20,20))];record['instances']=2
    worker.envelope(target,identity,record)
    with pytest.raises(ValueError,match='Overlapping'):
        worker._run_images(images,output,protocol,predictor,reader)


def test_inventory_binds_names_not_just_count_and_source_pixels(tmp_path):
    images=tmp_path/'images';images.mkdir()
    for name in ('a.jpeg','b.jpeg'):(images/name).write_bytes(name.encode())
    checksum=worker.fingerprint(['a.jpeg','b.jpeg'])
    first=worker.inventory(images,expected_count=2,expected_names_sha=checksum)
    assert set(first)=={'a.jpeg','b.jpeg'}
    (images/'b.jpeg').rename(images/'c.jpeg')
    with pytest.raises(ValueError,match='exact official'):
        worker.inventory(images,expected_count=2,expected_names_sha=checksum)


def test_native_image_reader_uses_original_grayscale_and_hash(tmp_path):
    image=np.full((2048,2048,3),100,np.uint8);image[20:40,20:40]=50
    path=tmp_path/'sample.jpeg';cv2.imwrite(str(path),image)
    actual=worker.read_image(path,worker.digest(path))
    assert np.array_equal(actual,cv2.imread(str(path),cv2.IMREAD_GRAYSCALE))
    with pytest.raises(ValueError,match='changed'):worker.read_image(path,'wrong hash')
    cv2.imwrite(str(path),np.zeros((50,50),np.uint8))
    with pytest.raises(ValueError,match='native 2048'):worker.read_image(path,worker.digest(path))


def test_output_cannot_overwrite_input_or_unidentified_results(scenario):
    images,output,protocol,reader,predictor=scenario
    with pytest.raises(ValueError,match='separate'):
        worker._run_images(images,images/'out',protocol,predictor,reader)
    output.mkdir();(output/'latest_result.json').write_text('{}')
    with pytest.raises(ValueError,match='no identity'):
        worker._run_images(images,output,protocol,predictor,reader)


def test_written_csv_checks_members_duplicates_and_overlap(tmp_path):
    mask=encode((10,10,20,20))['counts'];path=tmp_path/'pred.csv'
    def write(rows):
        with path.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['filament_id','segmentation_rle']);w.writerows(rows)
    write([['a_1',mask],['a_1',mask]])
    with pytest.raises(ValueError,match='duplicate'):worker.validate_csv(path,['a.jpeg'],2)
    write([['a_1',mask],['a_2',mask]])
    with pytest.raises(ValueError,match='Overlapping'):worker.validate_csv(path,['a.jpeg'],2)
    write([['other_1',mask]])
    with pytest.raises(ValueError,match='Unexpected'):worker.validate_csv(path,['a.jpeg'],1)


def test_native_coordinates_and_inclusive_minimum_area():
    p=probability([(1700,1800,15,20,.65),(100,100,13,23,.9)])
    masks=worker.decode_probability(p)
    assert len(masks)==1
    decoded=mu.decode(worker.validate_masks(masks)[0])
    assert decoded.shape==(2048,2048) and int(decoded.sum())==300
    assert np.all(decoded[1700:1715,1800:1820]==1)
    assert not decoded[1800:1820,1700:1715].any()


@pytest.fixture
def training(tmp_path,monkeypatch):
    root=tmp_path/'training';train=root/'train';train.mkdir(parents=True)
    cfg=dict(worker.REQUIRED_CONFIG,test_images=str(tmp_path/'images'))
    identity=dict(settings=cfg,code={name:worker.digest(Path(worker.__file__).with_name(name))
        for name in worker.SOURCE_FILES[1:]},runtime=dict(device='cuda',gpu='fixture'))
    token=worker.fingerprint(identity);monkeypatch.setattr(worker,'TRAINING_IDENTITY',token)
    checkpoint=tmp_path/'verified_copy/best_epoch100.pt';checkpoint.parent.mkdir()
    payload=dict(identity=token,epoch=100,config=cfg,
        validation=dict(threshold=.65,min_area=300,pq=worker.REVIEWED_PQ),
        model_state_dict={'weight':worker.torch.ones(1)})
    worker.torch.save(payload,checkpoint)
    monkeypatch.setattr(worker,'CHECKPOINT_SHA',worker.digest(checkpoint))
    monkeypatch.setattr(worker,'CHECKPOINT_SIZE',checkpoint.stat().st_size)
    worker.envelope(train/'identity.json',token,identity)
    worker.atomic_json(train/'config.json',cfg)
    worker.envelope(train/'state.json',token,dict(epoch=51,best_epoch=42,best_pq=.3,
        best=dict(path='checkpoints/best_epoch042.pt',sha256='0'*64)))
    # A stale mutable best selection cannot override the explicitly reviewed file.
    worker.atomic_json(train/'best_epoch.json',dict(epoch=42,best=dict(threshold=.35,min_area=100,pq=.3)))
    return root,identity,checkpoint,payload


def test_exact_checkpoint_ignores_mutable_best_pointer_and_preserves_training(training):
    root,identity,checkpoint,payload=training
    before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    metadata=worker.training_metadata(root,checkpoint)
    assert metadata['checkpoint']==str(checkpoint)
    assert metadata['validation']==payload['validation']
    assert metadata['checkpoint_selection']['selected_epoch']==100
    assert metadata['checkpoint_selection']['training_state']['best_epoch']==42
    assert 'model_state_dict' not in json.dumps(metadata)
    assert before=={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    (root/'train/state.json').unlink()
    assert worker.training_metadata(root,checkpoint)['validation']==payload['validation']


@pytest.mark.parametrize('field,value,message',[
    ('epoch',99,'payload differs'),('identity','another-run','payload differs'),
    ('validation_tile',768,'inference recipe differs'),('amp_dtype','bfloat16','inference recipe differs'),
    ('threshold',.35,'decoding recipe differs'),('min_area',100,'decoding recipe differs'),
    ('pq',.431,'validation PQ differs'),('pq',float('nan'),'validation PQ differs'),
    ('model_state_dict',{},'no model state'),
])
def test_verified_file_still_checks_exact_payload(training,monkeypatch,field,value,message):
    root,identity,checkpoint,payload=training
    if field in ('validation_tile','amp_dtype'):payload['config'][field]=value
    elif field in ('threshold','min_area','pq'):payload['validation'][field]=value
    else:payload[field]=value
    worker.torch.save(payload,checkpoint)
    monkeypatch.setattr(worker,'CHECKPOINT_SHA',worker.digest(checkpoint))
    monkeypatch.setattr(worker,'CHECKPOINT_SIZE',checkpoint.stat().st_size)
    with pytest.raises(ValueError,match=message):worker.training_metadata(root,checkpoint)


def test_checkpoint_bytes_and_links_checked_before_torch_load(training,monkeypatch,tmp_path):
    root,identity,checkpoint,payload=training
    monkeypatch.setattr(worker.torch,'load',lambda *a,**kw:pytest.fail('Unverified file loaded'))
    link=tmp_path/'link.pt';link.symlink_to(checkpoint)
    with pytest.raises(ValueError,match='Unsafe checkpoint link'):worker.training_metadata(root,link)
    original=checkpoint.read_bytes();checkpoint.write_bytes(b'x'*len(original))
    with pytest.raises(ValueError,match='checkpoint checksum/size'):worker.training_metadata(root,checkpoint)
    checkpoint.write_bytes(original[:-1])
    with pytest.raises(ValueError,match='checkpoint checksum/size'):worker.training_metadata(root,checkpoint)


def test_training_identity_config_and_helper_hashes_are_bound(training,monkeypatch):
    root,identity,checkpoint,payload=training
    cfg=dict(payload['config'],preprocessing='clahe')
    worker.atomic_json(root/'train/config.json',cfg)
    with pytest.raises(ValueError,match='recipe differs'):worker.training_metadata(root,checkpoint)
    worker.atomic_json(root/'train/config.json',payload['config'])
    identity['code']['solar_a100_train_standalone.py']='0'*64
    token=worker.fingerprint(identity);monkeypatch.setattr(worker,'TRAINING_IDENTITY',token)
    worker.envelope(root/'train/identity.json',token,identity)
    with pytest.raises(ValueError,match='helper differs'):worker.training_metadata(root,checkpoint)


def test_model_boundary_uses_pinned_inference_and_cached_run_never_builds_model(tmp_path,monkeypatch):
    images=tmp_path/'images';images.mkdir()
    image_path=images/'one.jpeg';cv2.imwrite(str(image_path),np.full((2048,2048),100,np.uint8))
    sources={image_path.name:worker.digest(image_path)}
    metadata=dict(config=dict(test_images=str(images)),checkpoint=str(tmp_path/'reviewed-best100.pt'),
        training_runtime=dict(device='cuda'),validation=dict(threshold=.65,min_area=300,pq=worker.REVIEWED_PQ),
        checkpoint_selection=dict(selected_epoch=100))
    monkeypatch.setattr(worker,'training_metadata',lambda *_:metadata)
    monkeypatch.setattr(worker,'inventory',lambda _:sources)
    monkeypatch.setattr(worker.torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(worker,'configure_cuda',lambda:None)
    monkeypatch.setattr(worker,'runtime_identity',lambda:dict(device='cuda',gpu='Tesla T4'))
    calls=[]
    def load(path):
        assert path==metadata['checkpoint'];calls.append('load')
        return dict(model_state_dict={'fixture':1})
    monkeypatch.setattr(worker,'load_reviewed_checkpoint',load)
    class Model:
        def load_state_dict(self,state,strict):
            assert state=={'fixture':1} and strict;calls.append('state')
        def to(self,device):assert str(device)=='cuda';return self
        def eval(self):calls.append('eval');return self
    def build(encoder,pretrained):
        assert encoder=='tu-convnext_tiny' and pretrained is False
        calls.append('build');return Model()
    monkeypatch.setattr(worker.base,'build_model',build)
    def limb(raw):
        assert np.all(raw==100);calls.append('limb');return raw-1
    monkeypatch.setattr(worker.base,'limb',limb)
    def inference(model,image,device,**kwargs):
        assert np.all(image==99) and str(device)=='cuda'
        assert kwargs==dict(tile=1024,overlap=.25,batch=1,amp_dtype=worker.torch.float16)
        calls.append('predict');return probability([(1700,1800,15,20,.65)])
    monkeypatch.setattr(worker.base,'predict_foreground',inference)
    args=(tmp_path/'train',images,tmp_path/'output',metadata['checkpoint'])
    result=worker.run(*args)
    assert calls==['load','build','state','eval','limb','predict']
    assert result['checkpoint_epoch']==100 and result['prediction_instances']==1
    calls.clear()
    again=worker.run(*args)
    assert calls==[] and again['csv_sha256']==result['csv_sha256']
