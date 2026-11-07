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
import chase_shape_test_inference as worker


def probability(rectangles=()):
    value=np.zeros((2048,2048),np.float32)
    for y,x,h,w,p in rectangles:value[y:y+h,x:x+w]=p
    return value


def encode(rectangle):
    return worker.decode_probability(probability([(*rectangle,.5)]))[0]


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
        return probability({'a.jpeg':[(10,10,20,20,.5)],'b.jpeg':[],
                            'c.jpeg':[(50,50,20,20,.6),(100,100,20,20,.4)]}[name])
    return images,output,protocol,reader,predictor


def test_validation_recipe_uses_exact_threshold_area_and_eight_connectivity():
    p=probability([(10,10,20,20,.35),(100,100,10,20,.9),(400,400,20,20,.349)])
    masks=worker.decode_probability(p)
    assert len(masks)==1 and int(mu.area(worker.validate_masks(masks)[0]))==400
    # Two diagonal 150px components are one eight-connected 300px instance.
    p=probability([(10,10,10,15,.6),(20,25,10,15,.6)])
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
    assert not (output/'submission_chase_shape_best82_native.csv').exists()
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


def test_cache_corruption_and_changed_runtime_do_not_silently_resume(scenario):
    images,output,protocol,reader,predictor=scenario
    result=worker._run_images(images,output,protocol,predictor,reader)
    previous=Path(result['csv']).read_bytes()
    changed=dict(protocol,runtime=dict(device='another-runtime'))
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
    first=dict(selected_epoch=82,selected_sha256=protocol['checkpoint_sha256'],
               selected_path='/drive/backup/best_epoch082.pt',training_state=dict(completed_epoch=51,best_epoch=42))
    second=dict(first,selected_path='/drive/restored/best_epoch082.pt',
                training_state=dict(completed_epoch=100,best_epoch=82))
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


@pytest.fixture
def training(tmp_path,monkeypatch):
    root=tmp_path/'training';train=root/'train';train.mkdir(parents=True)
    cfg=dict(encoder='tu-convnext_tiny',crop=1024,validation_tile=1024,
             preprocessing='limb_only',overlap=.25,infer_batch=1,test_images=str(tmp_path/'test'))
    identity=dict(settings=cfg,code={n:worker.digest(Path(worker.__file__).with_name(n)) for n in worker.SOURCE_FILES[1:]},
                  runtime=dict(device='cuda',gpu='fixture'))
    token=worker.fingerprint(identity);monkeypatch.setattr(worker,'TRAINING_IDENTITY',token)
    checkpoint=train/'checkpoints/best_epoch082.pt';checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'Fixture checkpoint bytes, never loaded')
    sha=worker.digest(checkpoint);monkeypatch.setattr(worker,'CHECKPOINT_SHA',sha)
    worker.envelope(train/'identity.json',token,identity)
    worker.envelope(train/'state.json',token,dict(best_epoch=82,best_pq=.431,
                    best=dict(path='checkpoints/best_epoch082.pt',sha256=sha)))
    worker.atomic_json(train/'config.json',cfg)
    worker.atomic_json(train/'best_epoch.json',dict(epoch=82,best=dict(threshold=.35,min_area=300,pq=.431)))
    return root,identity,checkpoint


def test_reviewed_checkpoint_selected_without_loading_last_or_training(training):
    root,identity,checkpoint=training
    metadata=worker.training_metadata(root)
    assert metadata['checkpoint']==str(checkpoint)
    assert metadata['validation']['threshold']==.35
    checkpoint.write_bytes(b'Changed weights')
    with pytest.raises(ValueError,match='checkpoint checksum'):worker.training_metadata(root)


def test_recorded_reviewed_selection_passes_without_mutating_pinned_constants():
    # Reviewed ZIP metadata, not the generated weight fixture above. This guard
    # test does not claim that the actual checkpoint bytes are present locally.
    recorded=dict(epoch=100,best_epoch=82,best=dict(path='checkpoints/best_epoch082.pt',
        sha256='8ea1198734dd434c42889aee2c79928afd8eb8ce4491e826c13d07e41e03d330'))
    worker.verify_reviewed_selection(recorded,'train/state.json')


@pytest.mark.parametrize('changed_epoch,changed_sha',[(80,None),(82,'0'*64),(80,'0'*64)])
def test_changed_drive_selection_reports_observed_and_expected_values(training,changed_epoch,changed_sha):
    root,identity,checkpoint=training
    path=root/'train/state.json'
    state=worker.read_envelope(path,worker.TRAINING_IDENTITY)
    original_sha=state['best']['sha256']
    state.update(epoch=100,best_epoch=changed_epoch)
    if changed_sha is not None:state['best']['sha256']=changed_sha
    worker.envelope(path,worker.TRAINING_IDENTITY,state)
    with pytest.raises(ValueError,match='Saved checkpoint selection differs') as error:
        worker.training_metadata(root)
    details=json.loads(str(error.value).split('\n',1)[1])
    assert details==dict(state_path=str(path),completed_epoch=100,
        expected_best_epoch=82,observed_best_epoch=changed_epoch,
        observed_best_path='checkpoints/best_epoch082.pt',expected_checkpoint_sha256=original_sha,
        observed_checkpoint_sha256=changed_sha or original_sha)
    assert checkpoint.read_bytes()==b'Fixture checkpoint bytes, never loaded'


def test_decoder_recipe_and_inference_helper_must_match_reviewed_run(training):
    root,identity,checkpoint=training
    path=root/'train/config.json';config=json.loads(path.read_text());config['validation_tile']=768
    worker.atomic_json(path,config)
    with pytest.raises(ValueError,match='recipe differs'):worker.training_metadata(root)


def test_bad_checkpoint_relative_path_is_not_loaded(training):
    root,identity,checkpoint=training
    token=worker.TRAINING_IDENTITY;state=worker.read_envelope(root/'train/state.json',token)
    state['best']['path']='../../outside.pt';worker.envelope(root/'train/state.json',token,state)
    with pytest.raises(ValueError,match='Unsafe checkpoint'):worker.training_metadata(root)


@pytest.fixture
def recovery(training,tmp_path,monkeypatch):
    root,identity,old_checkpoint=training
    state=dict(epoch=51,best_epoch=42,best_pq=.4092245503356435,
               best=dict(path='checkpoints/best_epoch042.pt',sha256='9'*64))
    worker.envelope(root/'train/state.json',worker.TRAINING_IDENTITY,state)
    worker.atomic_json(root/'train/best_epoch.json',dict(epoch=42,best=dict(threshold=.65,min_area=300,pq=state['best_pq'])))
    checkpoint=tmp_path/'recovered_copy/best_epoch082.pt';checkpoint.parent.mkdir()
    payload=dict(identity=worker.TRAINING_IDENTITY,epoch=82,
                 config=json.loads((root/'train/config.json').read_text()),
                 validation=dict(threshold=.35,min_area=300,pq=worker.REVIEWED_PQ,tp=472,fp=295,fn=236),
                 model_state_dict={'weight':worker.torch.ones(1)})
    worker.torch.save(payload,checkpoint)
    monkeypatch.setattr(worker,'CHECKPOINT_SHA',worker.digest(checkpoint))
    return root,checkpoint,payload


def test_exact_recovered_checkpoint_ignores_stale_best42_pointer_but_preserves_training(recovery):
    root,checkpoint,payload=recovery
    before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='Saved checkpoint selection differs'):
        worker.training_metadata(root)
    metadata=worker.training_metadata(root,checkpoint)
    assert metadata['checkpoint']==str(checkpoint)
    assert metadata['validation']==payload['validation']
    selection=metadata['checkpoint_selection']
    assert selection['mode']=='reviewed_file_override' and selection['selected_epoch']==82
    assert selection['training_state']['completed_epoch']==51 and selection['training_state']['best_epoch']==42
    assert selection['training_state_payload_sha256']==worker.fingerprint(worker.read_envelope(root/'train/state.json'))
    assert 'model_state_dict' not in json.dumps(metadata) and len(json.dumps(metadata))<8000
    assert before=={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_recovered_file_checksum_and_links_checked_before_torch_load(recovery,monkeypatch,tmp_path):
    root,checkpoint,payload=recovery
    monkeypatch.setattr(worker.torch,'load',lambda *a,**k:pytest.fail('Unverified checkpoint was loaded'))
    link=tmp_path/'link.pt';link.symlink_to(checkpoint)
    with pytest.raises(ValueError,match='Unsafe checkpoint link'):
        worker.training_metadata(root,link)
    checkpoint.write_bytes(b'a different checkpoint')
    with pytest.raises(ValueError,match='checkpoint checksum'):
        worker.training_metadata(root,checkpoint)


@pytest.mark.parametrize('field,value,message',[
    ('epoch',42,'payload differs'),('identity','another-run','payload differs'),
    ('tile',768,'inference recipe differs'),('threshold',.65,'decoding recipe differs'),
    ('min_area',100,'decoding recipe differs'),('pq',.4092245503356435,'validation PQ differs'),
    ('pq',float('nan'),'validation PQ differs'),('model_state_dict',None,'no model state'),
])
def test_recovery_checks_payload_after_file_hash(recovery,monkeypatch,field,value,message):
    root,checkpoint,payload=recovery
    if field=='tile':payload['config']['validation_tile']=value
    elif field in ('threshold','min_area','pq'):payload['validation'][field]=value
    else:payload[field]=value
    worker.torch.save(payload,checkpoint)
    # Rebind the generated fixture hash to exercise payload checks independently.
    monkeypatch.setattr(worker,'CHECKPOINT_SHA',worker.digest(checkpoint))
    with pytest.raises(ValueError,match=message):worker.training_metadata(root,checkpoint)


def test_model_boundary_loads_epoch82_without_pretraining_and_preprocesses_once(tmp_path,monkeypatch):
    images=tmp_path/'images';images.mkdir()
    image_path=images/'one.jpeg';cv2.imwrite(str(image_path),np.full((2048,2048),100,np.uint8))
    sources={image_path.name:worker.digest(image_path)}
    metadata=dict(config=dict(test_images=str(images)),checkpoint='reviewed-best82.pt',
                  training_runtime=dict(device='cuda'),validation=dict(threshold=.35,min_area=300,pq=.431),
                  checkpoint_selection=dict(selected_epoch=82,training_state=dict(completed_epoch=51,best_epoch=42)))
    monkeypatch.setattr(worker,'training_metadata',lambda *_:metadata)
    monkeypatch.setattr(worker,'inventory',lambda _:sources)
    monkeypatch.setattr(worker.torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(worker,'configure_cuda',lambda:None)
    monkeypatch.setattr(worker,'runtime_identity',lambda:dict(device='cuda',gpu='mock'))
    checkpoint=dict(identity=worker.TRAINING_IDENTITY,epoch=82,
                    validation=dict(threshold=.35,min_area=300),model_state_dict={'fixture':1})
    calls=[]
    def load(path,**kwargs):
        assert path=='reviewed-best82.pt' and kwargs==dict(map_location='cpu',weights_only=True)
        calls.append('load');return checkpoint
    monkeypatch.setattr(worker.torch,'load',load)
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
        calls.append('infer');return probability([(20,20,20,20,.5)])
    monkeypatch.setattr(worker.base,'predict_foreground',inference)
    output=tmp_path/'submission'
    result=worker.run(tmp_path/'training',None,output)
    assert calls==['load','build','state','eval','limb','infer']
    assert result['prediction_instances']==1
    metadata['checkpoint_selection']['training_state']=dict(completed_epoch=100,best_epoch=82)
    resumed=worker.run(tmp_path/'training',None,output)
    assert calls==['load','build','state','eval','limb','infer']
    assert resumed['identity']==result['identity']
    assert resumed['checkpoint_selection']['training_state']['completed_epoch']==100
    assert 'checkpoint_selection' not in worker.read_envelope(output/'identity.json')
