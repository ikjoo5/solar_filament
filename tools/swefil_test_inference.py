"""Inference-only test submission for the reviewed SWEFil mixed native epoch100.

Uses the validation-selected base-view recipe, not YOLO fusion or public RLEs.
Completed per-image predictions survive Colab resets; training is never resumed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import uuid
import zipfile

os.environ.setdefault('TORCH_CUDNN_V8_API_DISABLED', '1')
import cv2
import numpy as np
from pycocotools import mask as mu
import torch

import solar_a100_train_standalone as base
from chase_best84_training import configure_cuda
from chase_independent_decode import validate_masks

VERSION = 'swefil_epoch100_native_test_v1'
CHECKPOINT_SHA = '2b22ed327e2f64902bb7b9b3984586d1ba893f52ada74f557622cce4dac27e40'
REVIEWED_PQ = 0.41086632237070175
TRAINING_IDENTITY = '00c20ec35658f63e9cc6917cc9cdd30bdc060eee27445e1a0bf8b99c2c4ef405'
TEST_NAMES_SHA = 'f9b2272771d3f6666f6ef812f79968a27f3d448ac2f277eda46c854ad38d598f'
EXPECTED_IMAGES = 180
CHECKPOINT_EPOCH = 100
CHECKPOINT_SIZE = 128514783
CSV_NAME = 'submission_swefil_epoch100_native.csv'
ARCHIVE_NAME = 'swefil_epoch100_test_results.zip'
SOURCE_FILES = ('swefil_test_inference.py', 'solar_a100_train_standalone.py',
                'chase_best84_training.py', 'chase_independent_decode.py')
RECIPE = dict(encoder='tu-convnext_tiny', preprocessing='limb_only', tile=1024,
              overlap=.25, infer_batch=1, precision='float16', threshold=.65,
              min_area=300, connectivity=8, tta=False, yolo=False,
              public_predictions=False, checkpoint_epoch=100)
REQUIRED_CONFIG = dict(encoder='tu-convnext_tiny',crop=1024,validation_tile=1024,
                       preprocessing='limb_only',overlap=.25,infer_batch=1,
                       amp_dtype='float16',device='cuda')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    try:
        temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def envelope(path, identity, payload):
    atomic_json(path,dict(identity=identity,payload=payload,payload_sha256=fingerprint(payload)))


def read_envelope(path, identity=None):
    path=Path(path)
    if path.is_symlink():raise ValueError('Unsafe metadata link: '+str(path))
    value=json.loads(path.read_text())
    if ((identity is not None and value.get('identity')!=identity) or
            fingerprint(value.get('payload'))!=value.get('payload_sha256')):
        raise ValueError('Metadata identity/checksum differs: '+str(path))
    return value['payload']


def validate_checkpoint_payload(payload):
    """Check the exact reviewed model selection, independently of Drive pointers."""
    if (not isinstance(payload,dict) or payload.get('identity')!=TRAINING_IDENTITY
            or payload.get('epoch')!=CHECKPOINT_EPOCH):
        raise ValueError('Checkpoint payload differs from reviewed SWEFil epoch100')
    cfg=payload.get('config')
    if not isinstance(cfg,dict) or any(cfg.get(k)!=v for k,v in REQUIRED_CONFIG.items()):
        raise ValueError('Checkpoint inference recipe differs')
    validation=payload.get('validation')
    if (not isinstance(validation,dict) or validation.get('threshold')!=RECIPE['threshold']
            or validation.get('min_area')!=RECIPE['min_area']):
        raise ValueError('Checkpoint selected decoding recipe differs')
    pq=validation.get('pq')
    if not isinstance(pq,(float,int)) or not np.isfinite(pq) or abs(pq-REVIEWED_PQ)>1e-12:
        raise ValueError('Checkpoint validation PQ differs from reviewed epoch100')
    if not isinstance(payload.get('model_state_dict'),dict) or not payload['model_state_dict']:
        raise ValueError('Checkpoint has no model state')
    return dict(validation)


def load_reviewed_checkpoint(checkpoint):
    supplied=Path(checkpoint)
    if supplied.is_symlink():raise ValueError('Unsafe checkpoint link: '+str(supplied))
    if not supplied.is_file():raise FileNotFoundError('Reviewed epoch100 checkpoint missing: '+str(supplied))
    if supplied.stat().st_size!=CHECKPOINT_SIZE or digest(supplied)!=CHECKPOINT_SHA:
        raise ValueError('Epoch100 checkpoint checksum/size differs')
    payload=torch.load(supplied,map_location='cpu',weights_only=True)
    validate_checkpoint_payload(payload)
    return payload


def training_metadata(training_root,checkpoint):
    """Read reviewed metadata and verified weights; never run stored code snapshots."""
    root=Path(training_root).resolve();train=root/'train'
    identity=read_envelope(train/'identity.json',TRAINING_IDENTITY)
    if fingerprint(identity)!=TRAINING_IDENTITY:
        raise ValueError('This submission requires the reviewed SWEFil training identity')
    config_path=train/'config.json'
    if config_path.is_symlink():raise ValueError('Unsafe training config link')
    cfg=json.loads(config_path.read_text())
    if any(cfg.get(k)!=v or identity['settings'].get(k)!=v for k,v in REQUIRED_CONFIG.items()):
        raise ValueError('Saved training/evaluation recipe differs from reviewed native inference')
    for name in SOURCE_FILES[1:]:
        source=Path(__file__).with_name(name)
        if not source.is_file() or digest(source)!=identity['code'].get(name):
            raise ValueError('Inference helper differs from the evaluated training code: '+name)
    payload=load_reviewed_checkpoint(checkpoint)
    validation=dict(payload['validation']);del payload
    selection=dict(mode='reviewed_file',selected_path=str(Path(checkpoint).resolve()),
                   selected_epoch=CHECKPOINT_EPOCH,selected_sha256=CHECKPOINT_SHA,
                   selected_size=CHECKPOINT_SIZE)
    # The verified file is authoritative. A mutable best pointer is observation
    # only and is deliberately excluded from the prediction cache identity.
    state_path=train/'state.json'
    if state_path.exists() or state_path.is_symlink():
        state=read_envelope(state_path,TRAINING_IDENTITY)
        selection.update(training_state_payload_sha256=fingerprint(state),training_state=dict(
            completed_epoch=state.get('epoch'),best_epoch=state.get('best_epoch'),best_pq=state.get('best_pq')))
    return dict(root=str(root),checkpoint=str(Path(checkpoint).resolve()),
                checkpoint_sha256=CHECKPOINT_SHA,training_identity=TRAINING_IDENTITY,
                validation=validation,training_runtime=identity['runtime'],config=cfg,
                checkpoint_selection=selection)


def inventory(images, *, expected_count=EXPECTED_IMAGES, expected_names_sha=TEST_NAMES_SHA):
    images=Path(images).resolve()
    if not images.is_dir():raise FileNotFoundError('Test image directory missing: '+str(images))
    paths=sorted(p for p in images.iterdir() if p.suffix.lower() in ('.jpeg','.jpg','.png','.tif','.tiff','.bmp'))
    names=[p.name for p in paths]
    if len(names)!=expected_count or fingerprint(names)!=expected_names_sha:
        raise ValueError('Expected the exact official 180 test filenames; image inventory differs')
    if len({p.stem for p in paths})!=len(paths):raise ValueError('Repeated test image stems')
    result={}
    for index,path in enumerate(paths,1):
        if path.is_symlink() or not path.is_file():raise ValueError('Unsafe test image: '+str(path))
        result[path.name]=digest(path)
        if index%25==0 or index==len(paths):print(f'Test source verification {index}/{len(paths)}',flush=True)
    return result


def runtime_identity():
    value=dict(python=sys.version.split()[0],device='cuda',torch=str(torch.__version__),
               cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
               numpy=np.__version__,opencv=cv2.__version__)
    for name in ('segmentation-models-pytorch','timm','scipy','pycocotools'):
        value[name]=importlib.metadata.version(name)
    if torch.cuda.is_available():
        value.update(gpu=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)))
    return value


def read_image(path, checksum):
    payload=Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=checksum:raise ValueError('Test image changed since inventory')
    array=np.frombuffer(payload,np.uint8)
    original=cv2.imdecode(array,cv2.IMREAD_UNCHANGED)
    if original is None or original.dtype!=np.uint8 or original.shape[:2]!=(2048,2048):
        raise ValueError('Expected native 2048x2048 uint8 test image: '+str(path))
    if original.ndim not in (2,3) or (original.ndim==3 and original.shape[2]!=3):
        raise ValueError('Expected grayscale or RGB test image')
    image=cv2.imdecode(array,cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape!=(2048,2048):raise ValueError('Invalid grayscale test image')
    return image


def decode_probability(probability, recipe=None):
    recipe = RECIPE if recipe is None else recipe
    if (not isinstance(probability,np.ndarray) or probability.shape!=(2048,2048)
            or not np.issubdtype(probability.dtype,np.floating)
            or not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1):
        raise ValueError('Expected finite native2048 foreground probabilities in [0,1]')
    masks=[mu.encode(np.asfortranarray(mask.astype(np.uint8)))
           for mask in base.iter_components(probability,recipe['threshold'],recipe['min_area'])]
    return [dict(size=[2048,2048],counts=mask['counts'].decode('ascii'))
            for mask in validate_masks(masks)]


def verify_record(record, name, source_sha, recipe=None):
    recipe = RECIPE if recipe is None else recipe
    if record.get('file_name')!=name or record.get('image_sha256')!=source_sha:
        raise ValueError('Cached prediction belongs to another source image')
    masks=validate_masks(record['masks'])
    if record.get('instances')!=len(masks) or any(int(mu.area(m))<recipe['min_area'] for m in masks):
        raise ValueError('Cached instance count/area differs')
    return masks


@contextmanager
def locked(output):
    with (output/'.inference.lock').open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Another test inference process is using this output') from exc
        yield


def _run_images(images, output, protocol, predictor, image_reader=read_image, provenance=None,
                *, profile=None):
    """Testable persistence engine with an explicit, identity-bound export profile.

    The baseline defaults remain unchanged. Other reviewed entry points provide
    their own recipe and export names instead of mutating module globals.
    """
    profile = dict(recipe=RECIPE, csv_name=CSV_NAME, archive_name=ARCHIVE_NAME,
                   source_files=SOURCE_FILES) if profile is None else profile
    if set(profile) != {'recipe', 'csv_name', 'archive_name', 'source_files'}:
        raise ValueError('Invalid inference export profile')
    recipe = json.loads(canonical(profile['recipe']))
    if recipe != protocol.get('recipe'):
        raise ValueError('Inference profile identity/checksum differs from protocol')
    csv_name, archive_name = profile['csv_name'], profile['archive_name']
    source_files = tuple(profile['source_files'])
    if (not source_files or any(not isinstance(name, str) or Path(name).name != name
                for name in (csv_name, archive_name, *source_files))
            or not csv_name.endswith('.csv') or not archive_name.endswith('.zip')):
        raise ValueError('Invalid inference export filenames')
    export_profile = dict(csv_name=csv_name, archive_name=archive_name, source_files=list(source_files))
    if 'export_profile' in protocol and protocol['export_profile'] != export_profile:
        raise ValueError('Export profile identity/checksum differs from protocol')
    images,output=Path(images).resolve(),Path(output).resolve()
    if output==images or output in images.parents or images in output.parents:
        raise ValueError('Inference output must be separate from input images')
    output.mkdir(parents=True,exist_ok=True)
    identity=fingerprint(protocol);sources=protocol['source_images'];names=sorted(sources)
    with locked(output):
        identity_path=output/'identity.json'
        if identity_path.exists():
            if read_envelope(identity_path,identity)!=protocol:raise ValueError('Inference protocol differs')
        else:
            if any((output/n).exists() for n in ('predictions','latest_result.json',csv_name)):
                raise ValueError('Existing inference output has no identity; use a new folder')
            envelope(identity_path,identity,protocol)
        # Mutable Drive state/path observations document this invocation, but do
        # not alter predictions from the same verified model and image bytes.
        selection_path=output/'checkpoint_selection.json'
        if selection_path.is_symlink():raise ValueError('Unsafe checkpoint provenance link')
        if provenance is not None:atomic_json(selection_path,provenance)
        observation=json.loads(selection_path.read_text()) if selection_path.is_file() else None
        cache=output/'predictions'
        if cache.is_symlink():raise ValueError('Unsafe prediction cache directory')
        cache.mkdir(exist_ok=True)
        reports=[];started=time.monotonic()
        for index,name in enumerate(names,1):
            path=cache/(name+'.json')
            if path.exists():
                record=read_envelope(path,identity);verify_record(record,name,sources[name],recipe);status='cached'
            else:
                raw=image_reader(images/name,sources[name])
                masks=decode_probability(predictor(raw),recipe)
                record=dict(file_name=name,image_sha256=sources[name],masks=masks,instances=len(masks))
                verify_record(record,name,sources[name],recipe);envelope(path,identity,record);status='predicted'
            reports.append(dict(file_name=name,image_sha256=sources[name],instances=record['instances'],
                                cache_sha256=digest(path),completed=True))
            atomic_json(output/'progress.json',dict(status='running',identity=identity,
                images_completed=index,images_total=len(names),last_image=name))
            print(f'Test [{index}/{len(names)}] {name}: {status}, {record["instances"]} instances',flush=True)
        csv_path=output/csv_name
        temporary=csv_path.with_suffix('.pending.csv');count=0
        try:
            with temporary.open('w',newline='',encoding='utf-8') as stream:
                writer=csv.writer(stream);writer.writerow(['filament_id','segmentation_rle'])
                for name in names:
                    record=read_envelope(cache/(name+'.json'),identity)
                    masks=verify_record(record,name,sources[name],recipe)
                    for number,mask in enumerate(masks,1):
                        writer.writerow([f'{Path(name).stem}_{number}',mask['counts'].decode('ascii')]);count+=1
            # Validate the written interchange format, without altering masks.
            validate_csv(temporary,names,count,recipe)
            temporary.replace(csv_path)
        finally:temporary.unlink(missing_ok=True)
        result=dict(status='complete',identity=identity,output_root=str(output),
            images_completed=len(names),images_total=len(names),prediction_instances=count,
            empty_images=[r['file_name'] for r in reports if not r['instances']],
            checkpoint_epoch=100,checkpoint_sha256=protocol['checkpoint_sha256'],recipe=recipe,
            csv=str(csv_path),csv_sha256=digest(csv_path),elapsed_seconds=time.monotonic()-started,
            validation_reference=protocol.get('validation_reference'),
            checkpoint_selection=observation,
            scope='Native-only test predictions; no test ground truth or test PQ is available.',
            overlap_pixels=0,per_image=reports)
        if 'checkpoint_native_validation' in protocol:
            result['checkpoint_native_validation'] = protocol['checkpoint_native_validation']
        atomic_json(output/'submission_manifest.json',result)
        atomic_json(output/'progress.json',dict(status='complete',identity=identity,
                                               images_completed=len(names),images_total=len(names)))
        archive=output/archive_name;pending=archive.with_suffix('.pending.zip')
        try:
            with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as z:
                for name in ('identity.json','submission_manifest.json','progress.json',csv_path.name):
                    z.write(output/name,name)
                if selection_path.is_file():z.write(selection_path,selection_path.name)
                for name in names:
                    path=cache/(name+'.json');z.write(path,'predictions/'+path.name)
                for name in source_files:z.write(Path(__file__).with_name(name),'code/'+name)
            pending.replace(archive)
        finally:pending.unlink(missing_ok=True)
        result.update(archive=str(archive),archive_sha256=digest(archive))
        atomic_json(output/'latest_result.json',result)
        return result


def validate_csv(path,names,expected_instances,recipe=None):
    recipe = RECIPE if recipe is None else recipe
    allowed={Path(n).stem for n in names};groups={};seen=set()
    with Path(path).open(newline='',encoding='utf-8') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=['filament_id','segmentation_rle']:raise ValueError('Invalid submission columns')
        for row in reader:
            if set(row)!={'filament_id','segmentation_rle'} or any(v is None for v in row.values()):
                raise ValueError('Invalid submission row')
            identifier=row['filament_id'];stem,separator,number=identifier.rpartition('_')
            if not separator or stem not in allowed or not number.isdigit() or int(number)<1 or identifier in seen:
                raise ValueError('Unexpected/duplicate submission identifier')
            if number!=str(len(groups.get(stem,[]))+1):raise ValueError('Nonsequential instance identifier')
            seen.add(identifier);groups.setdefault(stem,[]).append(row['segmentation_rle'])
    if len(seen)!=expected_instances:raise ValueError('Submission instance count changed')
    for masks in groups.values():
        if any(int(mu.area(mask))<recipe['min_area'] for mask in validate_masks(masks)):
            raise ValueError('Submission instance is below minimum area')


def run(training_root,test_images,output,checkpoint):
    metadata=training_metadata(training_root,checkpoint)
    images=Path(test_images or metadata['config']['test_images']).resolve()
    output=Path(output).resolve();training_root=Path(training_root).resolve()
    if output==training_root or output in training_root.parents or training_root in output.parents:
        raise ValueError('Submission output must be separate from the training directory')
    if Path(metadata['checkpoint']).resolve().is_relative_to(output):
        raise ValueError('Submission output must not contain the recovered checkpoint')
    sources=inventory(images)
    if not torch.cuda.is_available():raise RuntimeError('Select a Colab T4 GPU runtime for model inference')
    configure_cuda();cv2.setNumThreads(2)
    runtime=runtime_identity()
    protocol=dict(version=VERSION,training_identity=TRAINING_IDENTITY,checkpoint_sha256=CHECKPOINT_SHA,
        recipe=RECIPE,source_images=sources,test_names_sha256=TEST_NAMES_SHA,
        source_code={name:digest(Path(__file__).with_name(name)) for name in SOURCE_FILES},
        runtime=runtime,validation_runtime=metadata['training_runtime'],
        runtime_differences={k:dict(validation=v,inference=runtime.get(k))
                            for k,v in metadata['training_runtime'].items() if runtime.get(k)!=v},
        validation_reference=metadata['validation'])
    print('Inference only: SWEFil epoch100, native1024, limb, threshold0.65, minimum area300.',flush=True)
    model=None;device=torch.device('cuda')
    def predict(raw):
        nonlocal model
        if model is None:
            payload=load_reviewed_checkpoint(metadata['checkpoint'])
            model=base.build_model(RECIPE['encoder'],pretrained=False)
            model.load_state_dict(payload['model_state_dict'],strict=True)
            del payload
            model=model.to(device).eval()
        return base.predict_foreground(model,base.limb(raw),device,
            tile=1024,overlap=.25,batch=1,amp_dtype=torch.float16)
    return _run_images(images,output,protocol,predict,provenance=metadata.get('checkpoint_selection'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root',type=Path,required=True)
    parser.add_argument('--test-images',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--result',type=Path)
    parser.add_argument('--checkpoint',type=Path,required=True,
                        help='Local verified copy of the exact reviewed SWEFil epoch100 weight file')
    args=parser.parse_args()
    result=run(args.training_root,args.test_images,args.output,args.checkpoint)
    if args.result:atomic_json(args.result,result)
    print(json.dumps({k:v for k,v in result.items() if k!='per_image'},indent=2),flush=True)


if __name__=='__main__':main()
