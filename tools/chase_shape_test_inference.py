"""Inference-only test submission for the reviewed CHASE-shape native epoch82.

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

VERSION = 'chase_shape_best82_native_test_v1'
CHECKPOINT_SHA = '8ea1198734dd434c42889aee2c79928afd8eb8ce4491e826c13d07e41e03d330'
TRAINING_IDENTITY = '9463860fe78c703085e279e40bf0bc950e177b9626d7e38074cbcb63b08dbc00'
TEST_NAMES_SHA = 'f9b2272771d3f6666f6ef812f79968a27f3d448ac2f277eda46c854ad38d598f'
EXPECTED_IMAGES = 180
SOURCE_FILES = ('chase_shape_test_inference.py', 'solar_a100_train_standalone.py',
                'chase_best84_training.py', 'chase_independent_decode.py')
RECIPE = dict(encoder='tu-convnext_tiny', preprocessing='limb_only', tile=1024,
              overlap=.25, infer_batch=1, precision='float16', threshold=.35,
              min_area=300, connectivity=8, tta=False, yolo=False,
              public_predictions=False, checkpoint_epoch=82)


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


def training_metadata(training_root):
    """Read only reviewed metadata; never execute the stored training snapshot."""
    root=Path(training_root).resolve();train=root/'train'
    identity=read_envelope(train/'identity.json',TRAINING_IDENTITY)
    if fingerprint(identity)!=TRAINING_IDENTITY:
        raise ValueError('This submission requires the reviewed CHASE-shape training run')
    state=read_envelope(train/'state.json',TRAINING_IDENTITY)
    if state.get('best_epoch')!=82 or state.get('best',{}).get('sha256')!=CHECKPOINT_SHA:
        raise ValueError('Expected the reviewed best_epoch082 checkpoint; refusing another model')
    relative=state['best']['path']
    checkpoint=(train/relative).resolve()
    if Path(relative).is_absolute() or not checkpoint.is_relative_to(train.resolve()):
        raise ValueError('Unsafe checkpoint path')
    if (train/relative).is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError('Best epoch82 weight file is missing: '+str(checkpoint))
    if digest(checkpoint)!=CHECKPOINT_SHA:
        raise ValueError('Epoch82 checkpoint checksum differs')
    cfg=json.loads((train/'config.json').read_text())
    required=dict(encoder=RECIPE['encoder'],crop=1024,validation_tile=1024,
                  preprocessing='limb_only',overlap=.25,infer_batch=1)
    if any(cfg.get(k)!=v or identity['settings'].get(k)!=v for k,v in required.items()):
        raise ValueError('Saved training/evaluation recipe differs from reviewed native inference')
    best=json.loads((train/'best_epoch.json').read_text())
    if best.get('epoch')!=82 or best['best']['threshold']!=.35 or best['best']['min_area']!=300:
        raise ValueError('Best-checkpoint selection metadata differs')
    if abs(best['best']['pq']-state['best_pq'])>1e-12:
        raise ValueError('Best validation metric and committed state differ')
    for name in SOURCE_FILES[1:]:
        source=Path(__file__).with_name(name)
        if not source.is_file() or digest(source)!=identity['code'].get(name):
            raise ValueError('Inference helper differs from the evaluated training code: '+name)
    return dict(root=str(root),checkpoint=str(checkpoint),checkpoint_sha256=CHECKPOINT_SHA,
                training_identity=TRAINING_IDENTITY,validation=best['best'],
                training_runtime=identity['runtime'],config=cfg)


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


def decode_probability(probability):
    if (not isinstance(probability,np.ndarray) or probability.shape!=(2048,2048)
            or not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1):
        raise ValueError('Expected finite native2048 foreground probabilities in [0,1]')
    masks=[mu.encode(np.asfortranarray(mask.astype(np.uint8)))
           for mask in base.iter_components(probability,RECIPE['threshold'],RECIPE['min_area'])]
    return [dict(size=[2048,2048],counts=mask['counts'].decode('ascii'))
            for mask in validate_masks(masks)]


def verify_record(record, name, source_sha):
    if record.get('file_name')!=name or record.get('image_sha256')!=source_sha:
        raise ValueError('Cached prediction belongs to another source image')
    masks=validate_masks(record['masks'])
    if record.get('instances')!=len(masks) or any(int(mu.area(m))<RECIPE['min_area'] for m in masks):
        raise ValueError('Cached instance count/area differs')
    return masks


@contextmanager
def locked(output):
    with (output/'.inference.lock').open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Another test inference process is using this output') from exc
        yield


def _run_images(images, output, protocol, predictor, image_reader=read_image):
    """Testable persistence engine. Public run fixes the real180/model recipe."""
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
            if any((output/n).exists() for n in ('predictions','latest_result.json','submission_chase_shape_best82_native.csv')):
                raise ValueError('Existing inference output has no identity; use a new folder')
            envelope(identity_path,identity,protocol)
        cache=output/'predictions'
        if cache.is_symlink():raise ValueError('Unsafe prediction cache directory')
        cache.mkdir(exist_ok=True)
        reports=[];started=time.monotonic()
        for index,name in enumerate(names,1):
            path=cache/(name+'.json')
            if path.exists():
                record=read_envelope(path,identity);verify_record(record,name,sources[name]);status='cached'
            else:
                raw=image_reader(images/name,sources[name])
                masks=decode_probability(predictor(raw))
                record=dict(file_name=name,image_sha256=sources[name],masks=masks,instances=len(masks))
                verify_record(record,name,sources[name]);envelope(path,identity,record);status='predicted'
            reports.append(dict(file_name=name,image_sha256=sources[name],instances=record['instances'],
                                cache_sha256=digest(path),completed=True))
            atomic_json(output/'progress.json',dict(status='running',identity=identity,
                images_completed=index,images_total=len(names),last_image=name))
            print(f'Test [{index}/{len(names)}] {name}: {status}, {record["instances"]} instances',flush=True)
        csv_path=output/'submission_chase_shape_best82_native.csv'
        temporary=csv_path.with_suffix('.pending.csv');count=0
        try:
            with temporary.open('w',newline='',encoding='utf-8') as stream:
                writer=csv.writer(stream);writer.writerow(['filament_id','segmentation_rle'])
                for name in names:
                    record=read_envelope(cache/(name+'.json'),identity)
                    masks=verify_record(record,name,sources[name])
                    for number,mask in enumerate(masks,1):
                        writer.writerow([f'{Path(name).stem}_{number}',mask['counts'].decode('ascii')]);count+=1
            # Validate the written interchange format, without altering masks.
            validate_csv(temporary,names,count)
            temporary.replace(csv_path)
        finally:temporary.unlink(missing_ok=True)
        result=dict(status='complete',identity=identity,output_root=str(output),
            images_completed=len(names),images_total=len(names),prediction_instances=count,
            empty_images=[r['file_name'] for r in reports if not r['instances']],
            checkpoint_epoch=82,checkpoint_sha256=protocol['checkpoint_sha256'],recipe=RECIPE,
            csv=str(csv_path),csv_sha256=digest(csv_path),elapsed_seconds=time.monotonic()-started,
            validation_reference=protocol.get('validation_reference'),
            scope='Native-only test predictions; no test ground truth or test PQ is available.',
            overlap_pixels=0,per_image=reports)
        atomic_json(output/'submission_manifest.json',result)
        atomic_json(output/'progress.json',dict(status='complete',identity=identity,
                                               images_completed=len(names),images_total=len(names)))
        archive=output/'chase_shape_best82_test_results.zip';pending=archive.with_suffix('.pending.zip')
        try:
            with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as z:
                for name in ('identity.json','submission_manifest.json','progress.json',csv_path.name):
                    z.write(output/name,name)
                for name in names:
                    path=cache/(name+'.json');z.write(path,'predictions/'+path.name)
                for name in SOURCE_FILES:z.write(Path(__file__).with_name(name),'code/'+name)
            pending.replace(archive)
        finally:pending.unlink(missing_ok=True)
        result.update(archive=str(archive),archive_sha256=digest(archive))
        atomic_json(output/'latest_result.json',result)
        return result


def validate_csv(path,names,expected_instances):
    allowed={Path(n).stem for n in names};groups={};seen=set()
    with Path(path).open(newline='',encoding='utf-8') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=['filament_id','segmentation_rle']:raise ValueError('Invalid submission columns')
        for row in reader:
            identifier=row['filament_id'];stem,separator,number=identifier.rpartition('_')
            if not separator or stem not in allowed or not number.isdigit() or int(number)<1 or identifier in seen:
                raise ValueError('Unexpected/duplicate submission identifier')
            seen.add(identifier);groups.setdefault(stem,[]).append(row['segmentation_rle'])
    if len(seen)!=expected_instances:raise ValueError('Submission instance count changed')
    for masks in groups.values():validate_masks(masks)


def run(training_root,test_images,output):
    metadata=training_metadata(training_root)
    images=Path(test_images or metadata['config']['test_images']).resolve()
    output=Path(output).resolve();training_root=Path(training_root).resolve()
    if output==training_root or output in training_root.parents or training_root in output.parents:
        raise ValueError('Submission output must be separate from the training directory')
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
    print('Inference only: epoch82, native1024, limb, threshold0.35, minimum area300.',flush=True)
    print('No training, synthesis, YOLO, TTA, public predictions or threshold search.',flush=True)
    model=None;device=torch.device('cuda')
    def predict(raw):
        nonlocal model
        if model is None:
            checkpoint=torch.load(metadata['checkpoint'],map_location='cpu',weights_only=True)
            if checkpoint.get('identity')!=TRAINING_IDENTITY or checkpoint.get('epoch')!=82:
                raise ValueError('Checkpoint payload differs from reviewed epoch82')
            validation=checkpoint.get('validation',{})
            if validation.get('threshold')!=.35 or validation.get('min_area')!=300:
                raise ValueError('Checkpoint selected decoding recipe differs')
            model=base.build_model(RECIPE['encoder'],pretrained=False)
            model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            del checkpoint
            model=model.to(device).eval()
        return base.predict_foreground(model,base.limb(raw),device,
            tile=1024,overlap=.25,batch=1,amp_dtype=torch.float16)
    return _run_images(images,output,protocol,predict)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root',type=Path,required=True)
    parser.add_argument('--test-images',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--result',type=Path)
    args=parser.parse_args()
    result=run(args.training_root,args.test_images,args.output)
    if args.result:atomic_json(args.result,result)
    print(json.dumps({k:v for k,v in result.items() if k!='per_image'},indent=2),flush=True)


if __name__=='__main__':main()
