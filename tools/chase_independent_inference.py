"""Independent old/new YOLO/native inference; never fuses the two models."""
import os
os.environ.setdefault('TORCH_CUDNN_V8_API_DISABLED','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import sys
import time

VERSION='057b2b49a0cdf3e0'
MANIFEST_SHA='31145ebed1889291a129f8ae3ed6c40833080e233a65ae094ba02245f4c615e2'
ROOT_NAME='chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1'
OLD_NATIVE_SHA='ae6b209f72355fd52d2cb07182e69028e97c703a1977ad1e7b38b27cf8cebe69'
OLD_YOLO_SHA='81c2a27d95821b5ede283169c0da8c08d360ecc4c2a60a1b9c6baf96d76bcccc'
ROLES=('old_yolo','new_yolo','old_native','new_native')
MODES=('raw','limb')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_sources(root):
    code=root/'code'/VERSION
    manifest=code/'source_manifest.json'
    if digest(manifest)!=MANIFEST_SHA:
        raise ValueError('Expected unchanged PNG-fix source manifest')
    value=json.loads(manifest.read_text())
    for folder,names in ((code,value['sources']),(root/'assets'/VERSION,value['assets'])):
        for name,sha in names.items():
            if digest(folder/name)!=sha:
                raise ValueError('Frozen code/asset changed: '+str(folder/name))
    return code,value


def make_protocol(*,role,mode,checkpoint_sha,model_sha,source_images,sources,
                  data_sha,runtime,helper_sha,decoder_sha,batch=1):
    if role not in ROLES or mode not in MODES:
        raise ValueError('Unknown model role or input mode')
    family='native' if role.endswith('native') else 'yolo'
    return dict(schema_version=1,model_role=role,model_family=family,input_mode=mode,
        checkpoint_sha256=checkpoint_sha,model_sha256=model_sha,
        model_digest_stage='loaded checkpoint before predictor backend setup',
        source_images=source_images,sources=sources,data_sha256=data_sha,runtime=runtime,
        helper_sha256=helper_sha,decoder_sha256=decoder_sha,
        preprocessing=('original grayscale; no limb/CLAHE/warp; not CHASE-domain matched' if mode=='raw'
                       else 'unchanged limb: radial brightness gain AND zero outside radius900 disk'),
        native_inference=dict(view='base',tile=1024,overlap=.25,batch=batch,
            normalization='repeat gray RGB /255, ImageNet mean/std',
            amp='bfloat16 if supported else float16 on CUDA',
            extra_foreground_union_thresholds=[.05,.1]),
        yolo_inference=dict(view='base',imgsz=2048,conf=.05,iou=.7,retina_masks=True,
            max_det=100,backend='unchanged InferenceProvider._result'),
        decode=(dict(foreground_threshold=.5,connectivity=8,min_area=100)
                if family=='native' else
                dict(confidence=.2,order='descending confidence; stable original-index ties',
                     overlap='earlier prediction owns pixel',min_area=1)),
        fusion=False,tta=False,training=False,
        metric='global all-pairs IoU strictly >0.5, 91 physical/101 annotator groups/708 GT')


def infer_record(predictor,image,filename,image_sha,identity,*,role,mode,device,batch=1):
    import numpy as np
    import torch
    import submission_inference as inference
    if role not in ROLES or mode not in MODES:
        raise ValueError('Unknown role/mode')
    if image is None or image.shape!=(2048,2048) or image.dtype!=np.uint8:
        raise ValueError('Expected original uint8 grayscale2048 image')
    pixels=image if mode=='raw' else inference.limb(image)
    with torch.inference_mode():
        if role.endswith('native'):
            predictor.eval()
            amp=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type=='cuda' else None
            heads=inference.predict_heads(predictor,pixels,device,tile=1024,
                overlap=.25,batch=batch,amp_dtype=amp)
            record=inference.prediction_record(heads,filename,image_sha,identity)
            record['foreground_low_threshold_unions']={f'{threshold:g}':inference.rle(heads[0]>=threshold)
                for threshold in (.05,.1)}
            return record
        result=predictor._result(pixels,2048)
        record=predictor._whole_record(result,Path(filename).stem,'base',2048)
        record['raw_box_count']=0 if result.boxes is None else len(result.boxes)
        record['max_det_reached']=record['raw_box_count']>=100
        return record


def decode_record(record,role):
    import chase_independent_decode as decoder
    if role.endswith('native'):
        masks=decoder.native_masks(record['unions']['foreground']['0.5'],min_area=100)
    else:
        masks=decoder.yolo_masks(record,confidence=.2)
    return decoder.validate_masks(masks)


def run_frames(*,role,names,image_hashes,images_root,conditions,predict,
               read_image=None,decode=decode_record):
    """Two independently identified modes; an interrupted image is safely repeated."""
    import cv2
    import numpy as np
    from submission_driver import read_envelope,write_envelope,json_ready
    if read_image is None:
        def read_image(path,sha):
            raw=Path(path).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=sha:
                raise ValueError('Source image changed during inference: '+str(path))
            return cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
    predictions={mode:{} for mode in MODES}
    reused={mode:0 for mode in MODES}
    for index,filename in enumerate(names,1):
        image=None
        for mode in MODES:
            identity,folder=conditions[mode]
            path=folder/'frames'/(Path(filename).stem+'.json.gz')
            value=read_envelope(path,identity,image_hashes[filename])
            valid=(isinstance(value,dict) and value.get('model_role')==role
                and value.get('input_mode')==mode and value.get('file_name')==filename
                and value.get('image_sha256')==image_hashes[filename]
                and isinstance(value.get('raw_prediction'),dict))
            masks=None
            if valid:
                try:
                    masks=decode(value['raw_prediction'],role)
                    if json_ready(masks)!=value.get('predictions'): masks=None
                except (KeyError,ValueError,TypeError,OverflowError): masks=None
            if masks is not None:
                reused[mode]+=1
            else:
                if image is None:
                    image=read_image(Path(images_root)/filename,image_hashes[filename])
                record=predict(image,filename,image_hashes[filename],identity,mode)
                masks=decode(record,role)
                value=dict(schema_version=1,model_role=role,input_mode=mode,
                    file_name=filename,image_sha256=image_hashes[filename],
                    raw_prediction=record,predictions=json_ready(masks))
                write_envelope(path,value,identity,image_hashes[filename])
            predictions[mode][Path(filename).stem]=masks
        print(f'{role} raw/limb [{index}/{len(names)}] {filename}',flush=True)
    return predictions,reused


def select_checkpoint(root,role,cfg,rows):
    if role=='old_native':
        return Path(cfg['initial_checkpoint']),OLD_NATIVE_SHA,rows['old_yolo_old_native']['model_sha256'],None
    if role=='old_yolo':
        return Path(cfg['yolo_checkpoint']),OLD_YOLO_SHA,None,None
    family='native' if role=='new_native' else 'yolo'
    result=json.loads((root/family/'result.json').read_text())
    if result.get('status')!='complete' or result.get('epochs_completed')!=60:
        raise ValueError('Requested new model must have completed epoch60')
    if family=='native':
        return root/'native/last.pt',rows['old_yolo_new_native']['native_checkpoint_sha256'],rows['old_yolo_new_native']['model_sha256'],result['identity']
    return root/'yolo/epoch060.pt',result['checkpoint_sha256'],None,result['identity']


def run(root,role,*,device='cuda',batch=1):
    root=Path(root).resolve()
    if root.name!=ROOT_NAME or role not in ROLES or batch!=1:
        raise ValueError('Use the existing experiment root, a known role and native batch1')
    code,manifest=verify_sources(root)
    sys.path.insert(0,str(code))
    import cv2
    import torch
    import chase_best84_evaluation as evaluation
    import submission_inference as inference
    import chase_independent_decode as decoder
    from submission_driver import atomic_json,export_csv
    selected_device=torch.device(device)
    if selected_device.type!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('Select a Colab T4/A100 GPU runtime for independent inference')
    cv2.setNumThreads(1)
    output=root/'independent_models_v1'/role
    output.mkdir(parents=True,exist_ok=True)
    with (output/'inference.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This model role is already running')
        started=time.monotonic()
        cfg=json.loads((root/'configs/comparison.json').read_text())
        cohort=evaluation.load_cohort(cfg)
        comparison=json.loads((root/'comparison/epoch060_comparison.json').read_text())
        rows={row['role']:row for row in comparison['results']}
        audit=json.loads((root/'comparison/validation/epoch060_old_yolo_old_native/identity.json').read_text())
        if (evaluation.fingerprint(audit['protocol'])!=rows['old_yolo_old_native']['identity']
                or audit['protocol']['data_sha256']!=cohort['source_sha256']):
            raise ValueError('Original validation data/identity changed')
        image_hashes=audit['protocol']['source_images']
        if set(image_hashes)!=set(cohort['names']) or len(image_hashes)!=91:
            raise ValueError('Expected exactly the original91 validation images')
        for filename,sha in image_hashes.items():
            if digest(Path(cfg['images'])/filename)!=sha:
                raise ValueError('Original validation image changed: '+filename)
        checkpoint,expected_sha,expected_model,training_id=select_checkpoint(root,role,cfg,rows)
        if digest(checkpoint)!=expected_sha:
            raise ValueError('Selected checkpoint differs: '+str(checkpoint))
        if role.endswith('native'):
            import chase_best84_training as training
            import solar_a100_train_standalone as legacy
            training.configure_cuda()
            payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
            expected_epoch=84 if role=='old_native' else 60
            if payload.get('epoch')!=expected_epoch or (training_id and payload.get('identity')!=training_id):
                raise ValueError('Native checkpoint epoch/identity differs')
            if role=='old_native' and payload.get('config',{}).get('encoder')!='tu-convnext_tiny':
                raise ValueError('Wrong original native architecture')
            predictor=legacy.build_model(encoder='tu-convnext_tiny',pretrained=False)
            predictor.load_state_dict(payload['model_state_dict'],strict=True)
            del payload
            model_sha=evaluation.model_digest(predictor)
            if model_sha!=expected_model:raise ValueError('Native model tensors differ from completed comparison')
            predictor=predictor.to(selected_device).eval()
        else:
            predictor=inference.InferenceProvider(checkpoint,None,device=str(selected_device),verify_weights=False)
            yolo=predictor._yolo()
            if yolo.task!='segment' or len(yolo.names)!=1:
                raise ValueError('Expected single-class YOLO segmentation weights')
            if role=='new_yolo' and (yolo.ckpt.get('completed_epoch')!=60 or
                    yolo.ckpt.get('chase_yolo_identity')!=training_id):
                raise ValueError('YOLO checkpoint epoch/identity differs')
            yolo.model.eval()
            model_sha=evaluation.model_digest(yolo.model)
        runtime=evaluation.runtime_identity(str(selected_device))
        for package in ('ultralytics','torchvision'):
            try:runtime[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:runtime[package]='not installed'
        conditions={};protocols={}
        for mode in MODES:
            protocol=make_protocol(role=role,mode=mode,checkpoint_sha=expected_sha,
                model_sha=model_sha,source_images=image_hashes,sources=manifest['sources'],
                data_sha=cohort['source_sha256'],runtime=runtime,helper_sha=digest(__file__),
                decoder_sha=digest(decoder.__file__),batch=batch)
            identity=evaluation.fingerprint(protocol)
            folder=output/identity
            (folder/'frames').mkdir(parents=True,exist_ok=True)
            identity_path=folder/'identity.json';metadata=dict(identity=identity,protocol=protocol)
            if identity_path.exists() and json.loads(identity_path.read_text())!=metadata:
                raise ValueError('Independent inference identity changed')
            if not identity_path.exists() and any((folder/'frames').glob('*.gz')):
                raise ValueError('Unidentified cached frames; preserve them')
            atomic_json(identity_path,metadata)
            conditions[mode]=(identity,folder);protocols[mode]=protocol
        def predict(image,filename,sha,identity,mode):
            return infer_record(predictor,image,filename,sha,identity,role=role,
                mode=mode,device=selected_device,batch=batch)
        predicted,reused=run_frames(role=role,names=cohort['names'],image_hashes=image_hashes,
            images_root=cfg['images'],conditions=conditions,predict=predict)
        results=[]
        for mode in MODES:
            identity,folder=conditions[mode]
            metrics=decoder.score_groups(cohort['groups'],predicted[mode])
            csv=folder/'predictions.csv'
            count=export_csv(csv,[dict(file_name=name,rles=predicted[mode][Path(name).stem])
                                  for name in cohort['names']])
            row=dict(metrics,schema_version=1,model_role=role,input_mode=mode,
                identity=identity,cache_root=str(folder),metrics_path=str(folder/'metrics.json'),
                csv_path=str(csv),csv_sha256=digest(csv),prediction_rows=count,
                physical_images=91,annotation_groups=101,gt_instances=708,
                reused_images=reused[mode],elapsed_role_seconds=time.monotonic()-started,
                checkpoint_sha256=expected_sha,model_sha256=model_sha,
                protocol=protocols[mode],fusion=False,training=False,
                scope='fixed development validation; standalone base view with minimal decoding; not leaderboard')
            atomic_json(folder/'metrics.json',row);results.append(row)
        if digest(checkpoint)!=expected_sha:raise ValueError('Checkpoint file changed during inference')
        if role.endswith('native') and evaluation.model_digest(predictor)!=model_sha:
            raise ValueError('Native model tensors changed during inference')
        report=dict(schema_version=1,model_role=role,status='complete',results=results)
        atomic_json(output/'role_results.json',report)
        print(json.dumps([dict(model_role=role,**{k:r[k]for k in
            ('input_mode','pq','sq','rq','tp_pairs','fp','fn')})for r in results],indent=2),flush=True)
        return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--model-role',choices=ROLES,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch',type=int,default=1)
    args=parser.parse_args()
    run(args.root,args.model_role,device=args.device,batch=args.batch)
