"""Optional inference-only CHASE transfer diagnosis; never edits training files."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import sys

MODE = 'raw_original_gray_v1'
VERSION = '057b2b49a0cdf3e0'
MANIFEST_SHA = '31145ebed1889291a129f8ae3ed6c40833080e233a65ae094ba02245f4c615e2'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def input_protocol(*, checkpoint_sha, model_sha, image_hashes, frozen_sources,
                   runtime, batch=1, mode=MODE, helper_sha=None):
    if mode not in (MODE, 'limb_v1'):
        raise ValueError('Unknown input mode')
    return dict(schema_version=1, input_mode=mode, checkpoint_sha256=checkpoint_sha,
        model_sha256=model_sha, source_images=image_hashes, sources=frozen_sources,
        helper_sha256=helper_sha or digest(__file__), runtime=runtime, native_batch=batch,
        tile=1024, overlap=.25, views=['base', 'hflip', 'vflip', 'hvflip'],
        normalization='unchanged RGB-repeat /255 and ImageNet mean/std',
        geometry='original2048 coordinates; no extra warp/crop/upscale',
        foreground_thresholds=[.2,.35,.5,.65,.8], component_min_area=100,
        scope='inference only, new epoch60 native weights; no parameter updates')


def predict_native_record(model, image, filename, image_sha, identity, *,
                          device, batch=1, mode=MODE):
    import numpy as np
    import torch
    import submission_inference as inference
    if image.shape != (2048,2048) or image.dtype != np.uint8:
        raise ValueError('Expected original uint8 grayscale2048 image')
    if mode not in (MODE, 'limb_v1'):
        raise ValueError('Unknown input mode')
    pixels = image if mode == MODE else inference.limb(image)
    amp = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type == 'cuda' else None
    model.eval()
    with torch.inference_mode():
        heads, average, unions = inference.predict_views(model, pixels, device,
            tile=1024, overlap=.25, batch=batch, amp_dtype=amp)
    record = inference.prediction_record(heads, filename, image_sha, identity)
    record['tta'] = inference.prediction_record(average, filename, image_sha, identity)
    record['view_unions'] = unions
    return record


def verify_sources(root):
    code = root/'code'/VERSION
    manifest = code/'source_manifest.json'
    if digest(manifest) != MANIFEST_SHA:
        raise ValueError('Expected unchanged PNG-fix training source manifest')
    data = json.loads(manifest.read_text())
    for folder, names in ((code,data['sources']), (root/'assets'/VERSION,data['assets'])):
        for name, sha in names.items():
            if digest(folder/name) != sha:
                raise ValueError('Frozen code/asset changed: '+str(folder/name))
    return code, data


def run(root, *, device='cuda', batch=1):
    root = Path(root).resolve()
    if root.name != 'chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1':
        raise ValueError('Use the existing PNG-fix experiment root')
    if type(batch) is not int or batch < 1:
        raise ValueError('Positive native batch required')
    code, manifest = verify_sources(root)
    sys.path.insert(0, str(code))
    import cv2
    import numpy as np
    import torch
    import solar_a100_train_standalone as legacy
    import chase_best84_training as training
    import chase_best84_evaluation as evaluation
    from submission_driver import (atomic_json, export_csv, finalize_frame,
        read_envelope, validate_predictions, write_envelope)

    selected_device = torch.device(device)
    if selected_device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This optional inference phase requires a Colab GPU')
    output = root/'input_diagnostic_v1'
    output.mkdir(parents=True, exist_ok=True)
    with (output/'raw_native.lock').open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Raw native diagnosis already running')
        comparison = json.loads((root/'comparison/epoch060_comparison.json').read_text())
        rows = {r['role']:r for r in comparison['results']}
        original = rows['old_yolo_new_native']
        result = json.loads((root/'native/result.json').read_text())
        checkpoint = root/'native/last.pt'
        checkpoint_sha = digest(checkpoint)
        if result.get('epochs_completed') != 60 or checkpoint_sha != original['native_checkpoint_sha256']:
            raise ValueError('Expected the exact completed native epoch60 checkpoint')
        cfg = json.loads((root/'configs/comparison.json').read_text())
        cohort = evaluation.load_cohort(cfg)
        caches = {name:evaluation.load_yolo_cache(path, cohort['names']) for name,path in (
            ('old_yolo',cfg['yolo_cache']), ('new_yolo',root/'comparison/yolo_epoch060.json.gz'))}
        for role,yolo in caches.items():
            audit = json.loads((root/'comparison/validation'/
                ('epoch060_'+role+'_new_native')/'identity.json').read_text())
            if (evaluation.fingerprint(audit['protocol']) != rows[role+'_new_native']['identity']
                    or audit['protocol']['yolo_cache_sha256'] != yolo['sha256']
                    or audit['protocol']['data_sha256'] != cohort['source_sha256']):
                raise ValueError('YOLO cache/validation cohort changed since comparison')
        images = caches['old_yolo']['image_sha256']
        if caches['new_yolo']['image_sha256'] != images:
            raise ValueError('YOLO caches use different source images')
        for name,sha in images.items():
            if digest(Path(cfg['images'])/name) != sha:
                raise ValueError('Original validation image changed: '+name)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if payload.get('epoch') != 60 or payload.get('identity') != result.get('identity'):
            raise ValueError('Native checkpoint epoch/identity changed')
        training.configure_cuda()
        model = legacy.build_model(encoder='tu-convnext_tiny', pretrained=False)
        model.load_state_dict(payload['model_state_dict'], strict=True)
        del payload
        model_sha = evaluation.model_digest(model)
        if model_sha != original['model_sha256']:
            raise ValueError('Native tensors do not match epoch60 comparison')
        model = model.to(selected_device).eval()
        protocol = input_protocol(checkpoint_sha=checkpoint_sha, model_sha=model_sha,
            image_hashes=images, frozen_sources=manifest['sources'],
            runtime=evaluation.runtime_identity(str(selected_device)), batch=batch)
        identity = evaluation.fingerprint(protocol)
        cache = output/'raw_native'/identity
        cache.mkdir(parents=True, exist_ok=True)
        identity_path = cache/'identity.json'
        metadata = dict(identity=identity,protocol=protocol)
        if identity_path.exists() and json.loads(identity_path.read_text()) != metadata:
            raise ValueError('Raw input cache identity changed; preserve existing cache')
        if not identity_path.exists() and any(cache.glob('*.json.gz')):
            raise ValueError('Unidentified raw native cache; preserve it')
        atomic_json(identity_path,metadata)
        predictions = {role:{} for role in caches}
        reused = 0
        for index,filename in enumerate(cohort['names'],1):
            stem = Path(filename).stem
            cached = cache/(stem+'.json.gz')
            record = read_envelope(cached,identity,images[filename])
            if evaluation._native_valid(record,stem,filename,images[filename],identity):
                reused += 1
            else:
                path = Path(cfg['images'])/filename
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != images[filename]:
                    raise ValueError('Source image changed during inference: '+filename)
                pixels = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
                if pixels is None: raise ValueError('Could not decode '+filename)
                record = predict_native_record(model,pixels,filename,images[filename],identity,
                    device=selected_device,batch=batch)
                if not evaluation._native_valid(record,stem,filename,images[filename],identity):
                    raise ValueError('Invalid raw native record: '+filename)
                write_envelope(cached,record,identity,images[filename])
            for role,yolo in caches.items():
                predictions[role][stem] = validate_predictions(finalize_frame(stem,
                    dict(yolo['frames'][stem],native=record)))
            print(f'Raw native inference [{index}/91] {filename}',flush=True)
        results = []
        for role,predicted in predictions.items():
            destination = cache/(role+'_raw_native_predictions.csv')
            exported = export_csv(destination,[dict(file_name=name,
                rles=predicted[Path(name).stem]) for name in cohort['names']])
            metric = evaluation.score_predictions(cohort['groups'],predicted)
            reference = rows[role+'_new_native']
            results.append(dict(role=role+'_raw_new_native',**metric,prediction_rows=exported,
                baseline_limb_pq=reference['pq'],delta_pq_vs_limb=metric['pq']-reference['pq'],
                csv_path=str(destination),csv_sha256=digest(destination)))
        summary = dict(schema_version=1,mode=MODE,native_identity=identity,
            native_cache_root=str(cache),model_sha256=model_sha,source_images=images,
            original_native_identity=original['native_identity'],checkpoint_sha256=checkpoint_sha,
            physical_images=91,reused_images=reused,results=results,
            scope='fixed development validation, not independent holdout or leaderboard',
            weights_updated=False,yolo_reinferred=False)
        if evaluation.model_digest(model) != model_sha or digest(checkpoint) != checkpoint_sha:
            raise ValueError('Native model/checkpoint changed during inference')
        atomic_json(cache/'summary.json',summary)
        atomic_json(output/'raw_native_summary.json',summary)
        print(json.dumps({k:summary[k] for k in ('mode','results','native_cache_root')},indent=2),flush=True)
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch',type=int,default=1)
    args = parser.parse_args()
    run(args.root,device=args.device,batch=args.batch)
