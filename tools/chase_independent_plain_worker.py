"""Independent inference worker delivered as readable Python, without encoded source."""
import os
os.environ.setdefault('TORCH_CUDNN_V8_API_DISABLED', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse, fcntl, hashlib, importlib.metadata, json, sys
from pathlib import Path

VERSION = '057b2b49a0cdf3e0'
MANIFEST_SHA = '31145ebed1889291a129f8ae3ed6c40833080e233a65ae094ba02245f4c615e2'
ROLES = ('old_yolo', 'new_yolo', 'old_native', 'new_native')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def decode_native(record, driver):
    # prediction_record uses base FG0.5, 8-connectivity, minimum area100.
    return driver.validate_predictions([c['rle'] for c in record['components']['0.5']])


def decode_yolo(record, driver, native, mu):
    output = []; covered = None
    for proposal in sorted(record['instances'], key=lambda p: -p['confidence']):
        if proposal['confidence'] < .2: continue
        rle = native._rle(dict(size=[2048, 2048], counts=proposal['rle']))
        if int(mu.area(rle)) == 0: continue
        candidate = native._trim(rle, covered)
        if candidate is not None:
            output.append(candidate); covered = mu.merge(output)
    return driver.validate_predictions(output)


def select_weight(root, role, cfg, rows):
    if role.endswith('native'):
        row = rows['old_yolo_'+('old_native' if role == 'old_native' else 'new_native')]
        path = Path(cfg['initial_checkpoint']) if role == 'old_native' else root/'native/last.pt'
        state = None if role == 'old_native' else json.loads((root/'native/result.json').read_text())
        if state and state['epochs_completed'] != 60: raise ValueError('Native epoch60 is required')
        return path, row['native_checkpoint_sha256'], row['model_sha256'], state
    if role == 'old_yolo':
        return Path(cfg['yolo_checkpoint']), '81c2a27d95821b5ede283169c0da8c08d360ecc4c2a60a1b9c6baf96d76bcccc', None, None
    state = json.loads((root/'yolo/result.json').read_text())
    if state['epochs_completed'] != 60: raise ValueError('YOLO epoch60 is required')
    return root/'yolo/epoch060.pt', state['checkpoint_sha256'], None, state


def run(root, role):
    root = Path(root).resolve()
    if root.name != 'chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1' or role not in ROLES:
        raise ValueError('Unexpected experiment or model role')
    code = root/'code'/VERSION
    if sha(code/'source_manifest.json') != MANIFEST_SHA: raise ValueError('Frozen source manifest differs')
    manifest = json.loads((code/'source_manifest.json').read_text())
    for folder, hashes in ((code, manifest['sources']), (root/'assets'/VERSION, manifest['assets'])):
        for name, expected in hashes.items():
            if sha(folder/name) != expected: raise ValueError('Changed frozen file: '+name)
    if {p.name for p in code.glob('*.py')} != set(manifest['sources']):
        raise ValueError('Unexpected module in frozen source folder')
    sys.path.insert(0, str(code))
    import cv2, numpy as np, torch
    from pycocotools import mask as mu
    import submission_driver as driver
    import submission_inference as inference
    import submission_native_recipe as native
    import chase_best84_evaluation as evaluation
    if not torch.cuda.is_available(): raise RuntimeError('Select a Colab T4 GPU runtime')
    device = torch.device('cuda'); cv2.setNumThreads(1)
    output = root/'independent_models_plain_v1'/role
    output.mkdir(parents=True, exist_ok=True)
    with (output/'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = json.loads((root/'configs/comparison.json').read_text())
        cohort = evaluation.load_cohort(cfg)
        rows = {r['role']: r for r in json.loads((root/'comparison/epoch060_comparison.json').read_text())['results']}
        control = json.loads((root/'comparison/validation/epoch060_old_yolo_old_native/identity.json').read_text())
        if evaluation.fingerprint(control['protocol']) != rows['old_yolo_old_native']['identity']:
            raise ValueError('Original comparison identity differs')
        if control['protocol']['data_sha256'] != cohort['source_sha256']:
            raise ValueError('Validation annotations/split differ')
        images = control['protocol']['source_images']
        if set(images) != set(cohort['names']) or len(images) != 91: raise ValueError('Expected validation91')
        for name, expected in images.items():
            if sha(Path(cfg['images'])/name) != expected: raise ValueError('Changed input image: '+name)
        checkpoint, checkpoint_sha, tensor_sha, state = select_weight(root, role, cfg, rows)
        if sha(checkpoint) != checkpoint_sha: raise ValueError('Checkpoint hash differs: '+str(checkpoint))
        is_native = role.endswith('native')
        if is_native:
            import solar_a100_train_standalone as legacy
            import chase_best84_training as training
            training.configure_cuda()
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if saved['epoch'] != (84 if role == 'old_native' else 60): raise ValueError('Wrong native epoch')
            if state and saved.get('identity') != state['identity']: raise ValueError('Wrong native training identity')
            model = legacy.build_model(encoder='tu-convnext_tiny', pretrained=False)
            model.load_state_dict(saved['model_state_dict'], strict=True); del saved
            if evaluation.model_digest(model) != tensor_sha: raise ValueError('Native tensor hash differs')
            model = model.to(device).eval()
        else:
            model = inference.InferenceProvider(checkpoint, None, device=str(device), verify_weights=False)
            yolo = model._yolo()
            if yolo.task != 'segment' or len(yolo.names) != 1: raise ValueError('Expected one-class YOLO segmentation')
            if state and (yolo.ckpt.get('completed_epoch') != 60 or yolo.ckpt.get('chase_yolo_identity') != state['identity']):
                raise ValueError('Wrong YOLO completed checkpoint')
            yolo.model.eval(); tensor_sha = evaluation.model_digest(yolo.model)
        runtime = evaluation.runtime_identity(str(device))
        for package in ('ultralytics', 'torchvision'):
            runtime[package] = importlib.metadata.version(package)
        conditions = {}
        for mode in ('raw', 'limb'):
            protocol = dict(schema_version=1, model_role=role, input_mode=mode,
                checkpoint_sha256=checkpoint_sha, model_sha256=tensor_sha,
                helper_sha256=sha(__file__), sources=manifest['sources'], runtime=runtime,
                source_images=images, data_sha256=cohort['source_sha256'],
                input='original grayscale' if mode == 'raw' else 'frozen limb gain + outside-disk zero',
                view='base', native_tile=1024, native_overlap=.25, native_batch=1,
                native_normalization='unchanged ImageNet normalization',
                native_decode='FG>=0.5;8-connected components;area>=100',
                yolo_inference='imgsz2048;conf0.05;NMS IoU0.7;max_det100;retina_masks',
                yolo_decode='confidence>=0.2;stable descending confidence pixel ownership',
                fusion=False, tta=False, training=False)
            identity = evaluation.fingerprint(protocol); folder = output/identity
            (folder/'frames').mkdir(parents=True, exist_ok=True)
            metadata = dict(identity=identity, protocol=protocol); path = folder/'identity.json'
            if path.exists() and json.loads(path.read_text()) != metadata: raise ValueError('Cache identity differs')
            driver.atomic_json(path, metadata)
            conditions[mode] = dict(identity=identity, folder=folder, protocol=protocol, predictions={}, reused=0)
        for index, name in enumerate(cohort['names'], 1):
            stem = Path(name).stem; image = None
            for mode, item in conditions.items():
                path = item['folder']/'frames'/(stem+'.json.gz')
                record = driver.read_envelope(path, item['identity'], images[name])
                if record is not None:
                    if record.get('file_name') != name or record.get('model_role') != role or record.get('input_mode') != mode:
                        raise ValueError('Cached model/input role differs')
                    raw = record['raw_prediction']; item['reused'] += 1
                else:
                    if image is None:
                        data = (Path(cfg['images'])/name).read_bytes()
                        if hashlib.sha256(data).hexdigest() != images[name]: raise ValueError('Input changed during inference')
                        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
                        if image is None or image.shape != (2048, 2048) or image.dtype != np.uint8:
                            raise ValueError('Expected original2048 grayscale image')
                    pixels = image if mode == 'raw' else inference.limb(image)
                    with torch.inference_mode():
                        if is_native:
                            amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                            heads = inference.predict_heads(model, pixels, device, tile=1024, overlap=.25, batch=1, amp_dtype=amp)
                            raw = inference.prediction_record(heads, name, images[name], item['identity'])
                            raw['foreground_low_threshold_unions'] = {str(t): inference.rle(heads[0] >= t) for t in (.05, .1)}
                        else:
                            raw = inference.InferenceProvider._whole_record(model._result(pixels, 2048), stem, 'base', 2048)
                    record = dict(model_role=role, input_mode=mode, file_name=name, raw_prediction=raw)
                    driver.write_envelope(path, record, item['identity'], images[name])
                masks = decode_native(raw, driver) if is_native else decode_yolo(raw, driver, native, mu)
                item['predictions'][stem] = masks
                print(f'{role}/{mode} [{index}/91] {name}', flush=True)
        results = []
        for mode, item in conditions.items():
            csv = item['folder']/'predictions.csv'
            driver.export_csv(csv, [dict(file_name=n, rles=item['predictions'][Path(n).stem]) for n in cohort['names']])
            metrics = evaluation.score_predictions(cohort['groups'], item['predictions'])
            row = dict(metrics, model_role=role, input_mode=mode, identity=item['identity'],
                cache_root=str(item['folder']), csv_sha256=sha(csv), reused_images=item['reused'],
                physical_images=91, annotation_groups=101, gt_instances=708, training=False, fusion=False)
            driver.atomic_json(item['folder']/'metrics.json', row); results.append(row)
        if sha(checkpoint) != checkpoint_sha: raise ValueError('Checkpoint changed during inference')
        if is_native and evaluation.model_digest(model) != tensor_sha: raise ValueError('Native tensors changed')
        driver.atomic_json(output/'role_results.json', dict(status='complete', model_role=role, results=results))
        print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    parser.add_argument('--role', choices=ROLES, required=True)
    args = parser.parse_args(); run(args.root, args.role)
