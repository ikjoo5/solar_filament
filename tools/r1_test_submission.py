"""Resumable original-images -> frozen YOLO+best84 -> R1 test inference.

No training, annotations, evaluation, or Kaggle submission occurs here. CLI:
python r1_test_submission.py --config config.json [--max-images 10]
The CPU-injectable run() API is used by focused cache/resume tests.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import argparse
import csv
import fcntl
import gc
import gzip
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import zipfile

# Set before any torch/Ultralytics import, including in the isolated child.
os.environ.setdefault('TORCH_CUDNN_V8_API_DISABLED', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import cv2
import numpy as np
from pycocotools import mask as mu
from experimental_consensus_tails import restore_consensus_tails, checked_rle, union_rle
from submission_driver import finalize_frame, validate_predictions, valid_stage, json_ready

YOLO_SHA = '81c2a27d95821b5ede283169c0da8c08d360ecc4c2a60a1b9c6baf96d76bcccc'
NATIVE_SHA = 'ae6b209f72355fd52d2cb07182e69028e97c703a1977ad1e7b38b27cf8cebe69'
OFFICIAL_NAMES_SHA = 'f9b2272771d3f6666f6ef812f79968a27f3d448ac2f277eda46c854ad38d598f'
SOURCE_FILES = ('r1_test_submission.py', 'experimental_consensus_tails.py',
                'submission_inference.py', 'submission_native_process.py', 'native_instance_training.py',
                'submission_driver.py', 'submission_yolo_recipe.py', 'submission_native_recipe.py')
INFERENCE_STAGES = ('geometric', 'preprocessing', 'tiled', 'native')
PACKAGES = ('numpy', 'opencv-python-headless', 'opencv-python', 'pycocotools',
            'torch', 'torchvision', 'ultralytics', 'segmentation-models-pytorch', 'timm', 'scipy')


@dataclass(frozen=True)
class Contract:
    """Production constants; alternative contracts exist only for injected CPU fixtures."""
    images: int = 180
    yolo_sha: str = YOLO_SHA
    native_sha: str = NATIVE_SHA


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.pending', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode()+b'\n')


def write_cache(path, payload):
    envelope = dict(payload=payload, sha256=fingerprint(payload))
    atomic_bytes(path, gzip.compress(canonical(envelope), mtime=0))
    return envelope


def read_cache(path, identity, image, stage):
    if not path.is_file():
        return None
    envelope = json.loads(gzip.decompress(path.read_bytes()))
    payload = envelope['payload']
    if envelope['sha256'] != fingerprint(payload):
        raise ValueError(f'Cache checksum mismatch: {path}')
    expected = dict(identity=identity, image=image['stem'], image_sha256=image['sha256'], stage=stage)
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Cache identity mismatch: {path}; choose a new output_root for changed inputs')
    return envelope


def validate_originals(masks):
    if masks and int(mu.area(mu.merge(masks))) != sum(int(mu.area(mask)) for mask in masks):
        raise ValueError('The frozen baseline contains overlapping masks')


def validate_outputs(originals, outputs):
    if len(originals) != len(outputs):
        raise ValueError('R1 changed instance count')
    validate_originals(outputs)
    changes = []
    for before, after in zip(originals, outputs):
        old_area, new_area = int(mu.area(before)), int(mu.area(after))
        if int(mu.area(mu.merge([before, after], intersect=True))) != old_area:
            raise ValueError('R1 removed an original pixel')
        if not old_area <= new_area <= 1.25*old_area:
            raise ValueError('R1 violated the25% area-growth limit')
        changes.append(dict(before_area=old_area, after_area=new_area, added_pixels=new_area-old_area,
                            changed=before['counts'] != after['counts']))
    return changes


def validate_geometry(views, stem):
    if len(views) != 4 or {view.get('view') for view in views} != {'base', 'scale', 'hflip', 'vflip'}:
        raise ValueError('Expected exactly the original four geometric views')
    for view in views:
        if view.get('image') != stem or view.get('shape') != [2048, 2048] or view.get('coordinate_space') != 'original':
            raise ValueError('Geometric support belongs to another image or coordinate space')
        for candidate in view['instances']:
            confidence = float(candidate['confidence'])
            if not np.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError('Invalid support confidence')
            checked_rle(dict(size=[2048, 2048], counts=candidate['rle']))


def validate_native(record, image, identity):
    for key, value in dict(file_name=image['name'], image_sha256=image['sha256'], identity=identity,
                           size=[2048, 2048]).items():
        if record.get(key) != value:
            raise ValueError('Native support metadata mismatch: '+key)
    if not valid_stage('native', record, image['stem']):
        raise ValueError('Baseline inference requires the full native rich record')
    for branch in (record, record['tta']):
        for head, thresholds in (('foreground', ('.2', '.35', '.5', '.65', '.8')), ('spine', ('.3', '.5', '.7'))):
            for threshold in thresholds:
                union_rle(branch['unions'][head]['0'+threshold])
        for threshold in ('0.2', '0.35', '0.5', '0.65', '0.8'):
            for component in branch['components'][threshold]:
                mask = checked_rle(component['rle'])
                if int(mu.area(mask)) != component['area']:
                    raise ValueError('Native component area mismatch')
    if set(record['view_unions']) != {'base', 'hflip', 'vflip', 'hvflip'}:
        raise ValueError('Native full record requires all four view unions')
    for value in record['view_unions'].values():union_rle(value)


def validate_stage(stage, data, image, identity):
    if not valid_stage(stage, data, image['stem']):
        raise ValueError('Incomplete frozen inference stage: '+stage)
    if stage == 'native':return validate_native(data, image, identity)
    if stage == 'geometric':return validate_geometry(data, image['stem'])
    for view in data:
        if view.get('shape') != [2048, 2048] or view.get('coordinate_space') != 'original':
            raise ValueError('YOLO view is not in original coordinates')
        for candidate in view['instances']:
            confidence = float(candidate['confidence'])
            if not np.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError('Invalid YOLO confidence')
            checked_rle(dict(size=[2048, 2048], counts=candidate['rle']))


def prepare(config, contract):
    required = {'output_root', 'images', 'expected_names_path', 'yolo_checkpoint', 'native_checkpoint'}
    allowed = required | {'device', 'native_batch', 'precision', 'expected_images'}
    if required-set(config) or set(config)-allowed:
        raise ValueError('Config must contain required path fields and no unknown keys: '+str(sorted(required-set(config))))
    cfg = dict(device='cuda', native_batch=1, precision='fp16', expected_images=contract.images)
    cfg.update(config)
    if cfg['precision'] != 'fp16' or cfg['native_batch'] != 1 or cfg['expected_images'] != contract.images:
        raise ValueError('Frozen run requires fp16, native_batch1 and the declared complete test set')
    for key in required:
        cfg[key] = str(Path(cfg[key]).expanduser().resolve())
    for key, expected in (('yolo_checkpoint', contract.yolo_sha),
                          ('native_checkpoint', contract.native_sha)):
        if sha(cfg[key]) != expected:
            raise ValueError('Wrong frozen input SHA-256: '+key)
    expected_names = json.loads(Path(cfg['expected_names_path']).read_text())
    if isinstance(expected_names, dict):
        expected_names = expected_names['test_images']
    if (not isinstance(expected_names, list) or len(expected_names) != contract.images
            or any(not isinstance(name, str) or Path(name).name != name for name in expected_names)
            or len(set(expected_names)) != contract.images):
        raise ValueError('Official test names must list each expected basename exactly once')
    if contract == Contract() and fingerprint(sorted(expected_names)) != OFFICIAL_NAMES_SHA:
        raise ValueError('Names differ from the pinned official180-image test set')
    image_root = Path(cfg['images'])
    paths = sorted(path for path in image_root.rglob('*') if path.is_file() and path.suffix.lower() in ('.jpeg', '.jpg', '.png'))
    if len(paths) != contract.images or {path.name for path in paths} != set(expected_names):
        raise ValueError('Image folder does not exactly match the official test-name manifest')
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError('Duplicate image stems')
    images = []
    for path in paths:
        source_bytes = path.read_bytes()
        # Match frozen process_frame exactly: decode original bytes as grayscale, without resizing.
        pixels = cv2.imdecode(np.frombuffer(source_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if pixels is None or pixels.dtype != np.uint8 or pixels.shape != (2048, 2048):
            raise ValueError('Expected original grayscale uint8 2048x2048 image: '+path.name)
        images.append(dict(name=path.name, stem=path.stem, path=str(path),
                           sha256=hashlib.sha256(source_bytes).hexdigest()))
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    sources = {name: sha(Path(__file__).with_name(name)) for name in SOURCE_FILES}
    protocol = dict(version=2, config=cfg, sources=sources, packages=versions,
        python=list(sys.version_info[:3]), environment={key: os.environ.get(key) for key in
            ('TORCH_CUDNN_V8_API_DISABLED', 'CUBLAS_WORKSPACE_CONFIG')},
        models=dict(yolo=contract.yolo_sha, native=contract.native_sha),
        expected_names_sha256=sha(cfg['expected_names_path']),
        images=images, precision='fp16', pipeline='frozen YOLO4+8+18 and best84 full rich -> finalize_frame -> R1 once',
        baseline='Generated from the same inference stages; no submission CSV input',
        runtime_binding='Every support stage binds runtime_device.json; live GPU facts checked only before new inference',
        output_policy='Same-run baseline IDs/count/pixels preserved by R1, disjoint, <=25% growth; blank generated masks stay blank')
    return cfg, images, protocol


def native_record_from_heads(base, average, view_unions, *, file_name, image_sha256, identity):
    """Use the frozen full native decoder for the shared baseline and R1."""
    from submission_inference import prediction_record
    if base.shape != (2, 2048, 2048) or average.shape != base.shape:
        raise ValueError('Expected original native two-head probabilities')
    if not np.isfinite(base).all() or not np.isfinite(average).all():
        raise ValueError('Nonfinite native probabilities')
    record = prediction_record(base, file_name, image_sha256, identity)
    record['tta'] = prediction_record(average, file_name, image_sha256, identity)
    record['view_unions'] = view_unions
    return record


class SupportEngine:
    """YOLO first; unload it before an environment-isolated persistent native worker."""
    def __init__(self, cfg):
        from submission_native_process import NativeProcessClient
        self.cfg, self.provider = cfg, None
        # Environment captured before importing/constructing YOLO.
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--native-support-server',
                   '--checkpoint', cfg['native_checkpoint'], '--device', cfg['device'], '--batch', '1']
        self.client = NativeProcessClient(cfg['native_checkpoint'], device=cfg['device'], batch=1, _command=command)

    def _provider(self):
        if self.provider is None:
            from submission_inference import InferenceProvider
            self.provider = InferenceProvider(self.cfg['yolo_checkpoint'], None, device=self.cfg['device'],
                native_batch=1, progress=lambda stage, view: print(f'  {stage}: {view}', flush=True))
        return self.provider

    def geometric(self, stem, pixels):return self._provider().geometric(stem, pixels)
    def preprocessing(self, stem, pixels):return self._provider().preprocessing(stem, pixels)
    def tiled(self, stem, pixels):return self._provider().tiled(stem, pixels)

    def finish_geometric(self):
        if self.provider is not None:
            self.provider._yolo_model = None
            self.provider = None
            gc.collect()
            import torch
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()

    def native(self, stem, pixels, **metadata):
        return self.client.native(stem, pixels, **metadata)

    def close(self):
        self.finish_geometric()
        self.client.close()


def native_support_server(argv):
    from submission_native_process import PROTOCOL
    from submission_inference import InferenceProvider, limb, predict_views
    from native_instance_training import configure_cuda
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-support-server', action='store_true')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch', type=int, default=1)
    args = parser.parse_args(argv)
    if args.batch != 1:
        raise ValueError('Frozen native support batch is1')
    configure_cuda()
    provider = InferenceProvider(None, args.checkpoint, device=args.device, native_batch=1)
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line); request_id = request.get('id')
            if request.get('protocol') != PROTOCOL:
                raise ValueError('Unknown support-worker protocol')
            if request.get('command') == 'close':
                break
            if request.get('command') != 'native':
                raise ValueError('Unknown support-worker command')
            pixels = np.load(request['input_path'], allow_pickle=False)
            corrected = limb(pixels)
            amp = torch.float16 if provider.device.type == 'cuda' else None
            base, average, view_unions = predict_views(provider._native(), corrected, provider.device, batch=1, amp_dtype=amp,
                progress=lambda view: print('  native: '+view, file=sys.stderr, flush=True))
            record = native_record_from_heads(base, average, view_unions, file_name=request['file_name'],
                image_sha256=request['image_sha256'], identity=request['identity'])
            atomic_json(Path(request['output_path']), record)
            response = dict(protocol=PROTOCOL, status='ok', id=request_id, output_path=request['output_path'])
        except Exception as error:
            response = dict(protocol=PROTOCOL, status='error', id=request_id,
                            error_type=type(error).__name__, message=str(error))
        print(json.dumps(response), flush=True)


def device_facts(device, injected):
    if injected:
        return dict(injected_CPU_fixture=True, device=device)
    import torch
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('New production inference requires a CUDA GPU; completed caches do not')
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    properties = torch.cuda.get_device_properties(index)
    try:
        driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            check=True, capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        driver = None
    return dict(torch=torch.__version__, torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        gpu_name=properties.name, capability=[properties.major, properties.minor],
        memory_bytes=properties.total_memory, driver_versions=driver, device=device,
        precision='fp16', native_depthwise_cudnn_fallback=True)


def run(config, *, max_images=None, engine_factory=None, contract=None):
    """Single local runtime process may own an output root; process death releases the lock."""
    if 'output_root' not in config:
        raise ValueError('output_root is required')
    key = hashlib.sha256(str(Path(config['output_root']).expanduser().resolve()).encode()).hexdigest()
    locks = Path(tempfile.gettempdir())/'solar_r1_locks'
    locks.mkdir(exist_ok=True)
    with (locks/(key+'.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another cell/process is already using this output_root') from error
        try:
            return _run_locked(config, max_images=max_images, engine_factory=engine_factory, contract=contract)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _run_locked(config, *, max_images=None, engine_factory=None, contract=None):
    """Infer all full-pipeline stages, freeze baseline, apply R1 once, export both."""
    contract = contract or Contract()
    if contract != Contract() and engine_factory is None:
        raise ValueError('Non-production contracts require an injected test engine')
    if max_images is not None and (not isinstance(max_images, int) or max_images < 1):
        raise ValueError('max_images must be a positive integer')
    cv2.setNumThreads(1)
    cfg, images, protocol = prepare(config, contract)
    if engine_factory is None and not cfg['device'].startswith('cuda'):
        raise ValueError('Production config device must be CUDA for the frozen fp16 pipeline')
    root = Path(cfg['output_root']); root.mkdir(parents=True, exist_ok=True)
    identity = fingerprint(protocol)
    protocol_path = root/'protocol.json'
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        if previous != dict(identity=identity, protocol=protocol):
            raise ValueError('Inputs/config/code/runtime changed; use a new output_root instead of mixing cached outputs')
    else:
        if any((root/name).exists() for name in ('cache', 'result.json', 'submission_pipeline_R1.csv')):
            raise ValueError('Existing outputs lack their protocol manifest')
        atomic_json(protocol_path, dict(identity=identity, protocol=protocol))
    runtime_path = root/'runtime_device.json'
    runtime_envelope = json.loads(runtime_path.read_text()) if runtime_path.is_file() else None
    if runtime_envelope and (runtime_envelope.get('sha256') != fingerprint(runtime_envelope['payload'])
            or runtime_envelope['payload'].get('identity') != identity):
        raise ValueError('Runtime device snapshot checksum/identity mismatch')
    runtime_signature = runtime_envelope['sha256'] if runtime_envelope else None
    def path(image, stage):return root/'cache'/image['stem']/(stage+'.json.gz')
    def load(image, stage):
        envelope = read_cache(path(image, stage), identity, image, stage)
        if envelope and (runtime_signature is None or envelope['payload'].get('runtime_signature') != runtime_signature):
            raise ValueError('Cached stage is not bound to its runtime device snapshot')
        return envelope
    def save(image, stage, data):
        if runtime_signature is None:raise ValueError('Missing inference runtime snapshot')
        return write_cache(path(image, stage), dict(identity=identity, image=image['stem'], image_sha256=image['sha256'],
            stage=stage, data=data, runtime_signature=runtime_signature))
    def load_stages(image):
        stages = {stage: load(image, stage) for stage in INFERENCE_STAGES}
        if any(value is None for value in stages.values()):
            raise ValueError('A downstream cache is missing one of its full-pipeline inference stages')
        for stage, value in stages.items():validate_stage(stage, value['payload']['data'], image, identity)
        return stages
    def baseline_from_cache(image, stage_records):
        baseline = load(image, 'baseline')
        if baseline is None:return None, None
        supports = {stage: value['sha256'] for stage, value in stage_records.items()}
        if baseline['payload']['data']['support_sha256'] != supports:
            raise ValueError('Baseline cache no longer matches its four inference stages')
        masks = validate_predictions(baseline['payload']['data']['masks'])
        return baseline, masks
    completed = {}; baselines = {}; selected = []; stats = Counter()
    for image in images:
        final = load(image, 'final')
        if final:
            stage_records = load_stages(image)
            baseline, originals = baseline_from_cache(image, stage_records)
            data = final['payload']['data']
            if baseline is None or data['baseline_sha256'] != baseline['sha256']:
                raise ValueError('R1 cache no longer matches its same-run baseline')
            if data['support_sha256'] != {stage: value['sha256'] for stage, value in stage_records.items()}:
                raise ValueError('R1 cache no longer matches its shared inference stages')
            masks = validate_predictions(data['masks'])
            validate_outputs(originals, masks)
            completed[image['stem']] = data; baselines[image['stem']] = originals
            stats['reused_final_images'] += 1
        elif max_images is None or len(selected) < max_images:
            selected.append(image)
    print(f'Validated {len(images)} official images; cached {len(completed)}; processing {len(selected)}.', flush=True)
    engine = None
    def get_engine():
        nonlocal engine, runtime_envelope, runtime_signature
        if engine is None:
            facts = device_facts(cfg['device'], engine_factory is not None)
            payload = dict(identity=identity, facts=facts)
            candidate = dict(payload=payload, sha256=fingerprint(payload))
            if runtime_envelope and runtime_envelope != candidate:
                raise ValueError('CUDA/GPU/cuDNN runtime changed; restore the original runtime or choose a new output_root')
            if runtime_envelope is None:
                atomic_json(runtime_path, candidate)
                runtime_envelope, runtime_signature = candidate, candidate['sha256']
            engine = (engine_factory or SupportEngine)(cfg)
        return engine
    def pixels(image):
        source = Path(image['path']).read_bytes()
        if hashlib.sha256(source).hexdigest() != image['sha256']:
            raise ValueError('Image changed after validation: '+image['name'])
        value = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_GRAYSCALE)
        if value is None or value.shape != (2048, 2048) or value.dtype != np.uint8:
            raise ValueError('Invalid original image: '+image['name'])
        return value
    try:
        # Infer the original30 YOLO views, then free that model before native36 tiles.
        for number, image in enumerate(selected, 1):
            image_pixels = None
            for stage in INFERENCE_STAGES[:3]:
                record = load(image, stage)
                if record is None:
                    print(f'YOLO {number}/{len(selected)} {image["name"]}: {stage}', flush=True)
                    if image_pixels is None:image_pixels = pixels(image)
                    value = getattr(get_engine(), stage)(image['stem'], image_pixels)
                    validate_stage(stage, value, image, identity); save(image, stage, value)
                    stats['new_'+stage+'_images'] += 1
                else:
                    validate_stage(stage, record['payload']['data'], image, identity)
                    stats['reused_'+stage+'_images'] += 1
        if engine is not None:engine.finish_geometric()
        for number, image in enumerate(selected, 1):
            native = load(image, 'native')
            if native is None:
                print(f'Native full {number}/{len(selected)} {image["name"]}', flush=True)
                value = get_engine().native(image['stem'], pixels(image), file_name=image['name'],
                    image_sha256=image['sha256'], identity=identity)
                validate_native(value, image, identity); save(image, 'native', value)
                stats['new_native_images'] += 1
            else:
                validate_native(native['payload']['data'], image, identity)
                stats['reused_native_images'] += 1
            stage_records = load_stages(image)
            stages = {stage: value['payload']['data'] for stage, value in stage_records.items()}
            supports = {stage: value['sha256'] for stage, value in stage_records.items()}
            baseline, originals = baseline_from_cache(image, stage_records)
            if baseline is None:
                # Exact original frozen YOLO/native finalizer; no new thresholds or alternate recipe.
                originals = validate_predictions(finalize_frame(image['stem'], stages))
                baseline = save(image, 'baseline', dict(masks=json_ready(originals), support_sha256=supports))
                stats['new_baseline_images'] += 1
            else:
                stats['reused_baseline_images'] += 1
            output, counter = restore_consensus_tails(originals, stages['geometric'], stages['native'])
            validate_outputs(originals, output)
            data = dict(masks=json_ready(output), counters=dict(counter), support_sha256=supports,
                        baseline_sha256=baseline['sha256'])
            save(image, 'final', data)
            completed[image['stem']] = data; baselines[image['stem']] = originals
            print(f'Pipeline + R1 saved {len(completed)}/{len(images)} {image["name"]}', flush=True)
    finally:
        if engine is not None:engine.close()
    for key, expected in (('yolo_checkpoint', contract.yolo_sha), ('native_checkpoint', contract.native_sha),
                          ('expected_names_path', protocol['expected_names_sha256'])):
        if sha(cfg[key]) != expected:raise ValueError('Input changed during the run: '+key)
    for image in images:
        if sha(image['path']) != image['sha256']:raise ValueError('Image changed during the run: '+image['name'])
    for name, expected in protocol['sources'].items():
        if sha(Path(__file__).with_name(name)) != expected:raise ValueError('Inference source changed during the run: '+name)
    baseline_output = root/'submission_pipeline_baseline.csv'; refined_output = root/'submission_pipeline_R1.csv'
    if len(completed) != contract.images:
        result = dict(status='paused', identity=identity, completed_images=len(completed), expected_images=contract.images,
                      submission_written=False, counters=dict(stats), output_root=str(root))
        atomic_json(root/'result.json', result)
        atomic_json(root/'latest_results.json', dict(result, status='partial', baseline_csv=str(baseline_output),
            r1_csv=str(refined_output), archive=None))
        print('Paused safely; no partial submission CSV was produced.', flush=True)
        return result
    # Generate matched IDs from the same-run baseline, with no dummy rows for empty images.
    streams = {method: io.StringIO(newline='') for method in ('baseline', 'R1')}
    writers = {method: csv.writer(stream) for method, stream in streams.items()}
    for writer in writers.values():writer.writerow(['filament_id', 'segmentation_rle'])
    changes = []; all_counters = Counter(); masks_total = 0
    for image in images:
        stem = image['stem']; originals = baselines[stem]
        outputs = validate_predictions(completed[stem]['masks'])
        image_changes = validate_outputs(originals, outputs)
        all_counters.update(completed[stem]['counters']); masks_total += len(originals)
        for index, (before, after, change) in enumerate(zip(originals, outputs, image_changes), 1):
            identifier = f'{stem}_{index}'
            writers['baseline'].writerow([identifier, before['counts'].decode()])
            writers['R1'].writerow([identifier, after['counts'].decode()])
            changes.append(dict(filament_id=identifier, image=stem, **change))
    if len(changes) != masks_total:raise ValueError('Incomplete final mask pairing')
    atomic_bytes(baseline_output, streams['baseline'].getvalue().encode())
    atomic_bytes(refined_output, streams['R1'].getvalue().encode())
    change_stream = io.StringIO(newline='')
    columns = ['filament_id', 'image', 'before_area', 'after_area', 'added_pixels', 'changed']
    change_writer = csv.DictWriter(change_stream, fieldnames=columns)
    change_writer.writeheader(); change_writer.writerows(changes)
    atomic_bytes(root/'change_audit.csv', change_stream.getvalue().encode())
    anchored_images = sum(bool(values) for values in baselines.values())
    manifest = dict(status='complete', identity=identity, official_images=contract.images,
        anchored_images=anchored_images, empty_baseline_images=contract.images-anchored_images,
        mask_count=masks_total, changed_masks=sum(row['changed'] for row in changes),
        added_pixels=sum(row['added_pixels'] for row in changes), inference_counters=dict(stats), r1_counters=dict(all_counters),
        historical_pipeline_public_LB=0.39, current_baseline_public_LB=None, new_public_LB=None,
        comparison='Same-run frozen YOLO+best84 baseline vs that baseline plus R1; shared inference outputs',
        historical_CSV_identity_not_claimed=True, new_score_not_measured=True,
        invariants=dict(same_ids=True, same_mask_count=True, baseline_pixels_preserved=True,
            masks_disjoint=True, maximum_area_ratio=1.25, generated_empty_baselines_preserved=True),
        source_sha256=protocol['sources'], inference_runtime=runtime_envelope,
        output_sha256={p.name: sha(p) for p in (baseline_output, refined_output, root/'change_audit.csv')})
    atomic_json(root/'run_manifest.json', manifest)
    descriptor, temporary_name = tempfile.mkstemp(prefix='results.', suffix='.pending', dir=root)
    os.close(descriptor); temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in ('submission_pipeline_baseline.csv', 'submission_pipeline_R1.csv', 'change_audit.csv', 'run_manifest.json', 'protocol.json'):
                archive.write(root/name, name)
            if runtime_path.is_file():archive.write(runtime_path, runtime_path.name)
        os.replace(temporary, root/'results.zip')
    finally:
        temporary.unlink(missing_ok=True)
    result = dict(status='complete', identity=identity, completed_images=contract.images, submission_written=True,
        submission=str(refined_output), baseline=str(baseline_output), archive=str(root/'results.zip'),
        archive_sha256=sha(root/'results.zip'), changed_masks=manifest['changed_masks'], added_pixels=manifest['added_pixels'],
        expected_images=contract.images, mask_count=masks_total)
    atomic_json(root/'result.json', result)
    atomic_json(root/'latest_results.json', dict(result, baseline_csv=str(baseline_output), r1_csv=str(refined_output)))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result



def main(argv=None):
    def stop(_signum, _frame):
        raise KeyboardInterrupt('Stopped; completed stage caches are preserved for resume')
    signal.signal(signal.SIGTERM, stop)
    argv = sys.argv[1:] if argv is None else argv
    if '--native-support-server' in argv:
        return native_support_server(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--max-images', type=int)
    args = parser.parse_args(argv)
    return run(json.loads(args.config.read_text()), max_images=args.max_images)


if __name__ == '__main__':main()
