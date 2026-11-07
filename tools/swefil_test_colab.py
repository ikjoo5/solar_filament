"""Durable Colab launcher for reviewed SWEFil epoch100 test inference.

Only source code and launcher settings are copied. Training, synthesis, weights,
test pixels and prediction inputs are never bundled or modified by this launcher.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import uuid


SCOPE = 'swefil_epoch100_native_t4_v1'
STATE_NAME = 'test_bootstrap_state.json'
SOURCE_FILES = (
    'tools/swefil_test_colab.py', 'tools/swefil_test_inference.py',
    'tools/solar_a100_train_standalone.py', 'tools/chase_best84_training.py',
    'tools/chase_independent_decode.py', 'tools/verified_drive_checkpoints.py',
)
BASE = Path('/content/drive/MyDrive/filament-segmentation-2026')
DEFAULT_CACHE = Path('/content/swefil_test_checkpoint_cache')
TRAINING_IDENTITY = '00c20ec35658f63e9cc6917cc9cdd30bdc060eee27445e1a0bf8b99c2c4ef405'
CHECKPOINT_SHA = '2b22ed327e2f64902bb7b9b3984586d1ba893f52ada74f557622cce4dac27e40'
CHECKPOINT_SIZE = 128514783
CHECKPOINT_EPOCH = 100
REMOTE_FIELDS = 'id,name,mimeType,size,sha256Checksum,appProperties,trashed'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    try:
        pending.write_bytes(payload)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def separate(output, protected):
    output, protected = Path(output).resolve(), Path(protected).resolve()
    if output == protected or output in protected.parents or protected in output.parents:
        raise ValueError('Inference output must be separate from training/test inputs: '+str(protected))


def settings(training_root, test_images, output, checkpoint=None, checkpoint_file_id=None):
    training_root, test_images, output = map(lambda p: Path(p).resolve(),
                                            (training_root, test_images, output))
    separate(output, training_root)
    separate(output, test_images)
    config = dict(scope=SCOPE, training_root=str(training_root), test_images=str(test_images),
                output_root=str(output), expected_epoch=CHECKPOINT_EPOCH, expected_images=180,
                checkpoint_sha256=CHECKPOINT_SHA, checkpoint_size=CHECKPOINT_SIZE,
                training_identity=TRAINING_IDENTITY, inference_only=True, native_only=True,
                checkpoint_override=str(Path(checkpoint).absolute()) if checkpoint else None,
                checkpoint_file_id=checkpoint_file_id)
    return config


def restore(output, training_root=None, test_images=None, checkpoint=None, checkpoint_file_id=None):
    """Restore an existing inference snapshot without touching training state."""
    output = Path(output).resolve()
    pointer = output/STATE_NAME
    if pointer.is_symlink():
        raise ValueError('Unsafe inference bootstrap pointer')
    state = json.loads(pointer.read_text())
    if state.get('scope') != SCOPE or state.get('output_root') != str(output):
        raise ValueError('Saved bootstrap belongs to another inference experiment')
    version = state.get('version', '')
    if not re.fullmatch('[0-9a-f]{16}', version):
        raise ValueError('Invalid source version')
    code = output/'code'/version
    manifest_path = code/'source_manifest.json'
    if code.is_symlink() or not code.resolve().is_relative_to(output):
        raise ValueError('Unsafe source snapshot directory')
    if manifest_path.is_symlink() or sha(manifest_path) != state.get('manifest_sha256'):
        raise ValueError('Saved source manifest checksum differs')
    manifest = json.loads(manifest_path.read_text())
    if set(manifest) != set(SOURCE_FILES) or hashlib.sha256(canonical(manifest)).hexdigest()[:16] != version:
        raise ValueError('Saved source inventory/version differs')
    for name, expected in manifest.items():
        path = code/name
        if path.is_symlink() or not path.resolve().is_relative_to(code.resolve()):
            raise ValueError('Unsafe saved source path: '+name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Saved source checksum differs: '+name)
    config_path = output/'inference_launcher_config.json'
    if config_path.is_symlink() or sha(config_path) != state.get('config_sha256'):
        raise ValueError('Saved inference settings checksum differs')
    config = json.loads(config_path.read_text())
    wanted = settings(training_root or config['training_root'],
                      test_images or config['test_images'], output,
                      config.get('checkpoint_override'),
                      config.get('checkpoint_file_id'))
    if config != wanted:
        raise ValueError('Inference paths/settings changed; use the saved paths or a new output')
    return config, code, state


def install(repository, training_root, test_images, output, checkpoint=None, checkpoint_file_id=None):
    """Pin a small complete source bundle on Drive on the first invocation."""
    repository, output = Path(repository).resolve(), Path(output).resolve()
    if (output/STATE_NAME).exists():
        return restore(output, training_root, test_images, checkpoint, checkpoint_file_id)
    if any((output/name).exists() for name in ('identity.json', 'latest_result.json', 'cache', 'train')):
        raise ValueError('Existing results lack an inference bootstrap identity; preserve them and use a new output')
    config = settings(training_root, test_images, output, checkpoint, checkpoint_file_id)
    payloads = {}
    for name in SOURCE_FILES:
        path = repository/name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(repository):
            raise FileNotFoundError('Missing inference source: '+str(path))
        payload = path.read_bytes()
        compile(payload, name, 'exec')
        payloads[name] = payload
    manifest = {name:hashlib.sha256(raw).hexdigest() for name,raw in payloads.items()}
    version = hashlib.sha256(canonical(manifest)).hexdigest()[:16]
    code = output/'code'/version
    config_bytes, manifest_bytes = canonical(config)+b'\n', canonical(manifest)+b'\n'
    try:
        commit = subprocess.check_output(['git','-C',str(repository),'rev-parse','HEAD'],text=True).strip()
        if not re.fullmatch('[0-9a-f]{40,64}',commit):
            raise ValueError('Invalid repository commit')
        dirty = bool(subprocess.check_output(['git','-C',str(repository),'status','--porcelain'],text=True).strip())
    except subprocess.CalledProcessError:
        commit, dirty = None, None
    state = dict(scope=SCOPE, version=version, output_root=str(output),
        source_commit=commit, source_worktree_dirty=dirty,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        config_sha256=hashlib.sha256(config_bytes).hexdigest())
    writes = {code/name:raw for name,raw in payloads.items()}
    writes.update({code/'source_manifest.json':manifest_bytes,
                   output/'inference_launcher_config.json':config_bytes,
                   output/STATE_NAME:canonical(state)+b'\n'})
    for path,raw in writes.items():
        if path.is_symlink() or not path.resolve().is_relative_to(output):
            raise ValueError('Unsafe snapshot destination: '+str(path))
        if path.exists() and (not path.is_file() or path.read_bytes()!=raw):
            raise ValueError('Preserving changed existing file: '+str(path))
    for path,raw in writes.items():
        if not path.exists():
            atomic_write(path,raw)
    return restore(output, training_root, test_images, checkpoint, checkpoint_file_id)


def reviewed_checkpoint_record(training_root):
    """Read only matching best-weight pointers; mounted checkpoint paths are unused."""
    training = Path(training_root)/'train'
    for name in ('summary.json', 'state.json'):
        path = training/name
        if not path.is_file() or path.is_symlink():
            continue
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or value.get('identity') != TRAINING_IDENTITY:
            continue
        payload = value if name == 'summary.json' else value.get('payload', value)
        if not isinstance(payload, dict) or payload.get('best_epoch') != CHECKPOINT_EPOCH:
            continue
        record = payload.get('best_checkpoint_record' if name == 'summary.json' else 'best')
        if not isinstance(record, dict):
            continue
        if (record.get('sha256') == CHECKPOINT_SHA and record.get('size') == CHECKPOINT_SIZE
                and isinstance(record.get('file_id'), str) and record['file_id']):
            return record
    raise ValueError('No matching reviewed epoch100 best record in train/summary.json or state.json. '
                     'Supply --checkpoint-file-id for the reviewed Drive API weight, or --checkpoint '
                     'for an exact local copy; mounted train/checkpoints is not used.')


def verify_checkpoint_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Reviewed checkpoint must be a regular file: '+str(path))
    if path.stat().st_size != CHECKPOINT_SIZE or sha(path) != CHECKPOINT_SHA:
        raise ValueError('Checkpoint bytes differ from the reviewed epoch100 SHA-256/size')
    return path.resolve()


def make_drive_service():
    """Use Colab credentials after the notebook calls auth.authenticate_user()."""
    import google.auth
    from googleapiclient.discovery import build
    credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/drive.readonly'])
    return build('drive', 'v3', credentials=credentials, cache_discovery=False)


def validate_remote_checkpoint(item, file_id, record=None):
    """Reject inconsistent server metadata before fetching any checkpoint bytes."""
    try:
        size = int(item.get('size', -1))
    except (TypeError, ValueError) as error:
        raise ValueError('Drive checkpoint size is invalid') from error
    if (item.get('id') != file_id or item.get('trashed') is not False
            or size != CHECKPOINT_SIZE
            or str(item.get('mimeType', '')).startswith('application/vnd.google-apps.')):
        raise ValueError('Drive checkpoint ID, trash state, type or size differs')
    server_sha = item.get('sha256Checksum')
    if server_sha is not None and server_sha != CHECKPOINT_SHA:
        raise ValueError('Drive checkpoint server SHA-256 differs')
    properties = item.get('appProperties', {})
    if not isinstance(properties, dict):
        raise ValueError('Drive checkpoint appProperties are invalid')
    expected = dict(run_identity=TRAINING_IDENTITY, kind='best',
                    epoch=str(CHECKPOINT_EPOCH), sha256=CHECKPOINT_SHA)
    # Original published records have these properties. An explicit file ID may
    # point to a copied binary with no appProperties; its actual hash still has
    # to match. Any published properties that are present must agree.
    if record is not None:
        if record.get('name') != item.get('name') or any(properties.get(k) != v for k,v in expected.items()):
            raise ValueError('Drive checkpoint name or appProperties differ from the reviewed record')
    elif any(k in properties and properties[k] != v for k,v in expected.items()):
        raise ValueError('Drive checkpoint appProperties differ from reviewed epoch100')


class _CheckpointWriter:
    """Hash the actual streamed bytes while bounding the local download size."""
    def __init__(self, stream):
        self.stream, self.size, self.hash = stream, 0, hashlib.sha256()

    def write(self, data):
        if self.size + len(data) > CHECKPOINT_SIZE:
            raise ValueError('Downloaded checkpoint exceeds reviewed size')
        written = self.stream.write(data)
        if written != len(data):
            raise OSError('Short local checkpoint write')
        self.hash.update(data)
        self.size += len(data)
        return written


def download_checkpoint(training_root, checkpoint=None, checkpoint_file_id=None, *,
                        cache_root=DEFAULT_CACHE, service=None, downloader=None):
    """Resolve one reviewed checkpoint with read-only API calls and atomic caching."""
    if checkpoint is not None:
        return verify_checkpoint_file(checkpoint)
    cache = Path(cache_root)
    if cache.is_symlink() or cache.resolve().is_relative_to('/content/drive'):
        raise ValueError('Use a regular local checkpoint cache outside mounted Drive')
    cache.mkdir(parents=True, exist_ok=True)
    target = cache/(CHECKPOINT_SHA+'.pt')
    if target.is_symlink():
        raise ValueError('Checkpoint cache target must not be a symlink')
    if target.is_file():
        try:
            result = verify_checkpoint_file(target)
        except ValueError:
            pass
        else:
            print('Reusing hash-verified local epoch100 checkpoint.', flush=True)
            return result
    record = None if checkpoint_file_id is not None else reviewed_checkpoint_record(training_root)
    file_id = checkpoint_file_id if checkpoint_file_id is not None else record['file_id']
    if not isinstance(file_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', file_id):
        raise ValueError('Supply a Drive file ID, not a URL or mounted path')
    if service is None:
        service = make_drive_service()
    item = service.files().get(fileId=file_id, fields=REMOTE_FIELDS,
                               supportsAllDrives=True).execute()
    validate_remote_checkpoint(item, file_id, record)
    if downloader is None:
        from verified_drive_checkpoints import download_file
        downloader = download_file
    print('Downloading reviewed epoch100 checkpoint through the read-only Drive API.', flush=True)
    fd, name = tempfile.mkstemp(prefix='epoch100-', suffix='.pending', dir=cache)
    pending = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            sink = _CheckpointWriter(stream)
            downloader(service, file_id, sink)
            stream.flush()
            os.fsync(stream.fileno())
        if sink.size != CHECKPOINT_SIZE or sink.hash.hexdigest() != CHECKPOINT_SHA:
            raise ValueError('Downloaded checkpoint bytes failed reviewed SHA-256/size verification')
        pending.replace(target)
        return target.resolve()
    finally:
        pending.unlink(missing_ok=True)


def ensure_dependencies(output):
    """Keep Colab's installed CUDA torch/torchvision and NumPy ABI unchanged."""
    preserved = {}
    for package in ('torch','torchvision'):
        try:
            preserved[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError('Select a Colab GPU runtime with its provided torch/torchvision') from error
    try:
        preserved['numpy'] = importlib.metadata.version('numpy')
    except importlib.metadata.PackageNotFoundError:
        pass
    needed = []
    for package,version in [('segmentation-models-pytorch','0.5.0'),('timm','1.0.20'),('pycocotools','2.0.11')]:
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != version:
            needed.append(package+'=='+version)
    for module,package in [('numpy','numpy'),('cv2','opencv-python-headless'),('scipy','scipy'),
                           ('googleapiclient','google-api-python-client'),
                           ('google.auth','google-auth'),('google_auth_httplib2','google-auth-httplib2')]:
        try:
            present = importlib.util.find_spec(module) is not None
        except (ModuleNotFoundError, ValueError):
            present = False
        if not present:
            needed.append(package)
    if needed:
        constraint = Path(output)/'keep_colab_runtime.txt'
        constraint.write_text(''.join(k+'=='+v+'\n' for k,v in preserved.items()))
        print('Installing inference dependencies:', ', '.join(needed),flush=True)
        subprocess.run([sys.executable,'-m','pip','install','--upgrade-strategy','only-if-needed',
                        '-c',str(constraint),*needed],check=True)
    if any(importlib.metadata.version(k)!=v for k,v in preserved.items()):
        raise RuntimeError('A preserved Colab runtime package changed during installation')


def tail(path, lines=12):
    path = Path(path)
    if not path.is_file():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-20000))
        return '\n'.join(stream.read().decode('utf-8','replace').splitlines()[-lines:])


def stop_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def monitored(command, log_path, cwd, env):
    print('Full inference log:',log_path,flush=True)
    with Path(log_path).open('ab',buffering=0) as stream:
        process = subprocess.Popen(command,cwd=cwd,env=env,stdout=stream,
                                   stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                try:
                    status = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(tail(log_path,8),flush=True)
            if status:
                print(tail(log_path,40),flush=True)
                raise subprocess.CalledProcessError(status,command)
        finally:
            stop_group(process)
    print(tail(log_path),flush=True)


def checked_result(output):
    output = Path(output).resolve()
    result_path = output/'latest_result.json'
    if result_path.is_symlink():
        raise ValueError('Unsafe inference result pointer')
    result = json.loads(result_path.read_text())
    if Path(result.get('output_root','')).resolve()!=output:
        raise ValueError('Inference result belongs to another output directory')
    if result.get('status') != 'complete' or result.get('images_completed') != 180:
        raise RuntimeError('All 180 images must complete before a submission is downloaded')
    if (result.get('checkpoint_epoch') != CHECKPOINT_EPOCH
            or result.get('checkpoint_sha256') != CHECKPOINT_SHA):
        raise ValueError('Result is not from the reviewed epoch100 checkpoint')
    for key in ('csv','archive'):
        path = Path(result[key]).resolve()
        if not path.is_relative_to(output) or not path.is_file() or Path(result[key]).is_symlink():
            raise ValueError('Missing or unsafe inference result: '+key)
        if sha(path)!=result.get(key+'_sha256'):
            raise ValueError('Inference result checksum differs: '+key)
    return result


def run(training_root, test_images, output, repository=None, checkpoint=None, checkpoint_file_id=None):
    output = Path(output).resolve()
    if (output/STATE_NAME).is_file():
        config,code,state = restore(output,training_root,test_images,checkpoint,checkpoint_file_id)
    else:
        repository = Path(repository).resolve() if repository else Path(__file__).resolve().parents[1]
        config,code,state = install(repository,training_root,test_images,output,checkpoint,checkpoint_file_id)
    training = Path(config['training_root'])
    for name in ('train/config.json','train/identity.json'):
        if not (training/name).is_file():
            raise FileNotFoundError('Training metadata needed to identify epoch100: '+str(training/name))
    if not Path(config['test_images']).is_dir():
        raise FileNotFoundError('Original test_images directory: '+config['test_images'])
    # Locators can change after a failed lookup: every result must still match
    # the same reviewed model SHA. Training/test/output paths remain pinned.
    explicit_checkpoint = checkpoint
    active_file_id = checkpoint_file_id
    if checkpoint is None and checkpoint_file_id is None:
        explicit_checkpoint = config.get('checkpoint_override')
        active_file_id = config.get('checkpoint_file_id')
    print('Inference only: SWEFil epoch100, limb-only, threshold0.65, area300, official180images.',flush=True)
    print('Saved source version:',state['version'],flush=True)
    ensure_dependencies(output)
    selected = download_checkpoint(training, explicit_checkpoint, active_file_id)
    env = dict(os.environ,PYTHONUNBUFFERED='1',TORCH_CUDNN_V8_API_DISABLED='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command = [sys.executable,'-u',str(code/'tools/swefil_test_inference.py'),
        '--training-root',config['training_root'],'--test-images',config['test_images'],
        '--output',str(output),'--result',str(output/'latest_result.json')]
    command += ['--checkpoint',str(selected)]
    monitored(command,output/'inference.log',code/'tools',env)
    result = checked_result(output)
    print('Submission CSV:',result['csv'],flush=True)
    print('Review ZIP:',result['archive'],flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root',type=Path,default=BASE/'fixed_split_experiments/swefil_all_mixed_native_t4_v1')
    parser.add_argument('--test-images',type=Path,default=BASE/'MAGFiLO_1.0_Kaggle_2026/test/test_images')
    parser.add_argument('--output',type=Path,default=BASE/'submissions'/SCOPE)
    parser.add_argument('--checkpoint',type=Path,help='Exact SHA-256-verified reviewed epoch100 local weight file')
    parser.add_argument('--checkpoint-file-id',help='Drive file ID for reviewed epoch100 weights if metadata mirrors are stale')
    args = parser.parse_args()
    def interrupt(number,frame):
        raise KeyboardInterrupt('Interrupted by signal '+str(number))
    for number in (signal.SIGINT,signal.SIGTERM):
        signal.signal(number,interrupt)
    run(args.training_root,args.test_images,args.output,checkpoint=args.checkpoint,
        checkpoint_file_id=args.checkpoint_file_id)


if __name__ == '__main__':
    main()
