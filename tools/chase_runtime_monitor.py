"""Small, stdlib-only launcher for the already restored FixPNG experiment.

This file belongs OUTSIDE code/<version>: the original training source identity
and all checkpoints remain unchanged. Resource observations diagnose a failure;
they do not establish its cause or change the training budget.
"""
from collections import deque
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

VERSION = '057b2b49a0cdf3e0'
ROOT_NAME = 'chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1'
SOURCES = {
 'chase_both_colab_bootstrap.py':'a1dff76c609e6381d832c6e0d64ec70cc221ecceb43065ccebb31682224f28ae',
 'chase_transfer_data.py':'fd5456d892c82de761b814296be5ab572c5c4bb337dfab6d41719f73c5927671',
 'chase_best84_training.py':'5899d2b74e3eb49f47a30b38bafe9d8b76bc7a3b6ff11a6c6e8c01f214eb6b17',
 'chase_best84_evaluation.py':'4e9459622af54f738186969a9778a30515ec7181e411775b331a2c8f49df2316',
 'chase_yolo_training.py':'f5e789f9cdd1cbbb97d190e4351a07360183d1071cdc809c2b7dcbbb19cf3778',
 'chase_yolo_predict.py':'4c86aff11ce14c27dd73579d898a27bd35e08df5461daf29e9f9cc42877f018f',
 'solar_a100_train_standalone.py':'d2777ba00c44aadca600ee4c535d9c88ed3a5094331b71536d0d90cb393c7963',
 'targeted_synthetic_training.py':'0c94ec171c57352986577ef4fe0d926243794eb731ab02d6a620019883626b49',
 'submission_inference.py':'7c6a9d829025801482350b2f009a51a7ffb1eaa3d270cdce80a6a0691f815227',
 'submission_driver.py':'53b10530fc31062db6633484d18f798a7ddb68757a4a13ab5c24c578c9a5568e',
 'submission_yolo_recipe.py':'28bfdd616ad30fe8eeb43d6d288e729936add4334acee58770f2240f4fcc36ad',
 'submission_native_recipe.py':'cd861374baa377eca6b0cdd5af1de4b53a21927d9d8d51f11327a7c484244606',
}
ASSETS = {
 'frozen_yolo_validation.json.gz':'9cfe08c74432b393f213d1d97fffed239da77a929fbe02b720aec84bf7939a3b',
 'train_split.csv':'aa0733424815731c5c108ada91e45525eb17ca3a42b377bf19e3089d86aede7e',
 'val_split.csv':'54c6a9aac5a3b70b3ecc6721109a8b0fe8ec020e90cb56e3848fccc92eadc61f',
 'chase_ahas2024_2025_manifest.json':'0321fccc3f887783bb70d6257d34aa1c5be65df55c27c9d13b35d2979315eb4d',
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name+'.'+str(os.getpid())+'.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    pending.replace(path)


def canonical_settings(root):
    root = Path(root); base = root.parents[1]
    return dict(root=str(root), assets=str(root/'assets'/VERSION),
        images=str(base/'MAGFiLO_1.0_Kaggle_2026/train/train_images'),
        annotations=str(base/'MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'),
        yolo_checkpoint='/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/best.pt',
        native_checkpoint='/content/drive/MyDrive/SolarFilament_NativeUNet_A100/20261028T030808Z_bedf3e21/best.pt',
        chase_root=str(base/'external_CHASE/ahas_2024_2025'),
        cache_root='/content/chase_only_both_alignment_60ep_fullpass_cache')


def validate_run(root):
    root = Path(root)
    if root.name != ROOT_NAME:
        raise ValueError('Use the existing FixPNG experiment folder; do not substitute an older run')
    code, assets = root/'code'/VERSION, root/'assets'/VERSION
    for folder, expected in ((code, SOURCES), (assets, ASSETS)):
        for name, sha in expected.items():
            path = folder/name
            if not path.is_file():
                raise FileNotFoundError('실행 자료가 없습니다. 작은 Runtime ZIP 설치 셀을 먼저 실행하세요: '+str(path))
            if digest(path) != sha:
                raise ValueError('기존 코드/자료가 변경됐습니다. 가중치는 보존합니다: '+str(path))
    manifest_path = code/'source_manifest.json'
    expected_manifest = dict(sources=SOURCES, assets=ASSETS)
    if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != expected_manifest:
        raise ValueError('FixPNG source_manifest.json does not match the pinned sources/assets')
    if {p.name for p in code.glob('*.py')} != set(SOURCES):
        raise ValueError('Unexpected Python module in the immutable training-code folder')
    path = root/'bootstrap_settings.json'
    if path.is_file():
        settings = json.loads(path.read_text())
        if settings.get('root') != str(root) or settings.get('assets') != str(assets):
            raise ValueError('Stored settings point to a different experiment/version')
        if set(settings) != set(canonical_settings(root)):
            raise ValueError('Unexpected saved bootstrap settings; nothing was overwritten')
    else:
        settings = canonical_settings(root)
        # Code and assets were independently verified above; no checkpoint is loaded.
        atomic_json(path, settings)
        print('검증한 FixPNG 코드의 실행 경로 설정을 복원했습니다.', flush=True)
    return settings, code


def tail_lines(path, count=20, limit=128*1024):
    path = Path(path)
    if not path.is_file(): return []
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-limit))
        return stream.read(limit).decode('utf-8', errors='replace').splitlines()[-count:]


def read_small_json(path, limit=8*1024**2):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > limit: return None
    try: return json.loads(path.read_text())
    except (ValueError, OSError): return None


def describe_previous(root):
    root = Path(root)
    print('이전 Drive 로그 마지막 부분:', flush=True)
    lines = tail_lines(root/'training.log')
    print('\n'.join(line[-500:] for line in lines) if lines else '(저장된 로그 없음)', flush=True)
    for arm in ('yolo', 'native'):
        history = read_small_json(root/arm/'history.json')
        result = read_small_json(root/arm/'result.json')
        epoch = history[-1].get('epoch') if isinstance(history, list) and history else None
        completed = result.get('epochs_completed') if isinstance(result, dict) else None
        print(f'{arm}: history 마지막 완료epoch={epoch}, result 완료epoch={completed}', flush=True)
    previous = read_small_json(root/'diagnostics/latest_heartbeat.json')
    if previous:
        keys = ('time_utc', 'phase', 'last_output', 'available_ram_bytes', 'boot_id', 'uptime_seconds', 'exit_code')
        print('이전 실행 관측:', json.dumps({k:previous.get(k) for k in keys}, ensure_ascii=False), flush=True)


def ensure_dependencies():
    for module in ('torch', 'torchvision'):
        if importlib.util.find_spec(module) is None:
            raise RuntimeError('PyTorch가 포함된 Colab 런타임을 사용하세요: '+module)
    wanted = {'ultralytics':'8.4.182', 'segmentation-models-pytorch':'0.5.0', 'timm':'1.0.20'}
    install = []
    for package, version in wanted.items():
        try: current = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: current = None
        if current != version: install.append(package+'=='+version)
    for module, package in [('cv2','opencv-python-headless'), ('numpy','numpy'), ('scipy','scipy'),
                             ('requests','requests'), ('pycocotools','pycocotools'), ('PIL','Pillow')]:
        if importlib.util.find_spec(module) is None: install.append(package)
    if install:
        constraints = Path('/content/chase_resume_torch_constraints.txt')
        constraints.write_text(''.join(p+'=='+importlib.metadata.version(p)+'\n' for p in ('torch','torchvision')))
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--constraint', str(constraints), *install], check=True)


def _text(path):
    try: return Path(path).read_text().strip()
    except OSError: return None


def resource_snapshot(pgid=None, *, include_gpu=False, disk_path='/content'):
    fields = {}
    for line in (_text('/proc/meminfo') or '').splitlines():
        key, _, value = line.partition(':')
        if key in ('MemTotal', 'MemAvailable'): fields[key] = int(value.split()[0])*1024
    rss, members = 0, []
    if pgid is not None:
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit(): continue
            try:
                if os.getpgid(int(entry.name)) != pgid: continue
                for line in (entry/'status').read_text().splitlines():
                    if line.startswith('VmRSS:'): rss += int(line.split()[1])*1024
                members.append(int(entry.name))
            except (OSError, ValueError): pass
    snapshot = dict(time_utc=datetime.now(timezone.utc).isoformat(), launcher_pid=os.getpid(),
        boot_id=_text('/proc/sys/kernel/random/boot_id'), uptime_seconds=None,
        available_ram_bytes=fields.get('MemAvailable'), total_ram_bytes=fields.get('MemTotal'),
        process_group=pgid, process_group_rss_sum_bytes=rss, process_pids=members,
        rss_note='Sum of RSS may count shared pages more than once; not private memory')
    uptime = _text('/proc/uptime')
    if uptime: snapshot['uptime_seconds'] = float(uptime.split()[0])
    try: snapshot['local_disk_free_bytes'] = shutil.disk_usage(disk_path).free
    except OSError: snapshot['local_disk_free_bytes'] = None
    cgroup = {}
    for name in ('memory.current', 'memory.max', 'memory.events'):
        value = _text('/sys/fs/cgroup/'+name)
        if value is not None: cgroup[name] = value
    snapshot['cgroup_v2'] = cgroup
    if not cgroup:
        snapshot['cgroup_v1'] = {name:_text('/sys/fs/cgroup/memory/'+name) for name in
            ('memory.usage_in_bytes', 'memory.limit_in_bytes', 'memory.failcnt')}
    if include_gpu:
        try:
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
            snapshot['gpu_mib_used_total_util_percent'] = gpu.stdout.strip() if gpu.returncode == 0 else 'unavailable'
        except (OSError, subprocess.TimeoutExpired): snapshot['gpu_mib_used_total_util_percent'] = 'unavailable'
    return snapshot


def stop_owned_group(process):
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: return
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait()


def monitored_command(command, root, *, cwd, environment, phase='all', interval=10., screen_interval=30.,
                      snapshot=resource_snapshot):
    root = Path(root); diagnostics = root/'diagnostics'; diagnostics.mkdir(parents=True, exist_ok=True)
    done = threading.Event(); state = dict(last_output='', phase=phase)
    tail = deque(maxlen=60); log_lock = threading.Lock()
    with (root/'training.log').open('a', encoding='utf-8', buffering=128*1024) as log:
        log.write('\n=== Short launcher '+datetime.now(timezone.utc).isoformat()+' ===\n')
        log.flush()
        process = subprocess.Popen(command, cwd=cwd, env=environment, start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors='replace')
        def observe():
            next_screen = 0.; next_gpu = 0.; last_gpu = None
            while not done.is_set():
                now = time.monotonic()
                try:
                    record = snapshot(process.pid, include_gpu=now >= next_gpu)
                    if now >= next_gpu:
                        next_gpu = now+30.
                        last_gpu = record.get('gpu_mib_used_total_util_percent')
                    record['gpu_mib_used_total_util_percent'] = last_gpu
                    record.update(state)
                    with log_lock: log.flush()
                    with (diagnostics/'runtime_resources.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(record, ensure_ascii=False)+'\n')
                    atomic_json(diagnostics/'latest_heartbeat.json', record)
                    if now >= next_screen:
                        available = record.get('available_ram_bytes')
                        amount = f'{available/1024**3:.2f} GiB' if available is not None else 'unknown'
                        gpu = str(last_gpu or 'unknown').replace('\n', '; ')
                        print('실행 중 | RAM 여유 '+amount+' | GPU 사용/전체MiB,활용% '+gpu+
                              ' | '+state['last_output'][-180:], flush=True)
                        next_screen = now+screen_interval
                except Exception as error:
                    if now >= next_screen:
                        print('자원 관측 기록 오류:', str(error), flush=True); next_screen = now+screen_interval
                done.wait(interval)
        observer = threading.Thread(target=observe, daemon=True); observer.start()
        last_flush = time.monotonic()
        try:
            for line in process.stdout:
                with log_lock:
                    log.write(line)
                    if time.monotonic()-last_flush >= 5:
                        log.flush(); last_flush = time.monotonic()
                tail.append(line); state['last_output'] = line.strip()[-500:]
                if line.startswith(('[1]', '[2]', '[3]', '[4]', '[5]', 'Traceback', 'RuntimeError:',
                        'ValueError:', 'FileNotFoundError:', 'CUDA out of memory', '전체 파이프라인', '비교:')):
                    print(line.rstrip(), flush=True)
            code = process.wait(); state['exit_code'] = code
            if code:
                stop_owned_group(process)
                print('실제 오류 마지막 출력:\n'+''.join(tail), flush=True)
                raise subprocess.CalledProcessError(code, command)
        except BaseException:
            stop_owned_group(process)
            raise
        finally:
            done.set(); observer.join(timeout=7); process.stdout.close()
            with log_lock: log.flush()
            last = read_small_json(diagnostics/'latest_heartbeat.json') or {}
            last.update(state, time_utc=datetime.now(timezone.utc).isoformat(), exit_code=process.poll())
            try: atomic_json(diagnostics/'latest_heartbeat.json', last)
            except OSError as error: print('최종 자원 관측 저장 실패:', error, flush=True)
    return code


def run_resume(root, run_until=60, arms=('yolo','native'), phase='all', *, install_dependencies=True):
    root = Path(root)
    if type(run_until) is not int or not 1 <= run_until <= 60: raise ValueError('run_until must be1–60')
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= {'yolo','native'}: raise ValueError('Invalid arms')
    if phase not in ('all', 'prepare', 'train', 'compare'): raise ValueError('Invalid phase')
    settings, code = validate_run(root)
    diagnostics = root/'diagnostics'; diagnostics.mkdir(exist_ok=True)
    with (diagnostics/'launcher.lock').open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('같은 실험의 짧은 실행 셀이 이미 동작 중입니다.')
        with (root/'orchestrator.lock').open('a+') as original_lock:
            try: fcntl.flock(original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise RuntimeError('기존 학습 프로세스가 아직 실행 중입니다. 중복 실행하지 않습니다.')
            fcntl.flock(original_lock, fcntl.LOCK_UN)
        describe_previous(root)
        if install_dependencies: ensure_dependencies()
        print('기존 FixPNG 가중치·설정으로 재개합니다. 전체60epoch, epoch당695장.', flush=True)
        print('화면 출력은 줄이고 전체 로그와 자원 관측은 Drive에 저장합니다:', diagnostics, flush=True)
        command = [sys.executable, '-u', str(code/'chase_both_colab_bootstrap.py'),
            '--settings', str(root/'bootstrap_settings.json'), '--run-until', str(run_until),
            '--arms', ','.join(arms), '--phase', phase]
        environment = dict(os.environ, PYTHONUNBUFFERED='1', TORCH_CUDNN_V8_API_DISABLED='1',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        monitored_command(command, root, cwd=code, environment=environment, phase=phase)
    result = read_small_json(root/'latest_result.json')
    if isinstance(result, dict):
        if result.get('comparison_available') and result.get('comparison_report'):
            report = read_small_json(result['comparison_report'])
            if isinstance(report, dict):
                for row in report.get('results', [])[:4]:
                    print(row.get('role'), 'PQ', row.get('pq'), 'FP', row.get('fp'), 'FN', row.get('fn'), flush=True)
        print('결과 ZIP:', result.get('archive'), flush=True)
    return result
