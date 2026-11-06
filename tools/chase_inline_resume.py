"""Small plain-text resume helper; keep outside the immutable training code."""
import fcntl, hashlib, importlib.metadata, importlib.util, json, os, re
import signal, subprocess, sys, time
from pathlib import Path


def _tail(path, count=3):
    if not path.is_file(): return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-12000))
        text = stream.read().decode('utf-8', 'replace')
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
    return '\n'.join(text.splitlines()[-count:])[-6000:]


def _stop(process):
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: return
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait()


def _validate(root):
    version = '057b2b49a0cdf3e0'
    if root.name != 'chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1':
        raise ValueError('기존 PNG 수정판 실험 폴더를 사용해야 합니다.')
    code, assets = root/'code'/version, root/'assets'/version
    manifest = code/'source_manifest.json'
    expected = '31145ebed1889291a129f8ae3ed6c40833080e233a65ae094ba02245f4c615e2'
    if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
        raise ValueError('먼저 기존 코드 복원을 완료해야 합니다: '+str(manifest))
    data = json.loads(manifest.read_text())
    for folder, files in [(code, data['sources']), (assets, data['assets'])]:
        for name, sha in files.items():
            path = folder/name
            if not path.is_file(): raise FileNotFoundError(str(path))
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != sha: raise ValueError('파일 내용이 변경됐습니다: '+str(path))
    if {p.name for p in code.glob('*.py')} != set(data['sources']):
        raise ValueError('학습 코드 폴더에 다른 Python 파일이 있습니다.')
    path = root/'bootstrap_settings.json'
    if not path.exists():
        base = root.parents[1]
        settings = dict(root=str(root), assets=str(assets),
            images=str(base/'MAGFiLO_1.0_Kaggle_2026/train/train_images'),
            annotations=str(base/'MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'),
            yolo_checkpoint='/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/best.pt',
            native_checkpoint='/content/drive/MyDrive/SolarFilament_NativeUNet_A100/20261028T030808Z_bedf3e21/best.pt',
            chase_root=str(base/'external_CHASE/ahas_2024_2025'),
            cache_root='/content/chase_only_both_alignment_60ep_fullpass_cache')
        pending = path.with_suffix('.pending')
        pending.write_text(json.dumps(settings, ensure_ascii=False, indent=2)+'\n'); pending.replace(path)
    settings = json.loads(path.read_text())
    if settings.get('root') != str(root) or settings.get('assets') != str(assets):
        raise ValueError('저장된 설정의 실험 경로가 다릅니다.')
    return code


def _dependencies():
    for name in ('torch', 'torchvision'):
        if importlib.util.find_spec(name) is None:
            raise RuntimeError('PyTorch가 설치된 Colab GPU 런타임을 선택하세요.')
    packages = []
    for name, version in [('ultralytics','8.4.182'), ('segmentation-models-pytorch','0.5.0'), ('timm','1.0.20')]:
        try: current = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: current = None
        if current != version: packages.append(name+'=='+version)
    for module, name in [('cv2','opencv-python-headless'), ('numpy','numpy'), ('scipy','scipy'),
                         ('requests','requests'), ('pycocotools','pycocotools'), ('PIL','Pillow')]:
        if importlib.util.find_spec(module) is None: packages.append(name)
    if packages:
        constraints = Path('/content/chase_resume_torch_constraints.txt')
        constraints.write_text(''.join(name+'=='+importlib.metadata.version(name)+'\n'
                                       for name in ('torch', 'torchvision')))
        subprocess.run([sys.executable,'-m','pip','install','-q','--constraint',str(constraints),*packages],check=True)


def _run_logged(command, root, code, interval=30):
    log = root/'training.log'
    heartbeat = root/'inline_heartbeat.json'
    environment = dict(os.environ, PYTHONUNBUFFERED='1', TORCH_CUDNN_V8_API_DISABLED='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with log.open('ab', buffering=0) as stream:
        stream.write(('\n=== 작은 셀 재개 '+time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())+' ===\n').encode())
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   cwd=code, env=environment, start_new_session=True)
        try:
            while True:
                status = process.poll()
                last = _tail(log)
                record = dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                              pid=process.pid, exit_code=status, last_output=last)
                pending = heartbeat.with_suffix('.pending')
                pending.write_text(json.dumps(record, ensure_ascii=False)); pending.replace(heartbeat)
                print(('실행 중 | ' if status is None else '종료 | ')+last[-600:], flush=True)
                if status is not None: break
                time.sleep(interval)
            if status:
                print(_tail(log, 35), flush=True)
                raise subprocess.CalledProcessError(status, command)
        except BaseException:
            _stop(process)
            raise
    return status


def run_inline(root, run_until=60, install_dependencies=True):
    root = Path(root)
    if type(run_until) is not int or not 1 <= run_until <= 60: raise ValueError('에폭은1~60입니다.')
    code = _validate(root)
    with (root/'inline_launcher.lock').open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('같은 실행 셀이 이미 실행 중입니다.')
        with (root/'orchestrator.lock').open('a+') as original:
            try: fcntl.flock(original, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise RuntimeError('기존 학습이 실행 중입니다. 중복 실행하지 않습니다.')
            fcntl.flock(original, fcntl.LOCK_UN)
        if install_dependencies: _dependencies()
        print('기존 완료 에폭부터 재개합니다. 전체 로그:',root/'training.log',flush=True)
        command = [sys.executable,'-u',str(code/'chase_both_colab_bootstrap.py'),
                   '--settings',str(root/'bootstrap_settings.json'),'--run-until',str(run_until),
                   '--arms','yolo,native','--phase','all']
        _run_logged(command, root, code)
    result = json.loads((root/'latest_result.json').read_text())
    print('결과 ZIP:',result.get('archive'))
    return result
