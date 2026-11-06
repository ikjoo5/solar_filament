"""Drive-persistent launcher for eight inference-only, independent conditions."""
from datetime import datetime, timezone
import fcntl
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
import time
import uuid
import zipfile

ROLES = ('old_yolo', 'new_yolo', 'old_native', 'new_native')
ROOT_NAME = 'chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1'
HISTORICAL_LIMB = {
    'old_yolo': dict(pq=.3966367650756154, tp_pairs=440, fp=350, fn=268),
    'new_yolo': dict(pq=.09147125781727895, tp_pairs=99, fp=550, fn=609),
    'old_native': dict(pq=.406801992949031, tp_pairs=485, fp=414, fn=223),
    'new_native': dict(pq=.06644953613444879, tp_pairs=71, fp=526, fn=637),
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    temporary.write_bytes(encoded(value)); temporary.replace(path)


def dependencies():
    for name in ('torch', 'torchvision'):
        if importlib.util.find_spec(name) is None:
            raise RuntimeError('PyTorch가 설치된 Colab T4 런타임을 선택하세요: '+name)
    packages = []
    for name, version in [('ultralytics', '8.4.182'),
                          ('segmentation-models-pytorch', '0.5.0'), ('timm', '1.0.20')]:
        try: installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: installed = None
        if installed != version:
            packages.append(name+'=='+version)
    for module, package in [('cv2', 'opencv-python-headless'), ('numpy', 'numpy'),
                            ('scipy', 'scipy'), ('pycocotools', 'pycocotools'),
                            ('PIL', 'Pillow'), ('requests', 'requests')]:
        if importlib.util.find_spec(module) is None:
            packages.append(package)
    if packages:
        constraints = Path('/content/chase_independent_torch_constraints.txt')
        constraints.write_text(''.join(name+'=='+importlib.metadata.version(name)+'\n'
                                       for name in ('torch', 'torchvision')))
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        '--constraint', str(constraints), *packages], check=True)


def tail(path, count=3):
    if not path.is_file(): return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-16384))
        value = stream.read().decode('utf-8', 'replace')
    value = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', value)
    return '\n'.join(value.splitlines()[-count:])[-8000:]


def stop_owned(process):
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: return
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait()


def run_worker(command, log, *, cwd, role, interval=30):
    environment = dict(os.environ, PYTHONUNBUFFERED='1', TORCH_CUDNN_V8_API_DISABLED='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with log.open('ab', buffering=0) as stream:
        stream.write(('\n=== independent '+role+' '+datetime.now(timezone.utc).isoformat()+' ===\n').encode())
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                status = process.poll()
                print(role+' | '+('실행 중 | ' if status is None else '종료 | ')+tail(log)[-700:], flush=True)
                atomic_json(log.with_suffix('.heartbeat.json'),
                            dict(role=role, pid=process.pid, exit_code=status,
                                 utc=datetime.now(timezone.utc).isoformat(), last_output=tail(log)))
                if status is not None: break
                time.sleep(interval)
            if status:
                print(tail(log, 40), flush=True)
                raise subprocess.CalledProcessError(status, command)
        except BaseException:
            stop_owned(process)
            raise


def validate_helpers(code, expected):
    required = {'chase_independent_inference.py', 'chase_independent_decode.py',
                'chase_independent_launcher.py'}
    if set(expected) != required or {p.name for p in code.glob('*.py')} != required:
        raise ValueError('독립 검증 코드 구성이 다릅니다.')
    for name, value in expected.items():
        if digest(code/name) != value:
            raise ValueError('저장된 독립 검증 코드가 바뀌었습니다: '+name)


def collect_results(root, code, expected):
    base = root/'independent_models_v1'
    records, entries = [], {}

    def add(path):
        path = Path(path).resolve()
        if not path.is_relative_to(base.resolve()) or path.suffix not in ('.json', '.gz', '.csv', '.py'):
            raise ValueError('예상하지 못한 결과 경로: '+str(path))
        relative = path.relative_to(base.resolve()).as_posix()
        entries[relative] = dict(path=path, name=relative, bytes=path.stat().st_size, sha256=digest(path))

    for name in expected: add(code/name)
    for role in ROLES:
        path = base/role/'role_results.json'
        value = json.loads(path.read_text())
        rows = value.get('results', [])
        if (value.get('status') != 'complete' or value.get('model_role') != role
                or len(rows) != 2 or {r.get('input_mode') for r in rows} != {'raw', 'limb'}):
            raise ValueError('두 입력 조건의 완료 결과가 필요합니다: '+role)
        add(path)
        for row in rows:
            identity = row['identity']
            if not isinstance(identity, str) or not re.fullmatch('[0-9a-f]{64}', identity):
                raise ValueError('잘못된 결과 식별자')
            directory = base/role/identity
            if Path(row['cache_root']).resolve() != directory.resolve():
                raise ValueError('모델별 결과 경로가 다릅니다.')
            metadata = json.loads((directory/'identity.json').read_text())
            if metadata.get('identity') != identity:
                raise ValueError('결과 식별자가 다릅니다.')
            protocol = metadata['protocol']
            if protocol.get('model_role') != role or protocol.get('input_mode') != row['input_mode']:
                raise ValueError('모델/입력 조건이 섞여 있습니다.')
            fingerprint = hashlib.sha256(json.dumps(protocol, ensure_ascii=False,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            if (fingerprint != identity
                    or protocol.get('helper_sha256') != expected['chase_independent_inference.py']
                    or protocol.get('decoder_sha256') != expected['chase_independent_decode.py']):
                raise ValueError('현재 코드와 완료된 추론 코드가 다릅니다.')
            if json.loads((directory/'metrics.json').read_text()) != row:
                raise ValueError('역할별 결과와 입력별 결과 기록이 다릅니다.')
            if digest(directory/'predictions.csv') != row['csv_sha256']:
                raise ValueError('완료된 예측 CSV가 변경됐습니다.')
            names = protocol['source_images']
            frames = sorted((directory/'frames').glob('*.json.gz'))
            if (len(names) != 91 or len(frames) != 91
                    or {p.name for p in frames} != {Path(n).stem+'.json.gz' for n in names}):
                raise ValueError('검증 이미지 91장의 캐시가 필요합니다.')
            for filename in ('identity.json', 'metrics.json', 'predictions.csv'): add(directory/filename)
            for frame in frames: add(frame)
            if row.get('model_role', role) != role:
                raise ValueError('집계 결과의 모델 역할이 다릅니다.')
            records.append(dict(row, model_role=role))
    by_role = {(row['model_role'], row['input_mode']): row for row in records}
    input_deltas = {role: by_role[role, 'raw']['pq']-by_role[role, 'limb']['pq'] for role in ROLES}
    weight_deltas = {family+'_'+mode: by_role['new_'+family, mode]['pq']-by_role['old_'+family, mode]['pq']
                     for family in ('yolo', 'native') for mode in ('raw', 'limb')}
    historical = {role: dict(cached_reference=reference,
                            fresh_minus_cached_pq=by_role[role, 'limb']['pq']-reference['pq'])
                  for role, reference in HISTORICAL_LIMB.items()}
    summary = dict(schema_version=1, conditions=8, models=4, independent_models=True,
                   training_performed=False, source_sha256=expected, results=records,
                   raw_minus_limb_pq=input_deltas, chase_minus_original_pq=weight_deltas,
                   historical_limb_advisory=historical,
                   historical_note='Cached limb results are advisory controls. Runtime differences may change fresh inference; primary raw/limb comparisons use this fresh run.',
                   scope='Fixed development validation91; no fusion; not leaderboard scores')
    atomic_json(base/'summary.json', summary); add(base/'summary.json')
    return summary, list(entries.values())


def package_results(base, entries, summary, *, limit=24*1024**2):
    manifest = dict(schema_version=1, protocol='independent old/new x YOLO/native x raw/limb',
                    conditions=summary['conditions'], models=summary['models'],
                    files=[{k:v for k,v in entry.items() if k != 'path'} for entry in entries])
    overhead = len(encoded(manifest))+16384
    groups, current, size = [], [], overhead
    for entry in entries:
        cost = entry['bytes']+2*len(entry['name'].encode())+200
        if cost+overhead > limit: raise ValueError('단일 결과 파일이 ZIP 분할 제한보다 큽니다: '+entry['name'])
        if current and size+cost > limit:
            groups.append(current); current, size = [], overhead
        current.append(entry); size += cost
    if current: groups.append(current)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
    output = Path(base)/'exports'/stamp; output.mkdir(parents=True)
    paths = []
    for number, group in enumerate(groups, 1):
        path = output/f'chase_independent_{stamp}_part{number:02d}.zip'
        pending = path.with_suffix('.pending')
        with zipfile.ZipFile(pending, 'w', zipfile.ZIP_STORED) as archive:
            archive.writestr('export_manifest.json', encoded(dict(manifest, part=number, parts=len(groups))))
            for entry in group:
                data = entry['path'].read_bytes()
                if hashlib.sha256(data).hexdigest() != entry['sha256']:
                    raise ValueError('결과 파일이 내보내는 중 변경됐습니다: '+entry['name'])
                archive.writestr(entry['name'], data)
        if pending.stat().st_size > limit:
            pending.unlink(); raise ValueError('ZIP 분할 크기 제한 초과')
        pending.replace(path); paths.append(str(path))
    atomic_json(Path(base)/'latest_export.json', dict(archives=paths, conditions=8, completed=True))
    return paths


def run_experiment(root, code, expected, *, install_dependencies=True, worker=run_worker):
    root, code = Path(root).resolve(), Path(code).resolve()
    if root.name != ROOT_NAME: raise ValueError('기존 CHASE PNG 수정판 실험 경로를 사용하세요.')
    base = root/'independent_models_v1'; base.mkdir(parents=True, exist_ok=True)
    if not code.is_relative_to((base/'code').resolve()): raise ValueError('검증 코드는 별도 code 폴더에 있어야 합니다.')
    validate_helpers(code, expected)
    with (base/'launcher.lock').open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('같은 런타임에서 독립 검증이 이미 실행 중입니다.')
        if install_dependencies: dependencies()
        log = base/'inference.log'
        print('독립 모델 4개 × 입력 2조건. 추가학습 없이 검증91장을 추론합니다.', flush=True)
        print('전체 로그:', log, flush=True)
        for role in ROLES:
            command = [sys.executable, '-u', str(code/'chase_independent_inference.py'),
                       '--root', str(root), '--model-role', role, '--device', 'cuda', '--batch', '1']
            worker(command, log, cwd=code, role=role)
        summary, entries = collect_results(root, code, expected)
        for row in summary['results']:
            print(f"{row['model_role']:12s} {row['input_mode']:4s} PQ={row['pq']:.6f} "
                  f"TP={row['tp_pairs']} FP={row['fp']} FN={row['fn']}", flush=True)
        print('같은 실행의 원본 입력 PQ − limb 입력 PQ:', summary['raw_minus_limb_pq'], flush=True)
        print('이전 limb 캐시 대비 변화(참고용):',
              {role: row['fresh_minus_cached_pq'] for role, row in summary['historical_limb_advisory'].items()}, flush=True)
        archives = package_results(base, entries, summary)
        for archive in archives: print('업로드할 결과 ZIP:', archive, flush=True)
        return dict(archives=archives, summary=str(base/'summary.json'))
