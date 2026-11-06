"""Build one readable Colab cell without encoded source."""
import hashlib
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
DEST = HERE.parents[1]/'artifacts/downloads'
STEM = 'SolarFilament_CHASE_Independent_Plain'

START = """# 한 셀 전체를 복사하세요. 추가학습 없이 독립 추론합니다.
from google.colab import drive, files
drive.mount('/content/drive')
from pathlib import Path
from datetime import datetime, timezone
import fcntl, hashlib, importlib.metadata, importlib.util
import json, os, signal, subprocess, sys, time, uuid, zipfile
ROOT = Path('/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1')
BASE = ROOT/'independent_models_plain_v1'
"""

END = r'''
WORKER_SHA = hashlib.sha256(WORKER.encode()).hexdigest()
compile(WORKER, 'chase_independent_plain_worker.py', 'exec')
if not (ROOT/'configs/comparison.json').is_file():
    raise FileNotFoundError('기존 CHASE 학습 완료 폴더가 없습니다: '+str(ROOT))
CODE = BASE/'code'/WORKER_SHA[:16]
SCRIPT = CODE/'chase_independent_plain_worker.py'
if SCRIPT.exists() and SCRIPT.read_text() != WORKER:
    raise ValueError('저장된 검증 코드를 덮어쓰지 않습니다: '+str(SCRIPT))
CODE.mkdir(parents=True, exist_ok=True)
if not SCRIPT.exists():
    pending = SCRIPT.with_suffix('.pending'); pending.write_text(WORKER); pending.replace(SCRIPT)

def ensure_packages():
    for name in ('torch','torchvision'):
        if importlib.util.find_spec(name) is None: raise RuntimeError('Colab T4 런타임을 선택하세요.')
    packages = []
    for name, version in [('ultralytics','8.4.182'),('segmentation-models-pytorch','0.5.0'),('timm','1.0.20')]:
        try: current = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: current = None
        if current != version: packages.append(name+'=='+version)
    for name, package in [('cv2','opencv-python-headless'),('numpy','numpy'),('scipy','scipy'),('pycocotools','pycocotools'),('PIL','Pillow'),('requests','requests')]:
        if importlib.util.find_spec(name) is None: packages.append(package)
    if packages:
        path = Path('/content/chase_plain_constraints.txt')
        path.write_text(''.join(name+'=='+importlib.metadata.version(name)+'\n' for name in ('torch','torchvision')))
        subprocess.run([sys.executable,'-m','pip','install','-q','--constraint',str(path),*packages],check=True)

def log_tail(path, count=3):
    if not path.exists(): return ''
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-16000))
        return '\n'.join(stream.read().decode('utf-8','replace').splitlines()[-count:])

def stop(process):
    try: os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError: return
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait()

def execute(role):
    log = BASE/'inference.log'
    env = dict(os.environ,PYTHONUNBUFFERED='1',TORCH_CUDNN_V8_API_DISABLED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    with log.open('ab',buffering=0) as stream:
        stream.write(('\n=== '+role+' ===\n').encode())
        process = subprocess.Popen([sys.executable,'-u',str(SCRIPT),'--root',str(ROOT),'--role',role],
            cwd=CODE,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                print(role+' | '+log_tail(log)[-600:],flush=True); time.sleep(30)
            if process.returncode:
                print(log_tail(log,40),flush=True)
                raise subprocess.CalledProcessError(process.returncode,process.args)
        except BaseException:
            stop(process); raise

def export_results(roles):
    rows = []; entries = {}
    def add(path):
        path = Path(path).resolve()
        if not path.is_relative_to(BASE.resolve()): raise ValueError('결과 경로가 실험 폴더 밖입니다.')
        data = path.read_bytes(); name = path.relative_to(BASE.resolve()).as_posix()
        entries[name] = dict(name=name,path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    add(SCRIPT)
    for role in roles:
        path = BASE/role/'role_results.json'; report = json.loads(path.read_text())
        if report['status'] != 'complete' or report['model_role'] != role or len(report['results']) != 2 or {x['input_mode'] for x in report['results']} != {'raw','limb'}:
            raise ValueError('원본/limb 조건이 완료되지 않았습니다: '+role)
        add(path)
        for row in report['results']:
            folder = BASE/role/row['identity']
            if Path(row['cache_root']).resolve() != folder.resolve(): raise ValueError('캐시 경로가 다릅니다.')
            metadata = json.loads((folder/'identity.json').read_text()); protocol = metadata['protocol']
            fingerprint = hashlib.sha256(json.dumps(protocol,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            if metadata['identity'] != row['identity'] or fingerprint != row['identity'] or protocol['helper_sha256'] != WORKER_SHA or protocol['model_role'] != role or protocol['input_mode'] != row['input_mode']:
                raise ValueError('실행 조건/코드 식별자가 다릅니다.')
            if json.loads((folder/'metrics.json').read_text()) != row or hashlib.sha256((folder/'predictions.csv').read_bytes()).hexdigest() != row['csv_sha256']:
                raise ValueError('완료된 결과가 변경됐습니다.')
            frames = sorted((folder/'frames').glob('*.json.gz'))
            if len(frames) != 91 or {p.name for p in frames} != {Path(n).stem+'.json.gz' for n in protocol['source_images']}:
                raise ValueError('검증91장 캐시가 완전하지 않습니다.')
            for name in ('identity.json','metrics.json','predictions.csv'): add(folder/name)
            for path in frames: add(path)
            rows.append(row)
            print(role,row['input_mode'],f"PQ={row['pq']:.6f}",f"TP={row['tp_pairs']} FP={row['fp']} FN={row['fn']}",flush=True)
    summary = dict(conditions=8,independent_models=4,training=False,results=rows,
        scope='Fixed validation91; not independent holdout or leaderboard',
        cached_limb_reference=dict(old_yolo=.3966367650756154,new_yolo=.09147125781727895,old_native=.406801992949031,new_native=.06644953613444879))
    path = BASE/'summary.json'; path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)); add(path)
    listing = [{k:v for k,v in e.items() if k != 'path'} for e in entries.values()]
    overhead = len(json.dumps(listing).encode())+16384; limit = 24*1024**2
    groups = []; group = []; size = overhead
    for entry in entries.values():
        cost = entry['bytes']+2*len(entry['name'].encode())+200
        if cost+overhead > limit: raise ValueError('단일 결과가 ZIP 분할 제한보다 큽니다.')
        if group and size+cost > limit: groups.append(group); group=[]; size=overhead
        group.append(entry); size+=cost
    if group: groups.append(group)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]
    destination = BASE/'exports'/stamp; destination.mkdir(parents=True)
    paths = []
    for number, group in enumerate(groups,1):
        path = destination/f'chase_independent_plain_{stamp}_part{number:02d}.zip'
        pending = path.with_suffix('.pending')
        with zipfile.ZipFile(pending,'w',zipfile.ZIP_STORED) as archive:
            archive.writestr('export_manifest.json',json.dumps(dict(part=number,parts=len(groups),files=listing,conditions=8,worker_sha256=WORKER_SHA)))
            for entry in group:
                data = entry['path'].read_bytes()
                if hashlib.sha256(data).hexdigest() != entry['sha256']: raise ValueError('내보내는 중 결과가 변경됐습니다.')
                archive.writestr(entry['name'],data)
        if pending.stat().st_size > limit: pending.unlink(); raise ValueError('ZIP 크기 제한 초과')
        pending.replace(path); paths.append(path)
    return paths

ROLES = ('old_yolo','new_yolo','old_native','new_native')
with (BASE/'launcher.lock').open('a+') as lock:
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('같은 런타임에서 검증이 이미 실행 중입니다.')
    ensure_packages()
    print('모델4개 × 입력2조건: 추가학습 없이 독립 검증합니다. 로그:',BASE/'inference.log')
    for role in ROLES: execute(role)
    archives = export_results(ROLES)
for archive in archives:
    print('이 채팅에 올릴 결과 ZIP:',archive)
    try: files.download(str(archive))
    except Exception as exc: print('Drive에 저장된 결과:',archive,str(exc))
'''


def build(dest=DEST, worker_path=None):
    dest = Path(dest); dest.mkdir(parents=True,exist_ok=True)
    worker_path = Path(worker_path or HERE/'chase_independent_plain_worker.py')
    worker = worker_path.read_text()
    if "'''" in worker: raise ValueError('Worker contains delivery delimiter')
    compile(worker,str(worker_path),'exec')
    digest = hashlib.sha256(worker.encode()).hexdigest()
    cell = START+"WORKER = r'''"+worker+"'''\n"+END
    compile(cell,'chase_plain_colab.py','exec')
    if len(cell.encode()) > 24*1024: raise ValueError('Plain cell too large')
    path = dest/(STEM+'_CopyCell.txt'); path.write_text(cell)
    notebook = dict(nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',kernelspec=dict(display_name='Python 3',name='python3',language='python')),
        cells=[dict(cell_type='code',id='independent-plain',metadata={},execution_count=None,outputs=[],source=cell.splitlines(keepends=True))])
    (dest/(STEM+'_OneCell.ipynb')).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    result = dict(copy_cell=str(path),worker_sha256=digest,cell_bytes=len(cell.encode()),worker_bytes=len(worker.encode()),
        encoded_source=False,extra_training=False,output_namespace='independent_models_plain_v1')
    (dest/(STEM+'_manifest.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__ == '__main__': print(json.dumps(build(),ensure_ascii=False,indent=2))
