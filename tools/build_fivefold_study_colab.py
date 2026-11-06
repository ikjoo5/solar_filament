"""Package the fresh fivefold study and all code needed after a Colab reset."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
ANN_SHA = '5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1'
RUNTIME_SHA = '6d60b5705fb0fbde03a6fd2922c8b83839fa49b5a78cc3434e7dcf4a1b4a08b4'
MODULES = ('grouped_cv_training.py', 'prepare_grouped_fivefold.py',
    'evaluate_grouped_oof.py', 'submission_native_recipe.py', 'fivefold_study.py',
    'fivefold_rich_inference.py', 'evaluate_fivefold_study.py',
    'fivefold_yolo_training.py', 'fivefold_full_pipeline.py',
    'submission_inference.py', 'submission_yolo_recipe.py', 'fixed_split_runner.py')

LAUNCHER = r'''"""Drive-persistent launcher for equal-budget fivefold experiments."""
import argparse, hashlib, importlib.metadata, json, os, subprocess, sys, tempfile, zipfile
from pathlib import Path

def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');pending.replace(path)

def preserve(path,raw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_bytes()!=raw: raise ValueError('Existing different file preserved: '+str(path))
    if not path.exists():
        pending=path.with_name(path.name+'.pending');pending.write_bytes(raw);pending.replace(path)

def prepare_layout(root,annotation):
    from prepare_grouped_fivefold import build_plan,write_outputs
    plan=build_plan(json.loads(annotation.read_text()))
    plan['sources_sha256']={'annotations':sha(annotation)}
    with tempfile.TemporaryDirectory() as temporary:
        generated=Path(temporary)/'splits';write_outputs(plan,generated)
        for path in sorted(generated.rglob('*')):
            if path.is_file():
                relative=path.relative_to(generated)
                name=Path('fold_plan.json') if str(relative)=='plan.json' else relative
                preserve(root/name,path.read_bytes())
    for info in plan['fold_summaries']:
        fold=info['fold']
        roles={role:[row for row in plan['observations'] if info['roles'][row['physical_identifier']]==role]
               for role in ('train','validation','embargo')}
        record=dict(fold=fold,annotations_sha256=sha(annotation),fold_plan_sha256=sha(root/'fold_plan.json'),
            temporal_block_days=27,embargo_days=3,summary={k:v for k,v in info.items() if k!='roles'},
            images={role:[r['file_name'] for r in rows] for role,rows in roles.items()},
            initialization='Fresh generic ImageNet; no best84 or oldfold0 weights',synthetic_ratio=0)
        preserve(root/f'fold{fold}/split_manifest.json',(json.dumps(record,ensure_ascii=False,indent=2)+'\n').encode())
    return plan

def execute(command,log_path,environment=None):
    with log_path.open('a',buffering=1) as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,bufsize=1,env=environment)
        try:
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line)
            code=process.wait()
        except BaseException:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
    if code: raise RuntimeError(f'Execution failed(exit{code}). Completed checkpoints are preserved. Log: {log_path}')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--stage',type=int);parser.add_argument('--profile',choices=['tta','tta_clahe'])
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();root=Path(args.root)
    settings=json.loads((root/'launch_config.json').read_text())
    previous=json.loads((root/'current_request.json').read_text()) if (root/'current_request.json').exists() else {}
    stage=args.stage if args.stage is not None else previous.get('stage_epoch',30)
    profile=args.profile or previous.get('inference_profile','tta_clahe')
    if not 1<=stage<=100:raise ValueError('Stage must be1..100; initialstage30perfold')
    pointer=json.loads((root/'current_code.json').read_text());code_dir=root/'code'/pointer['version']
    manifest=json.loads((code_dir/'source_manifest.json').read_text())
    work=Path('/content/solar_fivefold_study_v2');work.mkdir(parents=True,exist_ok=True)
    for name,expected in manifest.items():
        source=code_dir/name
        if sha(source)!=expected:raise ValueError('Stored study source changed: '+name)
        (work/name).write_bytes(source.read_bytes())
    sys.path.insert(0,str(work))
    base=Path(settings['base']);annotation=base/'MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    if sha(annotation)!=settings['annotations_sha256']:raise ValueError('Official annotation checksum differs')
    plan=prepare_layout(root,annotation)
    save(root/'current_request.json',dict(stage_epoch=stage,inference_profile=profile,folds=list(range(5))))
    print(f'이번 실행:5개fold의YOLO/native각각{stage}epoch,총{stage*10}model-epoch. 완료된작업은재사용합니다.',flush=True)
    if args.prepare_only:
        print(json.dumps(dict(stage_epoch=stage,per_branch_epochs=5*stage,total_initial_model_epochs=10*stage,
              folds=[{k:v for k,v in row.items() if k!='roles'} for row in plan['fold_summaries']]),indent=2));return
    pins={'segmentation-models-pytorch':'0.5.0','timm':'1.0.20','pycocotools':'2.0.11',
          'ultralytics':'8.4.182','huggingface-hub':'0.34.4'}
    needed=[]
    for name,version in pins.items():
        try:actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:actual=None
        if actual!=version:needed.append(name+'=='+version)
    for module,package in [('cv2','opencv-python-headless'),('scipy','scipy')]:
        try:__import__(module)
        except ImportError:needed.append(package)
    if needed:subprocess.run([sys.executable,'-m','pip','install','-q',*needed],check=True)
    subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(), "Select a T4 GPU runtime"; print("GPU:",torch.cuda.get_device_name(0),"torch:",torch.__version__)'],check=True)
    config=dict(study_root=str(root),images=str(base/'MAGFiLO_1.0_Kaggle_2026/train/train_images'),
        annotations=str(annotation),fold_plan=str(root/'fold_plan.json'),splits_root=str(root),
        base_runtime=str(work/'base_runtime.py'),cache_root='/content/solar_fivefold_study_v2_cache',
        stage_epoch=stage,folds=[0,1,2,3,4],inference_profile=profile)
    policy=dict(fivefold_role=['inference_preprocessing','postprocessing'],
        fixed_split_role=['architecture','loss','optimizer','learning_rate','training_augmentation','other_retraining'],
        fixed_train=str(Path('/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/train_split.csv')),
        fixed_validation=str(Path('/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/val_split.csv')),
        baseline='Selected production YOLO26s+native recipe, historicalPQ0.472593/publicLB0.39',
        stages_per_model=stage,folds=5,model_branches=2,total_initial_model_epochs=stage*5*2,
        new_architecture_weights_are_not_imported_into_existing_OOF_models=True)
    save(root/'experiment_policy.json',policy)
    old_run=Path('/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes')
    fixed=dict(images=config['images'],annotations=str(annotation),
        train_split=str(old_run/'train_split.csv'),val_split=str(old_run/'val_split.csv'),
        base_runtime=str(code_dir/'base_runtime.py'),output_root=str(base/'fixed_split_experiments/CHANGE_EXPERIMENT_NAME'),
        cache_root='/content/solar_fixed_split_cache',device='cuda',encoder='tu-convnext_tiny',pretrained=True,
        epochs=30,crop=1024,batch=1,accumulate=4,steps_per_epoch=400,workers=0,infer_batch=1,
        validate_every=5,lr=1e-4,spine_weight=.1,synthetic_ratio=.2,warmup_epochs=3,real_finish_epochs=6,
        synthetic_images=str(base/'synthetic_copy_paste_v2/images'),
        synthetic_annotations=str(base/'synthetic_copy_paste_v2/synthetic_annotations.json'),
        synthetic_split=str(base/'synthetic_copy_paste_v2/split_info.json'),include_empty_synthetic=True,
        thresholds=[.5],min_areas=[300],overlap=.25,seed=42,resume_dir=None,
        expected_source_sha256=dict(annotations=settings['annotations_sha256'],
            train_split='aa0733424815731c5c108ada91e45525eb17ca3a42b377bf19e3089d86aede7e',
            val_split='54c6a9aac5a3b70b3ecc6721109a8b0fe8ec020e90cb56e3848fccc92eadc61f',
            synthetic_annotations='5e7ad5f20fa12f0c820a82a315ba60e802480d54b4b3ecebd9fba25ae8fc6aea',
            synthetic_split='e8c9f3848c7b1fe1283167d973e459a5df7cd6336cc1d871268a34024d1ea881'))
    save(root/'fixed_split_model_template.json',fixed)
    logs=root/'logs';logs.mkdir(exist_ok=True)
    environment=dict(os.environ);environment.pop('TORCH_CUDNN_V8_API_DISABLED',None)
    print(f'두분기 각각{stage}epoch:5fold전체 총{stage*10}model-epoch. 모델재학습실험은 기존616/91분할로 진행합니다.',flush=True)
    for fold in range(5):
        request=dict(config,folds=[fold]);config_path=work/f'native_request_fold{fold}.json';save(config_path,request)
        execute([sys.executable,'-u',str(work/'fivefold_study.py'),'--config',str(config_path),
                 '--result',str(root/f'stage{stage:03d}_run_result.json')],logs/f'native_fold{fold}_stage{stage:03d}.log',environment)
        trained=json.loads((root/f'fold{fold}/config.json').read_text())
        yolo_cfg=dict(trained,study_root=str(root),stage_epoch=stage)
        yolo_request=work/f'yolo_request_fold{fold}.json';save(yolo_request,yolo_cfg)
        execute([sys.executable,'-u',str(work/'fivefold_yolo_training.py'),'--config',str(yolo_request),
            '--result',str(root/f'fold{fold}/yolo/stage{stage:03d}.json')],logs/f'yolo_fold{fold}_stage{stage:03d}.log',environment)
        full_cfg=dict(study_root=str(root),stage_epoch=stage,fold=fold,images=config['images'],device='cuda')
        full_request=work/f'full_request_fold{fold}.json';save(full_request,full_cfg)
        execute([sys.executable,'-u',str(work/'fivefold_full_pipeline.py'),'--config',str(full_request),
            '--result',str(root/f'fold{fold}/stage{stage:03d}/full_export_result.json')],logs/f'full_fold{fold}_stage{stage:03d}.log',environment)
    source_result=json.loads((root/f'stage{stage:03d}_run_result.json').read_text())
    stamp=hashlib.sha256(json.dumps(dict(source=source_result,comparer=sha(work/'fivefold_full_pipeline.py')),
                                    sort_keys=True).encode()).hexdigest()[:12]
    comparison=root/'comparisons'/f'stage{stage:03d}_{stamp}'
    report_path=comparison/'comparison.json'
    from fivefold_full_pipeline import evaluate as evaluate_full
    full_report=report_path
    evaluate_full(str(annotation),str(root/'fold_plan.json'),str(root),stage,str(full_report))
    report=json.loads(full_report.read_text())
    archive=root/f'fivefold_stage{stage:03d}_results.zip';pending=archive.with_name(archive.name+'.pending')
    with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ('launch_config.json','study_config.json','fold_plan.json','image_fold_manifest.csv',
                     'experiment_policy.json','fixed_split_model_template.json',full_report.name,
                     f'stage{stage:03d}_plan.json',f'stage{stage:03d}_progress.json'):
            if (root/name).is_file():z.write(root/name,name)
        for fold in range(5):
            folder=root/f'fold{fold}'
            for name in ('config.json','history.json','train_split.csv','val_split.csv',
                         'excluded_embargo.csv','split_manifest.json'):
                if (folder/name).is_file():z.write(folder/name,str((folder/name).relative_to(root)))
            for name in ('stage_summary.json','export_summary.json','rich_records.jsonl','full_export_result.json'):
                source=folder/f'stage{stage:03d}'/name
                if source.is_file():z.write(source,str(source.relative_to(root)))
            for source in (folder/'yolo/config.json',folder/'yolo/history.json',folder/'yolo'/f'stage{stage:03d}.json'):
                if source.is_file():z.write(source,str(source.relative_to(root)))
            full=folder/f'stage{stage:03d}'/'full_pipeline'
            for source in sorted(full.rglob('*')):
                if source.is_file() and 'records' not in source.relative_to(full).parts:z.write(source,str(source.relative_to(root)))
        for source in sorted(comparison.rglob('*')):
            if source.is_file():z.write(source,str(source.relative_to(root)))
    pending.replace(archive)
    save(root/'latest_result.json',dict(stage_epoch=stage,folds=[0,1,2,3,4],
         archive=str(archive),comparison=str(report_path),status=report['status']))
    print('결과 ZIP:',archive,flush=True)
    print('YOLO+native의추론파이프라인을복원했습니다. 새fold가중치와30epoch예산이므로 이전PQ를보장하지않습니다.',flush=True)

if __name__=='__main__':
    import fcntl
    destination=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    key=hashlib.sha256(str(destination).encode()).hexdigest()[:16]
    with (Path('/tmp')/('solar_fivefold_master_'+key+'.lock')).open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('This fivefold study is already running in this runtime') from error
        main()
'''

INITIAL = r'''# 새5-fold실험: 각fold의YOLO/native각30epoch(총300model-epoch), 5개순차실행.
STAGE_EPOCH = 30
INFERENCE_PROFILE = 'tta_clahe'
from pathlib import Path
from google.colab import drive,files
import base64,hashlib,json,os,signal,subprocess,sys,zlib
drive.mount('/content/drive')
BASE=Path('/content/drive/MyDrive/filament-segmentation-2026')
ROOT=BASE/'fivefold_study_v2'
PAYLOAD=__PAYLOAD__
raw=zlib.decompress(base64.b64decode(PAYLOAD))
if hashlib.sha256(raw).hexdigest()!=__SHA__:raise ValueError('복사셀코드checksum이 다릅니다.')
sources=json.loads(raw);version=__SHA__[:12]
CODE=ROOT/'code'/version;CODE.mkdir(parents=True,exist_ok=True)
def preserve(path,content):
    if path.exists() and path.read_bytes()!=content:raise ValueError('내용이다른기존파일을보존했습니다: '+str(path))
    if not path.exists():
        pending=path.with_name(path.name+'.pending');pending.write_bytes(content);pending.replace(path)
for name,source in sources.items():preserve(CODE/name,source.encode())
manifest={name:hashlib.sha256((CODE/name).read_bytes()).hexdigest() for name in sources}
preserve(CODE/'source_manifest.json',(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode())
settings=dict(base=str(BASE),annotations_sha256=__ANN_SHA__,folds=[0,1,2,3,4],initial_stage_epoch=30)
preserve(ROOT/'launch_config.json',(json.dumps(settings,indent=2)+'\n').encode())
(ROOT/'current_code.json').write_text(json.dumps(dict(version=version))+'\n')
command=[sys.executable,'-u',str(CODE/'study_launcher.py'),'--root',str(ROOT),
         '--stage',str(STAGE_EPOCH),'--profile',INFERENCE_PROFILE]
__EXECUTE__
result=json.loads((ROOT/'latest_result.json').read_text())
print('업로드할결과:',result['archive']);files.download(result['archive'])
'''

PROCESS = r'''process=subprocess.Popen(command,start_new_session=True)
try:
    code=process.wait()
    if code:raise subprocess.CalledProcessError(code,command)
except BaseException:
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait()
    raise'''

RESUME = r'''# 새v2실험재개: 폴드번호를선택할필요없이5개중완료된작업을확인해이어갑니다.
STAGE_EPOCH=None  # None=직전목표(기본30); 후속학습시60등공통목표로변경
from pathlib import Path
from google.colab import drive,files
import hashlib,json,os,signal,subprocess,sys
drive.mount('/content/drive')
ROOT=Path('/content/drive/MyDrive/filament-segmentation-2026/fivefold_study_v2')
if not (ROOT/'current_code.json').is_file():raise FileNotFoundError('새5-fold최초실행셀을먼저실행하세요.')
version=json.loads((ROOT/'current_code.json').read_text())['version'];CODE=ROOT/'code'/version
manifest=json.loads((CODE/'source_manifest.json').read_text())
for name,expected in manifest.items():
    if hashlib.sha256((CODE/name).read_bytes()).hexdigest()!=expected:raise ValueError('저장된코드가변경됐습니다: '+name)
command=[sys.executable,'-u',str(CODE/'study_launcher.py'),'--root',str(ROOT)]
if STAGE_EPOCH is not None:command+=['--stage',str(STAGE_EPOCH)]
__EXECUTE__
result=json.loads((ROOT/'latest_result.json').read_text())
print('업로드할결과:',result['archive']);files.download(result['archive'])
'''


def build():
    sources={name:(ROOT/'tools'/name).read_text() for name in MODULES}
    runtime=b'\n'+(ROOT/'tools/solar_a100_train_standalone.py').read_bytes()
    if hashlib.sha256(runtime).hexdigest()!=RUNTIME_SHA:raise ValueError('Reviewed runtime differs')
    sources.update({'base_runtime.py':runtime.decode(),'study_launcher.py':LAUNCHER})
    for name,source in sources.items():compile(source,name,'exec')
    raw=json.dumps(sources,sort_keys=True,ensure_ascii=False).encode();sha=hashlib.sha256(raw).hexdigest()
    cell=INITIAL.replace('__PAYLOAD__',repr(base64.b64encode(zlib.compress(raw,9)).decode()))
    cell=cell.replace('__SHA__',repr(sha)).replace('__ANN_SHA__',repr(ANN_SHA)).replace('__EXECUTE__',PROCESS)
    resume=RESUME.replace('__EXECUTE__',PROCESS)
    compile(cell,'study-initial','exec');compile(resume,'study-resume','exec')
    folder=ROOT/'notebooks'
    (folder/'SolarFilament_5Fold_Study_v2_CopyCell.py').write_text(cell)
    (folder/'SolarFilament_5Fold_Study_v2_Resume.py').write_text(resume)
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',
        kernelspec=dict(display_name='Python3',language='python',name='python3')),
        cells=[dict(id='intro',cell_type='markdown',metadata={},source=[
            '# Fresh fivefold study\n5fold×YOLO/native각30epoch=300model-epoch. 모든fold를동일예산으로비교합니다.\n',
            '기존best84/기존fold0가중치를쓰지않습니다. 중단후재개가능합니다.\n']),
            dict(id='run',cell_type='code',execution_count=None,metadata={},outputs=[],source=cell.splitlines(True))])
    (folder/'SolarFilament_5Fold_Study_v2.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n')
    return dict(bundle_sha256=sha,version=sha[:12],source_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()},
                default_stage_epoch=30,folds=[0,1,2,3,4],total_initial_model_epochs=300,cell_bytes=len(cell.encode()))

if __name__=='__main__':print(json.dumps(build(),indent=2))
