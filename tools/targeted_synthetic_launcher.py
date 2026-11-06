"""Persistent fixed-split synthetic comparison; no dependency on a past Colab VM."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('TORCH_CUDNN_V8_API_DISABLED','1')
ARMS=('control_v2','targeted_v3')
EXPECTED={
    'annotations':'5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1',
    'train_split':'aa0733424815731c5c108ada91e45525eb17ca3a42b377bf19e3089d86aede7e',
    'val_split':'54c6a9aac5a3b70b3ecc6721109a8b0fe8ec020e90cb56e3848fccc92eadc61f',
    'yolo':'81c2a27d95821b5ede283169c0da8c08d360ecc4c2a60a1b9c6baf96d76bcccc',
    'synthetic_annotations':'5e7ad5f20fa12f0c820a82a315ba60e802480d54b4b3ecebd9fba25ae8fc6aea',
    'synthetic_split':'e8c9f3848c7b1fe1283167d973e459a5df7cd6336cc1d871268a34024d1ea881',
}
NATIVE_SHA='ae6b209f72355fd52d2cb07182e69028e97c703a1977ad1e7b38b27cf8cebe69'

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    pending.replace(path)

def validate_configuration(cfg,phase,cap,arms):
    if phase not in ('all','prepare','train','evaluate'):raise ValueError('Unknown phase')
    if not arms or len(arms)!=len(set(arms)) or not set(arms)<=set(ARMS):raise ValueError('Invalid arms')
    if not isinstance(cfg['epochs'],int) or not 1<=cap<=cfg['epochs']:
        raise ValueError('run-until must be within the fixed epochs schedule')
    if cfg.get('batch',1)!=1 or cfg.get('nbs',1)!=1:
        raise ValueError('This T4 comparison uses batch1, no gradient accumulation')
    if int(cfg.get('imgsz',2048))%32 or int(cfg.get('imgsz',2048))<256:
        raise ValueError('imgsz must be a multiple of32 and at least256')
    if not 0<float(cfg.get('lr',1e-4))<.1:raise ValueError('Invalid learning rate')
    output=Path(cfg['output_root']).resolve();cache=Path(cfg['cache_root']).resolve()
    if output==cache or output in cache.parents or cache in output.parents:
        raise ValueError('Persistent output and temporary cache must be separate directories')
    required=['annotations','train_split','val_split','yolo']
    if phase in ('all','prepare','train'):required+=['synthetic_annotations','synthetic_split']
    hashes={}
    for key in required:
        path=Path(cfg[key])
        if not path.is_file():raise FileNotFoundError(f'{key}: {path}')
        hashes[key]=digest(path)
        if hashes[key]!=EXPECTED[key]:raise ValueError('Expected original reviewed file: '+key)
    if not Path(cfg['images']).is_dir():raise FileNotFoundError(cfg['images'])
    if phase in ('all','prepare','train'):
        if not Path(cfg['synthetic_images']).is_dir():raise FileNotFoundError(cfg['synthetic_images'])
        hard=json.loads(Path(cfg['hard_donors']).read_text())
        if hard['annotations_sha256']!=EXPECTED['annotations'] or hard['train_split_sha256']!=EXPECTED['train_split']:
            raise ValueError('Hard donor inventory belongs to different data')
        if len(hard['annotation_ids'])!=550:raise ValueError('Expected550 reviewed training donor IDs')
    # Fail before spending training time if the requested complete comparison
    # cannot load its frozen native model. Prepare-only remains CPU-capable.
    if phase!='prepare' and cfg.get('full_ensemble',True):
        native=Path(cfg['native'])
        if not native.is_file():raise FileNotFoundError('Frozen native best84 required: '+str(native))
        hashes['native']=digest(native)
        if hashes['native']!=NATIVE_SHA:raise ValueError('Expected original frozen native best84')
    if cfg.get('code_dir'):
        code=Path(cfg['code_dir']);manifest_path=code/'source_manifest.json'
        manifest=json.loads(manifest_path.read_text())
        if cfg.get('code_manifest_sha256') and digest(manifest_path)!=cfg['code_manifest_sha256']:
            raise ValueError('Saved code manifest changed')
        for name,expected in manifest.items():
            if Path(name).name!=name or digest(code/name)!=expected:
                raise ValueError('Saved source changed: '+name)
            local=Path(__file__).parent/name
            if not local.is_file() or digest(local)!=expected:
                raise ValueError('Restored local source differs: '+name)
    return hashes

def collect_stages(cfg,cap):
    import torch
    from targeted_synthetic_training import compact_checkpoint,atomic_torch
    stages={}
    for arm in ARMS:
        folder=Path(cfg['output_root'])/'arms'/arm
        config_path=folder/'training_config.json';stage=folder/f'epoch{cap:03d}.pt'
        if not config_path.is_file():continue
        identity=json.loads(config_path.read_text())['identity']
        if not stage.exists() and (folder/'last_full.pt').exists():
            last=torch.load(folder/'last_full.pt',map_location='cpu',weights_only=False)
            if last.get('synthetic_identity')==identity and last.get('completed_epoch')==cap:
                atomic_torch(stage,compact_checkpoint(last))
        if not stage.is_file():continue
        state=torch.load(stage,map_location='cpu',weights_only=False)
        if (state.get('synthetic_identity'),state.get('completed_epoch'),state.get('fixed_final_epoch'))!=(identity,cap,cfg['epochs']):
            raise ValueError('Checkpoint identity/epoch mismatch: '+str(stage))
        stages[arm]=dict(arm=arm,checkpoint=str(stage),checkpoint_sha256=digest(stage),
                         epochs_completed=cap,fixed_final_epoch=cfg['epochs'],identity=identity)
    return stages

def run(cfg,phase='all',run_until=None,arms=ARMS,result_path=None):
    cap=cfg['epochs'] if run_until is None else int(run_until)
    hashes=validate_configuration(cfg,phase,cap,list(arms))
    root=Path(cfg['output_root']);root.mkdir(parents=True,exist_ok=True)
    result_path=Path(result_path or root/'result.json')
    lock_name=hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:20]
    lock=(Path('/tmp')/('solar_targeted_'+lock_name+'.lock')).open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another cell is using this experiment in this runtime')
    started=time.time()
    result=dict(status='running',phase=phase,run_until=cap,fixed_final_epoch=cfg['epochs'],
                training={},evaluation=None,archives=[],output_root=str(root),input_sha256=hashes)
    atomic_json(result_path,result)
    try:
        from targeted_synthetic_training import train_arm,evaluate_checkpoints,export_results
        if phase in ('all','prepare','train'):
            from targeted_synthetic_data import prepare_dataset
            prepared=prepare_dataset(cfg)
            if prepared['real_samples']!=1053 or prepared['synthetic_samples']!=263:
                raise ValueError('Expected1053 original annotation records and263 synthetic slots per arm')
            if abs(prepared['synthetic_fraction']-263/1316)>1e-12:
                raise ValueError('Unequal or changed synthetic sampling ratio')
            result['dataset']=prepared;atomic_json(result_path,result)
            print(f"같은 원사진616장/주석1053개 + 합성263개: 군당1316샘플, 합성{prepared['synthetic_fraction']:.3%}",flush=True)
            print(f'두 군의 최종 목표는 각각{cfg["epochs"]}추가 epoch. 이번 요청{cap}. 기존 모델과5-fold는 수정하지 않습니다.',flush=True)
            if phase!='prepare':
                for arm in arms:
                    yaml=prepared['arms'][arm]['yaml']
                    result['training'][arm]=train_arm(cfg,arm,yaml,prepared['identity'],cap)
                    atomic_json(result_path,result)
        if phase in ('all','evaluate'):
            stages=collect_stages(cfg,cap)
            result['training'].update(stages)
            if set(stages)==set(ARMS):
                evaluation_cfg=dict(cfg,evaluation_tag=f'epoch{cap:03d}')
                result['evaluation']=evaluate_checkpoints(evaluation_cfg,{a:r['checkpoint'] for a,r in stages.items()})
            elif phase=='evaluate':
                raise FileNotFoundError(f'Both arms must have completed epoch{cap}; available: {list(stages)}')
            else:
                result['evaluation_pending']='Run the other arm to the same epoch before a paired comparison'
        if phase=='prepare':result['status']='prepared'
        elif result.get('evaluation_pending'):result['status']='waiting_for_other_arm'
        elif cap<cfg['epochs']:result['status']='stage_complete_final_training_remaining'
        else:result['status']='complete' if phase!='train' else 'training_complete_evaluation_pending'
        result['elapsed_seconds_this_launch']=time.time()-started
        atomic_json(result_path,result)
        archives=export_results(cfg)
        result['archives']=[str(p) for p in archives]
        atomic_json(result_path,result)
        if phase=='prepare':print('데이터 준비 완료. 이 단계에서는 학습하지 않았습니다.',flush=True)
        else:print('상태:',result['status'],'| 요청 epoch:',cap,'/ 최종 목표:',cfg['epochs'],flush=True)
        for path in result['archives']:print('결과 ZIP:',path,flush=True)
        return result
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--result')
    p.add_argument('--phase',choices=['all','prepare','train','evaluate'],default='all')
    p.add_argument('--run-until',type=int)
    p.add_argument('--arms',default=','.join(ARMS))
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text())
    run(cfg,args.phase,args.run_until,args.arms.split(','),args.result)

if __name__=='__main__':main()
