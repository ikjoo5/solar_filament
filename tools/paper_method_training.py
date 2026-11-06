"""Paired control/context fine-tuning of our native model, never external data.

The original reviewed runtime is loaded from an explicit SHA-pinned source file.
Its optimizer, sampler, mixed loss, official scorer and epoch resume are reused.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage
import torch

from paper_multiscale import METHODS,PaperUNet,load_base_weights

BASE_RUNTIME_SHA='6d60b5705fb0fbde03a6fd2922c8b83839fa49b5a78cc3434e7dcf4a1b4a08b4'
INITIAL_CHECKPOINT_SHA='ae6b209f72355fd52d2cb07182e69028e97c703a1977ad1e7b38b27cf8cebe69'
RESEARCH_KEYS=('paper_method','paper_context_hidden','paper_context_rates',
               'initial_checkpoint_sha256','research_source_sha256')


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.pending')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temporary.replace(path)


def assert_training_sources_are_official(config):
    external=Path(config['external_root']).resolve()
    for key in ('images','annotations','synthetic_images','synthetic_annotations','synthetic_split','train_split','val_split'):
        if Path(config[key]).resolve().is_relative_to(external):
            raise ValueError(f'External evaluation data cannot be a training input: {key}')
    if config.get('data_role','competition_training')!='competition_training':
        raise ValueError('This experiment trains only on the original competition training sources')


def suite_fingerprint(config,assets):
    keys=('images','annotations','synthetic_images','synthetic_annotations','synthetic_split','train_split','val_split',
        'crop','batch','accumulate','steps_per_epoch','epochs','lr','spine_weight','seed','synthetic_ratio','warmup_epochs',
        'real_finish_epochs','include_empty_synthetic','thresholds','min_areas','overlap','validate_every','encoder',
        'methods','paper_context_hidden','paper_context_rates','expected_source_sha256')
    value=dict(settings={k:config[k] for k in keys},assets=assets,initial_checkpoint_sha256=config['initial_checkpoint_sha256'])
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def load_runtime(source):
    if digest(source)!=BASE_RUNTIME_SHA:raise ValueError('The reviewed original runtime source differs')
    # This real filename/module name is importable by spawned DataLoader workers.
    # The portable launcher always restores it as base_runtime.py.
    if Path(source).name!='base_runtime.py':raise ValueError('Restore the reviewed source as base_runtime.py')
    if str(Path(source).parent) not in sys.path:sys.path.insert(0,str(Path(source).parent))
    name='base_runtime'
    spec=importlib.util.spec_from_file_location(name,source);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def make_runtime(config,method,journal,journal_path,assets):
    runtime=load_runtime(Path(config['base_runtime']))
    base_builder=runtime.build_model
    runtime.build_original_model=base_builder
    runtime.RESUME_KEYS=tuple(runtime.RESUME_KEYS)+RESEARCH_KEYS
    runtime.__file__=__file__
    initial=None
    def build(encoder='tu-convnext_tiny',pretrained=False):
        nonlocal initial
        base=base_builder(encoder,pretrained=False)
        model=PaperUNet(base,method,hidden=config['paper_context_hidden'],rates=config['paper_context_rates'])
        if not config.get('resume_dir'):
            if initial is None:initial=torch.load(config['initial_checkpoint'],map_location='cpu',weights_only=True)
            load_base_weights(model,initial)
            if initial.get('validation'):
                journal['warm_start_validation']=initial['validation']
        return model
    runtime.build_model=build
    original_save=runtime.atomic_json_save
    def save(path,value):
        path=Path(path)
        if path.name=='config.json':
            value['initialization']=('Completed-epoch optimizer/RNG resume' if config.get('resume_dir') else
                'Same original best84 warm start; fresh optimizer, paired fine-tuning control')
            value['experiment_scope']='Single native model; not the frozen YOLO+native ensemble publicLB score'
            for name,source in assets.items():
                destination=path.parent/name
                raw=Path(source).read_bytes()
                if destination.is_file() and destination.read_bytes()!=raw:raise ValueError('Saved research source changed')
                if not destination.is_file():destination.write_bytes(raw)
            journal['runs'][method]=dict(status='training',path=str(path.parent))
            atomic_json(journal_path,journal)
        original_save(path,value)
    runtime.atomic_json_save=save
    return runtime


def rle(mask):
    value=mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return dict(size=list(value['size']),counts=value['counts'].decode('ascii'))


def record(probability,name,checkpoint_sha):
    if probability.shape!=(2048,2048) or not np.isfinite(probability).all():
        raise ValueError('Native finite 2048-square probability required')
    components={}
    for threshold in (.35,.5,.65):
        label_image,number=ndimage.label(probability>=threshold,structure=np.ones((3,3),np.uint8))
        areas=np.bincount(label_image.ravel())
        values=[]
        for label in range(1,number+1):
            area=int(areas[label])
            if area<100:continue
            mask=label_image==label
            selected=probability[mask]
            values.append(dict(rle=rle(mask),area=area,stats=dict(foreground=dict(mean=float(selected.mean()),
                max=float(selected.max()),p90=float(np.percentile(selected,90))))))
        components[f'{threshold:g}']=values
    return dict(schema_version=1,file_name=name,size=[2048,2048],checkpoint_sha256=checkpoint_sha,
        components=components,unions=dict(foreground={'0.2':rle(probability>=.2)}))


def export_rich(runtime,config,run):
    """Prediction-only four-view export; same fields consumed by frozen postprocess."""
    run=Path(run);best=run/'best.pt';checkpoint=torch.load(best,map_location='cpu',weights_only=True)
    method=config['paper_method'];device=torch.device(config['device'])
    model=PaperUNet(runtime.build_original_model(config['encoder'],False),method,
        hidden=config['paper_context_hidden'],rates=config['paper_context_rates'])
    model.load_state_dict(checkpoint['model_state_dict']);model.to(device).eval()
    dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.type=='cuda' else None
    runtime.IMAGES=Path(config['images']);runtime.TRAIN=Path(config['train_split']);runtime.VAL=Path(config['val_split'])
    data=json.loads(Path(config['annotations']).read_text());_,names,_=runtime.prepare_split(data)
    store=runtime.ImageStore(Path(config['images']),Path(config['cache_root'])/'images_real','limb_only')
    output=run/'rich_validation';output.mkdir(exist_ok=True);checkpoint_sha=digest(best)
    combined=[]
    for index,name in enumerate(names,1):
        path=output/(Path(name).stem+'.json');image_sha=digest(Path(config['images'])/name)
        try:cached=json.loads(path.read_text()) if path.is_file() else None
        except (OSError,ValueError):cached=None
        if isinstance(cached,dict) and cached.get('checkpoint_sha256')==checkpoint_sha and cached.get('image_sha256')==image_sha:
            combined.append(cached);continue
        image=store.load(name);base=None;total=None;unions={}
        for view,axes in [('base',()),('hflip',(-1,)),('vflip',(-2,)),('hvflip',(-2,-1))]:
            pixels=np.ascontiguousarray(np.flip(image,axes)) if axes else image
            probability=runtime.predict_foreground(model,pixels,device,config['crop'],config['overlap'],config.get('infer_batch',4),dtype)
            if axes:probability=np.flip(probability,axes)
            if base is None:base=probability;total=probability.copy()
            else:total+=probability
            unions[view]=rle(probability>=.5)
        result=record(base,name,checkpoint_sha);result['tta']=record(total/np.float32(4),name,checkpoint_sha)
        result.update(view_unions=unions,image_sha256=image_sha,paper_method=method,
            export_scope='Minimal foreground fields for frozen native postprocessing; spine probabilities not exported')
        atomic_json(path,result);combined.append(result)
        print(f'{method} rich export [{index}/{len(names)}] {name}',flush=True)
    path=run/'raw_paper_views.jsonl';temporary=path.with_name(path.name+'.pending')
    temporary.write_text(''.join(json.dumps(value,separators=(',',':'))+'\n' for value in combined));temporary.replace(path)
    del model
    if device.type=='cuda':torch.cuda.empty_cache()
    return path


def run_suite(config):
    config=dict(config)
    assert_training_sources_are_official(config)
    if config['methods']!=['control','multiscale_context']:raise ValueError('Run the predeclared paired control/context methods')
    if config['thresholds']!=[.5] or config['min_areas']!=[300]:raise ValueError('Keep one fixed validation threshold/area for this comparison')
    if digest(config['initial_checkpoint'])!=INITIAL_CHECKPOINT_SHA:raise ValueError('Warm start is not the original best84 checkpoint')
    config['initial_checkpoint_sha256']=INITIAL_CHECKPOINT_SHA
    assets={name:str(Path(__file__).with_name(name)) for name in ('paper_method_training.py','paper_multiscale.py','base_runtime.py')}
    assets['base_runtime.py']=config['base_runtime']
    asset_hashes={name:digest(path) for name,path in assets.items()}
    config['research_source_sha256']=asset_hashes
    output=Path(config['output_root']);output.mkdir(parents=True,exist_ok=True)
    journal_path=output/'experiment_progress.json';identity=suite_fingerprint(config,asset_hashes)
    journal=json.loads(journal_path.read_text()) if journal_path.is_file() else dict(identity=identity,runs={})
    if journal['identity']!=identity:raise ValueError('Paired experiment settings or source changed; use a new output folder')
    atomic_json(journal_path,journal)
    for method in config['methods']:
        cfg=dict(config,paper_method=method,pretrained=False,output_root=str(output/method),resume_dir=None)
        previous=journal['runs'].get(method,{})
        previous_path=Path(previous['path']) if previous.get('path') else None
        if previous_path and (previous_path/'last.pt').is_file():cfg['resume_dir']=str(previous_path)
        runtime=make_runtime(cfg,method,journal,journal_path,assets)
        if previous.get('status') in ('train_complete','complete') and previous_path and (previous_path/'best.pt').is_file():
            result=dict(output=str(previous_path))
        else:
            result=runtime.run_training(cfg)
            journal['runs'][method].update(status='train_complete',path=result['output'])
            atomic_json(journal_path,journal)
        rich=export_rich(runtime,cfg,result['output'])
        metadata=json.loads((Path(result['output'])/'best_epoch.json').read_text())
        journal['runs'][method].update(status='complete',path=result['output'],best=metadata['best'],best_epoch=metadata['epoch'],rich_export=str(rich))
        atomic_json(journal_path,journal)
    control=journal['runs']['control']['best'];method=journal['runs']['multiscale_context']['best']
    comparison=dict(scope='Paired continued-finetuning single-model native PQ; not full ensemble or external validation',
        control=control,multiscale_context=method,delta_pq=method['pq']-control['pq'],
        warm_start_historical_validation=journal.get('warm_start_validation'),
        external_data_used_for_training=False,external_evaluation_status='Separate evaluator after external data eligibility is confirmed',
        settings=dict(epochs_per_arm=config['epochs'],initial_checkpoint_sha256=INITIAL_CHECKPOINT_SHA,
            validation_threshold=.5,minimum_area=300,lr=config['lr'],seed=config['seed']))
    reference=journal.get('warm_start_validation')
    if reference and reference.get('threshold')==.5 and reference.get('min_area')==300:
        comparison['context_minus_historical_warm_start_pq']=method['pq']-reference['pq']
        comparison['control_minus_historical_warm_start_pq']=control['pq']-reference['pq']
    atomic_json(output/'comparison.json',comparison)
    archive=output/'paper_multiscale_comparison.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in (journal_path,output/'comparison.json'):z.write(path,path.name)
        for method,info in journal['runs'].items():
            run=Path(info['path'])
            for name in ('config.json','history.json','best_epoch.json','best_validation.csv','raw_paper_views.jsonl'):
                if (run/name).is_file():z.write(run/name,method+'/'+name)
    return dict(archive=str(archive),comparison=comparison,output=str(output))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--result',required=True)
    args=parser.parse_args();result=run_suite(json.loads(Path(args.config).read_text()));atomic_json(args.result,result)


if __name__=='__main__':main()
