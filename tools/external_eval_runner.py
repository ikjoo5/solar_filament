"""Run the locked LB.39 pipeline on external data for evaluation only."""
from collections import defaultdict
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import zipfile

import cv2
import numpy as np

from external_evaluation_data import prepare_external_plan,score_external,annotation_rle
from submission_driver import (CHECKPOINT_SHA256,CODE_FILES,IsolatedInference,atomic_json,
    compact,digest,export_csv,process_frame)
from submission_inference import InferenceProvider
from submission_native_process import NativeProcessClient


def external_bootstrap(metrics,draws=5000):
    groups=defaultdict(lambda:np.zeros(4,np.float64))
    for row in metrics['per_group']:
        if row['gt_instances']:
            groups[row['time_block_27days']]+=np.asarray([row['matched_iou_sum'],row['tp_pairs'],row['fp'],row['fn']])
    if len(groups)<2:
        return dict(resampled_temporal_blocks=len(groups),percentile95=None,note='Too few temporal blocks for a useful block bootstrap.')
    values=np.stack(list(groups.values()));rng=np.random.default_rng(42)
    sampled=values[rng.integers(0,len(values),(draws,len(values)))].sum(axis=1)
    denominator=sampled[:,1]+.5*sampled[:,2]+.5*sampled[:,3]
    scores=np.divide(sampled[:,0],denominator,out=np.zeros(draws),where=denominator>0)
    return dict(resampled_temporal_blocks=len(groups),draws=draws,unit='27-day capture block, including all physical/rater variants',
        percentile95=np.quantile(scores,[.025,.975]).tolist(),note='Block count is not an exact effective sample size; residual temporal dependencies can remain.')


def run_external(config):
    import torch
    if not config.get('external_annotations') or not config.get('external_images'):
        raise ValueError('외부 MAGFiLO의 주석 JSON 경로와 이미지 폴더 경로가 필요합니다.')
    split=config.get('split','external_dev')
    if split not in ('external_dev','external_confirm'):raise ValueError('Choose external_dev or external_confirm explicitly')
    if not torch.cuda.is_available():raise ValueError('Colab A100 GPU required for the locked inference pipeline')
    cv2.setNumThreads(1);torch.set_num_threads(2)
    for name,expected in CHECKPOINT_SHA256.items():
        if digest(config[name+'_checkpoint'])!=expected:raise ValueError(f'Frozen checkpoint mismatch: {name}')
    plan=prepare_external_plan(config['external_annotations'],config['external_images'],
        config['official_train_images'],config['official_test_images'])
    assert plan['allow_training'] is False and plan['allow_pseudo_label_training'] is False
    base=Path(config['output']);base.mkdir(parents=True,exist_ok=True)
    atomic_json(base/'external_evaluation_plan.json',plan)
    selected=[row for row in plan['records'] if row['split']==split]
    print(json.dumps(plan['summary'],ensure_ascii=False,indent=2),flush=True)
    if not selected:
        raise ValueError(f'No independent samples remain in {split}; official train/validation/test overlap is never scored.')
    code={name:digest(Path(__file__).with_name(name)) for name in CODE_FILES+('external_evaluation_data.py','external_eval_runner.py')}
    runtime=dict(gpu=torch.cuda.get_device_name(0),amp='bfloat16' if torch.cuda.is_bf16_supported() else 'float16',
        versions={name:importlib.metadata.version(name) for name in ('torch','ultralytics','segmentation-models-pytorch','numpy','scipy','pycocotools')},
        native_cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    provenance=dict(data_role='external_evaluation_only',training=False,split=split,plan_sha256=hashlib.sha256(compact(plan).encode()).hexdigest(),
        checkpoints=CHECKPOINT_SHA256,code_sha256=code,runtime=runtime)
    identity=hashlib.sha256(compact(provenance).encode()).hexdigest()
    output=base/split;output.mkdir(exist_ok=True)
    identity_file=output/'identity.json'
    if identity_file.is_file() and json.loads(identity_file.read_text()).get('identity')!=identity:
        raise ValueError('External data/model/runtime changed; choose a fresh output folder rather than mix evaluations')
    atomic_json(identity_file,dict(identity=identity,provenance=provenance))
    for name in ('stages','completed'):(output/name).mkdir(exist_ok=True)
    native=NativeProcessClient(config['native_checkpoint'],device='cuda:0',batch=4)
    provider=IsolatedInference(InferenceProvider(config['yolo_checkpoint'],config['native_checkpoint'],device='cuda:0'),native)
    records=[];predictions={}
    try:
        with tempfile.TemporaryDirectory(prefix='external_eval_inputs_') as temporary:
            for index,item in enumerate(selected,1):
                # Exact file bytes, merely a local safe ID alias; no image re-encoding.
                source=Path(item['path']);raw=source.read_bytes()
                if hashlib.sha256(raw).hexdigest()!=item['file_sha256']:raise ValueError(f'External image changed: {source}')
                alias=Path(temporary)/(item['evaluation_key']+source.suffix.lower());alias.write_bytes(raw)
                record,reused=process_frame(alias,item['file_sha256'],identity,output,provider)
                alias.unlink();records.append(record);predictions[item['evaluation_key']]=record['rles']
                print(f'External {split} [{index}/{len(selected)}] {item["evaluation_key"]}: {record["instances"]} masks'+(' reused' if reused else ''),flush=True)
    finally:native.close()
    # All predictions are finished before target masks are rasterized or scored.
    metric=score_external(plan,predictions,split)
    metric['block_bootstrap']=external_bootstrap(metric)
    metric['model_identity']=identity
    metric['baseline_user_reported_public_lb']=.39
    metric['not_a_kaggle_submission']=True
    atomic_json(output/'metrics.json',metric)
    prediction_csv=output/f'{split}_predictions.csv';export_csv(prediction_csv,records)
    mapping={sid:(i,row['evaluation_key']) for i,row in enumerate(selected,1) for sid in row['source_image_ids']}
    source=json.loads(Path(plan['annotations']).read_text());counters=defaultdict(int)
    truth_csv=output/f'{split}_ground_truth.csv'
    group_numbers={sid:i for i,sid in enumerate(sorted(mapping),1)}
    with truth_csv.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle'])
        for ann in source['annotations']:
            sid=str(ann['image_id'])
            if sid not in mapping:continue
            counters[sid]+=1;key=mapping[sid][1]
            writer.writerow([f'r{group_numbers[sid]:06d}-{key}_{counters[sid]}',annotation_rle(ann)['counts'].decode('ascii')])
    # The reserved-confirmation targets are not exported during external_dev.
    archive=base/f'{split}_evaluation_results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in (prediction_csv,truth_csv,output/'metrics.json',identity_file,base/'external_evaluation_plan.json'):
            z.write(path,path.name)
    result=dict(archive=str(archive),split=split,physical_images=len(selected),pq=metric['official_nonblank_pq']['pq'],
        training_performed=False,external_data_used_for_training=False,summary=plan['summary'])
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--result',required=True)
    args=parser.parse_args();result=run_external(json.loads(Path(args.config).read_text()));atomic_json(args.result,result)


if __name__=='__main__':main()
