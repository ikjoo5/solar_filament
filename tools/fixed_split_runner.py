"""The separate616/91 lane for experiments requiring model retraining."""
import argparse
import json
from pathlib import Path
import torch
from grouped_cv_training import configure_cuda,load_runtime,atomic_json


def run(config):
    cfg=dict(config)
    if 'fivefold_study_v2' in str(cfg['output_root']):
        raise ValueError('Fixed-split experiments need a separate output folder')
    runtime=load_runtime(cfg['base_runtime'])
    # The reviewed helper has strict616/91 sibling/data guards and full resume.
    torch.cuda.is_bf16_supported=lambda *args,**kwargs:False
    if torch.cuda.is_available():configure_cuda()
    return runtime.run_training(cfg)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--result',required=True)
    args=parser.parse_args();atomic_json(args.result,run(json.loads(Path(args.config).read_text())))
