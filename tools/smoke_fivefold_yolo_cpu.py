"""Actual tiny Ultralytics training/resume API check; no competition score."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'tests'))
from test_grouped_cv_training import fixture
from test_fivefold_yolo_training import Store
import grouped_cv_training as native
import fivefold_yolo_training as training


class CPUFixtureYOLO:
    def __init__(self,path,task):
        self.model=YOLO('yolo26n-seg.yaml' if path==training.PRETRAINED else path,task=task)
    def add_callback(self,*args):return self.model.add_callback(*args)
    def train(self,**kwargs):
        kwargs.update(imgsz=64,device='cpu',amp=False,verbose=False,pretrained=False)
        return self.model.train(**kwargs)


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(dir='/workspace/artifacts') as folder:
        root=Path(folder);cfg=fixture(root);cfg.update(study_root=str(root/'study'),stage_epoch=1,yolo_device='cpu')
        annotation=Path(cfg['annotations']);data=json.loads(annotation.read_text())
        for ann in data['annotations']:ann['segmentation']=[[512,512,1536,512,1536,1536,512,1536]]
        annotation.write_text(json.dumps(data));plan=Path(cfg['fold_plan']);metadata=json.loads(plan.read_text())
        metadata['sources_sha256']['annotations']=native.digest(annotation)
        plan.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
        runtime=native.load_runtime(cfg['base_runtime'])
        with patch.object(runtime,'ImageStore',Store):
            first=training.run(cfg,_yolo=CPUFixtureYOLO,_runtime=runtime)
            second=training.run(dict(cfg,stage_epoch=2),_yolo=CPUFixtureYOLO,_runtime=runtime)
        state=torch.load(root/'study/fold0/yolo/last_full.pt',map_location='cpu',weights_only=False)
        assert state['cv_epoch']==2 and state['optimizer']['state']
        assert state['train_args']['epochs']==100
        assert state['cv_scheduler']['last_epoch']==1  # Two epochs, zero-based scheduler.
        report=dict(scope='Actual Ultralytics CPU training/resume smoke; no competition performance',
            fixture_architecture='YOLO26n',production_architecture='YOLO26s',fixture_imgsz=64,
            generic_weight_download=False,completed_epoch=2,optimizer_preserved=True,
            stage1_exists=Path(first['checkpoint']).exists(),stage2_exists=Path(second['checkpoint']).exists(),
            schedule_epochs=100,actual_stage_epochs=2,gpu_verified=False,runner_sha256=native.digest(training.__file__))
        native.atomic_json(ROOT/'docs/evidence/fivefold_yolo_cpu_training_smoke.json',report)
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
