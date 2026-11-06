"""Method wiring guards, unchanged initial predictions and no completed-run retrain."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

TOOLS=Path(__file__).resolve().parents[1]/'tools'
if str(TOOLS) not in sys.path:sys.path.insert(0,str(TOOLS))
import paper_method_training as paper


class Encoder(nn.Module):
    out_channels=[3,4]
    def __init__(self):super().__init__();self.conv=nn.Conv2d(3,4,1)
    def forward(self,x):return [x,self.conv(x)]


class Decoder(nn.Module):
    def forward(self,features):return features[-1]


class TinyBase(nn.Module):
    classification_head=None
    def __init__(self):
        super().__init__();self.encoder=Encoder();self.decoder=Decoder();self.segmentation_head=nn.Conv2d(4,2,1)
    def check_input_shape(self,x):pass
    def forward(self,x):return self.segmentation_head(self.decoder(self.encoder(x)))


class PaperTrainingTests(unittest.TestCase):
    def test_external_folder_is_never_accepted_for_any_training_source(self):
        config={key:'/allowed/'+key for key in ('images','annotations','synthetic_images','synthetic_annotations','synthetic_split','train_split','val_split')}
        config['external_root']='/data/external_MAGFiLO'
        paper.assert_training_sources_are_official(config)
        for key in tuple(k for k in config if k!='external_root'):
            wrong=dict(config);wrong[key]='/data/external_MAGFiLO/annotations.json'
            with self.assertRaises(ValueError):paper.assert_training_sources_are_official(wrong)
        with self.assertRaises(ValueError):paper.assert_training_sources_are_official(dict(config,data_role='external_evaluation_only'))

    def test_method_warm_start_resume_keys_and_source_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=TinyBase().eval();initial=root/'initial.pt'
            torch.save(dict(model_state_dict=base.state_dict()),initial)
            asset=root/'method.py';asset.write_text('# method fixture')
            config=dict(initial_checkpoint=str(initial),base_runtime=str(root/'base_runtime.py'),
                resume_dir=None,paper_context_hidden=4,paper_context_rates=[1,3,5])
            outputs=[];runtime=SimpleNamespace(RESUME_KEYS=('epochs',),build_model=lambda *a,**k:TinyBase(),
                atomic_json_save=lambda path,value:paper.atomic_json(path,value))
            journal=dict(runs={});journal_path=root/'progress.json'
            with patch.object(paper,'load_runtime',return_value=runtime):
                modified=paper.make_runtime(config,'multiscale_context',journal,journal_path,{'method.py':str(asset)})
            model=modified.build_model().eval();x=torch.randn(1,3,8,12)
            self.assertTrue(torch.equal(model(x),base(x)))
            self.assertIn('paper_method',modified.RESUME_KEYS);self.assertIn('research_source_sha256',modified.RESUME_KEYS)
            run=root/'run';run.mkdir();metadata={}
            modified.atomic_json_save(run/'config.json',metadata)
            self.assertEqual((run/'method.py').read_text(),asset.read_text())
            self.assertEqual(journal['runs']['multiscale_context']['path'],str(run))
            self.assertIn('warm start',metadata['initialization'])

    def test_native_export_records_preserve_holes_real_statistics_and_fixed_thresholds(self):
        probability=np.zeros((2048,2048),np.float32);probability[30:60,40:70]=.8;probability[40:45,50:55]=0
        result=paper.record(probability,'sample.jpeg','weights')
        self.assertEqual(set(result['components']),{'0.35','0.5','0.65'})
        component=result['components']['0.5'][0]
        self.assertEqual(component['area'],875);self.assertAlmostEqual(component['stats']['foreground']['mean'],.8,places=6)
        from pycocotools import mask as mu
        mask=mu.decode(component['rle']);self.assertEqual(int(mask[41,51]),0);self.assertEqual(int(mask[31,41]),1)
        with self.assertRaises(ValueError):paper.record(np.zeros((64,64)),'bad.jpeg','weights')

    def test_suite_fingerprint_locks_methods_budget_and_sources(self):
        keys=('images','annotations','synthetic_images','synthetic_annotations','synthetic_split','train_split','val_split',
            'crop','batch','accumulate','steps_per_epoch','epochs','lr','spine_weight','seed','synthetic_ratio','warmup_epochs',
            'real_finish_epochs','include_empty_synthetic','thresholds','min_areas','overlap','validate_every','encoder',
            'methods','paper_context_hidden','paper_context_rates','expected_source_sha256','initial_checkpoint_sha256')
        cfg={k:'fixture' for k in keys};reference=paper.suite_fingerprint(cfg,{'code':'hash'})
        self.assertNotEqual(reference,paper.suite_fingerprint(dict(cfg,epochs=200),{'code':'hash'}))
        self.assertNotEqual(reference,paper.suite_fingerprint(cfg,{'code':'changed'}))
        self.assertEqual(reference,paper.suite_fingerprint(dict(cfg,resume_dir='/mutable',workers=0),{'code':'hash'}))

    def test_completed_training_is_not_repeated_after_export_interruption(self):
        """A session expiry during export must not spend another 100 epochs."""
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            keys=('images','annotations','synthetic_images','synthetic_annotations','synthetic_split','train_split','val_split',
                'crop','batch','accumulate','steps_per_epoch','epochs','lr','spine_weight','seed','synthetic_ratio','warmup_epochs',
                'real_finish_epochs','include_empty_synthetic','overlap','validate_every','encoder','expected_source_sha256')
            config={k:'fixture' for k in keys}
            config.update(external_root=str(root/'external'),output_root=str(root/'runs'),
                initial_checkpoint=str(root/'initial.pt'),base_runtime=str(root/'base_runtime.py'),
                methods=['control','multiscale_context'],thresholds=[.5],min_areas=[300],
                paper_context_hidden=4,paper_context_rates=[1,3,5])
            calls=[];exports=[]
            def runtime(cfg,method,journal,journal_path,assets):
                def train(settings):
                    calls.append(method);run=root/method;run.mkdir()
                    journal['runs'][method]=dict(status='training',path=str(run))
                    paper.atomic_json(journal_path,journal)
                    (run/'best.pt').write_bytes(b'fixture');(run/'last.pt').write_bytes(b'fixture')
                    paper.atomic_json(run/'best_epoch.json',dict(epoch=10,best=dict(pq=.4 if method=='control' else .41)))
                    return dict(output=str(run))
                return SimpleNamespace(run_training=train)
            def export(runtime,cfg,run):
                exports.append(cfg['paper_method'])
                if len(exports)==1:raise RuntimeError('simulated expiry during export')
                path=Path(run)/'raw_paper_views.jsonl';path.write_text('{}\n');return path
            with patch.object(paper,'digest',return_value=paper.INITIAL_CHECKPOINT_SHA), \
                 patch.object(paper,'make_runtime',side_effect=runtime),patch.object(paper,'export_rich',side_effect=export):
                with self.assertRaisesRegex(RuntimeError,'expiry'):paper.run_suite(config)
                journal=json.loads((root/'runs/experiment_progress.json').read_text())
                self.assertEqual(journal['runs']['control']['status'],'train_complete')
                result=paper.run_suite(config)
                self.assertEqual(calls,['control','multiscale_context'])
                self.assertAlmostEqual(result['comparison']['delta_pq'],.01)
                self.assertTrue(Path(result['archive']).is_file())
                paper.run_suite(config)
                self.assertEqual(calls,['control','multiscale_context'])


if __name__=='__main__':unittest.main()
