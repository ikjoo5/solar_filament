"""Shared budgets, real optimizer continuation, and independent stage exports."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
sys.path.insert(0, str(ROOT/'tests'))
import fivefold_study as study
import grouped_cv_training as training
from test_grouped_cv_training import fixture, TinyDataset, TinyStore


def fake_export(runtime, model, cfg, directory, provenance):
    directory = Path(directory)
    records = directory/'rich_predictions.jsonl'
    with Path(cfg['val_split']).open() as stream:
        source = list(__import__('csv').DictReader(stream))
    names = sorted({row['file_name'] for row in source})
    records.write_text(''.join(json.dumps(dict(file_name=name, **provenance))+'\n' for name in names))
    summary = directory/'export_summary.json'
    summary.write_text(json.dumps(dict(image_coverage=len(names))))
    return dict(rich_predictions=str(records), summary=str(summary), image_coverage=len(names),
                inference_identity=study.fingerprint(provenance))


def settings(root):
    base = fixture(root)
    exporter = root/'export_source.py'; exporter.write_text('# pinned test exporter\n')
    cfg = dict(study_root=str(root/'fresh_study'), base_runtime=base['base_runtime'],
        annotations=base['annotations'], images=base['images'], fold_plan=base['fold_plan'],
        cache_root=base['cache_root'], rich_export_source=str(exporter),
        stage_epoch=1, folds=[0], training_settings=dict(device='cpu', crop=64,
            batch=1, accumulate=2, steps_per_epoch=2, workers=0, preflight_backward=False))
    return cfg


class StudyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_default_budget_all_five_and_plan_only_needs_no_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = settings(Path(folder)); cfg.pop('folds'); cfg.pop('stage_epoch')
            planned = study.run(dict(cfg, plan_only=True))
            self.assertEqual(planned['requested_folds'], [0,1,2,3,4])
            self.assertEqual(planned['stage_epoch'], 30)
            self.assertEqual(planned['total_initial_stage_epochs'], 150)
            self.assertEqual(planned['training_schedule_epochs'], 100)
            self.assertFalse(planned['training_performed'])
            for job in planned['jobs']:
                self.assertEqual(job['training_config']['run_until_epoch'],30)
                self.assertEqual(job['training_config']['validate_every'],100)
                self.assertEqual(job['training_config']['output_root'],cfg['study_root'])
                self.assertEqual(Path(job['stage_dir']).parts[-2:],(f"fold{job['fold']}",'stage030'))

    def test_training_continues_stage1_to_stage2_and_keeps_old_stage(self):
        calls = []
        def build(encoder, pretrained=True):
            calls.append(pretrained)
            return nn.Sequential(nn.Conv2d(3,2,1), nn.Dropout2d(.2))
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            cfg = settings(Path(folder)); runtime = training.load_runtime(cfg['base_runtime'])
            trained = []
            def train(config):
                trained.append(config['run_until_epoch'])
                return training.run(config, _runtime=runtime)
            with patch.multiple(runtime, build_model=build, ImageStore=TinyStore, MixedCropDataset=TinyDataset):
                first = study.run(cfg, _train=train, _export=fake_export, _runtime=runtime)
                self.assertEqual(first['status'],'partial_stage')
                self.assertFalse(first['complete_fivefold'])
                self.assertEqual(first['completed_folds'],[0])
                root = Path(cfg['study_root'])
                original_model = (root/'fold0/stage001/model.pt').read_bytes()
                second = study.run(dict(cfg,stage_epoch=2), _train=train, _export=fake_export, _runtime=runtime)
                self.assertEqual(second['stage_epoch'],2)
                self.assertEqual(trained,[1,2])
                self.assertEqual((root/'fold0/stage001/model.pt').read_bytes(),original_model)
                last = torch.load(root/'fold0/last.pt',weights_only=True)
                self.assertEqual(last['epoch'],2)
                self.assertEqual(len(last['history']),2)
                self.assertEqual(last['scheduler_state_dict']['last_epoch'],2)
                self.assertNotIn('optimizer_state_dict',torch.load(root/'fold0/stage002/model.pt',weights_only=True))
                # Completed export is reusable even after the full checkpoint advanced.
                study.run(cfg,_train=lambda c:self.fail('Training restarted'),
                          _export=fake_export,_runtime=runtime)
                # A preprocessing export variant reuses exactly the same model.
                alternate = study.run(dict(cfg,inference_profile='tta'),
                    _train=lambda c:self.fail('Preprocessing triggered training'),
                    _export=fake_export,_runtime=runtime)
                self.assertEqual(alternate['study_identity'],first['study_identity'])
                self.assertEqual(len(list((root/'fold0/stage001/variants').glob('tta_*/stage_summary.json'))),1)
                self.assertEqual(trained,[1,2])
                (root/'fold0/stage001/stage_summary.json').unlink()
                (root/'fold0/stage001/model.pt').unlink()
                with self.assertRaisesRegex(ValueError,'advanced past'):
                    study.run(cfg,_train=train,_export=fake_export,_runtime=runtime)

    def test_changed_learning_settings_or_old_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg=settings(Path(folder)); study.prepare_study(cfg)
            with self.assertRaisesRegex(ValueError,'metadata differs'):
                study.prepare_study(dict(cfg,training_settings=dict(cfg['training_settings'],lr=2e-4)))
            with self.assertRaisesRegex(ValueError,'old solar'):
                study.prepare_study(dict(cfg,initial_checkpoint='best84.pt'))
            with self.assertRaisesRegex(ValueError,'100epoch'):
                study.prepare_study(dict(cfg,training_settings=dict(epochs=30)))

    def test_all_five_folds_complete_at_one_common_budget(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            cfg=settings(Path(folder));cfg['folds']=[0,1,2,3,4]
            runtime=training.load_runtime(cfg['base_runtime'])
            def build(encoder,pretrained=True):
                return nn.Sequential(nn.Conv2d(3,2,1),nn.Dropout2d(.2))
            with patch.multiple(runtime,build_model=build,ImageStore=TinyStore,MixedCropDataset=TinyDataset):
                result=study.run(cfg,_train=lambda c:training.run(c,_runtime=runtime),
                                 _export=fake_export,_runtime=runtime)
            self.assertEqual(result['status'],'complete_stage')
            self.assertEqual(result['completed_folds'],[0,1,2,3,4])
            self.assertEqual({r['stage_epoch'] for r in result['summaries']},{1})
            names=[name for row in result['summaries'] for name in row['image_names']]
            self.assertEqual(len(names),12)
            self.assertEqual(len(set(names)),12)

    def test_new_exporter_source_reuses_weights_in_a_new_variant_folder(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            cfg=settings(Path(folder));runtime=training.load_runtime(cfg['base_runtime'])
            def build(encoder,pretrained=True):
                return nn.Sequential(nn.Conv2d(3,2,1),nn.Dropout2d(.2))
            with patch.multiple(runtime,build_model=build,ImageStore=TinyStore,MixedCropDataset=TinyDataset):
                first=study.run(cfg,_train=lambda c:training.run(c,_runtime=runtime),
                                _export=fake_export,_runtime=runtime)
                root=Path(cfg['study_root']); original=training.digest(root/'fold0/stage001/model.pt')
                Path(cfg['rich_export_source']).write_text('# changed inference exporter only\n')
                updated=study.run(cfg,_train=lambda c:self.fail('Export-only change restarted training'),
                                  _export=fake_export,_runtime=runtime)
            self.assertEqual(first['study_identity'],updated['study_identity'])
            self.assertEqual(training.digest(root/'fold0/stage001/model.pt'),original)
            self.assertEqual(len(list((root/'fold0/stage001/variants').glob('tta_clahe_*/stage_summary.json'))),1)


if __name__ == '__main__':
    unittest.main()
