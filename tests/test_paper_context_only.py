"""Fixed-baseline/context-only transition preserves existing runs and resume."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch

TOOLS = Path(__file__).resolve().parents[1]/'tools'
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import paper_context_only as context


class ContextOnlyTests(unittest.TestCase):
    def fixture(self, root):
        keys=('images','annotations','synthetic_images','synthetic_annotations','synthetic_split','train_split','val_split',
            'crop','batch','accumulate','steps_per_epoch','lr','spine_weight','seed','synthetic_ratio','warmup_epochs',
            'real_finish_epochs','include_empty_synthetic','overlap','validate_every','encoder','expected_source_sha256')
        cfg={key:'fixture' for key in keys}
        cfg.update(external_root=str(root/'external'), output_root=str(root/'results'),
            initial_checkpoint=str(root/'best84.pt'), base_runtime=str(root/'base_runtime.py'),
            methods=['control',context.METHOD],thresholds=[.5],min_areas=[300],epochs=100,
            paper_context_hidden=128,paper_context_rates=[1,3,5],validate_every=5,
            expected_source_sha256={})
        torch.save(dict(epoch=84,validation=dict(pq=.4,threshold=.5,min_area=300),config={}), cfg['initial_checkpoint'])
        return cfg

    def identity(self, cfg):
        values=dict(cfg,initial_checkpoint_sha256=context.legacy.INITIAL_CHECKPOINT_SHA)
        return context.legacy.suite_fingerprint(values,context.LEARNING_SOURCE_SHA)

    def dependencies(self, root, cfg, calls, *, fail_export=False):
        stack=ExitStack()
        actual_digest=context.legacy.digest
        stack.enter_context(patch.object(context.legacy,'digest',side_effect=lambda path:
            context.legacy.INITIAL_CHECKPOINT_SHA if str(path)==cfg['initial_checkpoint'] else actual_digest(path)))
        stack.enter_context(patch.object(context,'verified_assets',return_value=({},dict(context.LEARNING_SOURCE_SHA))))
        def make_runtime(settings,method,journal,journal_path,assets):
            self.assertEqual(method,context.METHOD)
            runtime=SimpleNamespace()
            def save(path,value):
                if Path(path).name=='config.json':
                    journal['runs'][method]=dict(status='training',path=str(Path(path).parent))
                    context.legacy.atomic_json(journal_path,journal)
                context.legacy.atomic_json(path,value)
            def train(current):
                calls.append(dict(current))
                run=Path(current['resume_dir']) if current['resume_dir'] else root/'new_context_run'
                run.mkdir(exist_ok=True)
                runtime.atomic_json_save(run/'config.json',current)
                start=context.completed_epochs(run)
                for epoch in range(start+1,101):
                    if epoch <= 23:
                        torch.save(dict(epoch=epoch,validation=dict(pq=.43,threshold=.5,min_area=300)),run/'best.pt')
                    torch.save(dict(epoch=epoch),run/'last.pt')
                    if epoch % 5 == 0:
                        runtime.package_results(run,dict(epochs_completed=epoch))
                return dict(output=str(run))
            def export_best(run,checkpoint):
                context.legacy.atomic_json(run/'best_epoch.json',dict(epoch=checkpoint['epoch'],best=checkpoint['validation']))
                (run/'best_validation.csv').write_text('fixture\n')
            runtime.atomic_json_save=save;runtime.run_training=train;runtime.export_best_predictions=export_best
            runtime.package_results=lambda run,summary: str(run/'small_results.zip')
            return runtime
        exports=[]
        def export(runtime,current,run):
            exports.append(context.METHOD)
            if fail_export and len(exports)==1:
                raise RuntimeError('session expiry during export')
            path=Path(run)/'raw_paper_views.jsonl';path.write_text('{}\n');return path
        stack.enter_context(patch.object(context.legacy,'make_runtime',side_effect=make_runtime))
        stack.enter_context(patch.object(context.legacy,'export_rich',side_effect=export))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def test_control_only_progress_is_ignored_and_never_modified(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg=self.fixture(root);output=Path(cfg['output_root']);output.mkdir()
            control=root/'old_control';control.mkdir();(control/'last.pt').write_bytes(b'old control preserved')
            old=output/'experiment_progress.json'
            context.legacy.atomic_json(old,dict(identity=self.identity(cfg),runs={'control':dict(status='training',path=str(control))}))
            before=old.read_bytes();calls=[]
            with self.dependencies(root,cfg,calls):
                result=context.run_context(cfg)
            self.assertEqual(len(calls),1);self.assertIsNone(calls[0]['resume_dir'])
            self.assertEqual(calls[0]['execution_methods'],[context.METHOD])
            self.assertEqual(calls[0]['initial_checkpoint'],cfg['initial_checkpoint'])
            self.assertEqual(old.read_bytes(),before)
            self.assertEqual((control/'last.pt').read_bytes(),b'old control preserved')
            self.assertFalse(result['comparison']['baseline_trained_again'])
            self.assertAlmostEqual(result['comparison']['delta_pq'],.03)
            saved=json.loads((Path(result['context_run'])/'config.json').read_text())
            self.assertEqual(saved['execution_methods'],[context.METHOD])
            self.assertIn('new context block',saved['initialization'])
            with zipfile.ZipFile(result['archive']) as archive:
                self.assertFalse(any(name.startswith('control/') for name in archive.namelist()))
                self.assertNotIn('best.pt',archive.namelist())

    def test_existing_context_resumes_same_path_with_same_learning_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg=self.fixture(root);output=Path(cfg['output_root']);output.mkdir()
            run=root/'existing_context';run.mkdir();torch.save(dict(epoch=7),run/'last.pt')
            old=output/'experiment_progress.json'
            context.legacy.atomic_json(old,dict(identity=self.identity(cfg),runs={context.METHOD:dict(status='training',path=str(run))}))
            before=old.read_bytes();calls=[]
            with self.dependencies(root,cfg,calls):
                result=context.run_context(cfg)
            self.assertEqual(calls[0]['resume_dir'],str(run))
            self.assertEqual(calls[0]['research_source_sha256'],context.LEARNING_SOURCE_SHA)
            self.assertEqual(calls[0]['epochs'],100)
            self.assertEqual(result['context_run'],str(run));self.assertEqual(old.read_bytes(),before)

    def test_existing_context_settings_drift_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg=self.fixture(root);output=Path(cfg['output_root']);output.mkdir()
            context.legacy.atomic_json(output/'experiment_progress.json',dict(identity=self.identity(cfg),runs={}))
            calls=[];changed=dict(cfg,lr=1e-2)
            with self.dependencies(root,cfg,calls):
                with self.assertRaisesRegex(ValueError,'Previous experiment settings'):
                    context.run_context(changed)
            self.assertEqual(calls,[])

    def test_invalid_original_baseline_fails_before_any_result_metadata_or_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg=self.fixture(root)
            torch.save(dict(epoch=100,validation=dict(pq=.6,threshold=.5,min_area=300)),cfg['initial_checkpoint'])
            calls=[]
            with self.dependencies(root,cfg,calls):
                with self.assertRaisesRegex(ValueError,'Original best84'):
                    context.run_context(cfg)
            self.assertEqual(calls,[]);self.assertFalse(Path(cfg['output_root']).exists())

    def test_export_interruption_never_repeats_completed_context_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg=self.fixture(root);calls=[]
            with self.dependencies(root,cfg,calls,fail_export=True):
                with self.assertRaisesRegex(RuntimeError,'session expiry'):
                    context.run_context(cfg)
                result=context.run_context(cfg)
                context.run_context(cfg)
                with self.assertRaisesRegex(ValueError,'settings/source changed'):
                    context.run_context(dict(cfg,lr=1e-2))
            self.assertEqual(len(calls),1)
            self.assertEqual(result['comparison']['best_context_epoch'],23)

    def test_source_tampering_is_rejected(self):
        with patch.object(context.legacy,'digest',return_value='changed'):
            with self.assertRaisesRegex(ValueError,'Learning source differs'):
                context.verified_assets(dict(base_runtime='/fake/base_runtime.py'))

    def test_metadata_tamper_is_detected_before_cached_baseline_or_completed_training_skip(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg=self.fixture(root);calls=[]
            for key in ('annotations','train_split','val_split','synthetic_annotations','synthetic_split'):
                path=root/(key+'.json');path.write_text('{}')
                cfg[key]=str(path)
                cfg['expected_source_sha256'][key]=context.legacy.digest(path)
            with self.dependencies(root,cfg,calls):
                context.run_context(cfg)
                Path(cfg['val_split']).write_text('{"changed":true}')
                with patch.object(context,'current_device_reference') as cached_baseline:
                    with self.assertRaisesRegex(ValueError,'Input metadata.*val_split'):
                        context.run_context(cfg)
                    cached_baseline.assert_not_called()
            self.assertEqual(len(calls),1)

    def test_pilot_pause_keeps_100_epoch_schedule_and_resumes_to_next_target(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg=self.fixture(root);calls=[]
            with self.dependencies(root,cfg,calls):
                first=context.run_context(dict(cfg,run_until_epoch=20))
                repeated=context.run_context(dict(cfg,run_until_epoch=20))
                self.assertEqual(len(calls),1)
                second=context.run_context(dict(cfg,run_until_epoch=40))
                self.assertEqual(len(calls),2)
                final=context.run_context(dict(cfg,run_until_epoch=100))
            self.assertEqual(first['comparison']['status'],'paused')
            self.assertEqual(first['comparison']['epochs_completed'],20)
            self.assertEqual(repeated['comparison']['epochs_completed'],20)
            self.assertEqual(second['comparison']['epochs_completed'],40)
            self.assertEqual(final['comparison']['status'],'complete')
            self.assertEqual(final['comparison']['epochs_completed'],100)
            self.assertTrue(all(call['epochs']==100 for call in calls))
            self.assertEqual(calls[1]['resume_dir'],first['context_run'])

    def test_current_device_baseline_is_evaluated_once_and_cached_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg=self.fixture(root)
            cfg.update(device='cpu',encoder='fixture',cache_root=str(root/'cache'),batch=1,infer_batch=1)
            annotation=root/'annotations.json';annotation.write_text('{}');cfg['annotations']=str(annotation)
            cfg['evaluate_baseline_current_device']=True
            torch.save(dict(epoch=84,validation=dict(pq=.4,threshold=.5,min_area=300),
                model_state_dict=torch.nn.Linear(1,1).state_dict()),cfg['initial_checkpoint'])
            reference=dict(validation=dict(pq=.4,threshold=.5,min_area=300))
            calls=[]
            def validate(*args):
                calls.append(args);return dict(pq=.399,threshold=.5,min_area=300),[],{}
            runtime=SimpleNamespace(build_model=lambda *a,**k:torch.nn.Linear(1,1),
                prepare_split=lambda data: ({'train'},['val.jpeg'],{'rater':[]}),
                ImageStore=lambda *a:None,validate_native=validate)
            with patch.object(context.legacy,'load_runtime',return_value=runtime):
                first=context.current_device_reference(cfg,reference,root,'identity')
                second=context.current_device_reference(cfg,reference,root,'identity')
                with self.assertRaisesRegex(ValueError,'device/precision/batch changed'):
                    context.current_device_reference(dict(cfg,infer_batch=2),reference,root,'identity')
            self.assertEqual(len(calls),1);self.assertEqual(first,second)
            self.assertEqual(first['historical_validation']['pq'],.4)
            self.assertEqual(first['validation']['pq'],.399)
            self.assertEqual(first['hardware']['infer_batch'],1)


if __name__=='__main__':
    unittest.main()
