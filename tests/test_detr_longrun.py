"""Real tiny HF DETR resume and failure recovery, with deterministic monitor scores."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import filament_detr as detr
import bbox_detr_continuation as continuation
import detr_longrun as longrun
from test_bbox_detr_continuation import source_fixture as original_fixture


def source_fixture(root, added=3, patience=10):
    old=original_fixture(root)
    study=root/'completed_study';study.mkdir()
    extension=dict(old,output_root=str(study/'detr_extension'),added_epochs=1)
    (study/'experiment_config.json').write_text(json.dumps(extension))
    continuation.train_extension(extension,1)
    return dict(source_root=str(study),output_root=str(root/'longrun'),
        cache_root=str(root/'longrun_cache'),device='cpu',max_epoch=2+added,
        patience=patience,min_delta=.001,continuation_image_size=64,
        continuation_batch=1,continuation_accumulate=2,continuation_precision='fp32',
        continuation_lr=5e-5,continuation_backbone_lr=5e-6)


def evaluation(scores):
    def evaluate(cfg,path,epoch):
        state=continuation.load_checkpoint(path)
        assert state['completed_epoch']==epoch
        return dict(score=dict(pq=scores.get(epoch,.4),tp=2,fp=1,fn=1),
                    checkpoint_sha256=detr.digest(path),selected_confidence=.8)
    return evaluate


def files_snapshot(root):
    return {str(path.relative_to(root)):detr.digest(path) for path in root.rglob('*') if path.is_file()}


class LongrunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_actual_hf_continuous_resume_and_source_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=source_fixture(root)
            source=Path(cfg['source_root']);original=root/'output'
            before=(files_snapshot(source),files_snapshot(original))
            scores=evaluation({2:.4,3:.42,4:.41,5:.43})
            full=dict(cfg,output_root=str(root/'continuous'))
            resumed=dict(cfg,output_root=str(root/'resumed'))
            # Bootstrap resume authenticates these exact, unnormalized bytes.
            (root/'resumed').mkdir()
            config_path=root/'resumed/experiment_config.json'
            config_bytes=json.dumps(resumed,sort_keys=True,indent=4).encode()
            config_path.write_bytes(config_bytes)
            longrun.run(full,5,scores)
            partial=longrun.run(resumed,3,scores)
            self.assertEqual(partial['status'],'paused')
            result=longrun.run(resumed,5,scores)
            a=continuation.load_checkpoint(root/'continuous/last_full.pt')
            b=continuation.load_checkpoint(root/'resumed/last_full.pt')
            self.assertTrue(continuation._tensor_states_equal(a['model'],b['model']))
            self.assertEqual(a['training_history'],b['training_history'])
            self.assertEqual(len(b['training_history']),3)
            self.assertTrue(all(float(v['step'])==10 for v in b['optimizer']['state'].values()))
            self.assertAlmostEqual(b['optimizer']['param_groups'][0]['lr'],1e-5)
            self.assertEqual(result['best_epoch'],5)
            self.assertEqual(result['best_pq'],.43)
            best=continuation.load_checkpoint(root/'resumed/best.pt')
            self.assertTrue(continuation._tensor_states_equal(best['model'],b['model']))
            self.assertIsNone(best['optimizer'])
            self.assertEqual(best['validation_selection']['selected_confidence'],.8)
            self.assertEqual(best['validation_selection']['score']['pq'],result['best_pq'])
            self.assertEqual(best['validation_selection']['epoch'],best['completed_epoch'])
            self.assertEqual(sorted(p.name for p in (root/'resumed').glob('*.pt')),['best.pt','last_full.pt'])
            with patch.object(detr,'build_detector',side_effect=AssertionError('terminal training restarted')):
                again=longrun.run(resumed,5,scores)
            self.assertEqual(again['best_checkpoint_sha256'],result['best_checkpoint_sha256'])
            self.assertEqual(config_path.read_bytes(),config_bytes)
            self.assertTrue((root/'resumed/resolved_config.json').is_file())
            self.assertEqual(before,(files_snapshot(source),files_snapshot(original)))

    def test_exact_ten_bad_epochs_and_baseline_best(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=source_fixture(root,added=14)
            scores=evaluation({2:.5})
            result=longrun.run(cfg,16,scores)
            self.assertEqual(result['status'],'early_stopped')
            self.assertEqual(result['completed_epoch'],12)
            self.assertEqual(result['bad_epochs'],10)
            self.assertEqual(result['best_epoch'],2)
            best=continuation.load_checkpoint(root/'longrun/best.pt')
            source=continuation.load_checkpoint(root/'completed_study/detr_extension/epoch002.pt')
            self.assertTrue(continuation._tensor_states_equal(best['model'],source['model']))
            with patch.object(detr,'build_detector',side_effect=AssertionError('stopped run restarted')):
                again=longrun.run(cfg,16,scores)
            self.assertEqual(again['bad_epochs'],10)
            control_path=root/'longrun/control_state.json';control=json.loads(control_path.read_text())
            control['bad_epochs']=0;control_path.write_text(json.dumps(control))
            with self.assertRaisesRegex(ValueError,'inconsistent'):
                longrun.run(cfg,16,scores)

    def test_pending_validation_replays_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=source_fixture(root)
            valid=evaluation({2:.4,3:.42,4:.43,5:.44})
            def interrupted(config,path,epoch):
                if epoch==3:raise RuntimeError('validation interrupted')
                return valid(config,path,epoch)
            with self.assertRaisesRegex(RuntimeError,'validation interrupted'):
                longrun.run(cfg,5,interrupted)
            last=root/'longrun/last_full.pt';before=detr.digest(last)
            control=json.loads((root/'longrun/control_state.json').read_text())
            self.assertEqual(control['last_validated_epoch'],2)
            self.assertEqual(continuation.load_checkpoint(last)['completed_epoch'],3)
            seen=[]
            def replay(config,path,epoch):
                seen.append(epoch)
                self.assertEqual(detr.digest(path),before)
                return valid(config,path,epoch)
            with patch.object(longrun,'_train_epoch',side_effect=AssertionError('trained before pending validation')):
                result=longrun.run(cfg,3,replay)
            self.assertEqual(seen,[3]);self.assertEqual(result['best_epoch'],3)
            self.assertEqual(detr.digest(last),before)

    def test_best_transaction_crash_before_and_after_control_commit(self):
        for after_commit in (False,True):
            with self.subTest(after_commit=after_commit),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);cfg=source_fixture(root)
                scores=evaluation({2:.4,3:.5,4:.45,5:.46})
                longrun.run(cfg,2,scores)
                atomic=detr.atomic_json
                def interrupted(path,value):
                    if Path(path).name=='control_state.json' and value['last_validated_epoch']==3:
                        if after_commit:atomic(path,value)
                        raise RuntimeError('controller commit interrupted')
                    return atomic(path,value)
                with patch.object(detr,'atomic_json',interrupted),self.assertRaisesRegex(RuntimeError,'controller commit'):
                    longrun.run(cfg,3,scores)
                self.assertTrue((root/'longrun/best_pending.pt').exists())
                with patch.object(longrun,'_train_epoch',side_effect=AssertionError('duplicate training')):
                    result=longrun.run(cfg,3,scores)
                self.assertEqual(result['best_epoch'],3)
                self.assertFalse((root/'longrun/best_pending.pt').exists())
                best=continuation.load_checkpoint(root/'longrun/best.pt')
                current=continuation.load_checkpoint(root/'longrun/last_full.pt')
                self.assertTrue(continuation._tensor_states_equal(best['model'],current['model']))

    def test_mid_epoch_retry_and_wrong_evaluation_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=source_fixture(root)
            scores=evaluation({2:.4,3:.5})
            longrun.run(cfg,2,scores);last=root/'longrun/last_full.pt';before=detr.digest(last)
            original=detr.DetectionDataset.__getitem__;calls=0
            def interrupted(dataset,index):
                nonlocal calls
                calls+=1
                if calls>1:raise RuntimeError('mid epoch interrupted')
                return original(dataset,index)
            with patch.object(detr.DetectionDataset,'__getitem__',interrupted),self.assertRaisesRegex(RuntimeError,'mid epoch'):
                longrun.run(cfg,3,scores)
            self.assertEqual(detr.digest(last),before)
            bad=lambda config,path,epoch:dict(score=dict(pq=.9),checkpoint_sha256='wrong')
            with self.assertRaisesRegex(ValueError,'SHA'):
                longrun.run(cfg,3,bad)
            self.assertEqual(json.loads((root/'longrun/control_state.json').read_text())['last_validated_epoch'],2)

    def test_raw_best_is_separate_from_significant_patience_anchor(self):
        state,is_best=longrun.monitor_update(None,60,.4,.001,10)
        self.assertTrue(is_best)
        for epoch,pq in [(61,.4004),(62,.4008)]:
            state,is_best=longrun.monitor_update(state,epoch,pq,.001,10)
            self.assertTrue(is_best)
        self.assertEqual(state['bad_epochs'],2)
        self.assertEqual(state['significant_anchor_epoch'],60)
        state,is_best=longrun.monitor_update(state,63,.4011,.001,10)
        self.assertTrue(is_best);self.assertEqual(state['bad_epochs'],0)
        self.assertEqual(state['significant_anchor_epoch'],63)
        with self.assertRaisesRegex(ValueError,'finite'):
            longrun.monitor_update(state,64,float('nan'),.001,10)


if __name__=='__main__':unittest.main()
