"""Actual tiny HF-DETR continuation, optimizer-state resume, and source isolation."""
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
from test_filament_detr import fixture


def source_fixture(root):
    cfg=fixture(root)
    cfg.update(detr_epochs=1)
    folder=Path(cfg['output_root']);folder.mkdir()
    (folder/'experiment_config.json').write_text(json.dumps(cfg,indent=2))
    detr.train_detr(cfg,1)
    return dict(base_root=str(folder),output_root=str(root/'extension'),added_epochs=2,
        continuation_lr=1e-4,continuation_backbone_lr=1e-5,
        continuation_image_size=64,continuation_batch=1,continuation_accumulate=2,
        continuation_precision='fp32',
        device='cpu',cache_root=str(root/'extension_cache'))


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_real_continuation_exact_resume_and_source_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=source_fixture(root)
            base=Path(cfg['base_root'])
            before={str(p.relative_to(base)):detr.digest(p) for p in base.rglob('*') if p.is_file()}
            continuous=dict(cfg,output_root=str(root/'continuous'))
            resumed=dict(cfg,output_root=str(root/'resumed'))
            continuation.train_extension(continuous,2)
            one=continuation.train_extension(resumed,1)
            self.assertEqual(one['cumulative_epochs_completed'],2)
            first=continuation.load_checkpoint(root/'resumed/last_full.pt')
            self.assertEqual(first['extension_history'][0]['optimizer_attempts'],2)
            self.assertTrue(all(float(v['step'])==4 for v in first['optimizer']['state'].values()))
            self.assertAlmostEqual(first['extension_history'][0]['next_lr'][0],5.5e-5)
            final=continuation.train_extension(resumed,2)
            a=continuation.load_checkpoint(root/'continuous/last_full.pt')
            b=continuation.load_checkpoint(root/'resumed/last_full.pt')
            maximum=max(float((a['model'][k].float()-b['model'][k].float()).abs().max()) for k in a['model'])
            self.assertEqual(maximum,0.)
            self.assertTrue(all(float(v['step'])==6 for v in b['optimizer']['state'].values()))
            self.assertEqual(b['completed_epoch'],3)
            self.assertEqual(b['added_epochs_completed'],2)
            self.assertEqual(b['fixed_final_epoch'],3)
            self.assertEqual(len(b['base_history']),1)
            self.assertEqual(len(b['extension_history']),2)
            self.assertEqual(b['scaler'],{})
            self.assertAlmostEqual(b['optimizer']['param_groups'][0]['lr'],1e-5)
            with patch.object(detr,'build_detector',side_effect=AssertionError('complete model rebuilt')):
                again=continuation.train_extension(resumed,2)
            self.assertEqual(final['checkpoint_sha256'],again['checkpoint_sha256'])
            props=continuation.export_extension_proposals(resumed,final['checkpoint'])
            self.assertEqual(len(props),1)
            with patch.object(detr,'build_detector',side_effect=AssertionError('cached export rebuilt')):
                self.assertEqual(props,continuation.export_extension_proposals(resumed,final['checkpoint']))
            after={str(p.relative_to(base)):detr.digest(p) for p in base.rglob('*') if p.is_file()}
            self.assertEqual(before,after)
            evidence=dict(status='passed_actual_tiny_HF_DETR_CPU_continuation',source_epochs=1,
                added_epochs=2,cumulative_epochs=3,continuous_vs_resumed_model_max_difference=maximum,
                optimizer_moments_preserved=True,source_files_byte_identical=True,
                continuation_precision='fp32',source_and_continuation_image_size=64,
                production_DETR_training=False,GPU_training=False,
                source_sha256=detr.digest(Path(continuation.__file__)))
            destination=Path(__file__).resolve().parents[1]/'docs/evidence/bbox_detr_continuation_cpu_smoke.json'
            destination.write_text(json.dumps(evidence,indent=2)+'\n')

    def test_interruption_identity_and_source_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=source_fixture(root)
            continuation.train_extension(cfg,1)
            last=Path(cfg['output_root'])/'last_full.pt';before=detr.digest(last)
            original=detr.DetectionDataset.__getitem__;calls=0
            def interrupt(dataset,index):
                nonlocal calls
                calls+=1
                if calls>1:raise RuntimeError('mid-epoch interruption fixture')
                return original(dataset,index)
            with patch.object(detr.DetectionDataset,'__getitem__',interrupt),self.assertRaisesRegex(RuntimeError,'mid-epoch'):
                continuation.train_extension(cfg,2)
            self.assertEqual(detr.digest(last),before)
            self.assertFalse((Path(cfg['output_root'])/'epoch003.pt').exists())
            with self.assertRaisesRegex(ValueError,'changed'):
                continuation.train_extension(dict(cfg,continuation_lr=2e-4),2)
            with self.assertRaisesRegex(ValueError,'changed'):
                continuation.train_extension(dict(cfg,continuation_image_size=96),2)
            source=Path(cfg['base_root'])/'detr/last_full.pt'
            state=continuation.load_checkpoint(source)
            key=next(k for k,v in state['model'].items() if v.is_floating_point())
            state['model'][key]+=1
            detr.atomic_torch(source,state)
            with self.assertRaisesRegex(ValueError,'disagree'):
                continuation.train_extension(cfg,2)

    def test_changed_source_pixels_and_output_location_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=source_fixture(root)
            with self.assertRaisesRegex(ValueError,'separate'):
                continuation.normalize_config(dict(cfg,output_root=cfg['base_root']))
            with self.assertRaisesRegex(ValueError,'separate'):
                continuation.normalize_config(dict(cfg,output_root=str(Path(cfg['base_root'])/'extension')))
            continuation.train_extension(cfg,0)
            source_config=json.loads((Path(cfg['base_root'])/'experiment_config.json').read_text())
            victim=next(Path(source_config['images']).glob('*.jpeg'));victim.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                continuation.train_extension(cfg,1)

    def test_bf16_is_explicit_and_never_silently_falls_back(self):
        with self.assertRaisesRegex(ValueError,'CPU fixtures'):
            continuation._precision({'continuation_precision':'bf16'},torch.device('cpu'))
        with patch.object(torch.cuda,'is_bf16_supported',return_value=False),self.assertRaisesRegex(RuntimeError,'A100'):
            continuation._precision({'continuation_precision':'bf16'},torch.device('cuda'))
        with patch.object(torch.cuda,'is_bf16_supported',return_value=True):
            p=continuation._precision({'continuation_precision':'bf16'},torch.device('cuda'))
        self.assertTrue(p['autocast']);self.assertFalse(p['scaler']);self.assertEqual(p['dtype'],torch.bfloat16)

    def test_actual_higher_resolution_and_changed_microbatch_are_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=source_fixture(root)
            cfg.update(continuation_image_size=96,continuation_batch=2,continuation_accumulate=2)
            sizes=[];original=detr._resized_normalized
            def observed(pixels,size):
                sizes.append(size)
                return original(pixels,size)
            with patch.object(detr,'_resized_normalized',observed):
                result=continuation.train_extension(cfg,1)
                continuation.export_extension_proposals(cfg,result['checkpoint'])
            self.assertTrue(sizes);self.assertEqual(set(sizes),{96})
            state=continuation.load_checkpoint(Path(cfg['output_root'])/'last_full.pt')
            self.assertEqual(state['extension_history'][0]['batches'],2)
            self.assertEqual(state['extension_history'][0]['optimizer_attempts'],1)
            identity=json.loads((Path(cfg['output_root'])/'training_identity.json').read_text())
            self.assertEqual(identity['protocol']['source_protocol']['settings']['detr_image_size'],64)
            self.assertEqual(identity['protocol']['continuation']['image_size'],96)

    def test_new_stage_accepts_source_runtime_transition_but_resume_rejects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=source_fixture(root)
            source_document=Path(cfg['base_root'])/'detr/training_identity.json'
            source_before=detr.digest(source_document)
            original_runtime=detr.runtime_versions()
            changed_runtime=dict(original_runtime,fixture_runtime_transition='new_A100_stage_simulation')
            with patch.object(detr,'runtime_versions',return_value=changed_runtime):
                continuation.train_extension(cfg,1)
            self.assertEqual(detr.digest(source_document),source_before)
            doc=json.loads((Path(cfg['output_root'])/'training_identity.json').read_text())
            self.assertEqual(doc['protocol']['source_protocol']['runtime'],original_runtime)
            self.assertEqual(doc['protocol']['runtime'],changed_runtime)
            with self.assertRaisesRegex(ValueError,'runtime/schedule changed'):
                continuation.train_extension(cfg,2)

    def test_actual_hf_bbox_head_stays_fp32_inside_bf16_and_keeps_gradients(self):
        cfg=dict(strict=False,detr_tiny_fixture=True,detr_pretrained=False,detr_image_size=64,device='cpu')
        model=detr.build_detector(cfg,False)
        # HF's native tiny ResNet naming differs from timm's layer2/3/4 freeze
        # allowlist. Unfreeze only this fixture to test gradient flow end-to-end.
        model.model.backbone.requires_grad_(True)
        keys=set(model.state_dict())
        result_model=continuation.install_fp32_bbox_head(model)
        self.assertIs(result_model,model)
        forward=model.bbox_predictor.forward
        self.assertIs(continuation.install_fp32_bbox_head(model).bbox_predictor.forward,forward)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            result=model(pixel_values=torch.randn(2,3,64,64))
        self.assertEqual(result.pred_boxes.dtype,torch.float32)
        self.assertEqual(result.logits.dtype,torch.bfloat16)
        result.pred_boxes.square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.bbox_predictor.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.model.backbone.parameters() if p.requires_grad))
        self.assertEqual(set(model.state_dict()),keys)
        restored=continuation.install_fp32_bbox_head(detr.build_detector(cfg,False))
        restored.load_state_dict(model.state_dict())
        for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,restored.state_dict()[key]))


if __name__=='__main__':unittest.main()
