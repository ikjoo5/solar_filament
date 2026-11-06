"""Optional real-library CPU check: RUN_ULTRALYTICS_RESUME_SMOKE=1 python -m unittest ..."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import chase_yolo_training as mod


@unittest.skipUnless(os.environ.get('RUN_ULTRALYTICS_RESUME_SMOKE') == '1',
                     'Optional actual Ultralytics CPU optimizer/resume check')
class RealResumeSmoke(unittest.TestCase):
    def test_interrupted_training_matches_uninterrupted_live_and_ema(self):
        from ultralytics import YOLO
        import yaml
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='chase_yolo_resume_') as directory:
            root = Path(directory)
            model = YOLO('yolo26n-seg.yaml')
            architecture = dict(model.model.yaml, nc=1, scale='n'); del model
            model_yaml = root/'yolo26n-seg.yaml'
            model_yaml.write_text(yaml.safe_dump(architecture))
            model = YOLO(str(model_yaml), task='segment')
            model.model.names = {0:'filament'}; model.save(root/'initial.pt'); del model
            records = []
            for domain in ('chase',):
                for index in range(2):
                    image = root/'data/images/train'/f'{domain}{index}.png'
                    label = root/'data/labels/train'/f'{domain}{index}.txt'
                    image.parent.mkdir(parents=True, exist_ok=True); label.parent.mkdir(parents=True, exist_ok=True)
                    pixels = np.full((64,64,3), 128, np.uint8)
                    cv2.rectangle(pixels, (15+index,15), (40,35), (20,20,20), -1)
                    cv2.imwrite(str(image), pixels)
                    label.write_text('0 0.23 0.23 0.64 0.23 0.64 0.56 0.23 0.56\n')
                    records.append(dict(domain=domain, physical=f'{domain}{index}', image=str(image), label=str(label)))
            dataset = root/'data/dataset.yaml'
            dataset.write_text(json.dumps(dict(path=str(root/'data'), train='images/train',
                val='images/train', names={0:'filament'})))
            cfg = dict(mod.DEFAULTS, output_root=str(root/'resumed'), yolo_checkpoint=str(root/'initial.pt'),
                images=str(root/'data/images'), annotations='unused', train_split='unused', val_split='unused',
                chase_root='unused', chase_manifest='unused', cache_root=str(root/'cache'), epochs=2,
                training_domain='chase_only', epoch_sampling='full_pass', real_finish_epochs=0, steps_per_epoch=2,
                chase_ratio=1., device='cpu', amp=False)
            original_options = mod.training_options
            def tiny_options(config, path):
                result = original_options(config, path)
                result.update(imgsz=64, verbose=False, project=str(Path(config['output_root'])/'scratch'))
                return result
            with patch.object(mod, 'ORIGINAL_SHA', mod.digest(root/'initial.pt')), \
                 patch.object(mod, 'prepare_dataset', return_value=(dataset, dict(identity='tiny', records=records))), \
                 patch.object(mod, 'training_options', side_effect=tiny_options):
                paused = mod.train(cfg, 1); self.assertEqual(paused['status'], 'paused')
                mod.train(cfg, 2)
                mod.train(dict(cfg, output_root=str(root/'continuous')), 2)
            resumed = torch.load(root/'resumed/last_full.pt', map_location='cpu', weights_only=False)
            full = torch.load(root/'continuous/last_full.pt', map_location='cpu', weights_only=False)
            self.assertEqual(resumed['history'][0]['sample_counts'], {'gong':0, 'chase':2})
            self.assertEqual(resumed['history'][1]['sample_counts'], {'gong':0, 'chase':2})
            self.assertEqual(resumed['history'][0]['unique_chase_observations'],2)
            self.assertEqual(resumed['history'][1]['unique_chase_observations'],2)
            self.assertEqual(paused['epoch_sampling'],'full_pass')
            self.assertEqual(paused['effective_steps_per_epoch'],2)
            for kind in ('live_state', 'ema_full_state'):
                for key in full[kind]: torch.testing.assert_close(resumed[kind][key], full[kind][key], rtol=0, atol=0)
            self.assertEqual(resumed['scheduler_state'], full['scheduler_state'])
            self.assertEqual(resumed['updates'], full['updates'])


if __name__ == '__main__': unittest.main()
