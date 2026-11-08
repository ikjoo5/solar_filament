"""Coverage, update boundaries and fail-closed resume for the full-pass detector."""
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import fullpass_yolo_training as mod


class FakeDataset:
    def __init__(self, records):
        self.im_files = [row['image'] for row in records]
        self.labels = ['unchanged']; self.transforms = []
    def __len__(self): return len(self.im_files)
    def __getitem__(self, index): return index


def records(counts=None):
    counts = counts or dict(real=616, swefil=250, synthetic=616)
    rows = []
    for domain, count in counts.items():
        for image in range(count):
            groups = [f'g{image}']
            if domain == 'real' and image == 0: groups.append('other_observer')
            for group in groups:
                for tile, (top, left) in enumerate(sorted(mod.TILE_ORIGINS)):
                    rows.append(dict(domain=domain, file_name=f'{image}.png', group_id=group,
                        top=top, left=left, image=f'/tmp/fullpass/{domain}/{image}_{group}_{tile}.png'))
    return rows


class FullPassYoloTests(unittest.TestCase):
    def test_every_image_has_all_four_tiles_each_epoch_and_observers_cycle(self):
        source = records(); data = mod.FullPassYoloDataset(FakeDataset(source), source)
        self.assertEqual(len(data), 5928)
        self.assertEqual(data.labels, ['unchanged'])
        prior = None; observer_sequence = []
        for epoch in (0, 1, 2, 99):
            data.set_epoch(epoch)
            self.assertEqual(len(set(data.plan)), 5928)
            self.assertEqual(data.domain_counts, dict(real=2464, swefil=1000, synthetic=2464))
            grouped = {}
            for domain, image, group, top, left in data.sample_keys:
                grouped.setdefault((domain, image), []).append((group, top, left))
            self.assertEqual(len(grouped), 1482)
            for variants in grouped.values():
                self.assertEqual(len({item[0] for item in variants}), 1)
                self.assertEqual({item[1:] for item in variants}, mod.TILE_ORIGINS)
            observer_sequence.append(grouped['real', '0.png'][0][0])
            repeat = mod.FullPassYoloDataset(FakeDataset(source), source)
            repeat.set_epoch(epoch); self.assertEqual(data.plan, repeat.plan)
            if prior is not None: self.assertNotEqual(prior, data.plan)
            prior = list(data.plan)
        self.assertNotEqual(observer_sequence[0], observer_sequence[1])
        self.assertEqual(observer_sequence[0], observer_sequence[2])

    def test_observer_selection_matches_native_loader(self):
        from fullpass_training_data import group_for_image
        for epoch in (1,2,3,100):
            for name in ('A.jpeg', 'B.jpeg'):
                groups = ['annotator2', 'annotator1', 'annotator0']
                self.assertEqual(mod.selected_group(groups,42,'real',name,epoch),
                    group_for_image(groups,42,'real',name,epoch))

    def test_missing_and_duplicate_variants_or_tiles_fail_instead_of_sampling_less(self):
        source = records(dict(real=1))
        with self.assertRaisesRegex(ValueError, 'four nonoverlapping'):
            mod.FullPassYoloDataset(FakeDataset(source[:-1]), source[:-1])
        with self.assertRaisesRegex(ValueError, 'own image alias'):
            mod.FullPassYoloDataset(FakeDataset(source+[source[0]]), source+[source[0]])
        with self.assertRaisesRegex(ValueError, 'discarded or added'):
            mod.FullPassYoloDataset(FakeDataset(source[:-1]), source)

    def test_order_does_not_depend_on_ultralytics_filesystem_sort_order(self):
        source = records(dict(real=2, swefil=1))
        data = mod.FullPassYoloDataset(FakeDataset(list(reversed(source))), source)
        data.set_epoch(15)
        self.assertEqual([record for record in data.sample_keys],
            [mod.record_key(list(reversed(source))[data[i]]) for i in range(len(data))])

    def test_augmentation_preserves_all_tile_pixels_and_schedule(self):
        options = mod.training_options(dict(cache_root='/tmp/test'), '/tmp/data.yaml', '/tmp/run')
        for key in ('mosaic','mixup','cutmix','copy_paste','degrees','translate',
                    'scale','shear','perspective','multi_scale'):
            self.assertEqual(options[key], 0, key)
        self.assertEqual(options['epochs'],100)
        self.assertEqual((options['batch'],options['nbs'],options['workers'],options['imgsz']), (1,4,0,1024))
        self.assertEqual(options['optimizer'],'AdamW')
        self.assertEqual(options['warmup_epochs'],0)
        self.assertFalse(options['cache']); self.assertFalse(options['val'])
        for changes in (dict(yolo_batch=2),dict(yolo_imgsz=512),dict(yolo_accumulate=1)):
            with self.assertRaises(ValueError): mod.training_options(changes,'a','b')

    def test_accumulation_averages_full_and_last_partial_window(self):
        parameter = torch.nn.Parameter(torch.tensor(0.))
        window = mod.AccumulationWindow(4)
        observed = []
        def step():
            observed.append(float(parameter.grad)); parameter.grad = None
        for index in range(6):
            (parameter*(index+1)).backward()
            updated = window.step([parameter], step, final=index==5)
            self.assertEqual(updated, index in (3,5))
        self.assertEqual(observed, [2.5, 5.5]); self.assertEqual(window.pending,0)

    def test_resume_window_is_same_at_any_epoch_boundary(self):
        def run(epochs, split=False):
            parameter = torch.nn.Parameter(torch.tensor(1.))
            optimizer = torch.optim.SGD([parameter],lr=.01)
            for epoch in range(epochs):
                # New instance at a completed epoch has no old partial gradients.
                window = mod.AccumulationWindow(4)
                for index in range(8):
                    (parameter.square()*(epoch+1)*(index+1)).backward()
                    window.step([parameter],lambda:(optimizer.step(),optimizer.zero_grad(set_to_none=True)))
                if split:
                    restored = torch.nn.Parameter(parameter.detach().clone())
                    resumed_optimizer = torch.optim.SGD([restored],lr=.01)
                    resumed_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
                    parameter, optimizer = restored, resumed_optimizer
            return parameter
        torch.testing.assert_close(run(3),run(3,True),rtol=0,atol=0)

    def test_generic_initialization_download_is_hash_checked_and_cached(self):
        raw = b'pinned generic model'
        import hashlib
        with tempfile.TemporaryDirectory() as folder, patch.object(mod,'PRETRAINED_SHA256',hashlib.sha256(raw).hexdigest()):
            with patch.object(mod,'urlopen',return_value=io.BytesIO(raw)) as download:
                result = mod.download_generic_weights(folder)
                self.assertEqual(result.read_bytes(),raw); download.assert_called_once()
            with patch.object(mod,'urlopen',side_effect=AssertionError('unexpected redownload')):
                self.assertEqual(mod.download_generic_weights(folder),result)
            result.unlink()
            with patch.object(mod,'urlopen',return_value=io.BytesIO(b'wrong')):
                with self.assertRaisesRegex(ValueError,'hash mismatch'): mod.download_generic_weights(folder)
            self.assertFalse(result.exists())

    def test_resume_checkpoint_must_agree_with_verified_remote_epoch_identity_and_history(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'last.pt'
            class Store:
                def restore(self, record): return path
            state=dict(epoch=2,last={},best_epoch=1,best_pq=.1)
            value=dict(identity='run',completed_epoch=2,epoch=1,best_epoch=1,best_pq=.1,
                history=[dict(epoch=1),dict(epoch=2)],live_state={},ema_full_state={},
                optimizer={},scheduler_state={},rng_state={})
            torch.save(value,path)
            self.assertEqual(mod._checked_resume(Store(),state,'run')[1]['completed_epoch'],2)
            for bad in (dict(identity='other'),dict(epoch=2),dict(history=[dict(epoch=2)]),dict(optimizer=None)):
                torch.save(dict(value,**bad),path)
                with self.assertRaises(ValueError): mod._checked_resume(Store(),state,'run')

    def test_export_best_epoch_uses_one_based_completed_epoch(self):
        import solar_a100_train_standalone as base
        payload=dict(epoch=1,completed_epoch=2,validation_predictions={})
        with patch.object(base,'export_best_predictions') as export:
            mod._export_predictions('/tmp/out',payload)
        self.assertEqual(export.call_args.args[1]['epoch'],2)
        self.assertEqual(payload['epoch'],1)


if __name__ == '__main__': unittest.main()
