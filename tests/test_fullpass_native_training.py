"""Native full-pass contracts: complete coverage and exact interrupted resume."""
from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import fullpass_native_training as worker
from test_swefil_mixed_training import (
    FakeVerifiedStore, ToyDataset, assert_nested_equal, toy_config,
    toy_model, toy_validator,
)


class ProgressStore(FakeVerifiedStore):
    def __init__(self, root):
        super().__init__(root)
        self.interrupt_progress = None
        self.progress_writes = []

    def latest_progress(self):
        path = self.root/'progress.json'
        return json.loads(path.read_text()) if path.exists() else None

    def publish_progress(self, path, metadata):
        checksum = worker.digest(path)
        filename = checksum+'.pt'
        shutil.copyfile(path, self.root/filename)
        record = dict(file_id=filename, sha256=checksum, size=Path(path).stat().st_size,
                      name=Path(path).name, path='progress.pt')
        value = dict(metadata, checkpoint=record)
        worker.atomic_json(self.root/'progress.json', value)
        self.progress_writes.append((metadata['epoch'], metadata['step']))
        if self.interrupt_progress == (metadata['epoch'], metadata['step']):
            raise InterruptedError('Runtime stopped after verified progress save')
        return value

    def clear_progress(self, completed_epoch):
        value = self.latest_progress()
        if value and value['epoch'] <= completed_epoch:
            (self.root/'progress.json').unlink()


def config(root):
    # The stale historical crop budget deliberately differs from dataset length.
    # The new engine must use all dataset rows, never steps_per_epoch.
    return dict(toy_config(root), steps_per_epoch=1, checkpoint_interval_updates=1)


def train(root, cap=4, store=None, dataset=None, factory=toy_model):
    store = store or ProgressStore(root/'server')
    return worker.run_native(config(root), dataset or ToyDataset(), toy_validator,
        root/'unet', 'fixture-v1', store, cap, model_factory=factory)


def checkpoint(root):
    store = ProgressStore(root/'server')
    return worker._load_remote(store, store.latest_state()['last'], 'fixture-v1')


class FullPassNativeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def assert_training_equal(self, left, right):
        for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
                    'rng_state', 'best_epoch', 'best_pq'):
            assert_nested_equal(self, left[key], right[key])
        self.assertEqual([r['train_loss'] for r in left['history']],
                         [r['train_loss'] for r in right['history']])

    def test_uses_every_tile_not_old_step_budget(self):
        root = self.root/'coverage'
        result = train(root, 1)
        last = checkpoint(root)
        row = last['history'][0]
        self.assertEqual(row['tiles_seen'], 6)
        self.assertEqual(row['microbatches'], 6)
        self.assertEqual(row['optimizer_update_attempts'], 3)
        self.assertEqual(result['samples_per_epoch'], 6)
        self.assertFalse((root/'latest_result.json').exists())
        self.assertTrue((root/'unet/latest_result.json').is_file())

    def test_completed_epoch_resume_matches_continuous(self):
        whole, resumed = self.root/'whole', self.root/'resumed'
        train(whole)
        train(resumed, 2)
        train(resumed)
        self.assert_training_equal(checkpoint(whole), checkpoint(resumed))

    def test_mid_epoch_resume_matches_continuous_optimizer_and_rng(self):
        whole, resumed = self.root/'whole', self.root/'resumed'
        train(whole)
        store = ProgressStore(resumed/'server')
        store.interrupt_progress = (2, 4)
        with self.assertRaisesRegex(InterruptedError, 'verified progress'):
            train(resumed, store=store)
        self.assertEqual(store.latest_state()['epoch'], 1)
        self.assertEqual(store.latest_progress()['step'], 4)
        store.interrupt_progress = None
        train(resumed, store=store)
        self.assert_training_equal(checkpoint(whole), checkpoint(resumed))
        self.assertIsNone(store.latest_progress())

    def test_first_epoch_resume_does_not_fetch_finished_tiles(self):
        root = self.root/'cursor'
        store = ProgressStore(root/'server')
        store.interrupt_progress = (1, 4)
        with self.assertRaises(InterruptedError):
            train(root, 1, store=store)
        self.assertIsNone(store.latest_state())
        store.interrupt_progress = None

        class TrackedDataset(ToyDataset):
            def __init__(self):
                self.seen = []
            def __getitem__(self, index):
                self.seen.append(index)
                return super().__getitem__(index)

        ds = TrackedDataset()
        fresh_flags = []
        def factory(fresh):
            fresh_flags.append(fresh)
            return toy_model(fresh)
        train(root, 1, store=store, dataset=ds, factory=factory)
        self.assertEqual(ds.seen, [4, 5])
        self.assertEqual(fresh_flags, [False])
        self.assertEqual(store.latest_state()['epoch'], 1)

    def test_failed_epoch_commit_resumes_verified_progress(self):
        whole, resumed = self.root/'whole', self.root/'resumed'
        train(whole)
        store = ProgressStore(resumed/'server')
        store.fail_epoch = 2
        with self.assertRaisesRegex(OSError, 'acknowledgement'):
            train(resumed, store=store)
        self.assertEqual(store.latest_state()['epoch'], 1)
        self.assertEqual(store.latest_progress()['epoch'], 2)
        store.fail_epoch = None
        train(resumed, store=store)
        self.assert_training_equal(checkpoint(whole), checkpoint(resumed))

    def test_completed_invocation_does_not_construct_model_or_probe_quota(self):
        root = self.root/'complete'
        store = ProgressStore(root/'server')
        train(root, 2, store=store)
        before = list(store.preflight_budgets)
        with patch.object(worker.base, 'make_epoch_loader', side_effect=AssertionError('No training loader')):
            result = train(root, 2, store=store, factory=lambda _: self.fail('No model initialization'))
        self.assertEqual(result['epochs_completed'], 2)
        self.assertEqual(store.preflight_budgets, before)

    def test_missing_remote_state_never_silently_restarts(self):
        root = self.root/'missing'
        train(root, 1)
        (root/'server/commit.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'refusing fresh restart'):
            train(root, 2)

    def test_invalid_cursor_and_recipe_fail_closed(self):
        root = self.root/'invalid'
        store = ProgressStore(root/'server')
        store.interrupt_progress = (1, 2)
        with self.assertRaises(InterruptedError):
            train(root, 1, store=store)
        value = store.latest_progress()
        value['base_epoch'] = 1
        worker.atomic_json(store.root/'progress.json', value)
        with self.assertRaisesRegex(ValueError, 'does not follow'):
            train(root, 1, store=store)
        value['base_epoch'] = 0
        worker.atomic_json(store.root/'progress.json', value)
        cfg = dict(config(root), lr=.9)
        with self.assertRaisesRegex(ValueError, 'resume settings changed: lr'):
            worker.run_native(cfg, ToyDataset(), toy_validator, root/'unet', 'fixture-v1', store,
                              1, model_factory=toy_model)

    def test_no_cap_mismatch_solar_warmstart_or_worker_rng(self):
        root = self.root/'bad'
        for changes, message in (
            ({'samples_per_epoch': 400}, 'complete dataset'),
            ({'initial_checkpoint': 'best84.pt'}, 'No solar checkpoint'),
            ({'workers': 2}, 'workers=0'),
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                worker.run_native(dict(config(root), **changes), ToyDataset(), toy_validator,
                    root/'unet', 'fixture-v1', ProgressStore(root/'server'), 1,
                    model_factory=toy_model)

    def test_all_skipped_optimizer_updates_cannot_complete_epoch(self):
        class AllSkippedScaler:
            def __init__(self, *args, **kwargs): self.value = 1024.
            def scale(self, loss): return loss
            def unscale_(self, optimizer): pass
            def get_scale(self): return self.value
            def step(self, optimizer): pass
            def update(self): self.value /= 2
            def state_dict(self): return {'value': self.value}
        root = self.root/'skipped'
        store = ProgressStore(root/'server')
        with patch.object(torch.amp, 'GradScaler', AllSkippedScaler):
            with self.assertRaisesRegex(FloatingPointError, 'All optimizer updates were skipped'):
                train(root, 1, store=store)
        self.assertIsNone(store.latest_state())


if __name__ == '__main__':
    unittest.main()
