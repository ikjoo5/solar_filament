"""CPU checks of CHASE supervision and completed-epoch training lifecycle."""
from pathlib import Path
import sys

import numpy as np
import unittest
import tempfile
from unittest.mock import patch
import torch
from torch import nn

TOOLS = Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0, str(TOOLS))
import chase_best84_training as worker


def raises(exception, match=None):
    case = unittest.TestCase()
    return case.assertRaisesRegex(exception, match) if match else case.assertRaises(exception)


def test_ignored_pixels_have_no_gradient_or_loss_effect():
    logits = torch.zeros((1, 2, 3, 4), requires_grad=True)
    truth = torch.zeros_like(logits); truth[0, 0, 1, 1] = 1
    valid = torch.zeros_like(logits); valid[:, 0] = 1; valid[0, 0, 1, 1] = 0
    first, _ = worker.masked_segmentation_loss(logits, truth, valid)
    changed = logits.detach().clone(); changed[0, 0, 1, 1] = -100
    second, _ = worker.masked_segmentation_loss(changed, truth, valid)
    assert first.item() == second.item()
    first.backward()
    assert logits.grad[0, 0, 1, 1].item() == 0
    assert torch.count_nonzero(logits.grad[:, 1]).item() == 0
    assert torch.count_nonzero(logits.grad[:, 0]).item() > 0


def test_real_spine_supervision_remains_active():
    logits = torch.zeros((1, 2, 4, 4), requires_grad=True)
    truth = torch.zeros_like(logits); truth[:, :, 1:3, 1:3] = 1
    loss, _ = worker.masked_segmentation_loss(logits, truth, torch.ones_like(logits))
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, 1]).item() > 0


def test_empty_supervision_is_rejected_before_weight_decay():
    value = torch.zeros((1, 2, 2, 2))
    with raises(ValueError, match='no supervised'):
        worker.masked_segmentation_loss(value, value, value)


def test_runtime_metadata_accepts_opencv_distribution_variants():
    versions = worker.runtime_versions()
    assert versions['opencv'] == worker.cv2.__version__
    assert versions['opencv-python'] or versions['opencv-python-headless']


class ToyGong:
    epoch = 1
    def __getitem__(self, index):
        rng = np.random.default_rng(100*self.epoch+index)
        image = torch.tensor(rng.normal(size=(3, 4, 4)), dtype=torch.float32)
        fg = (image[0] > 0).float(); spine = (image[1] > .7).float()*fg
        return image, torch.stack([fg, spine]), torch.tensor(1.)


class ToyChase(ToyGong):
    def __init__(self, count=7):
        self.records = [dict(id=f'physical_{i:03d}') for i in range(count)]
        self.seen = []
    def set_epoch(self, epoch): self.epoch = epoch
    def sample(self, index, epoch=None, record_index=None):
        self.seen.append(self.records[record_index]['id'])
        return self[index+1000*record_index]
    def __getitem__(self, index):
        image, target, _ = super().__getitem__(index)
        valid = torch.zeros_like(target); valid[0] = 1; valid[0, 0] = 0
        target[1] = 0
        return image, target, valid


def settings(root, domain='mixed'):
    return dict(worker.DEFAULTS, training_domain=domain, epoch_sampling='bounded', output_root=str(root), epochs=3, batch=2,
        steps_per_epoch=3, accumulate=2, real_finish_epochs=1, chase_ratio=.5,
        validate_every=1, seed=17, lr=.002, amp=False) if domain == 'mixed' else dict(
        worker.DEFAULTS, training_domain='chase_only', epoch_sampling='bounded', output_root=str(root), epochs=3,
        batch=2, steps_per_epoch=3, accumulate=2, validate_every=1, seed=17, lr=.002, amp=False)


def model():
    return nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU(), nn.Dropout2d(.2), nn.Conv2d(4, 2, 1))


def evaluator(model, cfg, label):
    # Include an RNG side effect to ensure validation state is checkpointed.
    torch.rand(1)
    return dict(pq=.2+int(label[-3:])*.01, fp=1, fn=2, tp_pairs=3)


def run(root, cap, *, fail_eval=None, fail_batch=False, domain='mixed'):
    root.mkdir(exist_ok=True); cfg = settings(root, domain)
    data = worker.ReplayDataset(ToyGong() if domain == 'mixed' else None, ToyChase(), cfg)
    if fail_batch:
        old = data.gong.__class__.__getitem__
        class BrokenGong(ToyGong):
            def __getitem__(self, index):
                if self.epoch == 2: raise RuntimeError('interrupted batch')
                return old(self, index)
        data.gong = BrokenGong()
    worker.seed_everything(8); net = model()
    def callback(*args):
        if args[-1] == fail_eval: raise RuntimeError('interrupted validation')
        return evaluator(*args)
    result = worker.run_training(net, data, cfg, 'fixture-v1', cap, callback,
                                 device=torch.device('cpu'))
    return net, result


def test_exact_replay_budget_and_final_real_only(tmp_path):
    cfg = settings(tmp_path)
    data = worker.ReplayDataset(ToyGong(), ToyChase(), cfg)
    assert len(data.chase_indices) == 3
    chosen = data.chase_indices.copy(); data.set_epoch(1)
    assert data.chase_indices == chosen
    for index in range(len(data)):
        _, _, valid = data[index]
        assert bool(valid[1].any()) == (index not in chosen)
    data.set_epoch(3); assert data.chase_indices == set()


def test_all695_observations_are_sampled_by_epoch9(tmp_path):
    cfg = dict(worker.DEFAULTS, training_domain='mixed', chase_ratio=.2,
               epoch_sampling='bounded', steps_per_epoch=400, epochs=30,
               real_finish_epochs=6, output_root=str(tmp_path))
    chase = ToyChase(count=695)
    data = worker.ReplayDataset(ToyGong(), chase, cfg)
    seen = set()
    for epoch in range(1, 10):
        data.set_epoch(epoch)
        assert len(data.chase_indices) == len(data.chase_observation_ids) == 80
        assert len(set(data.chase_observation_ids)) == 80
        seen.update(data.chase_observation_ids)
        # IDs written to history must correspond to the actual selected record.
        for index, name in zip(sorted(data.chase_indices), data.chase_observation_ids):
            data[index]
            assert chase.seen[-1] == name
        if epoch == 8: assert len(seen) == 640
    assert len(seen) == 695
    repeated = worker.ReplayDataset(ToyGong(), ToyChase(count=695), cfg)
    repeated.set_epoch(9)
    assert repeated.chase_observation_ids == data.chase_observation_ids
    data.set_epoch(25)
    assert data.chase_observation_ids == []


def test_chase_only_covers695_by_epoch2_and_never_samples_gong(tmp_path):
    cfg = dict(worker.DEFAULTS, epoch_sampling='bounded', steps_per_epoch=400, output_root=str(tmp_path))
    chase = ToyChase(count=695)
    data = worker.ReplayDataset(None, chase, cfg)
    seen = set()
    for epoch in (1, 2):
        data.set_epoch(epoch)
        assert len(data.chase_indices) == len(data.chase_observation_ids) == 400
        for index in range(len(data)):
            _, _, valid = data[index]
            assert not valid[1].any()
        seen.update(data.chase_observation_ids)
    assert len(seen) == 695
    data.set_epoch(30)
    assert len(data.chase_indices) == 400
    assert data.gong is None


def test_chase_only_preparation_never_constructs_or_reads_gong_train(tmp_path):
    import solar_a100_train_standalone as legacy
    import chase_transfer_data
    import chase_best84_evaluation
    cfg = dict(worker.DEFAULTS, output_root=str(tmp_path), cache_root=str(tmp_path/'cache'),
        steps_per_epoch=7,
        images='/unavailable_gong_train/images', annotations='annotations.json',
        train_split='train.csv', val_split='val.csv', initial_checkpoint='best84.pt',
        chase_root=str(tmp_path/'chase'), chase_manifest='manifest.json', yolo_cache='yolo.json.gz')
    expected = {cfg[key]: value for key, value in worker.INPUT_SHA.items()}
    expected.update({'best84.pt': worker.INITIAL_SHA, 'manifest.json': 'manifest',
        'yolo.json.gz': 'yolo', '/unavailable_gong_train/images/val.jpeg': 'val-jpeg'})
    touched = []
    def fake_digest(path):
        touched.append(str(path))
        return expected[str(path)]
    records = [dict(id=f'chase_{i}') for i in range(7)]
    class PreparedChase(ToyChase):
        def __init__(self, records, **kwargs): self.records = records; self.seen = []
        def preflight(self): return dict(observations=len(self.records))
    cohort = dict(names=['val.jpeg'], groups={'ann': dict(masks=[1])})
    forbidden = AssertionError('GONG training was accessed')
    with patch.object(worker, 'digest', fake_digest), \
         patch.object(legacy, 'prepare_split', side_effect=forbidden), \
         patch.object(legacy, 'ImageStore', side_effect=forbidden), \
         patch.object(legacy, 'MixedCropDataset', side_effect=forbidden), \
         patch.object(chase_best84_evaluation, 'load_cohort', return_value=cohort), \
         patch.object(chase_transfer_data, 'prepare_chase', return_value=(records, 'source_report.json')), \
         patch.object(chase_transfer_data, 'ChaseDataset', PreparedChase):
        dataset, metadata = worker.prepare_inputs(cfg)
    assert dataset.gong is None
    assert metadata['gong_train_photos_used'] == metadata['train_photos'] == 0
    assert metadata['gong_image_hash_scope'] == 'validation_only'
    assert metadata['chase_train_photos'] == 7
    assert metadata['epoch_sampling'] == 'full_pass'
    assert metadata['effective_steps_per_epoch'] == len(dataset) == 7
    assert [name for name in touched if name.startswith(cfg['images'])] == [cfg['images']+'/val.jpeg']


def test_chase_only_resume_preserves_zero_gong_schedule(tmp_path):
    torch.set_num_threads(1)
    full, _ = run(tmp_path/'full', 3, domain='chase_only')
    run(tmp_path/'resume', 1, domain='chase_only')
    resumed, result = run(tmp_path/'resume', 3, domain='chase_only')
    assert_weights_equal(full, resumed)
    assert result['training_domain'] == 'chase_only'
    assert result['gong_training_crops'] == 0
    saved = torch.load(tmp_path/'resume'/'last.pt', weights_only=True)
    assert all(row['gong_crops'] == 0 and row['chase_crops'] == 6 for row in saved['history'])
    assert all(row['spine_loss'] == 0 for row in saved['history'])


def test_mixed_checkpoint_cannot_resume_as_chase_only(tmp_path):
    run(tmp_path/'run', 1, domain='mixed')
    path = tmp_path/'run'/'last.pt'; before = path.read_bytes()
    with raises(ValueError, match='different training domain'):
        run(tmp_path/'run', 3, domain='chase_only')
    assert path.read_bytes() == before


def test_full_pass_visits_every_photo_once_each_epoch(tmp_path):
    cfg = dict(worker.DEFAULTS, output_root=str(tmp_path))
    chase = ToyChase(count=695)
    data = worker.ReplayDataset(None, chase, cfg)
    expected = {row['id'] for row in chase.records}
    previous = None
    for epoch in (1, 2, 30, 60):
        data.set_epoch(epoch)
        names = list(data.chase_observation_ids)
        assert len(data) == len(names) == len(set(names)) == 695
        assert set(names) == expected and len(data.chase_indices) == 695
        if previous is not None: assert names != previous
        previous = names
        fresh = worker.ReplayDataset(None, ToyChase(695), cfg)
        fresh.set_epoch(epoch)
        assert fresh.chase_observation_ids == names
        for index in (0, 400, 694):
            data[index]
            assert chase.seen[-1] == names[index]


def test_full_pass_rejects_partial_budget(tmp_path):
    cfg = dict(worker.DEFAULTS, output_root=str(tmp_path), steps_per_epoch=400)
    with raises(ValueError, match='must equal'):
        worker.ReplayDataset(None, ToyChase(695), cfg)


def test_full_pass_resume_keeps_last_partial_accumulation_group(tmp_path):
    torch.set_num_threads(1)
    def execute(root, cap):
        root.mkdir(exist_ok=True)
        cfg = dict(worker.DEFAULTS, output_root=str(root), epochs=3, steps_per_epoch=7,
            batch=1, accumulate=4, validate_every=1, amp=False)
        data = worker.ReplayDataset(None, ToyChase(7), cfg)
        worker.seed_everything(8); net = model()
        result = worker.run_training(net, data, cfg, 'full-pass-fixture', cap,
            evaluator, device=torch.device('cpu'))
        return net, result
    continuous, _ = execute(tmp_path/'continuous', 3)
    execute(tmp_path/'resumed', 1)
    resumed, result = execute(tmp_path/'resumed', 3)
    assert_weights_equal(continuous, resumed)
    assert result['epoch_sampling'] == 'full_pass'
    assert result['effective_steps_per_epoch'] == 7 and result['gong_training_crops'] == 0
    saved = torch.load(tmp_path/'resumed'/'last.pt', weights_only=True)
    assert all(row['chase_crops'] == 7 and row['optimizer_attempts'] == 2 for row in saved['history'])
    assert all(len(set(row['chase_observation_ids'])) == 7 for row in saved['history'])


def test_bounded_checkpoint_cannot_resume_as_full_pass(tmp_path):
    run(tmp_path/'run', 1, domain='chase_only')
    path = tmp_path/'run'/'last.pt'; before = path.read_bytes()
    cfg = dict(settings(tmp_path/'run', 'chase_only'), epoch_sampling='full_pass', batch=1, steps_per_epoch=7)
    data = worker.ReplayDataset(None, ToyChase(7), cfg)
    with raises(ValueError, match='different epoch sampling'):
        worker.run_training(model(), data, cfg, 'fixture-v1', 3, evaluator, device=torch.device('cpu'))
    assert path.read_bytes() == before


def assert_weights_equal(first, second):
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key]), key


def test_completed_epoch_resume_matches_uninterrupted(tmp_path):
    torch.set_num_threads(1)
    full, full_result = run(tmp_path/'full', 3)
    _, paused = run(tmp_path/'resume', 1)
    assert paused['status'] == 'paused' and paused['fixed_final_epoch'] == 3
    resumed, resumed_result = run(tmp_path/'resume', 3)
    assert_weights_equal(full, resumed)
    assert resumed_result['best_epoch'] == full_result['best_epoch'] == 3
    saved = torch.load(tmp_path/'resume'/'last.pt', weights_only=True)
    assert saved['epoch'] == 3 and len(saved['history']) == 3
    assert saved['pending_eval'] is None
    assert [len(row['chase_observation_ids']) for row in saved['history']] == [3, 3, 0]
    assert saved['history'][-1]['chase_unique_observations_seen'] == 6


def test_pending_validation_repairs_without_retraining_completed_epoch(tmp_path):
    torch.set_num_threads(1)
    full, _ = run(tmp_path/'full', 3)
    with raises(RuntimeError, match='interrupted validation'):
        run(tmp_path/'resume', 3, fail_eval='epoch001')
    pending = torch.load(tmp_path/'resume'/'last.pt', weights_only=True)
    assert pending['epoch'] == pending['pending_eval'] == 1
    assert len(pending['history']) == 1
    resumed, _ = run(tmp_path/'resume', 3)
    assert_weights_equal(full, resumed)


def test_interrupted_training_replays_only_incomplete_epoch(tmp_path):
    torch.set_num_threads(1)
    full, _ = run(tmp_path/'full', 3)
    with raises(RuntimeError, match='interrupted batch'):
        run(tmp_path/'resume', 3, fail_batch=True)
    saved = torch.load(tmp_path/'resume'/'last.pt', weights_only=True)
    assert saved['epoch'] == 1 and len(saved['history']) == 1
    resumed, _ = run(tmp_path/'resume', 3)
    assert_weights_equal(full, resumed)


def test_bad_identity_and_backwards_stage_preserve_checkpoint(tmp_path):
    run(tmp_path/'run', 2)
    path = tmp_path/'run'/'last.pt'; original = path.read_bytes()
    cfg = settings(tmp_path/'run'); net = model()
    data = worker.ReplayDataset(ToyGong(), ToyChase(), cfg)
    with raises(ValueError, match='different experiment identity'):
        worker.run_training(net, data, cfg, 'wrong', 3, evaluator, device=torch.device('cpu'))
    with raises(ValueError, match='Completed epoch/history'):
        run(tmp_path/'run', 1)
    assert path.read_bytes() == original


def test_atomic_save_failure_keeps_previous_file(tmp_path):
    path = tmp_path/'last.pt'; path.write_bytes(b'previous-complete')
    def broken(value, path):
        Path(path).write_bytes(b'incomplete'); raise OSError('simulated storage failure')
    with patch.object(worker.torch, 'save', broken):
        with raises(OSError): worker.atomic_torch(path, {'epoch': 5})
    assert path.read_bytes() == b'previous-complete'
    assert not path.with_name('last.pt.pending').exists()


class ChaseTrainingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.addCleanup(self.folder.cleanup)

    def test_ignored_gradients(self): test_ignored_pixels_have_no_gradient_or_loss_effect()
    def test_gong_spine_gradients(self): test_real_spine_supervision_remains_active()
    def test_no_supervision(self): test_empty_supervision_is_rejected_before_weight_decay()
    def test_runtime_metadata(self): test_runtime_metadata_accepts_opencv_distribution_variants()
    def test_schedule(self): test_exact_replay_budget_and_final_real_only(self.root)
    def test_full_chase_coverage(self): test_all695_observations_are_sampled_by_epoch9(self.root)
    def test_chase_only_coverage(self): test_chase_only_covers695_by_epoch2_and_never_samples_gong(self.root)
    def test_chase_only_no_gong_io(self): test_chase_only_preparation_never_constructs_or_reads_gong_train(self.root)
    def test_chase_only_resume(self): test_chase_only_resume_preserves_zero_gong_schedule(self.root)
    def test_domain_resume_guard(self): test_mixed_checkpoint_cannot_resume_as_chase_only(self.root)
    def test_full_pass_coverage(self): test_full_pass_visits_every_photo_once_each_epoch(self.root)
    def test_full_pass_budget(self): test_full_pass_rejects_partial_budget(self.root)
    def test_full_pass_resume(self): test_full_pass_resume_keeps_last_partial_accumulation_group(self.root)
    def test_full_pass_resume_guard(self): test_bounded_checkpoint_cannot_resume_as_full_pass(self.root)
    def test_resume(self): test_completed_epoch_resume_matches_uninterrupted(self.root)
    def test_pending_validation(self): test_pending_validation_repairs_without_retraining_completed_epoch(self.root)
    def test_batch_interrupt(self): test_interrupted_training_replays_only_incomplete_epoch(self.root)
    def test_wrong_identity(self): test_bad_identity_and_backwards_stage_preserve_checkpoint(self.root)
    def test_atomic_save(self): test_atomic_save_failure_keeps_previous_file(self.root)


if __name__ == '__main__': unittest.main()
