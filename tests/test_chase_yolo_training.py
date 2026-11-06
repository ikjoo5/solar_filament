import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import chase_yolo_training as mod


class FakeDataset:
    def __init__(self, records):
        self.im_files = [row['image'] for row in records]
        self.labels = ['labels']; self.transforms = []
    def __len__(self): return len(self.im_files)
    def __getitem__(self, index): return index


def records(n=616, m=695):
    # One GONG image deliberately has two independent annotator variants.
    result = [dict(domain='gong', physical=f'g{i}', group=f'g{i}', image=f'/tmp/g{i}.png') for i in range(n)]
    result.append(dict(domain='gong', physical='g0', group='g0_rater2', image='/tmp/g0_rater2.png'))
    return result+[dict(domain='chase', physical=f'c{i}', group=f'c{i}', image=f'/tmp/c{i}.png') for i in range(m)]


class ChaseYoloTrainingTests(unittest.TestCase):
    def test_full_pass_visits_each695_once_and_has_reproducible_epoch_shuffle(self):
        source = [r for r in records() if r['domain'] == 'chase']
        cfg = mod.resolve_epoch_sampling(mod.DEFAULTS, dict(records=source, chase_source_count=695))
        a = mod.BalancedEpochDataset(FakeDataset(source), source, cfg)
        b = mod.BalancedEpochDataset(FakeDataset(source), source, cfg)
        previous = None
        for epoch in (0,1,29,59):
            a.set_epoch(epoch); b.set_epoch(epoch)
            self.assertEqual(len(a),695)
            self.assertEqual(a.domain_counts, {'gong':0,'chase':695})
            self.assertEqual(len(set(a.plan)),695)
            self.assertEqual(set(a.plan),set(range(695)))
            self.assertEqual(a.plan,b.plan)
            if previous is not None: self.assertNotEqual(a.plan,previous)
            previous = list(a.plan)

    def test_full_pass_derives_shorter_epoch_only_from_explicit_exclusions(self):
        source = [r for r in records() if r['domain'] == 'chase'][:-2]
        metadata = dict(records=source, chase_source_count=695, explicit_excluded_ids=['c693','c694'])
        cfg = mod.resolve_epoch_sampling(mod.DEFAULTS, metadata)
        self.assertEqual(cfg['requested_steps_per_epoch'],695)
        self.assertEqual(cfg['effective_steps_per_epoch'],693)
        self.assertEqual(cfg['accepted_chase_observations'],693)
        ds = mod.BalancedEpochDataset(FakeDataset(source),source,cfg)
        self.assertEqual(len(ds),693)
        self.assertEqual(len(set(ds.plan)),693)
        with self.assertRaisesRegex(ValueError,'without an explicit exclusion'):
            mod.resolve_epoch_sampling(mod.DEFAULTS, dict(metadata,explicit_excluded_ids=[]))

    def test_full_pass_rejects_old400_budget_and_duplicate_physical_records(self):
        source = [r for r in records() if r['domain'] == 'chase']
        with self.assertRaisesRegex(ValueError,'steps_per_epoch must equal'):
            mod.resolve_epoch_sampling(dict(mod.DEFAULTS,steps_per_epoch=400),dict(records=source))
        duplicate = dict(source[0],image='/tmp/duplicate_alias.png')
        with self.assertRaisesRegex(ValueError,'exactly one record'):
            mod.resolve_epoch_sampling(mod.DEFAULTS,dict(records=source+[duplicate]))

    def test_chase_only_covers_all695_by_epoch2_and_never_switches_to_gong(self):
        source = [r for r in records() if r['domain'] == 'chase']
        cfg = dict(mod.DEFAULTS, training_domain='chase_only', epoch_sampling='bounded',
            steps_per_epoch=400, chase_ratio=1., real_finish_epochs=0)
        ds = mod.BalancedEpochDataset(FakeDataset(source), source, cfg)
        seen = set()
        for epoch in range(30):
            ds.set_epoch(epoch)
            self.assertEqual(ds.domain_counts, {'gong':0, 'chase':400})
            self.assertEqual(len(ds.plan), 400)
            self.assertEqual(len(set(ds.plan)), 400)
            seen.update(name for domain,name in ds.sample_names)
            if epoch == 1: self.assertEqual(len(seen), 695)
            self.assertTrue(all(domain == 'chase' for domain,_ in ds.sample_names))

    def test_chase_only_rejects_accidental_gong_records(self):
        source = records(1,2)
        cfg = dict(mod.DEFAULTS, training_domain='chase_only', chase_ratio=1., real_finish_epochs=0)
        with self.assertRaisesRegex(ValueError, 'cannot contain GONG'):
            mod.BalancedEpochDataset(FakeDataset(source), source, cfg)

    def test_chase_only_preparation_never_opens_gong_images_or_annotations(self):
        import chase_transfer_data
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            split = root/'split.csv'; split.write_text('frozen split fixture')
            cfg = dict(mod.DEFAULTS, training_domain='chase_only', chase_ratio=1., real_finish_epochs=0,
                images=str(root/'MUST_NOT_READ_GONG_IMAGES'), annotations=str(root/'MUST_NOT_READ_GONG_LABELS'),
                train_split=str(split), val_split=str(split), chase_root=str(root/'chase'),
                chase_manifest=str(root/'chase_manifest.json'), cache_root=str(root/'cache'))
            source = [dict(id='source_chase_example')]
            pixels = np.zeros((16,16), np.uint8); pixels[4:9,4:9] = 1
            class FakeChase:
                def __init__(self, *args, **kwargs): pass
                def aligned(self, record): return pixels*100, pixels, np.ones_like(pixels)
            with patch.object(chase_transfer_data, 'prepare_chase', return_value=(source, root/'source_report.json')), \
                 patch.object(chase_transfer_data, 'ChaseDataset', FakeChase), \
                 patch.object(mod, 'INPUT_SHA', dict(annotations='intentionally_not_read',
                     train_split=mod.digest(split), val_split=mod.digest(split))):
                dataset, report = mod.prepare_dataset(cfg)
            self.assertEqual(report['training_domain'], 'chase_only')
            self.assertEqual(report['gong_training_photos'], 0)
            self.assertEqual(report['source_images_sha256'], {})
            self.assertNotIn('annotations', report['inputs'])
            self.assertTrue(all(r['domain'] == 'chase' for r in report['records']))
            self.assertFalse((dataset.parent/'physical_gong').exists())
            placeholder = Path(json.loads(dataset.read_text())['val']).read_text().strip()
            self.assertEqual(placeholder, report['records'][0]['image'])

    def test_sampler_has_exact_budget_and_covers_all_chase_in_nine_epochs(self):
        source = records(); cfg = dict(mod.DEFAULTS, training_domain='mixed_replay',
            epoch_sampling='bounded', steps_per_epoch=400, epochs=30, chase_ratio=.2, real_finish_epochs=6)
        ds = mod.BalancedEpochDataset(FakeDataset(source), source, cfg)
        seen = set()
        for epoch in range(9):
            ds.set_epoch(epoch)
            self.assertEqual(len(ds), 400)
            self.assertEqual(ds.domain_counts, {'gong':320, 'chase':80})
            self.assertEqual(len({p for d,p in ds.sample_names if d == 'gong'}), 320)
            seen.update(p for d,p in ds.sample_names if d == 'chase')
        self.assertEqual(len(seen), 695)

    def test_resumed_epoch_plan_and_real_finish_are_identical(self):
        source = records(); cfg = dict(mod.DEFAULTS, training_domain='mixed_replay',
            epoch_sampling='bounded', steps_per_epoch=400, epochs=30, chase_ratio=.2, real_finish_epochs=6)
        a = mod.BalancedEpochDataset(FakeDataset(source), source, cfg)
        b = mod.BalancedEpochDataset(FakeDataset(source), source, cfg)
        for epoch in range(30):
            a.set_epoch(epoch); b.set_epoch(epoch)
            self.assertEqual(a.plan, b.plan)
            self.assertEqual(a.domain_counts['chase'], 80 if epoch < 24 else 0)
            self.assertEqual([a[i] for i in range(len(a))], a.plan)
        self.assertEqual(a.labels, ['labels'])

    def test_small_positive_components_are_not_filtered(self):
        mask = np.zeros((32,32), np.uint8)
        mask[0,0] = 1; mask[3,4:9] = 1; mask[30:,30:] = 1
        lines, report = mod.binary_proxy_polygons(mask)
        self.assertEqual(len(lines), 3)
        restored = np.maximum.reduce([mod.polygon_mask(line, mask.shape) for line in lines])
        np.testing.assert_array_equal(restored, mask)
        self.assertEqual(sum(r['lost_pixels'] for r in report), 0)

    def test_holes_and_disconnected_gong_instance_are_reported_not_dropped(self):
        mask = np.zeros((32,32), np.uint8); mask[2:10,2:10] = 1; mask[4:8,4:8] = 0
        mask[20:24,20:25] = 1
        line, report = mod.mask_polygon(mask)
        self.assertEqual(len(line.split())%2, 1)
        self.assertEqual(report['contours'], 2)
        self.assertGreater(report['added_pixels'], 0)
        self.assertEqual(report['lost_pixels'], 0)

    def test_every_augmentation_is_disabled(self):
        opts = mod.training_options(dict(mod.DEFAULTS, cache_root='/tmp/test'), '/tmp/data.yaml')
        for key in ('mosaic','mixup','cutmix','copy_paste','hsv_h','hsv_s','hsv_v','degrees',
                    'translate','scale','shear','perspective','fliplr','flipud','multi_scale'):
            self.assertEqual(opts[key], 0, key)
        self.assertEqual(opts['augmentations'], [])
        self.assertEqual((opts['batch'],opts['nbs'],opts['workers']), (1,1,0))

    def test_validation_sibling_leak_is_rejected(self):
        data = {'images':[{'id':'1','file_name':'same.jpeg'}, {'id':'2','file_name':'same.jpeg'}]}
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder)/'train.csv',Path(folder)/'val.csv']
            for path, sid in zip(paths, ('1','2')):
                with path.open('w') as out:
                    writer = csv.writer(out); writer.writerow(['image_id','file_name']); writer.writerow([sid,'same.jpeg'])
            with self.assertRaisesRegex(ValueError, 'siblings'):
                mod.read_training_groups(data, *paths)

    def test_ultralytics_may_not_silently_drop_invalid_labels(self):
        source = records(2,2)
        cfg = dict(mod.DEFAULTS, training_domain='mixed_replay', epoch_sampling='bounded',
            steps_per_epoch=400, epochs=30, chase_ratio=.2, real_finish_epochs=6)
        with self.assertRaisesRegex(ValueError, 'discarded'):
            mod.BalancedEpochDataset(FakeDataset(source[:-1]), source, cfg)


if __name__ == '__main__': unittest.main()
