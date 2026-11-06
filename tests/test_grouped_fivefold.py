from datetime import date, timedelta
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_grouped_fivefold import build_plan


def fixture():
    images, annotations = [], []
    for offset in (1, 2, 26, 27, 28, 54, 55, 81, 108, 135, 162, 189):
        day = date(2000, 1, 1)+timedelta(days=offset)
        name = day.strftime('%Y%m%d')+'000000Bh.jpeg'
        for rater in (1, 2):
            identifier = f'{rater}-{name[:-5]}'
            images.append(dict(id=identifier, file_name=name))
            annotations.append(dict(image_id=identifier))
    return dict(images=images, annotations=annotations)


class FivefoldTests(unittest.TestCase):
    def test_each_observation_held_out_once_and_all_raters_together(self):
        plan = build_plan(fixture())
        self.assertEqual(len(plan['assignments']), 12)
        self.assertEqual(sum(f['validation_physical'] for f in plan['fold_summaries']), 12)
        self.assertEqual(sum(f['validation_annotation_groups'] for f in plan['fold_summaries']), 24)
        for fold in plan['fold_summaries']:
            self.assertEqual(fold['train_annotation_groups'], 2*fold['train_physical'])
        blocks = {}
        for row in plan['observations']:
            blocks.setdefault(row['block'], set()).add(plan['assignments'][row['physical_identifier']])
        self.assertTrue(all(len(folds) == 1 for folds in blocks.values()))

    def test_embargo_excludes_near_dates_even_across_group_boundaries(self):
        plan = build_plan(fixture())
        self.assertGreater(sum(f['embargo_physical'] for f in plan['fold_summaries']), 0)
        for fold in plan['fold_summaries']:
            self.assertGreater(fold['minimum_train_validation_day_distance'], 3)
            self.assertEqual(fold['train_physical']+fold['validation_physical']+fold['embargo_physical'], 12)

    def test_old_synthetic_pool_and_solar_checkpoints_are_not_reused(self):
        data = fixture()
        plan = build_plan(data, dict(train_image_ids=[r['id'] for r in data['images']]),
                          dict(images=[dict(id=1, file_name='synthetic_0.png')], annotations=[]))
        for fold in plan['fold_summaries']:
            self.assertFalse(fold['synthetic_enabled'])
            self.assertGreater(fold['old_synthetic_possible_donors_in_validation'], 0)
        self.assertFalse(plan['initialization_policy']['allow_old_best84'])
        self.assertFalse(plan['initialization_policy']['allow_old_yolo_checkpoint'])
        self.assertFalse(plan['training_performed'])

    def test_assignment_is_reproducible_under_annotation_reordering(self):
        data = fixture()
        other = dict(images=list(reversed(data['images'])), annotations=list(reversed(data['annotations'])))
        self.assertEqual(build_plan(data)['assignments'], build_plan(other)['assignments'])


if __name__ == '__main__':
    unittest.main()
