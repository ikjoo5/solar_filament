"""Official-file identities, including blank predictions, must be excluded."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from audit_official_image_manifest import audit_manifest, manifest_inventory


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.names = ['20200101000000Bh.jpeg', '20200102000000Bh.jpeg', '20200103000000Bh.jpeg']
        self.manifest = dict(train_images=self.names[:1], test_images=self.names[1:2])
        self.official = dict(images=[dict(file_name=self.names[0])])
        # Labels intentionally cannot be iterated: identity auditing must not need them.
        self.public = dict(images=[dict(file_name=name, id=str(i)) for i, name in enumerate(self.names)],
                           annotations=object())

    def test_unpredicted_test_is_excluded_and_candidates_remain_provisional(self):
        self.public['images'].append(dict(file_name=self.names[2], id='second_annotator'))
        report = audit_manifest(self.manifest, self.public, self.official)
        self.assertEqual(report['public_overlap_official_test'], 1)
        self.assertEqual(report['candidate_external_physical'], 1)
        self.assertEqual(report['candidate_external_annotation_groups'], 2)
        self.assertIsNone(report['actual_eligible_external_N'])
        self.assertFalse(report['scores_computed'])

    def test_duplicate_identity_and_cross_split_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            manifest_inventory(dict(train_images=[self.names[0], self.names[0].lower()],
                                    test_images=self.names[1:2]))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            manifest_inventory(dict(train_images=self.names[:1], test_images=self.names[:1]))

    def test_incomplete_or_misassigned_train_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'differ'):
            audit_manifest(dict(train_images=self.names[2:], test_images=self.names[1:2]),
                           self.public, self.official)
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            manifest_inventory(dict(train_images=[], test_images=self.names[1:2]))

    def test_no_candidate_is_zero_N_but_not_a_measured_score(self):
        self.public['images'].pop()
        report = audit_manifest(self.manifest, self.public, self.official)
        self.assertEqual(report['actual_eligible_external_N'], 0)
        self.assertEqual(report['status'], 'no_additional_external_observations')
        self.assertFalse(report['scores_computed'])


if __name__ == '__main__':
    unittest.main()
