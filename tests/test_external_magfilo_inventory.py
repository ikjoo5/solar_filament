"""A public dataset must not become external validation merely by being public."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import colab_inventory_external_magfilo as inventory


class InventoryTests(unittest.TestCase):
    def test_all_official_ids_including_blank_test_are_excluded(self):
        data = dict(images=[dict(file_name='20110101000000Bh.jpeg'),
                            dict(file_name='20110102000000Bh.jpeg'),
                            dict(file_name='20110102000000Bh.jpeg')], annotations=[])
        report = inventory.inventory_annotations(data,
            dict(physical_identifiers=['20110101000000bh']),
            dict(physical_identifiers=['20110102000000bh']))
        self.assertEqual(report['actual_eligible_external_N'], 0)
        self.assertEqual(report['status'], 'no_additional_external_observations')
        self.assertFalse(report['scores_computed'])
        self.assertEqual(report['public_annotation_groups'], 3)

    def test_unmatched_id_is_only_potential_until_pixel_audit(self):
        data = dict(images=[dict(file_name='20110103000000Bh.jpeg')], annotations=[])
        report = inventory.inventory_annotations(data,
            dict(physical_identifiers=['20110101000000bh']),
            dict(physical_identifiers=['20110102000000bh']))
        self.assertEqual(report['potential_external_before_byte_pixel_audit'], 1)
        self.assertIsNone(report['actual_eligible_external_N'])
        self.assertFalse(report['label_masks_used_for_selection'])

    def test_public_json_checksum_and_existing_file_protection(self):
        raw = json.dumps(dict(images=[], annotations=[])).encode()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'public.json'
            data, provenance = inventory.load_verified_annotations(path,
                opener=lambda *a,**k: io.BytesIO(raw), expected_md5=hashlib.md5(raw).hexdigest(), expected_bytes=len(raw))
            self.assertTrue(provenance['downloaded'])
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'checksum'):
                inventory.load_verified_annotations(path, expected_md5='wrong', expected_bytes=len(raw))
            self.assertEqual(path.read_bytes(), before)

    def test_official_directories_must_be_nonempty_and_disjoint(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Complete official'):
                inventory.directory_inventory(folder)
        data = dict(images=[], annotations=[])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            inventory.inventory_annotations(data, dict(physical_identifiers=['id']), dict(physical_identifiers=['id']))


if __name__ == '__main__':
    unittest.main()
