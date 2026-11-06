import io
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import download_external_magfilo_candidates as acquisition


def image_bytes(value):
    return cv2.imencode('.jpg', np.full((2048, 2048), value, np.uint8))[1].tobytes()


class AcquisitionTests(unittest.TestCase):
    def test_excludes_every_official_id_and_rejects_duplicates(self):
        keys = ['20110101000000Bh', '20110102000000Bh', '20110103000000Bh']
        data = dict(images=[dict(file_name=key+'.jpeg', url=f'https://gong2.nso.edu/HA/hag/{key[:6]}/{key[:8]}/{key}.jpg', width=2048, height=2048, id=key) for key in keys])
        manifest = dict(train_images=[keys[0]+'.jpeg'], test_images=[keys[1]+'.jpeg'])
        records, counts = acquisition.candidate_records(data, manifest)
        self.assertEqual([row['physical_identifier'] for row in records], [keys[2].lower()])
        manifest['test_images'].append(keys[1]+'.jpg')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            acquisition.candidate_records(data, manifest)

    def test_rejects_wrong_source_and_invalid_image_without_writing(self):
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            acquisition.checked_url('https://example.org/20110103000000Bh.jpg', '20110103000000bh')
        row = dict(physical_identifier='20110103000000bh', file_name='20110103000000Bh.jpeg', source_url='https://gong2.nso.edu/HA/hag/201101/20110103/20110103000000Bh.jpg')
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'decodable'):
                acquisition.acquire_one(row, folder, fetch=lambda row: b'<html>error</html>', retries=0)
            self.assertFalse((Path(folder)/row['file_name']).exists())

    def test_cache_tampering_and_different_existing_image_are_preserved(self):
        row = dict(physical_identifier='20110103000000bh', file_name='20110103000000Bh.jpeg', source_url='https://gong2.nso.edu/HA/hag/201101/20110103/20110103000000Bh.jpg')
        original, changed = image_bytes(12), image_bytes(34)
        with tempfile.TemporaryDirectory() as folder:
            prior = acquisition.acquire_one(row, folder, fetch=lambda row: original, retries=0)
            cached = acquisition.acquire_one(row, folder, previous=prior, fetch=lambda row: self.fail('Network used for verified cache'))
            self.assertEqual(cached['acquisition'], 'verified_cache')
            path = Path(folder)/row['file_name']
            path.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, 'Existing image changed'):
                acquisition.acquire_one(row, folder, previous=prior)
            self.assertEqual(path.read_bytes(), changed)
            with self.assertRaisesRegex(ValueError, 'differs'):
                acquisition.acquire_one(row, folder, fetch=lambda row: original, retries=0)
            self.assertEqual(path.read_bytes(), changed)


if __name__ == '__main__':
    unittest.main()
