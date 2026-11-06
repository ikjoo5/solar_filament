"""Bounded tests of exclusion, resumability and reserved split assignment."""
import copy
from datetime import date, timedelta
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import colab_prepare_external71 as cell
from external_evaluation_data import assign_external_splits


def image_bytes(value=0):
    return cv2.imencode('.png', np.full((2048, 2048), value, np.uint8))[1].tobytes()


class Response(io.BytesIO):
    def geturl(self):
        return 'https://gong2.nso.edu/HA/hag/202101/20210101/20210101000000Bh.jpg'


class External71CellTests(unittest.TestCase):
    def test_archive_resume_checks_bytes_and_preserves_different_local_file(self):
        raw = image_bytes()
        url = Response().geturl()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'20210101000000Bh.jpeg'
            info = cell.archive_image(path, url, opener=lambda *a, **k: Response(raw))
            same = cell.archive_image(path, url, opener=lambda *a, **k: self.fail('Unexpected redownload'))
            self.assertEqual(info, same)
            path.write_bytes(image_bytes(1))
            with self.assertRaisesRegex(ValueError, 'Cached image/source changed'):
                cell.archive_image(path, url)
            self.assertEqual(path.read_bytes(), image_bytes(1))
            path.with_name(path.name+'.source.json').unlink()
            with self.assertRaisesRegex(FileExistsError, 'preserved'):
                cell.archive_image(path, url, opener=lambda *a, **k: Response(raw))
            self.assertEqual(path.read_bytes(), image_bytes(1))

    def test_unknown_host_or_invalid_download_never_creates_image(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'20210101000000Bh.jpeg'
            with self.assertRaisesRegex(ValueError, 'Unexpected archive URL'):
                cell.archive_image(path, 'https://example.org/HA/hag/image.jpg',
                    opener=lambda *a, **k: self.fail('Untrusted host fetched'))
            with self.assertRaisesRegex(ValueError, '2048'):
                cell.archive_image(path, Response().geturl(), opener=lambda *a, **k: Response(b'not an image'))
            with self.assertRaisesRegex(ValueError, 'empty bytes'):
                cell.archive_image(path, Response().geturl(), opener=lambda *a, **k: Response(b''))
            self.assertFalse(path.exists())
        for url in ('https://gong2.nso.edu:8443/HA/hag/202101/20210101/20210101000000Bh.jpg',
                    'https://gong2.nso.edu/HA/hag/arbitrary.jpg', 'https://example.org/image.jpg'):
            with self.assertRaisesRegex(ValueError, 'Unexpected archive URL'):
                cell.ArchiveRedirect().redirect_request(None, None, 302, '', {}, url)

    def test_official_inventory_rejects_omission_or_duplicate_observation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'MAGFiLO_1.0_Kaggle_2026'
            for split, identifier in [('train', '20210101000000Bh'), ('test', '20210102000000Bh')]:
                directory = root/split/f'{split}_images'
                directory.mkdir(parents=True)
                (directory/(identifier+'.jpeg')).write_bytes(image_bytes())
            expected = {split: (1, cell.sha(identifier.lower().encode()))
                        for split, identifier in [('train', '20210101000000Bh'), ('test', '20210102000000Bh')]}
            inventory = cell.official_inventory(folder, expected)
            self.assertEqual(len(inventory['test']), 1)
            duplicate = root/'train/train_images/copy_20210101000000Bh.png'
            duplicate.write_bytes(b'irrelevant because metadata guard runs first')
            with self.assertRaisesRegex(ValueError, 'differs'):
                cell.official_inventory(folder, expected)

    def test_duplicate_exclusion_and_split_are_metadata_only_and_match_existing_policy(self):
        identifiers = ['20210201000000bh', '20210301000000bh', '20210401000000bh',
                       '20210501000000bh', '20210601000000bh', '20210701000000bh']
        groups = {k: [dict(id='rater1-'+k, file_name=k+'.jpeg')] for k in identifiers}
        acquired = {k: dict(file_sha256='file'+str(i), pixel_sha256='pixel'+str(i), bytes=10)
                    for i, k in enumerate(identifiers)}
        inventory = dict(train=[dict(physical_identifier='20210101000000bh',
                                     file_sha256='file0', pixel_sha256='pixel1')], test=[])
        # Two external observations decoding to the same pixels are both withheld.
        acquired[identifiers[3]]['pixel_sha256'] = acquired[identifiers[2]]['pixel_sha256']
        records, exclusions = cell.select_records(groups, acquired, inventory, Path('/tmp/external'))
        self.assertEqual([r['evaluation_key'] for r in records], identifiers[4:])
        self.assertEqual(len(exclusions), 4)
        self.assertEqual(records, assign_external_splits(copy.deepcopy(records)))
        self.assertTrue(all(r['time_clean_3days'] for r in records))
        # The standalone notebook is identical, so it can recover after /content expires.
        root = Path(__file__).resolve().parents[1]
        self.assertEqual((root/'tools/colab_prepare_external71.py').read_bytes(),
                         (root/'notebooks/External71_Prepare_CopyCell.py').read_bytes())

    def test_export_contains_only_external_labels_and_zip_has_metadata_only(self):
        names = [(date(2021, 1, 1)+timedelta(days=i)).strftime('%Y%m%d')+'000000Bh' for i in range(71)]
        official_names = ['20191201000000Bh', '20191202000000Bh']
        rows = [dict(id='rater-'+name, file_name=name+'.jpeg', width=2048, height=2048,
                     url=f'https://gong2.nso.edu/HA/hag/{name[:6]}/{name[:8]}/{name}.jpg')
                for name in official_names+names]
        data = dict(images=rows, annotations=[dict(image_id=r['id'], segmentation='inert label') for r in rows])
        raw = json.dumps(data).encode()
        inventory = {role: [dict(physical_identifier=name.lower(), file_sha256='official'+role,
                                  pixel_sha256='official'+role)] for role, name in zip(('train', 'test'), official_names)}
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(cell, 'PUBLIC_BYTES', len(raw)), patch.object(cell, 'PUBLIC_MD5', hashlib.md5(raw).hexdigest()), \
                 patch.object(cell, 'fetch', return_value=raw), patch.object(cell, 'official_inventory', return_value=inventory), \
                 patch.object(cell, 'archive_image', side_effect=lambda path, url: dict(file_sha256=path.stem,
                      pixel_sha256=path.stem, bytes=10)), patch('builtins.print'):
                archive = cell.run(folder)
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.namelist(), ['external71_plan.json'])
                plan = json.loads(z.read(z.namelist()[0]))
            self.assertEqual(plan['summary']['eligible_physical_images'], 71)
            self.assertFalse(plan['scores_computed'])
            self.assertFalse(plan['allow_training'])
            self.assertNotIn('segmentation', json.dumps(plan))
            selected = json.loads(Path(plan['annotations']).read_text())
            expected_ids = {'rater-'+name for name in names}
            self.assertEqual({r['id'] for r in selected['images']}, expected_ids)
            self.assertEqual({r['image_id'] for r in selected['annotations']}, expected_ids)


if __name__ == '__main__':
    unittest.main()
