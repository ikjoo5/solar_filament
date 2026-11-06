"""Submission contract and interruption/corruption recovery, without neural weights."""
import csv
import gzip
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
from pycocotools import mask as mu

TOOLS = Path(__file__).resolve().parents[1]/'tools'
if str(TOOLS) not in sys.path:
    sys.path.insert(0,str(TOOLS))
import submission_driver as driver


def rectangle(top=30,left=50):
    mask = np.zeros((2048,2048),np.uint8)
    mask[top:top+10,left:left+20] = 1
    return mu.encode(np.asfortranarray(mask))


class Provider:
    def __init__(self,fail=None):
        self.fail = fail
        self.calls = []

    def stage(self,name,stem=''):
        self.calls.append(name)
        if self.fail == name:
            raise RuntimeError('simulated runtime disconnect')
        return [{'image':stem,'view':name+str(i),'max_det_reached':False,'instances':[]}
                for i in range({'geometric':4,'preprocessing':8,'tiled':18}.get(name,1))]

    def geometric(self,stem,pixels):
        return self.stage('geometric',stem)

    def preprocessing(self,stem,pixels):
        return self.stage('preprocessing',stem)

    def tiled(self,stem,pixels):
        return self.stage('tiled',stem)

    def native(self,stem,pixels,**metadata):
        self.stage('native')
        return dict(metadata,unions={},components={},tta={},view_unions={})


class SubmissionDriverTests(unittest.TestCase):
    def frame(self,root):
        source=root/'test_images';source.mkdir()
        path=source/'20110109104734Ch.jpeg'
        self.assertTrue(cv2.imwrite(str(path),np.zeros((2048,2048),np.uint8)))
        output=root/'run';(output/'stages').mkdir(parents=True);(output/'completed').mkdir()
        return path,output

    def test_stage_interruption_resume_and_final_record_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path,output=self.frame(Path(directory));digest=driver.digest(path)
            broken=Provider(fail='tiled')
            with self.assertRaisesRegex(RuntimeError,'disconnect'):
                driver.process_frame(path,digest,'identity',output,broken,lambda stem,stages:[rectangle()])
            self.assertEqual(broken.calls,['geometric','preprocessing','tiled'])
            self.assertFalse(list((output/'completed').glob('*')))
            resumed=Provider()
            record,reused=driver.process_frame(path,digest,'identity',output,resumed,lambda stem,stages:[rectangle()])
            self.assertFalse(reused)
            self.assertEqual(resumed.calls,['tiled','native'])
            self.assertEqual(record['instances'],1)
            self.assertFalse(list((output/'stages').glob('*')))
            final_provider=Provider(fail='geometric')
            same,reused=driver.process_frame(path,digest,'identity',output,final_provider)
            self.assertTrue(reused);self.assertEqual(same,record);self.assertEqual(final_provider.calls,[])

    def test_corrupted_stage_recomputed_and_other_stages_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            path,output=self.frame(Path(directory));digest=driver.digest(path)
            with self.assertRaises(RuntimeError):
                driver.process_frame(path,digest,'id',output,Provider(fail='native'),lambda stem,stages:[])
            (output/'stages'/f'{path.stem}.preprocessing.json.gz').write_bytes(b'corrupt')
            provider=Provider()
            result,reused=driver.process_frame(path,digest,'id',output,provider,lambda stem,stages:[])
            self.assertEqual(provider.calls,['preprocessing','native'])
            self.assertFalse(reused);self.assertEqual(result['rles'],[])
            self.assertEqual(result['instances'],0)

    def test_corrupt_final_recovered_from_verified_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            path,output=self.frame(Path(directory));digest=driver.digest(path)
            expected,_=driver.process_frame(path,digest,'id',output,Provider(),lambda stem,stages:[rectangle()],True)
            (output/'completed'/f'{path.stem}.json.gz').write_bytes(b'corrupt final')
            provider=Provider(fail='geometric')
            actual,reused=driver.process_frame(path,digest,'id',output,provider,lambda stem,stages:[rectangle()],True)
            self.assertEqual(expected,actual);self.assertFalse(reused);self.assertEqual(provider.calls,[])

    def test_envelope_binds_source_and_pipeline_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'stage.json.gz'
            driver.write_envelope(path,{'rle':rectangle()},'pipeline','image')
            self.assertIsNotNone(driver.read_envelope(path,'pipeline','image'))
            self.assertIsNone(driver.read_envelope(path,'new-pipeline','image'))
            self.assertIsNone(driver.read_envelope(path,'pipeline','changed-image'))
            self.assertFalse(path.with_name(path.name+'.pending').exists())

    def test_compressed_body_bitflip_and_json_root_corruption_are_cache_misses(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'stage.json.gz'
            driver.write_envelope(path,{'rle':rectangle()},'pipeline','image')
            damaged=bytearray(path.read_bytes());damaged[10]^=255;path.write_bytes(damaged)
            self.assertIsNone(driver.read_envelope(path,'pipeline','image'))
            for value in (b'[]',b'null',b'"string"'):
                path.write_bytes(gzip.compress(value))
                self.assertIsNone(driver.read_envelope(path,'pipeline','image'))

    def test_structurally_invalid_stage_is_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            path,output=self.frame(Path(directory));digest=driver.digest(path)
            with self.assertRaises(RuntimeError):
                driver.process_frame(path,digest,'id',output,Provider(fail='native'),lambda stem,stages:[])
            driver.write_envelope(output/'stages'/f'{path.stem}.tiled.json.gz',{},'id',digest)
            provider=Provider()
            driver.process_frame(path,digest,'id',output,provider,lambda stem,stages:[])
            self.assertEqual(provider.calls,['tiled','native'])

    def test_manifest_corruption_recovers_but_identity_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'manifest.json'
            for value in ('{incomplete','[]','null'):
                path.write_text(value)
                self.assertIsNone(driver.load_manifest(path,'id'))
            driver.atomic_json(path,{'identity':'id','completed':{}})
            self.assertEqual(driver.load_manifest(path,'id')['identity'],'id')
            with self.assertRaises(ValueError):driver.load_manifest(path,'changed-pipeline')

    def test_csv_columns_ids_native_masks_and_blank_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'submission.csv'
            records=[dict(file_name='20110101.jpeg',rles=[rectangle(),rectangle(100,200)]),dict(file_name='20110102.jpeg',rles=[])]
            self.assertEqual(driver.export_csv(path,records),2)
            with path.open(newline='') as f:
                reader=csv.DictReader(f);rows=list(reader)
                self.assertEqual(reader.fieldnames,['filament_id','segmentation_rle'])
            self.assertEqual([r['filament_id'] for r in rows],['20110101_1','20110101_2'])
            self.assertEqual([driver.checked_rle(dict(size=[2048,2048],counts=r['segmentation_rle']))['counts'] for r in rows],
                             [r['counts'] for r in records[0]['rles']])

    def test_invalid_or_duplicate_csv_never_replaces_previous_complete_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'submission.csv';path.write_text('previous complete file')
            for records in ([dict(file_name='20110101.jpeg',rles=[rectangle(),rectangle()])],
                            [dict(file_name='20110101.jpeg',rles=[rectangle()])]*2):
                with self.assertRaises(ValueError):driver.export_csv(path,records)
                self.assertEqual(path.read_text(),'previous complete file')
                self.assertFalse(path.with_name(path.name+'.pending').exists())

    def test_rle_rejects_truncation_empty_and_wrong_dimensions(self):
        value=rectangle();self.assertEqual(int(mu.area(driver.checked_rle(value))),200)
        with self.assertRaises(ValueError):driver.checked_rle(dict(size=[2048,2048],counts=value['counts'][:-1]))
        with self.assertRaises(ValueError):driver.checked_rle(mu.encode(np.zeros((2048,2048),np.uint8,order='F')))
        with self.assertRaises(ValueError):driver.checked_rle(dict(size=[1024,1024],counts=value['counts']))

    def test_test_file_discovery_rejects_ambiguous_ids_and_train_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);test=root/'test_images';test.mkdir()
            (test/'20110101.jpeg').write_bytes(b'image');(test/'note.txt').write_text('ignore')
            self.assertEqual([p.name for p in driver.list_test_images(test)],['20110101.jpeg'])
            (test/'20110101.png').write_bytes(b'same stem')
            with self.assertRaises(ValueError):driver.list_test_images(test)
            train=root/'train_images';train.mkdir()
            with self.assertRaises(ValueError):driver.list_test_images(train)


if __name__ == '__main__':
    unittest.main()
