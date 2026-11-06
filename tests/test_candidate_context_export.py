"""CPU fixtures for descriptor/label separation and interrupted-session exports."""
import ast
import base64
from contextlib import redirect_stdout
import gzip
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import zlib

import cv2
import numpy as np
from pycocotools import mask as mu

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import export_candidate_context_features as export
import build_candidate_context_export_colab as builder


def rectangle(x,y,w,h,size=128):
    pixels=np.zeros((size,size),np.uint8);pixels[y:y+h,x:x+w]=1
    value=mu.encode(np.asfortranarray(pixels));return dict(size=list(value['size']),counts=value['counts'].decode())


class ContextTests(unittest.TestCase):
    def setUp(self):cv2.setNumThreads(1)

    def fixture(self,directory,empty_validation=False):
        base=Path(directory);root=base/'source';images=base/'photos';output=base/'export'
        root.mkdir();images.mkdir();(root/'proposals').mkdir()
        identity=dict(inputs=export.INPUTS,code=export.CODE,runtime=dict(source='fixture'))
        corpus_id=export.fingerprint(identity);records=[]
        for role,name,masks in [
            ('train','20110101000000Ch.jpeg',[rectangle(53,39,8,35),rectangle(67,63,5,5)]),
            ('validation','20110102000000Ch.jpeg',[] if empty_validation else [rectangle(81,41,9,30)])]:
            yy,xx=np.mgrid[:128,:128];image=(130+xx/8+yy/12).astype(np.uint8)
            for mask in masks:image[mu.decode(dict(size=mask['size'],counts=mask['counts'].encode())).astype(bool)]-=45
            cv2.imwrite(str(images/name),image)
            instances=[dict(rle=value['counts'],confidence=.2+.1*i,target=[.7,.9] if i==0 else [0.,0.]) for i,value in enumerate(masks)]
            raw=dict(identity=dict(corpus=corpus_id,image_sha=export.file_sha(images/name),name=name,role=role),instances=instances)
            path=root/'proposals'/(Path(name).stem+'.json.gz');export.save_gzip(path,raw)
            records.append(dict(name=name,role=role,proposal_count=len(masks),raw_sha=export.file_sha(path),
                                selected=[0] if masks else [],roi_count=int(bool(masks))))
        (root/'corpus_identity.json').write_text(json.dumps(identity))
        (root/'corpus.json').write_text(json.dumps(dict(identity=corpus_id,records=records,training_images=1,validation_images=1)))
        return root,images,output

    def run_export(self,root,images,output,**kwargs):
        with redirect_stdout(io.StringIO()):
            return export.run(root,images,output,expected_counts=(1,1),expected_totals=kwargs.pop('totals',(2,1)),expected_shape=(128,128),**kwargs)

    def test_descriptors_dark_sign_geometry_and_target_free_signature(self):
        image=np.full((128,128),160,np.uint8);value=rectangle(59,40,10,48)
        encoded=export.checked_rle(value['counts'],image.shape)
        image[mu.decode(encoded).astype(bool)]=90
        maps=export.make_maps(image);values=export.features(maps,encoded,.35);row=dict(zip(export.COLUMNS,values))
        self.assertEqual(len(values),102)
        self.assertEqual(row['components'],1)
        self.assertEqual(row['extent'],1.)
        self.assertAlmostEqual(row['bbox_width_fraction'],10/128,places=7)
        self.assertGreater(row['sigma0_far_dark_z'],0)
        self.assertLess(row['sigma0_inside_median'],row['sigma0_far_median'])
        self.assertTrue(all(v is None or np.isfinite(v) for v in values))
        import inspect
        self.assertEqual(list(inspect.signature(export.features).parameters),['maps','value','confidence'])
        self.assertFalse(any('target' in name or 'label' in name for name in export.COLUMNS))

    def test_constant_border_invalid_rings_and_checked_RLE(self):
        image=np.full((128,128),100,np.uint8);maps=export.make_maps(image)
        center=export.features(maps,export.checked_rle(rectangle(60,60,5,5)['counts'],image.shape),.1)
        row=dict(zip(export.COLUMNS,center));self.assertEqual(row['sigma0_far_dark_z'],0.)
        outside=export.features(maps,export.checked_rle(rectangle(0,0,2,2)['counts'],image.shape),.1)
        row=dict(zip(export.COLUMNS,outside));self.assertEqual(row['outside_disk_fraction'],1.)
        self.assertIsNone(row['sigma0_inside_mean']);self.assertIsNone(row['sigma0_near_dark_z'])
        with self.assertRaises(ValueError):export.checked_rle('!',image.shape)
        with self.assertRaises(ValueError):export.checked_rle(rectangle(1,1,2,2)['counts'],(64,64))

    def test_all_train_raw_rows_tiny_masks_and_separate_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root,images,output=self.fixture(temporary);result=self.run_export(root,images,output)
            self.assertEqual(result['rows_by_role'],dict(train=2,validation=1))
            train=export.read_gzip(next((output/'images').glob('20110101*.gz')))
            self.assertEqual([row['index'] for row in train['rows']],[0,1])
            self.assertEqual(train['rows'][0]['targets'],[.7,.9])
            self.assertEqual(len(train['rows'][1]['features']),len(export.COLUMNS))
            self.assertEqual(train['rows'][1]['rle_sha256'],export.sha(train['rows'][1]['rle_counts'].encode()))
            with zipfile.ZipFile(result['archives'][0]) as z:
                manifest=json.loads(z.read('export_manifest.json'))
                self.assertFalse(any(name.endswith(('.pt','.jpeg','.jpg','.npy')) for name in z.namelist()))
                for name,details in manifest['entries'].items():
                    if name in z.namelist():self.assertEqual(export.sha(z.read(name)),details['sha256'])
            self.assertFalse(result['model_training']);self.assertFalse(result['new_neural_inference'])

    def test_partial_pending_resume_and_stable_dataset_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root,images,output=self.fixture(temporary)
            first=self.run_export(root,images,output,limit_images=1)
            self.assertEqual(first['status'],'paused')
            (output/'images/20110102000000Ch.json.gz.pending').write_bytes(b'interrupted')
            with patch.object(export,'make_maps',wraps=export.make_maps) as maps:
                second=self.run_export(root,images,output)
                self.assertEqual(maps.call_count,1)
            self.assertEqual(second['status'],'complete');self.assertEqual(second['image_cache_reused'],1)
            with patch.object(export,'make_maps',side_effect=AssertionError('cache should be reused')):
                third=self.run_export(root,images,output)
            self.assertEqual(third['image_cache_reused'],2)
            self.assertEqual(second['export_id'],third['export_id']);self.assertEqual(second['archives'],third['archives'])
            pointer=json.loads((output/'current_code.json').read_text())
            self.assertEqual(export.file_sha(pointer['path']),pointer['sha256'])
            self.assertTrue(Path(pointer['path']).is_file())

    def test_changed_image_raw_source_or_cached_join_is_rejected(self):
        for change in ('image','raw','source','cache','features'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temporary:
                root,images,output=self.fixture(temporary);self.run_export(root,images,output)
                if change=='image':
                    path=images/'20110101000000Ch.jpeg';path.write_bytes(path.read_bytes()+b'changed')
                elif change=='raw':
                    path=root/'proposals/20110101000000Ch.json.gz';raw=export.read_gzip(path);raw['instances'][0]['target']=[0.,0.];export.save_gzip(path,raw)
                elif change=='source':
                    request=json.loads((output/'request.json').read_text());request['identity']['source_sha256']='wrong';export.atomic_json(output/'request.json',request)
                else:
                    path=output/'images/20110101000000Ch.json.gz';value=export.read_gzip(path)
                    if change=='cache':value['rows'][0]['targets']=[0.,0.]
                    else:value['rows'][0]['features']=value['rows'][0]['features'][:-1]
                    export.save_gzip(path,value)
                with self.assertRaises(ValueError):self.run_export(root,images,output)

    def test_target_changes_do_not_change_descriptors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root,images,output=self.fixture(temporary);first=self.run_export(root,images,output)
            before=export.read_gzip(output/'images/20110101000000Ch.json.gz')['rows'][0]
            path=root/'proposals/20110101000000Ch.json.gz';raw=export.read_gzip(path)
            raw['instances'][0]['target']=[0.,0.];export.save_gzip(path,raw)
            corpus=json.loads((root/'corpus.json').read_text());corpus['records'][0]['raw_sha']=export.file_sha(path)
            (root/'corpus.json').write_text(json.dumps(corpus))
            second_output=Path(temporary)/'separate';self.run_export(root,images,second_output)
            after=export.read_gzip(second_output/'images/20110101000000Ch.json.gz')['rows'][0]
            self.assertEqual(before['features'],after['features']);self.assertNotEqual(before['targets'],after['targets'])

    def test_empty_proposal_image_kept_in_cohort(self):
        with tempfile.TemporaryDirectory() as temporary:
            root,images,output=self.fixture(temporary,empty_validation=True)
            result=self.run_export(root,images,output,totals=(2,0))
            self.assertEqual(result['completed_physical_images'],2)
            self.assertEqual(result['rows_by_role'],dict(train=2,validation=0))
            self.assertEqual(export.read_gzip(output/'images/20110102000000Ch.json.gz')['rows'],[])

    def test_missing_known_directory_and_default_full_cohort_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root,images,output=self.fixture(temporary)
            with self.assertRaisesRegex(ValueError,'616/91'):export.load_corpus(root)
            with self.assertRaisesRegex(ValueError,'directory missing'):self.run_export(root,images/'absent',output)

    def test_archive_parts_are_bounded_complete_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);files=[]
            for i in range(3):
                path=base/f'{i}.gz';path.write_bytes(os.urandom(18000));files.append((f'images/{i}.gz',path))
            paths=export.write_parts(base,{'note.json':b'{}'},files,90000,'c'*64)
            self.assertEqual(len(paths),3);seen=set()
            for path in paths:
                self.assertLessEqual(Path(path).stat().st_size,90000)
                with zipfile.ZipFile(path) as z:
                    manifest=json.loads(z.read('export_manifest.json'));self.assertEqual(manifest['parts_total'],3)
                    for name in manifest['files_in_this_part']:
                        self.assertEqual(export.sha(z.read(name)),manifest['entries'][name]['sha256'])
                        if name.startswith('images/'):self.assertNotIn(name,seen);seen.add(name)
            self.assertEqual(seen,{name for name,_ in files})

    def test_builder_embeds_current_source_and_creates_short_resume(self):
        info=builder.build();cell=Path(info['copy_cell']).read_text()
        payload=next(n.args[0].value for n in ast.walk(ast.parse(cell)) if isinstance(n,ast.Call)
            and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode' and isinstance(n.args[0],ast.Constant))
        self.assertEqual(zlib.decompress(base64.b64decode(payload)),Path(export.__file__).read_bytes())
        resume=Path(info['resume_cell']).read_text();compile(resume,'resume','exec')
        self.assertIn('current_code.json',resume);self.assertIn('saved[\'sha256\']',resume)
        self.assertLess(len(resume),len(cell))


if __name__=='__main__':unittest.main()
