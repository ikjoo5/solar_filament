import ast
from contextlib import redirect_stdout
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from pycocotools import mask as mu

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import export_solar_pixel_quality as export


def encoded(mask):
    value=mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return dict(size=list(value['size']),counts=value['counts'].decode())


def canonical(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class PixelFeatureTests(unittest.TestCase):
    def setUp(self):cv2.setNumThreads(1)

    def test_dark_line_blob_bright_sign_and_rotation(self):
        yy,xx=np.mgrid[-64:65,-64:65].astype(np.float32)
        line=.7-.4*np.exp(-xx*xx/8)
        blob=.7-.4*np.exp(-(xx*xx+yy*yy)/8)
        bright=.7+.25*np.exp(-xx*xx/8)
        a=export.ridge_maps(line,(2.,));b=export.ridge_maps(blob,(2.,));c=export.ridge_maps(bright,(2.,))
        self.assertGreater(a['strength'][64,64],5*b['strength'][64,64])
        self.assertEqual(float(c['strength'][64,64]),0.)
        rotated=export.ridge_maps(np.rot90(line),(2.,))
        np.testing.assert_allclose(np.rot90(a['strength']),rotated['strength'],atol=1e-5)
        mask=np.abs(xx)<=2
        fa=export.mask_features(line,a,mask)
        fr=export.mask_features(np.rot90(line),rotated,np.rot90(mask))
        self.assertGreater(fa['ridge_orientation_coherence'],.99)
        self.assertAlmostEqual(fa['ridge_orientation_coherence'],fr['ridge_orientation_coherence'],places=5)
        self.assertAlmostEqual(fa['dark_z'],fr['dark_z'],places=5)

    def test_blank_border_and_missing_validity(self):
        gray=np.full((32,32),.5,np.float32);maps=export.ridge_maps(gray)
        mask=np.zeros_like(gray,bool);mask[:4,:5]=True
        features=export.mask_features(gray,maps,mask)
        self.assertEqual(features['dark_z'],0.)
        self.assertEqual(features['ridge_orientation_coherence'],0.)
        self.assertFalse(features['ridge_orientation_valid'])
        valid=np.zeros_like(mask)
        missing=export.mask_features(gray,maps,mask,valid,valid)
        self.assertIsNone(missing['dark_z']);self.assertIsNone(missing['ridge_inside_mean'])
        self.assertEqual(missing['outside_disk_fraction'],1.)
        self.assertEqual(missing['ridge_valid_fraction'],0.)

    def test_mask_coordinate_rotation(self):
        gray=np.arange(48*64,dtype=np.float32).reshape(48,64)/(48*64)
        mask=np.zeros_like(gray,bool);mask[17:24,43:51]=True
        a=export.mask_features(gray,export.ridge_maps(gray),mask)
        b=export.mask_features(np.rot90(gray),export.ridge_maps(np.rot90(gray)),np.rot90(mask))
        self.assertEqual(a['area'],56)
        self.assertAlmostEqual(a['intensity_inside_mean'],float(gray[17:24,43:51].mean()),places=6)
        self.assertAlmostEqual(a['intensity_inside_median'],b['intensity_inside_median'],places=6)

    def test_ring_excludes_neighbor_predictions(self):
        gray=np.full((32,32),.8,np.float32);mask=np.zeros_like(gray,bool);mask[12:16,12:16]=True
        occupied=np.zeros_like(mask);occupied[5:12,5:23]=True;gray[occupied]=.1
        values=export.mask_features(gray,export.ridge_maps(gray),mask,ring_exclusion=occupied)
        self.assertAlmostEqual(values['intensity_ring_median'],.8,places=6)
        self.assertLess(values['photometric_ring_pixels'],export.mask_features(gray,export.ridge_maps(gray),mask)['photometric_ring_pixels'])

    def test_roi_fast_path_matches_global_coordinates_and_rejects_bad_area(self):
        gray=np.linspace(0,1,80*96,dtype=np.float32).reshape(80,96)
        mask=np.zeros_like(gray,bool);mask[47:54,71:78]=True;maps=export.ridge_maps(gray)
        old=export.mask_features(gray,maps,mask)
        new=export.mask_features(gray,maps,mask,bbox=(71,47,7,7),pixel_area=49)
        self.assertEqual(old,new)
        with self.assertRaisesRegex(ValueError,'Mask area differs'):
            export.mask_features(gray,maps,mask,bbox=(71,47,7,7),pixel_area=48)
        with self.assertRaisesRegex(ValueError,'outside image'):
            export.mask_features(gray,maps,mask,bbox=(95,47,7,7),pixel_area=49)

    def test_limb_is_exact_frozen_function(self):
        source=(Path(__file__).resolve().parents[1]/'tools/submission_inference.py').read_text()
        current=Path(export.__file__).read_text()
        for name in ('limb','limb_geometry'):
            nodes=[]
            for raw in (source,current):
                node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name==name)
                if isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body=node.body[1:]
                nodes.append(ast.dump(node,include_attributes=False))
            self.assertEqual(*nodes)

    def test_target_deduplication_and_novelty(self):
        original=np.zeros((2048,2048),np.uint8);original[100:120,100:120]=1
        novel=np.zeros_like(original);novel[1000:1020,1000:1020]=1
        a=encoded(original);b=encoded(novel)
        full=dict(file_name='image.jpeg',masks=dict(full_selected=[a]))
        view=lambda name,items:dict(image='image',shape=[2048,2048],coordinate_space='original',view=name,instances=items)
        raw=dict(geometric=[view('base',[dict(confidence=.3,rle=a['counts']),dict(confidence=.2,rle=b['counts'])])],
                 preprocessing=[view('limb',[dict(confidence=.99,rle=b['counts'])]),view('clahe',[dict(confidence=.4,rle=b['counts'])])],
                 tiled=[view('tile',[dict(confidence=.9,rle=b['counts'],edge_clipped=True)])])
        targets=export.target_masks(full,raw)
        self.assertEqual([t['role'] for t in targets],['baseline','novel_raw'])
        self.assertEqual(targets[1]['confidence'],.4)
        self.assertEqual(targets[1]['max_original_iou'],0.)

    def fixture(self,base):
        root=base/'study';snapshot=root/'current_epoch_evaluation/native030_yolo_current_v1';images=base/'images';images.mkdir()
        provider=Path(__file__).resolve().parents[1]/'tools/submission_inference.py'
        target=root/'code/5c1fb752ac88/submission_inference.py';target.parent.mkdir(parents=True);shutil.copyfile(provider,target)
        pixels=np.full((2048,2048),150,np.uint8)
        first='fold0_image000.png';cv2.imwrite(str(images/first),pixels);image_sha=export.sha(images/first)
        mask=np.zeros_like(pixels);mask[1000:1020,1000:1020]=1;rle=encoded(mask)
        empty=encoded(np.zeros_like(mask));native_sha='a'*64;yolo_sha='b'*64
        for fold,count in enumerate((143,142,140,141,141)):
            f=root/f'fold{fold}';stage=f/'stage030';stage.mkdir(parents=True)
            folder=snapshot/f'fold{fold}';folder.mkdir(parents=True)
            splits={}
            for role in ('train','val'):
                p=f/(role+'_split.csv');p.write_text('image_id,file_name\n'+f'{fold},placeholder.png\n');splits[role+'_split']=export.sha(p)
            (f/'yolo').mkdir();export.atomic_json(f/'yolo/config.json',dict(settings=dict(sources=splits)))
            payload=dict(native_sha256=native_sha,yolo_sha256=yolo_sha);identity=export.fingerprint(payload)
            export.atomic_json(folder/'snapshot_identity.json',dict(signature=identity,payload=payload))
            native=[];full=[]
            for i in range(count):
                name=f'fold{fold}_image{i:03d}.png'
                n=dict(file_name=name,fold=fold,stage_epoch=30,image_sha256=image_sha,checkpoint_sha256=native_sha,
                       tta=dict(unions=dict(foreground={'0.2':empty})))
                n['record_sha256']=canonical(n);native.append(n)
                row=dict(file_name=name,fold=fold,native_epoch=30,yolo_epoch=25 if fold==4 else 30,
                    identity=identity,image_sha256=image_sha,masks=dict(full_selected=[rle]))
                row['record_sha256']=export.fingerprint(row);full.append(row)
            native_path=stage/'rich_records.jsonl';native_path.write_text(''.join(json.dumps(r)+'\n' for r in native))
            full_path=folder/'full_records.jsonl';full_path.write_text(''.join(json.dumps(r)+'\n' for r in full))
            export.atomic_json(stage/'stage_summary.json',dict(training_config=dict(source_sha256=splits),
                rich_predictions_sha256=export.sha(native_path),image_names=[r['file_name'] for r in native]))
            export.atomic_json(folder/'summary.json',dict(identity=identity,records_sha256=export.sha(full_path),
                native_rich_sha256=export.sha(native_path),yolo_epoch=25 if fold==4 else 30,
                yolo=dict(checkpoint_sha256=yolo_sha)))
        raw=dict(identity=dict(image_sha256=image_sha,yolo_sha256=yolo_sha,provider_sha256=export.PROVIDER_SHA),
                 geometric=[],preprocessing=[],tiled=[])
        raw['checksum']=export.fingerprint(raw)
        p=root/'fold0/stage030/yolo_records'/('fold0_image000.json.gz');p.parent.mkdir();p.write_bytes(gzip.compress(json.dumps(raw).encode()))
        return root,snapshot,images

    def test_partial_run_resume_and_source_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);root,snapshot,images=self.fixture(base)
            out=root/'pixel_quality_exports/test'
            def zero_maps(gray):
                return {k:np.zeros_like(gray) for k in ('strength','cos2','sin2','anisotropy','scale')}
            original={p:export.sha(p) for p in root.rglob('*') if p.is_file()}
            with redirect_stdout(io.StringIO()):
                result=export.run(root,images,out,snapshot,limit_images=1,_extract=zero_maps)
                self.assertEqual(result['status'],'partial_pixel_features')
                self.assertEqual(result['completed_physical_images'],1)
                self.assertEqual(result['targets'],1)
                with patch.object(export,'ridge_maps',side_effect=AssertionError('Should reuse feature cache')):
                    again=export.run(root,images,out,snapshot,limit_images=1)
            self.assertEqual(result['targets'],again['targets'])
            self.assertEqual(export.read(out/'summary.json')['reused_images'],1)
            for p,digest in original.items():self.assertEqual(export.sha(p),digest)
            image=images/'fold0_image000.png';image.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'image bytes changed'):
                export.run(root,images,out,snapshot,limit_images=1)

    def test_archive_chunks_and_no_images_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary);p=out/'features.csv';raw=b'abcdef'*1000;p.write_bytes(raw)
            archives=export.package(out,[p],limit=2000)
            self.assertGreater(len(archives),1)
            manifest=export.read(out/'bundles/archive_manifest.json');restored=b''
            for piece in manifest['files'][0]['pieces']:
                with __import__('zipfile').ZipFile(out/'bundles'/piece['archive']) as z:
                    restored+=z.read(piece['member'])
                    self.assertIn('archive_manifest.json',z.namelist())
            self.assertEqual(restored,raw)
            self.assertFalse(manifest['images_and_weights_included'])


if __name__=='__main__':unittest.main()
