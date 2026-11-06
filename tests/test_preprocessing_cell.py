"""Preprocessing geometry, finite transforms and standalone Colab patch tests."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
from pycocotools import mask as mu


ROOT=Path(__file__).resolve().parents[1]


def parts():
    tree=ast.parse((ROOT/'tools/colab_preprocessing_cell.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) or (
        isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
        and n.targets[0].id in {'INITIALIZE','POSTPROCESS','INFERENCE'})]
    namespace={'ast':ast}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<reviewed-preprocessing-patch>','exec'),namespace)
    return namespace


def helper_namespace():
    p=parts()
    namespace=dict(np=np,cv2=cv2,mu=mu)
    exec(p['INITIALIZE']+'\n'+p['POSTPROCESS'],namespace)
    return namespace


class Tensor:
    def __init__(self,array):self.array=np.asarray(array)
    def float(self):return self
    def cpu(self):return self
    def numpy(self):return self.array


class PreprocessingTests(unittest.TestCase):
    def fixture(self):
        image=np.zeros((64,80),np.uint8)
        image[7:55,9:70]=120
        image[12:15,35:60]=70
        image[25:38,15:22]=200
        disk=np.zeros_like(image,dtype=bool);disk[7:55,9:70]=1
        return image,disk

    def test_base_exact_and_geometry_preserved(self):
        ns=helper_namespace();image,disk=self.fixture()
        for view in ns['PREPROCESS_PARAMS']:
            actual=ns['preprocess_variant'](image,view,disk)
            self.assertEqual(actual.shape,image.shape)
            self.assertEqual(actual.dtype,np.uint8)
            self.assertTrue(actual.flags.c_contiguous)
            self.assertTrue(np.all(actual[~disk]==0))
            self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_array_equal(ns['preprocess_variant'](image,'limb',disk),image)

    def test_constant_and_percentile_guard(self):
        ns=helper_namespace();image=np.full((64,80),111,np.uint8);disk=np.ones_like(image,bool)
        for view in ns['PREPROCESS_PARAMS']:
            actual=ns['preprocess_variant'](image,view,disk)
            self.assertTrue(np.isfinite(actual).all())
            self.assertEqual(actual.dtype,np.uint8)
        np.testing.assert_array_equal(ns['preprocess_variant'](image,'percentile',disk),image)
        with self.assertRaises(ValueError):ns['preprocess_variant'](image,'percentile',np.zeros_like(disk))
        with self.assertRaises(ValueError):ns['preprocess_variant'](image.astype(float),'limb',disk)
        with self.assertRaises(ValueError):ns['preprocess_variant'](image,'unknown',disk)

    def test_parameters_and_68_unique_candidates(self):
        ns=helper_namespace();specs=ns['PREP_SPECS']
        self.assertEqual(len(ns['PREPROCESS_PARAMS']),8)
        self.assertEqual(len(specs),68)
        self.assertEqual(len(ns['predictions']),68)
        self.assertEqual(sum(len(s['views'])==1 for s in specs),48)
        self.assertEqual(sum(s['name'].startswith('contrast') for s in specs),6)
        self.assertEqual(sum(s['name'].startswith('smooth') for s in specs),6)
        self.assertEqual(sum(s['name'].startswith('all8') for s in specs),8)
        self.assertTrue(all(s['required']<=len(s['views']) for s in specs))

    def test_roi_native_rle_preserves_holes_and_boundaries(self):
        ns=helper_namespace();mask=np.ones((13,19),bool);mask[3:8,4:12]=0
        for top,left in [(0,0),(2035,2029),(20,30)]:
            rle=ns['encode_native_roi'](mask,top,left)
            expected=np.zeros((2048,2048),np.uint8);expected[top:top+13,left:left+19]=mask
            np.testing.assert_array_equal(mu.decode(rle),expected)

    def test_patch_compiles_previous_source_and_keeps_exports(self):
        original=ast.parse((ROOT/'tools/colab_paste_evaluate.py').read_text())
        source=next(ast.literal_eval(n.value) for n in original.body if isinstance(n,ast.Assign)
                    and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='SCRIPT')
        patch=parts()['Patch']()
        worker=ast.unparse(ast.fix_missing_locations(patch.visit(ast.parse(source))))
        compile(worker,'<preprocessing-worker>','exec')
        self.assertEqual(patch.counts,dict(init=1,post=1,loop=1,summary=1))
        self.assertIn('preprocessing_8view_v1',worker)
        self.assertIn('raw_preprocess_views.jsonl',worker)
        self.assertIn('result',worker)

    def test_native_eight_view_loop_and_real_jsonl(self):
        p=parts();image=np.zeros((2048,2048),np.uint8)
        image[100:125,200:230]=130
        image[600:610,700:710]=80
        disk=np.zeros_like(image,bool);disk[50:2000,50:2000]=1
        calls=[]

        class Model:
            def predict(self,source,**kwargs):
                calls.append(kwargs)
                mask=source[:,:,0]>0
                return [SimpleNamespace(masks=SimpleNamespace(data=Tensor(mask[None])),
                                        boxes=SimpleNamespace(conf=Tensor([.7])))]

        with tempfile.TemporaryDirectory() as folder:
            ns=dict(np=np,cv2=cv2,mu=mu,json=json,Path=Path,disk=disk,
                    val_names=['sample.jpeg'],IMAGES=Path(folder),output=Path(folder),
                    model=Model(),limb=lambda x:image)
            (Path(folder)/'sample.jpeg').write_bytes(b'placeholder')
            exec(p['INITIALIZE']+'\n'+p['POSTPROCESS'],ns)
            exec(p['INFERENCE'],ns)
            records=[json.loads(line) for line in (Path(folder)/'raw_preprocess_views.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),8)
            self.assertEqual({r['view'] for r in records},set(ns['PREPROCESS_PARAMS']))
            self.assertEqual(len(calls),8)
            for record in records:
                self.assertEqual(record['shape'],[2048,2048])
                self.assertEqual(record['coordinate_space'],'original')
                self.assertIn('parameters',record)
                self.assertEqual(record['instances'][0]['confidence'],.7)
            self.assertTrue(all('sample' in prediction for prediction in ns['predictions'].values()))
            self.assertTrue(all(call['imgsz']==2048 and call['conf']==.05 for call in calls))


if __name__=='__main__':unittest.main()
