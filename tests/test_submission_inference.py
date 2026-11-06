"""Portable inference parity with the already executed Colab helpers."""
import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import cv2
import numpy as np
from pycocotools import mask as mu
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import submission_inference as provider


def literal(file,name):
    tree = ast.parse((ROOT/'tools'/file).read_text())
    return next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
                and any(isinstance(target,ast.Name) and target.id==name for target in node.targets))


def load(name,file):
    spec = importlib.util.spec_from_file_location(name,ROOT/'tools'/file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


original_native = types.ModuleType('submission_original_native_fixture')
exec(compile(literal('colab_export_native_predictions_cell.py','EXPORT_SOURCE'),'<reviewed-export>','exec'),original_native.__dict__)
native_source = load('submission_original_training_fixture','solar_a100_train_standalone.py')
original_tile = dict(np=np,cv2=cv2,mu=mu)
exec(compile(literal('colab_tiled_inference_cell.py','HELPERS'),'<reviewed-tile>','exec'),original_tile)
prep_tree = ast.parse(literal('colab_preprocessing_cell.py','INITIALIZE'))
prep_nodes = [node for node in prep_tree.body if
    (isinstance(node,ast.FunctionDef) and node.name=='preprocess_variant') or
    (isinstance(node,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='PREPROCESS_PARAMS' for v in node.targets))]
original_prep = dict(np=np,cv2=cv2,disk=provider.limb_geometry()[1])
exec(compile(ast.Module(body=prep_nodes,type_ignores=[]),'<reviewed-preprocess>','exec'),original_prep)


class AsymmetricModel(torch.nn.Module):
    def forward(self,x):
        yy = torch.linspace(-.5,.7,x.shape[-2],dtype=x.dtype,device=x.device)[None,:,None]
        xx = torch.linspace(.1,.9,x.shape[-1],dtype=x.dtype,device=x.device)[None,None,:]
        return torch.stack([x[:,0]*.7+yy,x[:,1]*-.2+xx],dim=1)


def result(masks,confidence):
    return types.SimpleNamespace(masks=types.SimpleNamespace(data=torch.from_numpy(masks)),
        boxes=types.SimpleNamespace(conf=torch.tensor(confidence,dtype=torch.float32)))


class PortableInferenceTests(unittest.TestCase):
    def test_limb_and_all_preprocessing_variants_are_exact(self):
        image=np.random.default_rng(91).integers(0,256,(2048,2048),np.uint8)
        expected=native_source.limb(image)
        actual=provider.limb(image)
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(provider.PREPROCESS_PARAMS,original_prep['PREPROCESS_PARAMS'])
        for view in provider.PREPROCESS_PARAMS:
            np.testing.assert_array_equal(provider.preprocess_variant(actual,view),
                original_prep['preprocess_variant'](expected,view),err_msg=view)

    def test_native_heads_and_inverse_views_match_reviewed_export_exactly(self):
        image=np.random.default_rng(19).integers(0,256,(48,64),np.uint8)
        model=AsymmetricModel();device=torch.device('cpu')
        actual=provider.predict_views(model,image,device,32,.25,2,None)
        expected=original_native.predict_views(native_source,model,image,device,32,.25,2,None)
        np.testing.assert_array_equal(actual[0],expected[0])
        np.testing.assert_array_equal(actual[1],expected[1])
        self.assertEqual(actual[2],expected[2])
        self.assertEqual(provider.prediction_record(actual[1],'fixture.jpeg','pixels','run'),
            original_native.prediction_record(expected[1],'fixture.jpeg','pixels','run'))

    def test_tiled_result_encoding_matches_reviewed_native_coordinates_and_edges(self):
        masks=np.zeros((3,1024,1024),np.float32)
        masks[0,2:17,100:160]=1
        masks[1,970:1024,1000:1024]=1
        # Third detection is empty: reviewed behavior records and skips it.
        sample=result(masks,[.7,.9,.2])
        for top,left in [(0,0),(512,512),(1024,1024)]:
            actual=provider.InferenceProvider._tile_record(sample,'fixture',1536,top,left)
            expected=original_tile['encode_result'](sample,'fixture',1536,top,left)
            self.assertEqual(actual,expected)
            for item in actual['instances']:
                self.assertEqual(mu.decode(dict(size=[2048,2048],counts=item['rle'])).shape,(2048,2048))
        with self.assertRaises(ValueError):
            provider.InferenceProvider._tile_record(result(np.zeros((1,64,64),np.float32),[.5]),'bad',1536,0,0)

    def test_whole_frame_inverse_flip_and_sizes_preserve_native_pixels(self):
        mask=np.zeros((2048,2048),np.float32);mask[23:42,1700:1727]=1
        for view,transform in [('base',lambda x:x),('hflip',np.fliplr),('vflip',np.flipud)]:
            raw=provider.InferenceProvider._whole_record(result(transform(mask)[None].copy(),[.9]),'fixture',view,2048)
            self.assertEqual(raw['image'],'fixture')
            self.assertEqual(raw['coordinate_space'],'original')
            np.testing.assert_array_equal(mu.decode(dict(size=[2048,2048],counts=raw['instances'][0]['rle'])),mask)

    def test_stage_view_order_prediction_options_and_single_limb_input(self):
        class FakeYolo:
            def __init__(self):self.calls=[]
            def predict(self,pixels,**kw):
                self.calls.append((pixels.shape,kw))
                return [types.SimpleNamespace(masks=None,boxes=None)]
        image=np.zeros((2048,2048),np.uint8)
        model=FakeYolo()
        p=provider.InferenceProvider(None,None,'cpu',verify_weights=False,yolo_model=model)
        geometric=p.geometric('fixture',image)
        prep=p.preprocessing('fixture',image)
        tiled=p.tiled('fixture',image)
        self.assertEqual([r['view'] for r in geometric],['base','scale','hflip','vflip'])
        self.assertEqual([r['input_size'] for r in geometric],[2048,2560,2048,2048])
        self.assertEqual([r['view'] for r in prep],list(provider.PREPROCESS_PARAMS))
        self.assertEqual([r['view'] for r in tiled],
            [f'tile{s}_y{y}_x{x}' for y in (0,512,1024) for x in (0,512,1024) for s in (1536,2048)])
        self.assertEqual(len(model.calls),30)
        for shape,kw in model.calls:
            self.assertEqual(shape[-1],3)
            self.assertEqual(kw['conf'],.05);self.assertEqual(kw['iou'],.7)
            self.assertTrue(kw['retina_masks']);self.assertEqual(kw['max_det'],100)
        with self.assertRaises(ValueError):provider.limb(np.zeros((1024,1024),np.uint8))


if __name__=='__main__':
    unittest.main()
