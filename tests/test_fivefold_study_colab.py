import ast
import base64
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import build_fivefold_study_colab as builder


class StudyColabTests(unittest.TestCase):
    def test_new_bundle_contains_full_recipe_and_no_old_drive_dependency(self):
        meta=builder.build();source=(ROOT/'notebooks/SolarFilament_5Fold_Study_v2_CopyCell.py').read_text()
        tree=ast.parse(source)
        literals={node.targets[0].id:ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
            and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('PAYLOAD','STAGE_EPOCH','INFERENCE_PROFILE')}
        raw=zlib.decompress(base64.b64decode(literals['PAYLOAD']));parts=json.loads(raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),meta['bundle_sha256'])
        self.assertEqual(literals['STAGE_EPOCH'],30);self.assertEqual(meta['total_initial_model_epochs'],300)
        for name in ('base_runtime.py','fivefold_yolo_training.py','fivefold_full_pipeline.py','submission_yolo_recipe.py','submission_native_recipe.py'):
            self.assertIn(name,parts);compile(parts[name],name,'exec')
        self.assertEqual(hashlib.sha256(parts['base_runtime.py'].encode()).hexdigest(),builder.RUNTIME_SHA)
        self.assertNotIn('SolarFilament_NativeUNet_A100',source)
        self.assertIn('fivefold_study_v2',source)
        self.assertIn('for fold in range(5)',parts['study_launcher.py'])
        policy=next(node.value for node in ast.walk(ast.parse(parts['study_launcher.py']))
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='policy')
        lanes={kw.arg:kw.value for kw in policy.keywords}
        self.assertIn('architecture',ast.literal_eval(lanes['fixed_split_role']))
        self.assertIn('postprocessing',ast.literal_eval(lanes['fivefold_role']))

    def test_all_fold_folders_have_reproducible_separation_manifests(self):
        namespace={'__name__':'fixture'};exec(builder.LAUNCHER,namespace)
        annotation=Path('/workspace/datasets/solar_filament/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);plan=namespace['prepare_layout'](root,annotation)
            self.assertEqual(sum(f['validation_physical'] for f in plan['fold_summaries']),707)
            before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            namespace['prepare_layout'](root,annotation)
            self.assertEqual(before,{str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()})
            for fold in range(5):
                for name in ('train_split.csv','val_split.csv','excluded_embargo.csv','split_manifest.json'):
                    self.assertTrue((root/f'fold{fold}'/name).exists())
                manifest=json.loads((root/f'fold{fold}/split_manifest.json').read_text())
                self.assertFalse(set(manifest['images']['train'])&set(manifest['images']['validation']))
            (root/'fold0/val_split.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError,'preserved'):namespace['prepare_layout'](root,annotation)


if __name__=='__main__':unittest.main()
