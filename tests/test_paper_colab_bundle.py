"""The Colab handoff contains the actual current research sources and fixed inputs."""
import ast
import base64
import hashlib
import importlib.util
import json
import lzma
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paper_colab_builder', ROOT/'tools/build_paper_method_colab.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PaperColabBundleTests(unittest.TestCase):
    def test_paste_cell_and_notebook_embed_current_sources(self):
        cell, raw = builder.build_cell()
        self.assertEqual(cell, (ROOT/'notebooks/SolarFilament_PaperMethods_CopyCell.py').read_text())
        notebook = json.loads((ROOT/'notebooks/SolarFilament_PaperMethods_A100.ipynb').read_text())
        self.assertEqual(cell, ''.join(notebook['cells'][1]['source']))
        tree = ast.parse(cell)
        payload = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'PAYLOAD' for target in node.targets))
        self.assertEqual(raw, lzma.decompress(base64.b64decode(payload)))
        self.assertIn(hashlib.sha256(raw).hexdigest(), cell)
        sources = json.loads(raw)
        self.assertEqual(set(sources), set(builder.MODULES))
        for name, source in sources.items():
            self.assertEqual(source, (ROOT/'tools'/name).read_text())
            self.assertNotIn('/workspace', source)
            compile(source, name, 'exec')

    def test_known_drive_paths_and_t4_context_only_budget(self):
        cell, _ = builder.build_cell()
        tree = ast.parse(cell)
        names = ('BASE', 'YOLO_RUN', 'NATIVE_RUN', 'OUTPUT', 'RUN_UNTIL_EPOCH', 'config')
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)]
        scope = dict(work=Path('/content/solar_paper_methods'))
        exec(compile(ast.Module(body=assignments, type_ignores=[]), 'config-only', 'exec'), scope)
        config = scope['config']
        self.assertEqual(config['methods'], ['control', 'multiscale_context'])
        self.assertEqual(config['execution_methods'], ['multiscale_context'])
        self.assertTrue(config['evaluate_baseline_current_device'])
        self.assertEqual(config['run_until_epoch'],20)
        self.assertEqual(config['epochs'], 100)
        self.assertEqual((config['batch'], config['accumulate'], config['steps_per_epoch']), (1, 4, 400))
        self.assertEqual(config['infer_batch'],1)
        self.assertEqual(config['crop'],1024)
        self.assertTrue(config['output_root'].endswith('/context_only_t4_v1'))
        self.assertEqual((config['thresholds'], config['min_areas']), ([0.5], [300]))
        self.assertEqual(config['synthetic_ratio'], 0.2)
        self.assertEqual(config['initial_checkpoint'], scope['NATIVE_RUN']+'/best.pt')
        self.assertTrue(config['images'].endswith('/MAGFiLO_1.0_Kaggle_2026/train/train_images'))
        self.assertEqual(config['external_root'], scope['BASE']+'/external_MAGFiLO')
        self.assertEqual(config['data_role'], 'competition_training')
        for name in ('images', 'annotations', 'synthetic_images', 'synthetic_annotations',
                     'synthetic_split', 'train_split', 'val_split'):
            self.assertFalse(Path(config[name]).is_relative_to(Path(config['external_root'])))
        self.assertEqual(len(config['expected_source_sha256']), 5)
        self.assertEqual(config['expected_source_sha256']['train_split'],
            'aa0733424815731c5c108ada91e45525eb17ca3a42b377bf19e3089d86aede7e')

    def test_fresh_subprocess_pinned_runtime_and_small_result_archive(self):
        cell, _ = builder.build_cell()
        self.assertIn(builder.BASE_RUNTIME_SHA, cell)
        self.assertIn('segmentation-models-pytorch==0.5.0', cell)
        self.assertIn('subprocess.run([sys.executable', cell)
        self.assertIn('str(work/"paper_context_only.py")',cell)
        self.assertIn('files.download(result["archive"])', cell)
        self.assertEqual(cell.count('files.download('), 1)
        self.assertNotIn('import torch', cell)
        self.assertNotIn('ultralytics', cell)

    def test_transition_embeds_current_driver_and_restores_saved_sources_after_restart(self):
        import tempfile
        cell,raw=builder.build_transition_cell()
        self.assertEqual(cell,(ROOT/'notebooks/SolarFilament_ContextOnly_SwitchCell.py').read_text())
        self.assertEqual(raw,(ROOT/'tools/paper_context_only.py').read_bytes())
        tree=ast.parse(cell)
        payload=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in node.targets))
        self.assertEqual(raw,lzma.decompress(base64.b64decode(payload)))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'saved_control';run.mkdir()
            oldoutput=root/'old_output';oldoutput.mkdir()
            (oldoutput/'experiment_progress.json').write_text(json.dumps(dict(runs={'control':dict(path=str(run))})))
            (run/'config.json').write_text(json.dumps(dict(epochs=100,batch=2,accumulate=2,steps_per_epoch=200)))
            for name in ('paper_method_training.py','paper_multiscale.py'):
                (run/name).write_bytes((ROOT/'tools'/name).read_bytes())
            (run/'base_runtime.py').write_bytes(b'\n'+(ROOT/'tools/solar_a100_train_standalone.py').read_bytes())
            scope=dict(Path=Path,json=json,hashlib=hashlib,BASE=root,NATIVE=root/'native',
                OUTPUT=root/'new_t4_output',PREVIOUS=oldoutput,OLD=root/'expired_local_runtime',WORK=root/'new_runtime')
            nodes=[]
            skip={'BASE','NATIVE','OUTPUT','PREVIOUS','OLD','WORK'}
            for node in tree.body:
                if isinstance(node,(ast.Import,ast.ImportFrom)):continue
                if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) \
                    and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='drive':continue
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in node.targets):break
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in skip for t in node.targets):continue
                nodes.append(node)
            exec(compile(ast.Module(body=nodes,type_ignores=[]),'restore-only','exec'),scope)
            self.assertEqual(scope['config']['batch'],2,'Restoration must read saved source config before explicit T4 override')
            for name in ('paper_method_training.py','paper_multiscale.py','base_runtime.py'):
                self.assertEqual((scope['WORK']/name).read_bytes(),(run/name).read_bytes())
            self.assertFalse(scope['OLD'].exists(),'No old local runtime is needed after restart')


if __name__ == '__main__':
    unittest.main()
