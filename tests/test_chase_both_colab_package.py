"""CPU-only orchestration and delivery checks; no fabricated model scores."""
from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import zipfile
import zlib

TOOLS = Path(__file__).resolve().parents[1]/'tools'
spec = importlib.util.spec_from_file_location('chase_both_colab_bootstrap', TOOLS/'chase_both_colab_bootstrap.py')
bootstrap = importlib.util.module_from_spec(spec); spec.loader.exec_module(bootstrap)


class Orchestration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/'study'
        self.settings = dict(root=str(self.root), assets=str(Path(self.temp.name)/'assets'),
            images='/fixture/images', annotations='/fixture/annotations.json',
            native_checkpoint='/fixture/old_native.pt', yolo_checkpoint='/fixture/old_yolo.pt',
            chase_root='/fixture/chase', cache_root='/fixture/cache')
        self.calls = []

    def tearDown(self): self.temp.cleanup()

    def runner(self, command, **kwargs):
        self.assertTrue(kwargs['check'])
        self.calls.append(command)
        script = Path(command[2]).name if len(command)>2 else ''
        if script in ('chase_yolo_training.py', 'chase_best84_training.py'):
            cfg = json.loads(Path(command[command.index('--config')+1]).read_text())
            cap = int(command[command.index('--run-until')+1])
            arm = 'yolo' if script == 'chase_yolo_training.py' else 'native'
            root = Path(cfg['output_root']); root.mkdir(parents=True, exist_ok=True)
            checkpoint = root/(f'epoch{cap:03d}.pt' if arm == 'yolo' else 'last.pt')
            checkpoint.write_bytes((arm+str(cap)).encode())
            result = dict(epochs_completed=cap, checkpoint=str(checkpoint),
                          checkpoint_sha256=bootstrap.digest(checkpoint), last=str(checkpoint))
            bootstrap.atomic_json(root/'result.json', result)
        if '--worker' in command and command[command.index('--worker')+1] == 'compare':
            cap = int(command[command.index('--run-until')+1])
            bootstrap.atomic_json(self.root/'comparison'/f'epoch{cap:03d}_comparison.json', {'fixture':True})

    def test_configs_isolate_models_and_share_native_inference(self):
        configs = bootstrap.configurations(self.settings)
        self.assertEqual(configs['native']['epochs'], 60)
        self.assertEqual(configs['yolo']['epochs'], 60)
        self.assertEqual(configs['native']['native_validation_cache_root'], configs['comparison']['native_validation_cache_root'])
        self.assertNotEqual(configs['native']['output_root'], configs['yolo']['output_root'])
        self.assertEqual(configs['native']['accumulate'], 4)
        self.assertEqual(configs['yolo']['nbs'], 1)
        self.assertEqual(configs['native']['chase_ratio'], 1.0)
        self.assertEqual(configs['native']['real_finish_epochs'], 0)
        self.assertEqual(configs['native']['training_domain'], 'chase_only')
        self.assertEqual(configs['yolo']['chase_ratio'], 1.0)
        self.assertEqual(configs['yolo']['real_finish_epochs'], 0)
        self.assertEqual(configs['yolo']['training_domain'], 'chase_only')
        for arm in ('native', 'yolo'):
            self.assertEqual(configs[arm]['steps_per_epoch'], 695)
            self.assertEqual(configs[arm]['epoch_sampling'], 'full_pass')

    @mock.patch.object(bootstrap, 'validate_paths')
    def test_full_phase_orders_independent_trainers_then_comparison(self, validation):
        result = bootstrap.run(self.settings, 60, ['yolo','native'], 'all', runner=self.runner)
        scripts = [Path(c[2]).name for c in self.calls]
        self.assertEqual(scripts, [self.calls[0][2], 'chase_both_colab_bootstrap.py',
            'chase_yolo_training.py', 'chase_best84_training.py', 'chase_yolo_predict.py',
            'chase_both_colab_bootstrap.py'])
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['comparison_available'])
        prediction = self.calls[-2]
        self.assertIn('--checkpoint-sha256', prediction)
        self.assertTrue(Path(result['archive']).is_file())
        with zipfile.ZipFile(result['archive']) as z:
            self.assertFalse(any(n.endswith('.pt') for n in z.namelist()))

    @mock.patch.object(bootstrap, 'validate_paths')
    def test_epoch30_is_partial_with_fixed60_schedule(self, validation):
        result = bootstrap.run(self.settings, 30, ['yolo','native'], 'all', runner=self.runner)
        self.assertEqual(result['status'], 'paused')
        self.assertEqual(result['fixed_epochs_per_model'], 60)
        self.assertTrue(result['comparison_available'])
        for arm in ('native', 'yolo'):
            cfg = json.loads((self.root/'configs'/f'{arm}.json').read_text())
            self.assertEqual(cfg['epochs'], 60)
            self.assertEqual(cfg['steps_per_epoch'], 695)
            self.assertEqual(cfg['epoch_sampling'], 'full_pass')

    @mock.patch.object(bootstrap, 'validate_paths')
    def test_prepare_requires_no_gpu_or_training(self, validation):
        result = bootstrap.run(self.settings, 60, ['yolo','native'], 'prepare', runner=self.runner)
        self.assertEqual(len(self.calls), 1)
        self.assertIn('prepare', self.calls[0])
        self.assertEqual(result['status'], 'prepared')
        self.assertFalse(result['comparison_available'])

    @mock.patch.object(bootstrap, 'validate_paths')
    def test_single_arm_does_not_claim_combined_evaluation(self, validation):
        result = bootstrap.run(self.settings, 10, ['yolo'], 'all', runner=self.runner)
        self.assertFalse(result['comparison_available'])
        self.assertEqual(result['status'], 'paused')
        self.assertNotIn('chase_yolo_predict.py', [Path(c[2]).name for c in self.calls])

    @mock.patch.object(bootstrap, 'validate_paths')
    def test_configuration_change_does_not_replace_saved_run(self, validation):
        bootstrap.run(self.settings, 10, ['yolo'], 'train', runner=self.runner)
        changed = dict(self.settings, images='/different/images')
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            bootstrap.run(changed, 60, ['yolo'], 'train', runner=self.runner)
        saved = json.loads((self.root/'configs/native.json').read_text())
        self.assertEqual(saved['images'], '/fixture/images')

    @mock.patch.object(bootstrap, 'validate_paths')
    def test_child_failure_is_not_reported_as_complete(self, validation):
        def fail(*args, **kwargs): raise RuntimeError('actual child failure')
        with self.assertRaisesRegex(RuntimeError, 'actual child failure'):
            bootstrap.run(self.settings, 60, ['yolo','native'], 'all', runner=fail)
        self.assertFalse((self.root/'latest_result.json').exists())

    def test_no_hidden_schedule_extension(self):
        with self.assertRaises(ValueError):
            bootstrap.run(self.settings, 61, ['yolo','native'], 'all', runner=self.runner)
        with self.assertRaises(ValueError):
            bootstrap.run(self.settings, 100, ['yolo','native'], 'all', runner=self.runner)
        with self.assertRaises(ValueError):
            bootstrap.run(self.settings, 0, ['yolo','native'], 'all', runner=self.runner)


class DeliveredPayload(unittest.TestCase):
    def test_actual_cell_restores_sources_and_rejects_mutation(self):
        import sys
        downloads = TOOLS.parents[1]/'artifacts/downloads'
        path = downloads/'SolarFilament_CHASE_Only_BothModels_60Epoch_FullPass_FixPNG_CopyCell.txt'
        if not path.exists(): self.skipTest('Final package not built yet')
        prefix = path.read_text().split('# Colab의 CUDA PyTorch/torchvision')[0]
        google = types.ModuleType('google'); colab = types.ModuleType('google.colab')
        colab.drive = types.SimpleNamespace(mount=lambda path: None)
        colab.files = types.SimpleNamespace(download=lambda path: None)
        google.colab = colab
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(sys.modules, {'google':google,'google.colab':colab}):
            prefix = prefix.replace('/content/drive', temp+'/drive').replace('/content/chase_only_both_alignment_60ep_fullpass_cache', temp+'/cache')
            base = Path(temp)/'drive/MyDrive/filament-segmentation-2026'
            old_root = base/'fixed_split_experiments/chase_only_both_alignment_t4_60ep_fullpass_v1'
            source_root = base/'external_CHASE/ahas_2024_2025'
            cache_root = Path(temp)/'cache'
            # Model the actual stopped-preparation state: immutable old run
            # metadata plus an already downloaded ZIP and partial Drive PNGs.
            retained = {
                old_root/'bootstrap_settings.json': b'{"previous":"settings"}',
                old_root/'experiment.json': b'{"previous":"experiment"}',
                old_root/'configs/native.json': b'{"previous":"native"}',
                old_root/'code/ff1af6fc182be562/source_manifest.json': b'{"previous":"manifest"}',
                source_root/'images/already_saved.png': b'partial-source-fixture',
                cache_root/'download/ahas_21582245_data.zip': b'existing-local-archive-fixture',
            }
            for retained_path, value in retained.items():
                retained_path.parent.mkdir(parents=True, exist_ok=True)
                retained_path.write_bytes(value)
            first = {}; exec(compile(prefix, '<actual-delivery-prefix>', 'exec'), first)
            second = {}; exec(compile(prefix, '<actual-delivery-prefix>', 'exec'), second)
            self.assertEqual(first['CODE'], second['CODE'])
            self.assertEqual(first['ROOT'], base/'fixed_split_experiments/chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1')
            self.assertNotEqual(first['ROOT'], old_root)
            self.assertEqual(first['CHASE_ROOT'], source_root)
            self.assertEqual(first['CACHE_ROOT'], cache_root)
            for retained_path, value in retained.items():
                self.assertEqual(retained_path.read_bytes(), value)
            manifest = json.loads((first['CODE']/'source_manifest.json').read_text())
            self.assertGreater(len(manifest['sources']), 8)
            altered = first['CODE']/'chase_both_colab_bootstrap.py'
            altered.write_text(altered.read_text()+'\n# accidental change\n')
            with self.assertRaisesRegex(ValueError, '변경'):
                exec(compile(prefix, '<actual-delivery-prefix>', 'exec'), {})

    def test_final_notebook_payload_matches_sources(self):
        downloads = TOOLS.parents[1]/'artifacts/downloads'
        notebook = downloads/'SolarFilament_CHASE_Only_BothModels_60Epoch_FullPass_FixPNG_OneCell.ipynb'
        if not notebook.exists(): self.skipTest('Final package not built yet')
        data = json.loads(notebook.read_text())
        code_cells = [cell for cell in data['cells'] if cell['cell_type']=='code']
        self.assertEqual(len(code_cells), 1)
        code = ''.join(code_cells[0]['source']); ast.parse(code)
        self.assertEqual(code, (downloads/'SolarFilament_CHASE_Only_BothModels_60Epoch_FullPass_FixPNG_CopyCell.txt').read_text())
        tree = ast.parse(code)
        payload = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'b64decode':
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    payload = node.args[0].value; break
        self.assertIsNotNone(payload)
        raw = zlib.decompress(base64.b64decode(payload)); bundle = json.loads(raw)
        manifest = json.loads((downloads/'SolarFilament_CHASE_Only_BothModels_60Epoch_FullPass_FixPNG_manifest.json').read_text())
        self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest['bundle_sha256'])
        self.assertEqual(manifest['counts']['epochs_each'], 60)
        self.assertEqual(manifest['counts']['gong_train'], 0)
        self.assertEqual(manifest['counts']['samples_each_epoch'], 695)
        self.assertEqual(manifest['counts']['sample_exposures_each_model'], 41700)
        self.assertEqual(manifest['counts']['total_sample_exposures'], 83400)
        self.assertEqual(manifest['epoch_sampling'], 'full_pass')
        self.assertIn('chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1', code)
        self.assertIn("CACHE_ROOT = Path('/content/chase_only_both_alignment_60ep_fullpass_cache')", code)
        self.assertIn("CHASE_ROOT = BASE/'external_CHASE/ahas_2024_2025'", code)
        self.assertIn("PHASE = 'all'", code)
        self.assertIn('RUN_UNTIL_EPOCH = 60', code)
        for name, text in bundle['sources'].items():
            self.assertEqual(text, (TOOLS/name).read_text(), name)
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), manifest['source_sha256'][name])
        for name, value in bundle['assets'].items():
            raw_asset = base64.b64decode(value)
            self.assertEqual(hashlib.sha256(raw_asset).hexdigest(), manifest['asset_sha256'][name])


if __name__ == '__main__': unittest.main()
