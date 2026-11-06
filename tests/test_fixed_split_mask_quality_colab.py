"""Check fresh-session restoration and launcher wiring without Drive or a GPU."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import build_fixed_split_mask_quality_colab as builder


class QualityColabTests(unittest.TestCase):
    def launch_fixture(self, temporary):
        metadata = builder.build()
        cell = Path(metadata['copy_cell']).read_text()
        drive_root = temporary/'drive'
        content = temporary/'content'
        base = drive_root/'filament-segmentation-2026'
        data = base/'MAGFiLO_1.0_Kaggle_2026/train'
        (data/'train_images').mkdir(parents=True)
        (data/'MAGFiLO_1.0_Annotations_kaggle2026_train.json').write_text('{}')
        old = drive_root/'SolarFilament_LimbOnly_YOLO26_mixed_FullRes'
        old.mkdir()
        for name in ('best.pt', 'train_split.csv', 'val_split.csv'):
            (old/name).write_text('preserve '+name)
        cell = cell.replace('/content/drive/MyDrive', str(drive_root))
        cell = cell.replace('/content/solar_fixed_mask_quality', str(content/'solar_fixed_mask_quality'))
        commands = []
        downloads = []
        google = types.ModuleType('google')
        colab = types.ModuleType('google.colab')
        google.colab = colab
        colab.drive = types.SimpleNamespace(mount=lambda _: None)
        colab.files = types.SimpleNamespace(download=downloads.append)

        class FakeProcess:
            def __init__(self, command, **kwargs):
                commands.append(command)
                phase = command[command.index('--phase')+1]
                self.stdout = io.StringIO('fixture phase '+phase+'\n')
                self.pid = 12345
                if phase == 'train':
                    output = Path(command[command.index('--output')+1])
                    until = int(command[command.index('--run-until')+1])
                    total = int(command[command.index('--epochs')+1])
                    (output/'summary.json').write_text(json.dumps(dict(
                        status='complete' if until == total else 'paused', epochs_completed=until)))
                    (output/'mask_quality_results.zip').write_bytes(b'fixture')

            def wait(self, timeout=None):
                return 0

        def run(run_until=None):
            selected_cell = cell if run_until is None else cell.replace('RUN_UNTIL = 40 ', f'RUN_UNTIL = {run_until} ')
            with patch.dict(sys.modules, {'google': google, 'google.colab': colab}), \
                    patch('subprocess.Popen', FakeProcess), \
                    patch('subprocess.run', return_value=types.SimpleNamespace(returncode=0)), \
                    patch('importlib.metadata.version', side_effect=lambda name:
                        {'ultralytics': '8.4.182', 'timm': '1.0.20', 'pycocotools': '2.0.11'}[name]), \
                    redirect_stdout(io.StringIO()):
                exec(compile(selected_cell, 'fixture-copy-cell', 'exec'), {})
        return metadata, base, old, content, commands, downloads, run

    def test_fresh_runtime_restores_all_imports_and_reuses_durable_source(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata, base, old, content, commands, downloads, run = self.launch_fixture(Path(directory))
            before = {path.name: path.read_bytes() for path in old.iterdir()}
            run()
            self.assertEqual([command[command.index('--phase')+1] for command in commands], ['prepare', 'train'])
            for command in commands:
                for option, value in [('--epochs', '40'), ('--run-until', '40'), ('--batch-size', '16')]:
                    self.assertEqual(command[command.index(option)+1], value)
                self.assertIn('fixed_split_experiments/mask_quality_v1', command[command.index('--output')+1])
                work = Path(command[2]).parent
                for name in builder.MODULES:
                    self.assertEqual(hashlib.sha256((work/name).read_bytes()).hexdigest(), metadata['sources_sha256'][name])
            self.assertEqual(len(downloads), 1)
            self.assertEqual(before, {path.name: path.read_bytes() for path in old.iterdir()})
            import shutil
            shutil.rmtree(content)
            run()
            self.assertEqual(len(commands), 4)
            self.assertEqual(before, {path.name: path.read_bytes() for path in old.iterdir()})
            self.assertTrue(Path(commands[-1][2]).is_file())

    def test_twenty_epoch_review_preserves_same_forty_epoch_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, _, commands, _, run = self.launch_fixture(Path(directory))
            run(run_until=20)
            run(run_until=40)
            self.assertEqual([command[command.index('--run-until')+1] for command in commands], ['20', '20', '40', '40'])
            self.assertTrue(all(command[command.index('--epochs')+1] == '40' for command in commands))
            self.assertEqual(len({command[command.index('--output')+1] for command in commands}), 1)
            self.assertEqual(len({command[2] for command in commands}), 1)

    def test_changed_durable_code_is_rejected_before_child_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            _, base, _, _, commands, _, run = self.launch_fixture(Path(directory))
            run()
            path = next((base/'fixed_split_experiments/mask_quality_v1/code').glob('*/submission_inference.py'))
            path.write_text('changed')
            commands.clear()
            with self.assertRaisesRegex(ValueError, '저장된 코드가 변경'):
                run()
            self.assertEqual(commands, [])


if __name__ == '__main__':
    unittest.main()
