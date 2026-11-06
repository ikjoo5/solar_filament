"""The bootstrap must restore code, preserve sources and survive /content loss."""
import ast
import base64
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
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import build_grouped_cv_colab as builder


class FakeProcess:
    def __init__(self, command, **kwargs):
        assert kwargs['start_new_session'] is True
        assert command[command.index('--folds')+1] == '0'
        self.pid = 99999999
    def wait(self, **kwargs):
        return 0


class ColabBundleTests(unittest.TestCase):
    def test_payload_matches_reviewed_sources_and_notebook(self):
        result = builder.build()
        source = (ROOT/'notebooks/SolarFilament_GroupedCV_T4_CopyCell.py').read_text()
        tree = ast.parse(source)
        values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id in ('PAYLOAD','FOLDS','EPOCHS','RUN_UNTIL_EPOCH')}
        raw = zlib.decompress(base64.b64decode(values['PAYLOAD']))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), result['bundle_sha256'])
        bundled = json.loads(raw)
        for name in ('grouped_cv_training.py', 'prepare_grouped_fivefold.py'):
            self.assertEqual(bundled[name], (ROOT/'tools'/name).read_text())
        self.assertEqual(values['FOLDS'], [0])
        self.assertEqual((values['EPOCHS'], values['RUN_UNTIL_EPOCH']), (100,100))
        notebook = json.loads((ROOT/'notebooks/SolarFilament_GroupedCV_T4.ipynb').read_text())
        self.assertEqual(''.join(notebook['cells'][1]['source']), source)

    def test_initial_restore_and_short_resume_need_no_previous_content_directory(self):
        builder.build()
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            base = Path(folder); experiment = base/'grouped_cv_t4_v1'
            code = experiment/'code'; code.mkdir(parents=True)
            (code/'base_runtime.py').write_bytes(b'\n'+(ROOT/'tools/solar_a100_train_standalone.py').read_bytes())
            (experiment/'latest_results.json').write_text(json.dumps([dict(archive='fixture.zip')]))
            downloads = []
            google = types.ModuleType('google'); colab = types.ModuleType('google.colab')
            colab.drive = types.SimpleNamespace(mount=lambda *a,**k: None)
            colab.files = types.SimpleNamespace(download=downloads.append)
            google.colab = colab
            source = (ROOT/'notebooks/SolarFilament_GroupedCV_T4_CopyCell.py').read_text()
            source = source.replace("BASE = Path('/content/drive/MyDrive/filament-segmentation-2026')",
                                    'BASE = Path('+repr(str(base))+')')
            resume = builder.RESUME.replace("ROOT = Path('/content/drive/MyDrive/filament-segmentation-2026/grouped_cv_t4_v1')",
                                            'ROOT = Path('+repr(str(experiment))+')')
            with patch.dict(sys.modules, {'google':google,'google.colab':colab}), patch('subprocess.Popen', FakeProcess):
                exec(compile(source,'initial-cell','exec'), {'__name__':'__main__'})
                config = json.loads((experiment/'launch_config.json').read_text())
                self.assertEqual(config['folds'],[0])
                self.assertEqual(config['epochs'],100)
                self.assertTrue((code/'grouped_cv_training.py').is_file())
                exec(compile(resume,'resume-cell','exec'), {'__name__':'__main__'})
                self.assertEqual(downloads, ['fixture.zip','fixture.zip'])
                target = code/'grouped_cv_training.py'
                target.write_text(target.read_text()+'\n# changed\n')
                with self.assertRaisesRegex(ValueError, '코드가 달라'):
                    exec(compile(resume,'resume-cell','exec'), {'__name__':'__main__'})

    def test_launcher_json_is_valid_and_does_not_replace_torch(self):
        namespace = {'__name__':'fixture'}
        exec(builder.LAUNCHER, namespace)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'config.json'
            namespace['save'](path, {'a':1})
            self.assertEqual(json.loads(path.read_text()), {'a':1})
        self.assertNotIn("'torch':", builder.LAUNCHER)
        self.assertIn("environment.pop('TORCH_CUDNN_V8_API_DISABLED', None)", builder.LAUNCHER)
        self.assertIn("source = root/'code'/name", builder.LAUNCHER)


if __name__ == '__main__':
    unittest.main()
