"""Lean delivery preserves the existing experiment and never embeds its assets."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

TOOLS = Path(__file__).resolve().parents[1]/'tools'


def module(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS/(name+'.py'))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


builder = module('build_chase_lean_colab')
installer = module('chase_lean_install')


class LeanRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.shared.name)
        monitor = cls.folder/'monitor.py'
        monitor.write_text("def run_resume(root, **kwargs):\n    return {'fixture': True}\n")
        cls.report = builder.build(cls.folder/'delivery', monitor)
        cls.package = Path(cls.report['runtime_zip'])
        cls.sha = cls.report['runtime_zip_sha256']

    @classmethod
    def tearDownClass(cls): cls.shared.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'MyDrive/filament-segmentation-2026/fixed_split_experiments'/installer.ROOT_NAME

    def install(self, package=None, sha=None):
        return installer.install_runtime(package or self.package, self.root, sha or self.sha)

    def test_small_cell_and_runtime_contains_no_notebook_or_payload_string(self):
        cell = Path(self.report['copy_cell']).read_text()
        self.assertLess(len(cell.encode()), 4000)
        self.assertNotIn('base64', cell)
        self.assertNotIn('zlib.decompress', cell)
        self.assertIn(self.sha, cell)
        notebook = json.loads(Path(self.report['notebook']).read_text())
        self.assertEqual(len(notebook['cells']), 1)
        self.assertEqual(''.join(notebook['cells'][0]['source']), cell)
        with zipfile.ZipFile(self.package) as archive:
            self.assertFalse(any(name.endswith(('.ipynb', 'CopyCell.txt')) for name in archive.namelist()))
            self.assertEqual(len([name for name in archive.namelist() if name.startswith('code/') and name.endswith('.py')]), 12)

    def test_initial_install_and_repeat_preserve_training_state(self):
        launcher = Path(self.install())
        self.assertEqual(launcher.read_text(), "def run_resume(root, **kwargs):\n    return {'fixture': True}\n")
        self.assertNotIn('/code/', str(launcher))
        settings_path = self.root/'bootstrap_settings.json'
        settings = json.loads(settings_path.read_text())
        self.assertEqual(settings['cache_root'], '/content/chase_only_both_alignment_60ep_fullpass_cache')
        self.assertEqual(settings['root'], str(self.root))
        self.assertEqual(settings['assets'], str(self.root/'assets'/installer.VERSION))
        snapshots = {}
        for name, value in [('native/last.pt', b'preserved-native-optimizer'),
                            ('yolo/last_full.pt', b'preserved-yolo-EMA'),
                            ('native/history.json', b'[{"epoch": 12}]'),
                            ('configs/native.json', b'preserved-configuration'),
                            ('training.log', b'preserved-existing-log')]:
            path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)
            snapshots[path] = (value, path.stat().st_mtime_ns)
        settings_bytes = settings_path.read_bytes()
        self.assertEqual(Path(self.install()), launcher)
        self.assertEqual(settings_path.read_bytes(), settings_bytes)
        for path, (value, modified) in snapshots.items():
            self.assertEqual(path.read_bytes(), value)
            self.assertEqual(path.stat().st_mtime_ns, modified)

    def test_exact_legacy_source_identity_and_paths(self):
        self.install()
        legacy = json.loads((builder.DEST/(builder.LEGACY_STEM+'_manifest.json')).read_text())
        for folder, key in (('code', 'source_sha256'), ('assets', 'asset_sha256')):
            for name, sha in legacy[key].items():
                self.assertEqual(hashlib.sha256((self.root/folder/installer.VERSION/name).read_bytes()).hexdigest(), sha)
        expected = json.dumps(dict(sources=legacy['source_sha256'], assets=legacy['asset_sha256']), sort_keys=True, indent=2).encode()
        self.assertEqual((self.root/'code'/installer.VERSION/'source_manifest.json').read_bytes(), expected)

    def test_changed_existing_settings_blocks_all_new_writes(self):
        self.root.mkdir(parents=True)
        original = b'{"user":"preserve this"}'
        (self.root/'bootstrap_settings.json').write_bytes(original)
        with self.assertRaisesRegex(ValueError, 'Preserved existing'):
            self.install()
        self.assertEqual((self.root/'bootstrap_settings.json').read_bytes(), original)
        self.assertFalse((self.root/'code').exists())

    def test_changed_existing_source_is_preserved(self):
        self.install()
        source = self.root/'code'/installer.VERSION/'chase_best84_training.py'
        source.write_bytes(b'changed source preserved')
        with self.assertRaisesRegex(ValueError, 'Preserved existing'):
            self.install()
        self.assertEqual(source.read_bytes(), b'changed source preserved')

    def test_existing_user_paths_are_preserved_without_reset(self):
        self.install()
        path = self.root/'bootstrap_settings.json'
        settings = json.loads(path.read_text())
        settings['native_checkpoint'] = '/previously-configured/user/best84.pt'
        settings['images'] = '/previously-configured/user/train_images'
        raw = json.dumps(settings, indent=4).encode()
        path.write_bytes(raw)
        self.install()
        self.assertEqual(path.read_bytes(), raw)

    def test_wrong_archive_hash_rejects_before_extraction(self):
        with self.assertRaisesRegex(ValueError, 'checksum differs'):
            self.install(sha='0'*64)
        self.assertFalse(self.root.exists())

    def test_path_traversal_even_in_manifest_is_rejected(self):
        bad = Path(self.tmp.name)/'bad.zip'
        with zipfile.ZipFile(self.package) as original:
            entries = {name:original.read(name) for name in original.namelist()}
        manifest = json.loads(entries['install_manifest.json'])
        value = b'do not write'; name = '../outside.txt'
        manifest['files'][name] = hashlib.sha256(value).hexdigest()
        entries[name] = value
        entries['install_manifest.json'] = json.dumps(manifest).encode()
        with zipfile.ZipFile(bad, 'w') as archive:
            for name, value in entries.items(): archive.writestr(name, value)
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            self.install(bad, hashlib.sha256(bad.read_bytes()).hexdigest())
        self.assertFalse(self.root.exists())
        self.assertFalse((self.root.parent/'outside.txt').exists())

    def test_real_small_cell_uploads_once_then_reuses_drive_archive(self):
        # Exercise the generated cell with a CPU fixture monitor. Model launch
        # is deliberately stubbed; installing and importing the ZIP is real.
        cell = Path(self.report['copy_cell']).read_text().replace(
            "BASE = Path('/content/drive/MyDrive/filament-segmentation-2026')",
            'BASE = Path('+repr(str(self.root.parents[1]))+')')
        upload = mock.Mock(return_value={'runtime.zip': self.package.read_bytes()})
        colab = types.ModuleType('google.colab')
        colab.drive = types.SimpleNamespace(mount=mock.Mock())
        colab.files = types.SimpleNamespace(upload=upload, download=mock.Mock())
        google = types.ModuleType('google'); google.colab = colab
        with mock.patch.dict(sys.modules, {'google':google, 'google.colab':colab}):
            exec(compile(cell, 'small_cell.py', 'exec'), {'__name__':'__main__'})
            checkpoint = self.root/'native/last.pt'
            checkpoint.parent.mkdir(); checkpoint.write_bytes(b'keep-training-progress')
            exec(compile(cell, 'small_cell.py', 'exec'), {'__name__':'__main__'})
        self.assertEqual(upload.call_count, 1)
        self.assertEqual(checkpoint.read_bytes(), b'keep-training-progress')


if __name__ == '__main__': unittest.main()
