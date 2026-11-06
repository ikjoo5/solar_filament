"""Exercise migration from the actual released pre-PNG Drive payload."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

TOOLS = Path(__file__).resolve().parents[1]/'tools'
spec = importlib.util.spec_from_file_location('recovery_builder', TOOLS/'build_chase_nozip_recovery.py')
builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)


class Recovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.section = builder.render_recovery_section()
        cls.scope = {}; exec(compile(cls.section, 'generated.py', 'exec'), cls.scope)
        cls.expected = cls.scope['EXPECTED']

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)/'filament-segmentation-2026'
        self.old = self.base/'fixed_split_experiments'/builder.OLD_ROOT
        self.root = self.old.parent/builder.ROOT_NAME
        with zipfile.ZipFile(builder.DEST/(builder.STEM+'_Package.zip')) as archive:
            for folder, hashes in self.expected.items():
                for name in hashes:
                    path = self.old/folder/builder.OLD_VERSION/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.read(folder+'/'+name))

    def recover(self): return self.scope['recover_sources'](self.base)

    def test_complete_old_payload_yields_exact_frozen_fixed_sources(self):
        before = {p:(p.read_bytes(),p.stat().st_mtime_ns) for p in self.old.rglob('*') if p.is_file()}
        root,code,assets = self.recover()
        self.assertEqual(root,self.root)
        with zipfile.ZipFile(builder.DEST/(builder.STEM+'_FixPNG_Package.zip')) as archive:
            for folder, hashes in self.expected.items():
                for name, sha in hashes.items():
                    value = (root/folder/builder.VERSION/name).read_bytes()
                    self.assertEqual(value,archive.read(folder+'/'+name))
                    self.assertEqual(hashlib.sha256(value).hexdigest(),sha)
        self.assertEqual(json.loads((code/'source_manifest.json').read_text()),
                         dict(sources=self.expected['code'],assets=self.expected['assets']))
        for p,(value,mtime) in before.items():
            self.assertEqual((p.read_bytes(),p.stat().st_mtime_ns),(value,mtime))

    def test_repeat_after_old_payload_removed_preserves_new_training_files(self):
        self.recover()
        snapshots = {}
        for name, value in [('yolo/last_full.pt',b'optimizer-EMA-RNG'),
                            ('native/last.pt',b'nativeoptimizer'),
                            ('native/history.json',b'[{"epoch": 12}]'),
                            ('bootstrap_settings.json',b'{"custom": "keep"}')]:
            p = self.root/name; p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value)
            snapshots[p]=(value,p.stat().st_mtime_ns)
        for p in self.root.rglob('*'):
            if p.is_file(): snapshots[p]=(p.read_bytes(),p.stat().st_mtime_ns)
        shutil.rmtree(self.old)
        self.recover()
        for p,(value,mtime) in snapshots.items():
            self.assertEqual((p.read_bytes(),p.stat().st_mtime_ns),(value,mtime))

    def test_changed_old_data_source_refuses_before_writing(self):
        p=self.old/'code'/builder.OLD_VERSION/'chase_transfer_data.py'
        p.write_bytes(p.read_bytes()+b'\n# edited\n')
        with self.assertRaises(ValueError): self.recover()
        self.assertFalse(self.root.exists())

    def test_changed_existing_target_is_not_overwritten_or_completed(self):
        p=self.root/'code'/builder.VERSION/'chase_transfer_data.py'
        p.parent.mkdir(parents=True);p.write_bytes(b'custom edit')
        with self.assertRaises(ValueError): self.recover()
        self.assertEqual(p.read_bytes(),b'custom edit')
        self.assertEqual([x for x in self.root.rglob('*') if x.is_file()],[p])

    def test_partial_target_recovers_from_old_without_touching_checkpoint(self):
        self.recover()
        code=self.root/'code'/builder.VERSION
        (code/'chase_transfer_data.py').unlink()
        checkpoint=self.root/'native/last.pt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'keep')
        stamp=checkpoint.stat().st_mtime_ns
        self.recover()
        self.assertEqual(hashlib.sha256((code/'chase_transfer_data.py').read_bytes()).hexdigest(),
                         self.expected['code']['chase_transfer_data.py'])
        self.assertEqual((checkpoint.read_bytes(),checkpoint.stat().st_mtime_ns),(b'keep',stamp))

    def test_missing_last_asset_does_not_leave_partial_code_install(self):
        asset=self.old/'assets'/builder.OLD_VERSION/'chase_ahas2024_2025_manifest.json'
        asset.unlink()
        with self.assertRaises(FileNotFoundError): self.recover()
        self.assertFalse(self.root.exists())

    def test_assembled_helper_accepts_exact_recovered_payload(self):
        report = builder.build(Path(self.tmp.name)/'delivery')
        cell = Path(report['copy_cell']).read_text()
        compile(cell, 'assembled_colab_cell.py', 'exec')
        self.assertLess(len(cell.encode()), 18000)
        self.assertNotIn('files.upload', cell)
        self.assertNotIn('base64', cell)
        notebook=json.loads(Path(report['notebook']).read_text())
        self.assertEqual(''.join(notebook['cells'][0]['source']),cell)
        helper=(TOOLS/'chase_inline_resume.py').read_text()
        scope=dict(self.scope);exec(compile(helper, 'inline_helper.py','exec'),scope)
        root,code,assets=scope['recover_sources'](self.base)
        self.assertEqual(scope['_validate'](root),code)
        settings_path=root/'bootstrap_settings.json'
        settings=json.loads(settings_path.read_text())
        self.assertEqual(settings['root'],str(root))
        self.assertEqual(settings['assets'],str(assets))
        snapshot=(settings_path.read_bytes(),settings_path.stat().st_mtime_ns)
        self.assertEqual(scope['_validate'](root),code)
        self.assertEqual((settings_path.read_bytes(),settings_path.stat().st_mtime_ns),snapshot)


if __name__ == '__main__': unittest.main()
