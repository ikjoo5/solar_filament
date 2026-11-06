import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

SOURCE = Path(__file__).resolve().parents[1]/'tools/chase_inline_resume.py'
spec = importlib.util.spec_from_file_location('chase_inline_resume', SOURCE)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
PACKAGE = Path('/workspace/artifacts/downloads/SolarFilament_CHASE_Light_Runtime.zip')


class InlineResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'fixed_split_experiments/chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1'
        self.root.mkdir(parents=True)
        with zipfile.ZipFile(PACKAGE) as archive:
            for name in archive.namelist():
                if name.startswith(('code/','assets/')):
                    path = self.root/name; path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(archive.read(name))

    def test_validate_restores_only_settings_preserves_checkpoint(self):
        checkpoint = self.root/'last.pt'; checkpoint.write_bytes(b'not a real checkpoint; never load')
        before = {str(p.relative_to(self.root)):(p.stat().st_mtime_ns,hashlib.sha256(p.read_bytes()).hexdigest())
                  for p in self.root.rglob('*') if p.is_file()}
        code = module._validate(self.root)
        self.assertEqual(code.name,'057b2b49a0cdf3e0')
        for name,value in before.items():
            p=self.root/name
            self.assertEqual((p.stat().st_mtime_ns,hashlib.sha256(p.read_bytes()).hexdigest()),value)
        settings=(self.root/'bootstrap_settings.json').read_bytes()
        module._validate(self.root)
        self.assertEqual(settings,(self.root/'bootstrap_settings.json').read_bytes())

    def test_corrupt_code_rejected_before_settings(self):
        (self.root/'code/057b2b49a0cdf3e0/chase_transfer_data.py').write_text('modified')
        with self.assertRaisesRegex(ValueError,'파일 내용'):
            module._validate(self.root)
        self.assertFalse((self.root/'bootstrap_settings.json').exists())

    def test_real_child_output_is_retained_without_flood(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code=module._run_logged([sys.executable,'-c','for i in range(1000): print("progress",i,flush=True)'],
                                    self.root,self.root,interval=.1)
        self.assertEqual(code,0)
        text=(self.root/'training.log').read_text()
        self.assertIn('progress 0\n',text); self.assertIn('progress 999\n',text)
        self.assertLess(len(output.getvalue().splitlines()),15)
        self.assertEqual(json.loads((self.root/'inline_heartbeat.json').read_text())['exit_code'],0)

    def test_real_child_failure_prints_reason(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(subprocess.CalledProcessError):
                module._run_logged([sys.executable,'-c','raise RuntimeError("real failure")'],
                                   self.root,self.root,interval=.1)
        self.assertIn('real failure',output.getvalue())
        self.assertEqual(json.loads((self.root/'inline_heartbeat.json').read_text())['exit_code'],1)


if __name__ == '__main__': unittest.main()
