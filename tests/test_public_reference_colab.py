"""Exercise the generated audit cell without Drive, GPU, pip, or a subprocess."""
import ast
import base64
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
import zlib

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import build_public_reference_audit_colab as builder


def snapshots(root):
    return {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file()}


class PublicReferenceColabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle_temp = tempfile.TemporaryDirectory()
        cls.bundle_root = Path(cls.bundle_temp.name)
        with patch.object(builder, 'DEST', cls.bundle_root), redirect_stdout(io.StringIO()):
            builder.main()
        cls.cell = (cls.bundle_root/(builder.STEM+'_CopyCell.txt')).read_text()

    @classmethod
    def tearDownClass(cls):
        cls.bundle_temp.cleanup()

    def prepare(self, root, complete=True):
        run_root = root/'inference'
        run_root.mkdir()
        (run_root/'result.json').write_text(json.dumps({'status':'complete' if complete else 'paused'}))
        (run_root/'submission_pipeline_R1.csv').write_text('filament_id,segmentation_rle\n')
        (run_root/'cache').mkdir()
        (run_root/'cache/immutable.bin').write_bytes(b'inference cache is immutable')
        audit_root = root/'diagnostics'
        return run_root, audit_root

    def execute(self, run_root, audit_root, *, absent=(), status=0):
        tree = ast.parse(self.cell)
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == 'RUN_ROOT':node.value = ast.Constant(str(run_root))
                elif node.targets[0].id == 'AUDIT_ROOT':node.value = ast.Constant(str(audit_root))
        compiled = compile(ast.fix_missing_locations(tree), '<generated_colab_audit>', 'exec')
        calls, downloads, pip_calls, mounts = [], [], [], []
        colab = ModuleType('google.colab')
        colab.drive = SimpleNamespace(mount=lambda path:mounts.append(path))
        colab.files = SimpleNamespace(download=lambda path:downloads.append(path))
        google = ModuleType('google'); google.colab = colab
        class FakeProcess:
            pid = 123456789
            def __init__(self, command, **kwargs):
                calls.append(command)
                self.stdout = io.StringIO('audit fixture completed\n' if not status else 'Traceback: audit failed\n')
                if not status:
                    output = Path(command[command.index('--output-root')+1])
                    output.mkdir(parents=True)
                    (output/'summary.json').write_text(json.dumps({'fixture':True,'PQ_computed':False}))
                    (output/'reference_instances.csv').write_text('image,diagnosis\nfixture,cached_candidate\n')
            def wait(self, timeout=None):return status
        def fake_pip(command, **kwargs):
            pip_calls.append(command)
            return subprocess.CompletedProcess(command,0)
        with patch.dict(sys.modules, {'google':google,'google.colab':colab}), \
                patch('subprocess.Popen', FakeProcess), patch('subprocess.run', fake_pip), \
                patch('importlib.util.find_spec', side_effect=lambda name:None if name in absent else object()), \
                patch('os.killpg', side_effect=ProcessLookupError), redirect_stdout(io.StringIO()):
            namespace = {}
            exec(compiled,namespace)
        return dict(calls=calls,downloads=downloads,pip_calls=pip_calls,mounts=mounts,namespace=namespace)

    def test_payload_and_packaged_sources_match_current_code(self):
        tree = ast.parse(self.cell)
        assignments = {node.targets[0].id:node.value.value for node in tree.body
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
            and isinstance(node.value,ast.Constant)}
        raw = zlib.decompress(base64.b64decode(assignments['payload']))
        bundle = json.loads(raw)
        digest = hashlib.sha256(raw).hexdigest()
        self.assertIn(digest,self.cell)
        self.assertEqual(assignments['version'],digest[:16])
        self.assertEqual(set(bundle['sources']),set(builder.MODULES))
        for name,source in bundle['sources'].items():
            self.assertEqual(source,(TOOLS/name).read_text())
            compile(source,name,'exec')
        public = base64.b64decode(bundle['assets']['reference_public055.csv'])
        self.assertEqual(hashlib.sha256(public).hexdigest(), '0ab3932122a563ba96f4d243934d5c00680dd49b9d1165912a4f180907d2c603')
        names = json.loads(base64.b64decode(bundle['assets']['official_test_names.json']))
        self.assertEqual(len(names['test_images']),180)
        with zipfile.ZipFile(self.bundle_root/(builder.STEM+'_Package.zip')) as archive:
            self.assertIsNone(archive.testzip())
            for name,source in bundle['sources'].items():self.assertEqual(archive.read('tools/'+name).decode(),source)

    def test_incomplete_run_stops_before_install_or_audit_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root,audit_root = self.prepare(Path(temporary),complete=False)
            before = snapshots(run_root)
            with self.assertRaisesRegex(RuntimeError,'아직 완료되지'):
                self.execute(run_root,audit_root,absent=('pycocotools',))
            self.assertFalse(audit_root.exists())
            self.assertEqual(snapshots(run_root),before)

    def test_success_archives_audit_and_repeat_uses_new_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root,audit_root = self.prepare(Path(temporary))
            before = snapshots(run_root)
            first = self.execute(run_root,audit_root)
            second = self.execute(run_root,audit_root)
            self.assertEqual(first['pip_calls'],[])
            self.assertEqual(second['pip_calls'],[])
            self.assertEqual(len(first['calls']),1)
            self.assertEqual(first['calls'][0][1],'-u')
            self.assertTrue(first['calls'][0][2].endswith('public_reference_audit.py'))
            first_zip, second_zip = first['downloads'][0],second['downloads'][0]
            self.assertNotEqual(first_zip,second_zip)
            for archive_path in (first_zip,second_zip):
                with zipfile.ZipFile(archive_path) as archive:
                    self.assertIsNone(archive.testzip())
                    self.assertIn('summary.json',archive.namelist())
                    self.assertIn('reference_instances.csv',archive.namelist())
                    self.assertIn('execution.log',archive.namelist())
                    self.assertIn('code/source_manifest.json',archive.namelist())
                    self.assertFalse(any(name.endswith('.pt') or 'submission_pipeline' in name for name in archive.namelist()))
            self.assertEqual(snapshots(run_root),before)

    def test_only_missing_analysis_dependency_is_installed(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root,audit_root = self.prepare(Path(temporary))
            result = self.execute(run_root,audit_root,absent=('pycocotools',))
            self.assertEqual(result['pip_calls'],[[sys.executable,'-m','pip','install','pycocotools']])

    def test_failure_does_not_package_or_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_root,audit_root = self.prepare(Path(temporary))
            before = snapshots(run_root)
            with self.assertRaisesRegex(RuntimeError,'진단 종료 코드 1'):
                self.execute(run_root,audit_root,status=1)
            self.assertEqual(list(audit_root.glob('review_*.zip')),[])
            self.assertEqual(snapshots(run_root),before)


if __name__ == '__main__':unittest.main()
