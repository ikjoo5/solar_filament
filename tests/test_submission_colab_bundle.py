"""The user's paste cell embeds the actual reviewed portable production code."""
import ast
import base64
import hashlib
import importlib.util
import json
import lzma
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('submission_builder',ROOT/'tools/build_submission_colab.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


class SubmissionBundleTests(unittest.TestCase):
    def test_generated_copy_cell_and_notebook_are_current(self):
        cell,raw=builder.build_cell()
        self.assertEqual(cell,(ROOT/'notebooks/SolarFilament_Submission_CopyCell.py').read_text())
        notebook=json.loads((ROOT/'notebooks/SolarFilament_Submission_Frozen.ipynb').read_text())
        self.assertEqual(cell,''.join(notebook['cells'][1]['source']))
        compile(cell,'copy-cell','exec')

    def test_integrity_pinned_payload_is_exact_reviewed_code(self):
        cell,raw=builder.build_cell()
        tree=ast.parse(cell)
        payload=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in n.targets))
        decoded=lzma.decompress(base64.b64decode(payload))
        self.assertEqual(decoded,raw)
        self.assertIn(hashlib.sha256(raw).hexdigest(),cell)
        files=json.loads(decoded)
        self.assertEqual(set(files),set(builder.MODULES))
        for name,source in files.items():
            self.assertEqual(source,(ROOT/'tools'/name).read_text())
            self.assertNotIn('/workspace',source)
            self.assertNotIn('filament_eval',source)
            compile(source,name,'exec')

    def test_known_paths_fresh_subprocess_and_one_csv_download(self):
        cell,_=builder.build_cell()
        self.assertIn('/MAGFiLO_1.0_Kaggle_2026/test/test_images',cell)
        self.assertIn('20261028T030808Z_bedf3e21',cell)
        self.assertIn('ultralytics==8.4.182',cell)
        self.assertIn('subprocess.run([sys.executable',cell)
        self.assertEqual(cell.count('files.download('),1)
        self.assertNotIn('import torch',cell)


if __name__=='__main__':unittest.main()
