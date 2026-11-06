"""Execute actual Colab cells with a fake child, never GPU/pip/Drive network."""
from contextlib import contextmanager, redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile
from pycocotools import mask as mu

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
import build_r1_submission_colab as builder
import r1_submission_colab_bootstrap as bootstrap


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Child:
    pid=987654321
    def __init__(self,code=0):
        self.code=code
        self.stdout=io.StringIO('fixture child stdout and stderr\n'+
            ('Traceback: ModuleNotFoundError: fixture_inner_error\n' if code else ''))
    def wait(self,timeout=None):return self.code
    def poll(self):return self.code
    def terminate(self):pass
    def kill(self):pass


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory(prefix='r1_colab_') as tmp:
        base=Path(tmp)/'Drive';images=base/'MAGFiLO_1.0_Kaggle_2026/test/test_images'
        images.mkdir(parents=True)
        checkpoints=Path(tmp)/'weights';checkpoints.mkdir()
        yolo=checkpoints/'yolo.pt';native=checkpoints/'native.pt'
        yolo.write_bytes(b'fixture yolo');native.write_bytes(b'fixture native')
        names=json.loads(builder.NAMES.read_text())['test_images']
        for name in names:(images/name).write_bytes(b'fixture image only')
        sources={name:(TOOLS/name).read_text() for name in builder.MODULES}
        assets={'official_test_names.json':builder.encoded(names)}
        # A fake newly inferred mask, never an input asset or historical submission.
        mask=mu.frPyObjects({'size':[2048,2048],'counts':[100,400,2048**2-500]},2048,2048)
        stream=io.StringIO(newline='');writer=csv.writer(stream)
        writer.writerow(['filament_id','segmentation_rle'])
        writer.writerow([Path(names[0]).stem+'_1',mask['counts'].decode()])
        fresh_baseline=stream.getvalue().encode()
        rendered=builder.render_cells(sources,assets)
        first=rendered['first'].replace("BASE = '/content/drive/MyDrive/filament-segmentation-2026'",'BASE = '+repr(str(base)))
        first=first.replace("YOLO_CHECKPOINT = '/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/best.pt'",'YOLO_CHECKPOINT = '+repr(str(yolo)))
        first=first.replace("NATIVE_CHECKPOINT = '/content/drive/MyDrive/SolarFilament_NativeUNet_A100/20261028T030808Z_bedf3e21/best.pt'",'NATIVE_CHECKPOINT = '+repr(str(native)))
        root=base/'submissions/best84_pipeline_r1_t4_v1'
        resume=rendered['resume'].replace("OUTPUT = '/content/drive/MyDrive/filament-segmentation-2026/submissions/best84_pipeline_r1_t4_v1'",'OUTPUT = '+repr(str(root)))
        downloads=[];commands=[];local=Path(tmp)/'content';mode={'status':'complete','code':0,'tamper':False}
        google=types.ModuleType('google');colab=types.ModuleType('google.colab')
        colab.drive=types.SimpleNamespace(mount=lambda *a,**k:None)
        colab.files=types.SimpleNamespace(download=lambda p:downloads.append(str(p)))
        google.colab=colab
        def popen(command,**kwargs):
            commands.append(command)
            cfg,_,work=bootstrap.restore(root,work_root=local)
            bootstrap.verify_inputs(cfg)
            assert all((work/n).read_text()==sources[n] for n in sources)
            if mode['code']:return Child(mode['code'])
            if mode['status']=='complete':
                baseline=root/'submission_pipeline_baseline.csv';r1=root/'submission_pipeline_R1.csv'
                baseline.write_bytes(fresh_baseline);r1.write_bytes(baseline.read_bytes())
                change=root/'change_audit.csv';change.write_text('filament_id,changed\n')
                manifest={'status':'complete','identity':'fixture-identity','baseline_source':'same_run_frozen_pipeline',
                    'output_sha256':{baseline.name:sha(baseline),r1.name:sha(r1),change.name:sha(change)},'mask_count':1,'official_images':180}
                (root/'run_manifest.json').write_text(json.dumps(manifest))
                archive=root/'results.zip'
                with zipfile.ZipFile(archive,'w') as z:
                    for p in (baseline,r1,root/'run_manifest.json'):z.write(p,p.name)
                result=dict(status='complete',identity='fixture-identity',completed_images=180,expected_images=180,
                    baseline_csv=str(baseline),r1_csv=str(r1),archive=str(archive),archive_sha256=sha(archive),submission_written=True)
                if mode['tamper']:r1.write_bytes(r1.read_bytes()+b'\n')
            else:
                result=dict(status='partial',completed_images=2,expected_images=180,archive=None,submission_written=False)
            (root/'latest_results.json').write_text(json.dumps(result))
            return Child()
        def run(source):
            output=io.StringIO()
            with patch.dict(sys.modules,{'google':google,'google.colab':colab}), \
                 patch('subprocess.Popen',side_effect=popen),patch('os.killpg'),redirect_stdout(output):
                exec(compile(source,'fixture_actual_colab_cell','exec'),{})
            return output.getvalue()
        yield types.SimpleNamespace(root=root,local=local,first=first,resume=resume,run=run,
            downloads=downloads,commands=commands,mode=mode,sources=sources,assets=assets,rendered=rendered,images=images,fresh_baseline=fresh_baseline)


class ColabBundleTests(unittest.TestCase):
    def test_only_official_names_asset_without_submission_input(self):
        with fixture() as f:
            bundle=json.loads(f.rendered['bundle_bytes'])
            self.assertEqual(set(bundle['assets_b64']),{'official_test_names.json'})
            self.assertEqual(len(json.loads(f.assets['official_test_names.json'])),180)
            self.assertNotIn('public055',f.rendered['bundle_bytes'].decode())
            self.assertIn('submission_driver.py',bundle['sources'])
            self.assertIn('submission_yolo_recipe.py',bundle['sources'])
            self.assertIn('submission_native_recipe.py',bundle['sources'])
            f.run(f.first)
            self.assertNotIn('baseline_csv',json.loads((f.root/'experiment_config.json').read_text()))

    def test_first_repeat_and_session_reset_resume(self):
        with fixture() as f:
            f.run(f.first)
            pointer=(f.root/'current_code.json').read_bytes()
            f.run(f.first)
            shutil.rmtree(f.local)
            f.run(f.resume)
            self.assertEqual((f.root/'current_code.json').read_bytes(),pointer)
            self.assertEqual(len(f.commands),3)
            self.assertEqual(len(f.downloads),9)
            self.assertEqual((f.root/'submission_pipeline_baseline.csv').read_bytes(),f.fresh_baseline)
            self.assertIn('fixture child stdout and stderr',(f.root/'colab_bootstrap.log').read_text())

    def test_partial_then_full_never_downloads_partial(self):
        with fixture() as f:
            f.mode['status']='partial';f.run(f.first.replace('MAX_IMAGES = None','MAX_IMAGES = 2'))
            self.assertEqual(f.downloads,[])
            self.assertFalse((f.root/'submission_pipeline_R1.csv').exists())
            self.assertIn('--max-images',f.commands[0])
            f.mode['status']='complete';f.run(f.resume)
            self.assertEqual(len(f.downloads),3)

    def test_immutable_names_and_source_tamper_rejected(self):
        for kind in ['asset','source','config']:
            with self.subTest(kind=kind),fixture() as f:
                f.run(f.first);f.downloads.clear()
                pointer=json.loads((f.root/'current_code.json').read_text())
                p={'asset':f.root/'assets/official_test_names.json',
                   'source':f.root/'code'/pointer['version']/'experimental_consensus_tails.py',
                   'config':f.root/'experiment_config.json'}[kind]
                p.write_bytes(p.read_bytes()+b'\n')
                with self.assertRaises(ValueError):f.run(f.resume)
                self.assertEqual(f.downloads,[])

    def test_wrong_config_cannot_overwrite_first_run(self):
        with fixture() as f:
            f.run(f.first);before=(f.root/'experiment_config.json').read_bytes()
            with self.assertRaises(ValueError):f.run(f.first.replace("precision='fp16'","precision='fp32'"))
            self.assertEqual((f.root/'experiment_config.json').read_bytes(),before)

    def test_inner_error_visible_in_persistent_log(self):
        with fixture() as f:
            f.mode['code']=1
            with self.assertRaisesRegex(RuntimeError,'종료 코드 1'):f.run(f.first)
            self.assertIn('ModuleNotFoundError: fixture_inner_error',(f.root/'colab_bootstrap.log').read_text())
            self.assertEqual(f.downloads,[])

    def test_all_missing_images_rejected_before_child(self):
        with fixture() as f:
            next(f.images.iterdir()).unlink()
            with self.assertRaisesRegex(FileNotFoundError,'test 이미지'):f.run(f.first)
            self.assertEqual(f.downloads,[])

    def test_changed_completed_output_is_not_downloaded(self):
        with fixture() as f:
            f.mode['tamper']=True
            with self.assertRaisesRegex(ValueError,'완료 후 파일이 변경'):f.run(f.first)
            self.assertEqual(f.downloads,[])

    def test_dependency_install_preserves_colab_torch(self):
        versions={'torch':'2.11.0+cu130','torchvision':'0.26.0+cu130',
                  'ultralytics':'8.4.100','segmentation-models-pytorch':'0.5.0',
                  'timm':'1.0.20','pycocotools':'2.0.11'}
        with tempfile.TemporaryDirectory() as tmp,patch('importlib.metadata.version',side_effect=versions.__getitem__), \
             patch('importlib.util.find_spec',return_value=object()),patch('subprocess.run') as run:
            bootstrap.ensure_dependencies(Path(tmp))
            command=run.call_args.args[0]
            self.assertIn('ultralytics==8.4.182',command)
            self.assertNotIn('torch',command)
            text=(Path(tmp)/'keep_colab_torch.txt').read_text()
            self.assertEqual(text,'torch==2.11.0+cu130\ntorchvision==0.26.0+cu130\n')


if __name__=='__main__':unittest.main()
