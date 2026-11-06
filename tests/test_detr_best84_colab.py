"""One-copy-cell seed durability, actual process failures and isolated imports."""
import ast
import base64
from contextlib import redirect_stdout
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import zipfile
import zlib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_detr_best84_colab as builder


def namespace():
    ns={'__name__':'detr_best84_bootstrap_fixture'}
    exec(compile(builder.BOOTSTRAP,'detr-best84-bootstrap','exec'),ns)
    return ns


class DetrBest84ColabTests(unittest.TestCase):
    def fixture(self,tmp):
        sources={name:'# fixture '+name+'\n' for name in builder.MODULES}
        sources[builder.BOOTSTRAP_NAME]=builder.BOOTSTRAP
        seed=gzip.compress(json.dumps({'frames':[],'scope':'fixture predictions only'}).encode(),mtime=0)
        seed_path=tmp/'seed.json.gz';seed_path.write_bytes(seed)
        with patch.object(builder,'ROOT',tmp/'repo'),patch.object(builder,'build_sources',return_value=sources):
            metadata=builder.build(downloads=tmp/'downloads',seed_path=seed_path)
        drive=tmp/'drive';content=tmp/'content'
        first=Path(metadata['copy_cell']).read_text().replace('/content/drive/MyDrive',str(drive))
        resume=Path(metadata['resume_cell']).read_text().replace('/content/drive/MyDrive',str(drive))
        parent=drive/'filament-segmentation-2026/fixed_split_experiments'
        root=parent/'detr_best84_t4_v1';source=parent/'detr_a100_e150_v1';base=parent/'detection_roi_detr_v1'
        source.mkdir(parents=True);base.mkdir()
        (source/'sentinel').write_text('completed source immutable')
        inputs=[drive/'SolarFilament_NativeUNet_A100/20261028T030808Z_bedf3e21/best.pt',
                base/'roi/epoch030.pt',
                drive/'filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json',
                drive/'SolarFilament_LimbOnly_YOLO26_mixed_FullRes/train_split.csv',
                drive/'SolarFilament_LimbOnly_YOLO26_mixed_FullRes/val_split.csv']
        for path in inputs:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture input')
        (drive/'filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/train_images').mkdir()
        google=types.ModuleType('google');colab=types.ModuleType('google.colab');google.colab=colab
        downloads=[];colab.drive=types.SimpleNamespace(mount=lambda _:None);colab.files=types.SimpleNamespace(download=downloads.append)
        commands=[];restored=[];ns=namespace()
        class FakeProcess:
            def __init__(self,command,**kwargs):
                commands.append(command)
                cfg,_,work=ns['restore'](Path(command[command.index('--root')+1]),content)
                restored.append((cfg,work));run_root=Path(cfg['output_root'])
                archive=run_root/'fixture_part01.zip';archive.write_bytes(b'fixture archive')
                count=int(command[-1]) if '--max-images' in command else 91
                (run_root/'latest_results.json').write_text(json.dumps(dict(status='complete' if count==91 else 'paused',
                    physical_images_completed=count,archives=[str(archive)])))
                self.pid=98765;self.stdout=io.StringIO('visible inner stdout and stderr\n')
            def wait(self,timeout=None):return 0
        def run(cell,cap=None):
            text=cell.replace('MAX_IMAGES = None ',f'MAX_IMAGES = {cap!r} ')
            output=io.StringIO()
            with patch.dict(sys.modules,{'google':google,'google.colab':colab}),patch('subprocess.Popen',FakeProcess),redirect_stdout(output):
                exec(compile(text,'detr-best84-colab-fixture','exec'),{})
            return output.getvalue()
        return dict(first=first,resume=resume,root=root,source=source,base=base,content=content,sources=sources,seed=seed,
                    commands=commands,restored=restored,downloads=downloads,run=run)

    def test_prefix_pause_and_complete_content_reset_preserve_seed_config(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));f['run'](f['first'],cap=2)
            config=(f['root']/'experiment_config.json').read_bytes()
            seed=(f['root']/'assets/prediction_seed.json.gz').read_bytes()
            shutil.rmtree(f['content'])
            output=f['run'](f['resume'])
            self.assertEqual(config,(f['root']/'experiment_config.json').read_bytes())
            self.assertEqual(seed,f['seed'])
            self.assertEqual(seed,(f['root']/'assets/prediction_seed.json.gz').read_bytes())
            self.assertIn('--max-images',f['commands'][0]);self.assertEqual(f['commands'][0][-1],'2')
            self.assertNotIn('--max-images',f['commands'][1])
            self.assertTrue(all('--run-until' not in command for command in f['commands']))
            for cfg,work in f['restored']:
                self.assertEqual((cfg['native_batch'],cfg['roi_batch']),(1,1))
                self.assertEqual((cfg['native_precision'],cfg['roi_precision']),('fp16','fp16'))
                self.assertTrue(cfg['strict']);self.assertFalse(cfg['include_selected_only'])
                for name,source in f['sources'].items():self.assertEqual((work/name).read_text(),source)
            self.assertIn('visible inner stdout and stderr',output)
            self.assertIn('visible inner stdout and stderr',(f['root']/'colab_bootstrap.log').read_text())
            self.assertEqual((f['source']/'sentinel').read_text(),'completed source immutable')
            self.assertEqual(len(f['downloads']),2)

    def test_first_cell_idempotent_and_invalid_prefix_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));f['run'](f['first'],cap=1)
            snapshot={str(p.relative_to(f['root'])):p.read_bytes() for p in f['root'].rglob('*')
                      if p.is_file() and (p.suffix in ('.py','.gz') or p.name in ('experiment_config.json','current_code.json'))}
            f['run'](f['first'])
            for name,content in snapshot.items():self.assertEqual((f['root']/name).read_bytes(),content)
            for cap in (0,92,-1,True,1.5):
                with self.assertRaisesRegex(ValueError,'MAX_IMAGES'):f['run'](f['resume'],cap=cap)
            self.assertEqual(len(f['commands']),2)

    def test_missing_input_does_not_lock_output_before_corrected_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory))
            wrong=f['first'].replace('/20261028T030808Z_bedf3e21/best.pt',
                '/20261028T030808Z_bedf3e21/typo.pt')
            with self.assertRaisesRegex(FileNotFoundError,'native_checkpoint'):
                f['run'](wrong)
            self.assertFalse((f['root']/'experiment_config.json').exists())
            self.assertFalse((f['root']/'current_code.json').exists())
            self.assertFalse(f['root'].exists())
            self.assertFalse(f['commands'])
            f['run'](f['first'],cap=2)
            self.assertEqual(len(f['commands']),1)
            cfg=json.loads((f['root']/'experiment_config.json').read_text())
            self.assertTrue(cfg['native_checkpoint'].endswith('/best.pt'))

    def test_manifest_asset_and_config_tampering_block_before_child(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));f['run'](f['first'])
            config=f['root']/'experiment_config.json';original=config.read_bytes();config.write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError,'설정'):f['run'](f['resume'])
            config.write_bytes(original)
            seed=f['root']/'assets/prediction_seed.json.gz';original=seed.read_bytes();seed.write_bytes(original+b'changed')
            with self.assertRaisesRegex(ValueError,'예측 입력'):f['run'](f['resume'])
            seed.write_bytes(original)
            pointer=json.loads((f['root']/'current_code.json').read_text())
            source=f['root']/'code'/pointer['version']/builder.MODULES[0];source.write_text('changed')
            with self.assertRaisesRegex(ValueError,'코드'):f['run'](f['resume'])
            self.assertEqual(len(f['commands']),1)

    def test_source_overlap_and_missing_inputs_are_explicit(self):
        ns=namespace()
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory))
            for protected in (f['source'],f['base']):
                for bad in (protected,protected/'nested',protected.parent):
                    first=f['first'].replace("OUTPUT = BASE + '/fixed_split_experiments/detr_best84_t4_v1'",
                        'OUTPUT = '+repr(str(bad)))
                    with self.assertRaisesRegex(ValueError,'OUTPUT'):f['run'](first)
            cfg={key:str(Path(directory)/key) for key in ('native_checkpoint','roi_checkpoint','annotations','train_split','val_split','seed_path','images')}
            with self.assertRaisesRegex(FileNotFoundError,'native_checkpoint'):ns['verify_inputs'](cfg)
            for key in cfg:
                if key=='images':Path(cfg[key]).mkdir()
                else:Path(cfg[key]).write_text('fixture')
            cfg.update(native_precision='fp16',roi_precision='fp16',native_batch=1,roi_batch=1)
            ns['verify_inputs'](cfg)
            cfg['native_precision']='bf16'
            with self.assertRaisesRegex(ValueError,'FP16'):ns['verify_inputs'](cfg)

    def test_early_inner_failure_is_visible_and_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);durable=root/'code';durable.mkdir()
            (durable/builder.BOOTSTRAP_NAME).write_text("import sys\nprint('fixture missing best84 checkpoint',file=sys.stderr,flush=True)\nraise SystemExit(1)\n")
            ns=dict(sys=sys,os=os,signal=signal,subprocess=subprocess,root=root,durable=durable,
                    MAX_IMAGES=None,datetime=datetime,timezone=timezone)
            output=io.StringIO()
            with redirect_stdout(output),self.assertRaisesRegex(RuntimeError,'전체 로그'):
                exec(compile(builder.EXECUTE,'detr-best84-execute','exec'),ns)
            self.assertIn('fixture missing best84 checkpoint',output.getvalue())
            self.assertIn('fixture missing best84 checkpoint',(root/'colab_bootstrap.log').read_text())
            self.assertEqual(ns['process'].poll(),1)

    def test_dependencies_keep_colab_torch_pair_without_unneeded_detectors(self):
        ns=namespace()
        def version(name):
            if name=='torch':return '2.11.0+cu130'
            if name=='torchvision':return '0.26.0+cu130'
            from importlib.metadata import PackageNotFoundError
            raise PackageNotFoundError(name)
        with tempfile.TemporaryDirectory() as directory,patch('importlib.metadata.version',side_effect=version),patch('importlib.util.find_spec',return_value=object()),patch('subprocess.run') as run:
            ns['ensure_dependencies'](Path(directory));command=run.call_args.args[0]
            self.assertIn('segmentation-models-pytorch==0.5.0',command)
            self.assertFalse(any(s.startswith(('torch==','torchvision==','transformers','ultralytics')) for s in command))
            self.assertEqual(Path(command[command.index('-c')+1]).read_text(),'torch==2.11.0+cu130\ntorchvision==0.26.0+cu130\n')
            self.assertIn("dtype=torch.float16",builder.BOOTSTRAP)
            self.assertNotIn('is_bf16_supported',builder.BOOTSTRAP)

    @unittest.skipUnless(os.name=='posix','requires POSIX process groups')
    def test_stopping_bootstrap_stops_inference_child(self):
        ns=namespace()
        with tempfile.TemporaryDirectory() as directory:
            tmp=Path(directory);root=tmp/'run';version='a'*16;durable=root/'code'/version;durable.mkdir(parents=True)
            pidfile=tmp/'worker_pid'
            worker="from pathlib import Path\nimport os,time\nPath("+repr(str(pidfile))+").write_text(str(os.getpid()))\ntime.sleep(60)\n"
            (durable/'detr_best84_pipeline.py').write_text(worker)
            manifest={'detr_best84_pipeline.py':hashlib.sha256(worker.encode()).hexdigest()}
            mpath=durable/'source_manifest.json';mpath.write_text(json.dumps(manifest))
            asset=root/'assets/prediction_seed.json.gz';asset.parent.mkdir();asset.write_bytes(b'fixture')
            cfg=dict(output_root=str(root),seed_path=str(asset),seed_sha256=ns['digest'](asset))
            cpath=root/'experiment_config.json';cpath.write_text(json.dumps(cfg))
            (root/'current_code.json').write_text(json.dumps(dict(version=version,manifest_sha256=ns['digest'](mpath),
                config_sha256=ns['digest'](cpath),seed_sha256=ns['digest'](asset))))
            module=tmp/'bootstrap.py';module.write_text(builder.BOOTSTRAP)
            wrapper=tmp/'wrapper.py';wrapper.write_text('import importlib.util,signal\nfrom pathlib import Path\n'
                +'spec=importlib.util.spec_from_file_location("b",'+repr(str(module))+')\n'
                +'b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)\n'
                +'old=b.restore\nb.restore=lambda root:old(root,Path('+repr(str(tmp/'content'))+'))\n'
                +'b.ensure_dependencies=lambda work:None\nb.check_gpu=lambda *args:None\nb.verify_inputs=lambda *args:None\n'
                +'signal.signal(signal.SIGTERM,b.interrupted)\nb.run(Path('+repr(str(root))+'),None)\n')
            process=subprocess.Popen([sys.executable,str(wrapper)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            try:
                deadline=time.monotonic()+8
                while not pidfile.exists() and time.monotonic()<deadline:
                    if process.poll() is not None:break
                    time.sleep(.05)
                self.assertTrue(pidfile.exists(),'inference worker did not start')
                worker_pid=int(pidfile.read_text());os.killpg(process.pid,signal.SIGTERM);process.communicate(timeout=12)
                with self.assertRaises(ProcessLookupError):os.kill(worker_pid,0)
            finally:
                if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.communicate(timeout=5)
                if pidfile.exists():
                    try:os.kill(int(pidfile.read_text()),signal.SIGKILL)
                    except ProcessLookupError:pass

    def test_actual_payload_notebook_and_readonly_seed_contract(self):
        if not all((builder.ROOT/'tools'/name).is_file() for name in builder.MODULES):self.skipTest('Engines still being authored')
        actual_root=builder.ROOT
        actual_sources=builder.build_sources()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(builder,'ROOT',Path(directory)/'repo'),patch.object(builder,'build_sources',return_value=actual_sources):
                metadata=builder.build(downloads=Path(directory)/'downloads')
            cell=Path(metadata['copy_cell']).read_text();tree=ast.parse(cell)
            payload=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in n.targets))
            raw=zlib.decompress(base64.b64decode(payload));bundle=json.loads(raw)
            self.assertEqual(hashlib.sha256(raw).hexdigest(),metadata['source_bundle_sha256'])
            self.assertEqual(set(bundle['sources']),set(builder.MODULES)|{builder.BOOTSTRAP_NAME})
            self.assertEqual(base64.b64decode(bundle['seed_gzip_b64']),builder.SEED_PATH.read_bytes())
            for name in builder.MODULES:self.assertEqual(bundle['sources'][name],(actual_root/'tools'/name).read_text())
            self.assertEqual(metadata['additional_training_epochs'],0)
            self.assertEqual(metadata['actual_public_LB'],.39)
            self.assertFalse(metadata['test_inference']);self.assertFalse(metadata['auto_Kaggle_submission'])
            notebook=json.loads(Path(metadata['notebook']).read_text())
            self.assertEqual(cell,''.join(notebook['cells'][1]['source']))
            self.assertEqual(Path(metadata['resume_cell']).read_text(),''.join(notebook['cells'][3]['source']))
            with zipfile.ZipFile(metadata['package']) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(json.loads(archive.read('source/source_manifest.json')),metadata['source_sha256'])
                self.assertEqual(archive.read('assets/prediction_seed.json.gz'),builder.SEED_PATH.read_bytes())

    def test_imports_in_isolated_folder_without_old_content(self):
        if not all((builder.ROOT/'tools'/name).is_file() for name in builder.MODULES):self.skipTest('Engines still being authored')
        with tempfile.TemporaryDirectory() as directory:
            isolated=Path(directory)
            for name,source in builder.build_sources().items():(isolated/name).write_text(source)
            script=('import sys; sys.path.insert(0,sys.argv[1]); '
                    'import detr_best84_pipeline,detr_best84_tiles; '
                    'assert callable(detr_best84_pipeline.run); '
                    'assert callable(detr_best84_tiles.infer_native_variants); print("isolated import ok")')
            result=subprocess.run([sys.executable,'-I','-c',script,str(isolated)],cwd=isolated,
                                  stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout)
            self.assertIn('isolated import ok',result.stdout)


if __name__=='__main__':unittest.main()
