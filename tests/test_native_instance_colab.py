"""Durable one-model experiment, fresh-session restore and visible failures."""
import ast
import base64
from contextlib import redirect_stdout
from datetime import datetime, timezone
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
import zlib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_native_instance_colab as builder


def bootstrap_namespace():
    namespace={'__name__':'bootstrap_fixture'}
    exec(compile(builder.BOOTSTRAP,'bootstrap-fixture','exec'),namespace)
    return namespace


class NativeInstanceColabTests(unittest.TestCase):
    def fixture(self,temporary):
        sources={name:'# native instance fixture '+name+'\n' for name in builder.MODULES}
        sources[builder.BOOTSTRAP_NAME]=builder.BOOTSTRAP
        with patch.object(builder,'ROOT',temporary/'repo'),patch.object(builder,'build_sources',return_value=sources):
            metadata=builder.build()
        drive=temporary/'drive';content=temporary/'content'
        first=Path(metadata['copy_cell']).read_text().replace('/content/drive/MyDrive',str(drive))
        resume=Path(metadata['resume_cell']).read_text().replace('/content/drive/MyDrive',str(drive))
        root=drive/'filament-segmentation-2026/fixed_split_experiments/native_instance_unet_v1/resnet34_instance'
        google=types.ModuleType('google');colab=types.ModuleType('google.colab');google.colab=colab
        downloads=[];colab.drive=types.SimpleNamespace(mount=lambda _:None);colab.files=types.SimpleNamespace(download=downloads.append)
        commands=[];restored=[];ns=bootstrap_namespace()
        class FakeProcess:
            def __init__(self,command,**kwargs):
                commands.append(command)
                cfg,_,work=ns['restore'](Path(command[command.index('--root')+1]),content)
                restored.append((cfg,work));run_root=Path(cfg['output_root'])
                archive=run_root/'fixture_results.zip';archive.write_bytes(b'fixture archive')
                (run_root/'result.json').write_text(json.dumps(dict(status='fixture_complete',epochs_completed=30,archives=[str(archive)])))
                self.pid=98765;self.stdout=io.StringIO('fixture child stdout and stderr\n')
            def wait(self,timeout=None):return 0
        def run(cell,until=30):
            text=cell.replace('RUN_UNTIL = 30 ',f'RUN_UNTIL = {until} ')
            output=io.StringIO()
            with patch.dict(sys.modules,{'google':google,'google.colab':colab}),patch('subprocess.Popen',FakeProcess),redirect_stdout(output):
                exec(compile(text,'colab-fixture','exec'),{})
            return output.getvalue()
        return dict(first=first,resume=resume,root=root,content=content,sources=sources,commands=commands,restored=restored,downloads=downloads,run=run)

    def test_fresh_resume_restores_after_content_deletion_without_reinitializing_config(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));f['run'](f['first'],10)
            original=(f['root']/'experiment_config.json').read_bytes()
            shutil.rmtree(f['content']);output=f['run'](f['resume'],30)
            self.assertEqual(original,(f['root']/'experiment_config.json').read_bytes())
            self.assertEqual([x[x.index('--run-until')+1] for x in f['commands']],['10','30'])
            for cfg,work in f['restored']:
                self.assertEqual((cfg['encoder'],cfg['head_mode'],cfg['epochs']),('resnet34','instance',30))
                self.assertEqual((cfg['crop'],cfg['batch'],cfg['accumulate'],cfg['steps_per_epoch']),(768,2,2,256))
                self.assertNotIn('yolo',cfg);self.assertNotIn('native',cfg);self.assertNotIn('synthetic_images',cfg)
                for name,source in f['sources'].items():self.assertEqual((work/name).read_text(),source)
            self.assertIn('fixture child stdout and stderr',output)
            self.assertIn('fixture child stdout and stderr',(f['root']/'colab_bootstrap.log').read_text())

    def test_changing_encoder_gets_its_own_experiment_but_old_settings_cannot_change(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));f['run'](f['first'])
            with self.assertRaisesRegex(ValueError,'새 OUTPUT'):
                f['run'](f['first'].replace('FINAL_EPOCHS = 30 ','FINAL_EPOCHS = 60 '))
            f['run'](f['first'].replace("ENCODER = 'resnet34'", "ENCODER = 'resnet18'"))
            roots={cfg['output_root'] for cfg,_ in f['restored']}
            self.assertEqual(len(roots),2)
            self.assertTrue(any(path.endswith('/resnet18_instance') for path in roots))
            self.assertEqual(len(f['commands']),2)

    def test_source_and_config_tampering_are_rejected_before_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));f['run'](f['first'])
            config=f['root']/'experiment_config.json';original=config.read_bytes();config.write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError,'설정'):f['run'](f['resume'])
            config.write_bytes(original)
            pointer=json.loads((f['root']/'current_code.json').read_text())
            source=f['root']/'code'/pointer['version']/builder.MODULES[0];source.write_text('changed')
            with self.assertRaisesRegex(ValueError,'코드'):f['run'](f['resume'])
            self.assertEqual(len(f['commands']),1)

    def test_outer_first_cell_streams_and_logs_early_stderr_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);durable=root/'code';durable.mkdir()
            (durable/builder.BOOTSTRAP_NAME).write_text("import sys\nprint('fixture missing input before worker',file=sys.stderr,flush=True)\nraise SystemExit(1)\n")
            namespace=dict(sys=sys,os=os,signal=signal,subprocess=subprocess,root=root,durable=durable,
                           PHASE='all',RUN_UNTIL=30,datetime=datetime,timezone=timezone)
            output=io.StringIO()
            with redirect_stdout(output),self.assertRaises(subprocess.CalledProcessError):
                exec(compile(builder.EXECUTE,'first-cell-execute','exec'),namespace)
            self.assertIn('fixture missing input before worker',output.getvalue())
            self.assertIn('fixture missing input before worker',(root/'colab_bootstrap.log').read_text())
            self.assertEqual(namespace['process'].poll(),1)

    def test_dependency_setup_constrains_existing_colab_cuda_torch_pair(self):
        ns=bootstrap_namespace()
        def version(name):
            if name=='torch':return '2.11.0+cu130'
            if name=='torchvision':return '0.26.0+cu130'
            from importlib.metadata import PackageNotFoundError
            raise PackageNotFoundError(name)
        with tempfile.TemporaryDirectory() as directory,patch('importlib.metadata.version',side_effect=version),patch('importlib.util.find_spec',return_value=object()),patch('subprocess.run') as run:
            ns['ensure_dependencies'](Path(directory));command=run.call_args.args[0]
            self.assertIn('segmentation-models-pytorch==0.5.0',command)
            self.assertIn('timm==1.0.20',command)
            self.assertNotIn('ultralytics',str(command))
            self.assertFalse(any(value.startswith(('torch==','torchvision==')) for value in command))
            self.assertEqual(Path(command[command.index('-c')+1]).read_text(),'torch==2.11.0+cu130\ntorchvision==0.26.0+cu130\n')

    def test_synthetic_or_model_weights_are_not_required_inputs(self):
        ns=bootstrap_namespace()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg={}
            for key in ('annotations','train_split','val_split'):
                path=root/key;path.write_text('fixture');cfg[key]=str(path)
            (root/'images').mkdir();cfg['images']=str(root/'images')
            ns['check_paths'](cfg)
            Path(cfg['val_split']).unlink()
            with self.assertRaisesRegex(FileNotFoundError,'val_split'):ns['check_paths'](cfg)

    @unittest.skipUnless(os.name=='posix','process group cancellation requires POSIX')
    def test_stopping_bootstrap_also_stops_its_separate_training_worker(self):
        ns=bootstrap_namespace()
        with tempfile.TemporaryDirectory() as directory:
            tmp=Path(directory);root=tmp/'run';version='a'*16;durable=root/'code'/version;durable.mkdir(parents=True)
            pidfile=tmp/'worker_pid';worker="from pathlib import Path\nimport os,time\nPath("+repr(str(pidfile))+").write_text(str(os.getpid()))\ntime.sleep(60)\n"
            (durable/'native_instance_training.py').write_text(worker)
            manifest={'native_instance_training.py':hashlib.sha256(worker.encode()).hexdigest()}
            mpath=durable/'source_manifest.json';mpath.write_text(json.dumps(manifest))
            cfg=dict(output_root=str(root),epochs=30,encoder='resnet34',head_mode='instance',steps_per_epoch=256,batch=2,accumulate=2)
            for key in ('annotations','train_split','val_split'):
                path=tmp/key;path.write_text('fixture');cfg[key]=str(path)
            (tmp/'images').mkdir();cfg['images']=str(tmp/'images')
            cpath=root/'experiment_config.json';cpath.write_text(json.dumps(cfg))
            (root/'current_code.json').write_text(json.dumps(dict(version=version,manifest_sha256=ns['digest'](mpath),config_sha256=ns['digest'](cpath))))
            module=tmp/'bootstrap.py';module.write_text(builder.BOOTSTRAP)
            wrapper=tmp/'wrapper.py';wrapper.write_text('import importlib.util,signal\nfrom pathlib import Path\n'
                +'spec=importlib.util.spec_from_file_location("fixture_bootstrap",'+repr(str(module))+')\n'
                +'b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)\n'
                +'original_restore=b.restore\nb.restore=lambda root: original_restore(root,Path('+repr(str(tmp/'content'))+'))\n'
                +'b.ensure_dependencies=lambda work: None\nb.check_gpu=lambda *args: None\n'
                +'signal.signal(signal.SIGTERM,b.interrupted)\nb.run(Path('+repr(str(root))+'),"train",1)\n')
            process=subprocess.Popen([sys.executable,str(wrapper)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            try:
                deadline=time.monotonic()+8
                while not pidfile.exists() and time.monotonic()<deadline:
                    if process.poll() is not None:break
                    time.sleep(.05)
                self.assertTrue(pidfile.exists(),'worker did not start')
                worker_pid=int(pidfile.read_text());os.killpg(process.pid,signal.SIGTERM);process.communicate(timeout=12)
                with self.assertRaises(ProcessLookupError):os.kill(worker_pid,0)
            finally:
                if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.communicate(timeout=5)
                if pidfile.exists():
                    try:os.kill(int(pidfile.read_text()),signal.SIGKILL)
                    except ProcessLookupError:pass

    def test_current_bundle_sources_and_notebook_match(self):
        if not all((builder.ROOT/'tools'/name).is_file() for name in builder.MODULES):
            self.skipTest('Implementation source modules still being authored')
        result=builder.build();cell=Path(result['copy_cell']).read_text();tree=ast.parse(cell)
        payload=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='PAYLOAD' for target in node.targets))
        raw=zlib.decompress(base64.b64decode(payload));sources=json.loads(raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),result['source_bundle_sha256'])
        self.assertEqual(set(sources),set(builder.MODULES)|{builder.BOOTSTRAP_NAME})
        for name in builder.MODULES:self.assertEqual(sources[name],(builder.ROOT/'tools'/name).read_text())
        self.assertEqual(sources[builder.BOOTSTRAP_NAME],builder.BOOTSTRAP)
        notebook=json.loads(Path(result['notebook']).read_text())
        self.assertEqual(cell,''.join(notebook['cells'][1]['source']))
        self.assertEqual(Path(result['resume_cell']).read_text(),''.join(notebook['cells'][3]['source']))


if __name__=='__main__':unittest.main()
