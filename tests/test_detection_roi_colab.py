"""DETR/ROI stage scopes, durable restore, visible failures and CUDA pair safety."""
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
import build_detection_roi_colab as builder


def bootstrap_namespace():
    namespace={'__name__':'bootstrap_fixture'}
    exec(compile(builder.BOOTSTRAP,'bootstrap-fixture','exec'),namespace)
    return namespace


class DetectionRoiColabTests(unittest.TestCase):
    def fixture(self,temporary):
        sources={name:'# detection ROI fixture '+name+'\n' for name in builder.MODULES}
        sources[builder.BOOTSTRAP_NAME]=builder.BOOTSTRAP
        with patch.object(builder,'ROOT',temporary/'repo'),patch.object(builder,'build_sources',return_value=sources):
            metadata=builder.build()
        drive=temporary/'drive';content=temporary/'content'
        first=Path(metadata['copy_cell']).read_text().replace('/content/drive/MyDrive',str(drive))
        resume=Path(metadata['resume_cell']).read_text().replace('/content/drive/MyDrive',str(drive))
        root=drive/'filament-segmentation-2026/fixed_split_experiments/detection_roi_detr_v1'
        google=types.ModuleType('google');colab=types.ModuleType('google.colab');google.colab=colab
        downloads=[];colab.drive=types.SimpleNamespace(mount=lambda _:None);colab.files=types.SimpleNamespace(download=downloads.append)
        commands=[];restored=[];ns=bootstrap_namespace()
        class FakeProcess:
            def __init__(self,command,**kwargs):
                commands.append(command)
                cfg,_,work=ns['restore'](Path(command[command.index('--root')+1]),content)
                restored.append((cfg,work));run_root=Path(cfg['output_root'])
                archive=run_root/'fixture_results.zip';archive.write_bytes(b'fixture archive')
                (run_root/'result.json').write_text(json.dumps(dict(status='fixture_complete',phase=command[command.index('--phase')+1],archives=[str(archive)])))
                self.pid=98765;self.stdout=io.StringIO('fixture bootstrap output and stderr\n')
            def wait(self,timeout=None):return 0
        def run(cell,roi_until=30,detr_until=30,phase='all'):
            text=cell.replace('ROI_UNTIL = 30 ',f'ROI_UNTIL = {roi_until} ').replace('DETR_UNTIL = 30 ',f'DETR_UNTIL = {detr_until} ').replace("PHASE = 'all'",'PHASE = '+repr(phase))
            output=io.StringIO()
            with patch.dict(sys.modules,{'google':google,'google.colab':colab}),patch('subprocess.Popen',FakeProcess),redirect_stdout(output):
                exec(compile(text,'colab-fixture','exec'),{})
            return output.getvalue()
        return dict(first=first,resume=resume,root=root,content=content,sources=sources,commands=commands,restored=restored,downloads=downloads,run=run)

    def test_default_all_has_two_real30_schedules_and_distinct_stage_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));f['run'](f['first'],roi_until=10,detr_until=20)
            original=(f['root']/'experiment_config.json').read_bytes()
            shutil.rmtree(f['content']);output=f['run'](f['resume'],30,30)
            self.assertEqual(original,(f['root']/'experiment_config.json').read_bytes())
            self.assertEqual([(c[c.index('--roi-until')+1],c[c.index('--detr-until')+1]) for c in f['commands']],[('10','20'),('30','30')])
            for cfg,work in f['restored']:
                self.assertEqual((cfg['roi_epochs'],cfg['detr_epochs']),(30,30))
                self.assertEqual((cfg['roi_size'],cfg['roi_context'],cfg['roi_batch'],cfg['roi_steps_per_epoch']),(384,1.5,4,256))
                self.assertEqual((cfg['detr_image_size'],cfg['detr_batch'],cfg['detr_accumulate']),(800,1,4))
                self.assertEqual(cfg['detr_model_name'],'facebook/detr-resnet-50')
                self.assertIn('yolo',cfg);self.assertNotIn('native',cfg);self.assertNotIn('synthetic_images',cfg)
                for name,source in f['sources'].items():self.assertEqual((work/name).read_text(),source)
            self.assertIn('fixture bootstrap output and stderr',output)
            self.assertIn('fixture bootstrap output and stderr',(f['root']/'colab_bootstrap.log').read_text())

    def test_phase_switch_does_not_mutate_saved_training_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));f['run'](f['first'],phase='yolo_roi')
            original=(f['root']/'experiment_config.json').read_bytes()
            f['run'](f['resume'],phase='compare_detr');f['run'](f['resume'],phase='evaluate')
            self.assertEqual(original,(f['root']/'experiment_config.json').read_bytes())
            self.assertEqual([x[x.index('--phase')+1] for x in f['commands']],['yolo_roi','compare_detr','evaluate'])
            with self.assertRaisesRegex(ValueError,'새 OUTPUT'):
                f['run'](f['first'].replace("ROI_ENCODER = 'resnet34'","ROI_ENCODER = 'resnet50'"))
            self.assertEqual(len(f['commands']),3)

    def test_source_or_config_tampering_blocks_before_child(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));f['run'](f['first'])
            config=f['root']/'experiment_config.json';original=config.read_bytes();config.write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError,'설정'):f['run'](f['resume'])
            config.write_bytes(original)
            pointer=json.loads((f['root']/'current_code.json').read_text())
            source=f['root']/'code'/pointer['version']/builder.MODULES[0];source.write_text('changed')
            with self.assertRaisesRegex(ValueError,'코드'):f['run'](f['resume'])
            self.assertEqual(len(f['commands']),1)

    def test_initial_failure_stderr_is_visible_and_in_persistent_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);durable=root/'code';durable.mkdir()
            (durable/builder.BOOTSTRAP_NAME).write_text("import sys\nprint('fixture failure before dependency install',file=sys.stderr,flush=True)\nraise SystemExit(1)\n")
            namespace=dict(sys=sys,os=os,signal=signal,subprocess=subprocess,root=root,durable=durable,
                           PHASE='all',ROI_UNTIL=30,DETR_UNTIL=30,datetime=datetime,timezone=timezone)
            output=io.StringIO()
            with redirect_stdout(output),self.assertRaises(subprocess.CalledProcessError):
                exec(compile(builder.EXECUTE,'first-execute','exec'),namespace)
            self.assertIn('fixture failure before dependency install',output.getvalue())
            self.assertIn('fixture failure before dependency install',(root/'colab_bootstrap.log').read_text())
            self.assertEqual(namespace['process'].poll(),1)

    def test_dependency_pins_include_actual_detr_without_replacing_cuda_torch_pair(self):
        ns=bootstrap_namespace()
        def version(name):
            if name=='torch':return '2.11.0+cu130'
            if name=='torchvision':return '0.26.0+cu130'
            from importlib.metadata import PackageNotFoundError
            raise PackageNotFoundError(name)
        with tempfile.TemporaryDirectory() as directory,patch('importlib.metadata.version',side_effect=version),patch('importlib.util.find_spec',return_value=object()),patch('subprocess.run') as run:
            ns['ensure_dependencies'](Path(directory));command=run.call_args.args[0]
            self.assertIn('transformers==4.57.1',command);self.assertIn('ultralytics==8.4.182',command)
            self.assertIn('segmentation-models-pytorch==0.5.0',command)
            self.assertFalse(any(value.startswith(('torch==','torchvision==')) for value in command))
            self.assertEqual(Path(command[command.index('-c')+1]).read_text(),'torch==2.11.0+cu130\ntorchvision==0.26.0+cu130\n')

    def test_only_original_data_and_existing_yolo_are_required_paths(self):
        ns=bootstrap_namespace()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg={}
            for key in ('annotations','train_split','val_split','yolo'):
                path=root/key;path.write_text('fixture');cfg[key]=str(path)
            (root/'images').mkdir();cfg['images']=str(root/'images')
            ns['check_paths'](cfg)
            Path(cfg['yolo']).unlink()
            with self.assertRaisesRegex(FileNotFoundError,'yolo'):ns['check_paths'](cfg)

    @unittest.skipUnless(os.name=='posix','process group cancellation requires POSIX')
    def test_stopping_bootstrap_reaps_separate_pipeline_worker(self):
        ns=bootstrap_namespace()
        with tempfile.TemporaryDirectory() as directory:
            tmp=Path(directory);root=tmp/'run';version='a'*16;durable=root/'code'/version;durable.mkdir(parents=True)
            pidfile=tmp/'worker_pid';worker="from pathlib import Path\nimport os,time\nPath("+repr(str(pidfile))+").write_text(str(os.getpid()))\ntime.sleep(60)\n"
            (durable/'roi_filament_pipeline.py').write_text(worker)
            manifest={'roi_filament_pipeline.py':hashlib.sha256(worker.encode()).hexdigest()}
            mpath=durable/'source_manifest.json';mpath.write_text(json.dumps(manifest))
            cfg=dict(output_root=str(root),roi_epochs=30,detr_epochs=30,roi_steps_per_epoch=256,roi_batch=4)
            for key in ('annotations','train_split','val_split','yolo'):
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
                +'signal.signal(signal.SIGTERM,b.interrupted)\nb.run(Path('+repr(str(root))+'),"yolo_roi",1,1)\n')
            process=subprocess.Popen([sys.executable,str(wrapper)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            try:
                deadline=time.monotonic()+8
                while not pidfile.exists() and time.monotonic()<deadline:
                    if process.poll() is not None:break
                    time.sleep(.05)
                self.assertTrue(pidfile.exists(),'pipeline worker did not start')
                worker_pid=int(pidfile.read_text());os.killpg(process.pid,signal.SIGTERM);process.communicate(timeout=12)
                with self.assertRaises(ProcessLookupError):os.kill(worker_pid,0)
            finally:
                if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.communicate(timeout=5)
                if pidfile.exists():
                    try:os.kill(int(pidfile.read_text()),signal.SIGKILL)
                    except ProcessLookupError:pass

    def test_real_bundle_source_notebook_and_default_budget_match(self):
        if not all((builder.ROOT/'tools'/name).is_file() for name in builder.MODULES):
            self.skipTest('Pipeline implementation is still being authored')
        result=builder.build();cell=Path(result['copy_cell']).read_text();tree=ast.parse(cell)
        payload=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='PAYLOAD' for target in node.targets))
        raw=zlib.decompress(base64.b64decode(payload));sources=json.loads(raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),result['source_bundle_sha256'])
        self.assertEqual(set(sources),set(builder.MODULES)|{builder.BOOTSTRAP_NAME})
        for name in builder.MODULES:self.assertEqual(sources[name],(builder.ROOT/'tools'/name).read_text())
        self.assertEqual(result['default_total_training_epochs'],60)
        notebook=json.loads(Path(result['notebook']).read_text())
        self.assertEqual(cell,''.join(notebook['cells'][1]['source']))
        self.assertEqual(Path(result['resume_cell']).read_text(),''.join(notebook['cells'][3]['source']))


if __name__=='__main__':unittest.main()
