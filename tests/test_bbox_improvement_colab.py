"""Durable bbox experiment cells, source guards, budgets and visible failures."""
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
import build_bbox_improvement_colab as builder


def namespace():
    ns={'__name__':'bbox_bootstrap_fixture'}
    exec(compile(builder.BOOTSTRAP,'bbox-bootstrap','exec'),ns)
    return ns


class BboxImprovementColabTests(unittest.TestCase):
    def fixture(self,tmp):
        sources={name:'# bbox fixture '+name+'\n' for name in builder.MODULES}
        sources[builder.BOOTSTRAP_NAME]=builder.BOOTSTRAP
        with patch.object(builder,'ROOT',tmp/'repo'),patch.object(builder,'build_sources',return_value=sources):metadata=builder.build()
        drive=tmp/'drive';content=tmp/'content'
        first=Path(metadata['copy_cell']).read_text().replace('/content/drive/MyDrive',str(drive))
        resume=Path(metadata['resume_cell']).read_text().replace('/content/drive/MyDrive',str(drive))
        root=drive/'filament-segmentation-2026/fixed_split_experiments/bbox_improvement_v1'
        google=types.ModuleType('google');colab=types.ModuleType('google.colab');google.colab=colab
        downloads=[];colab.drive=types.SimpleNamespace(mount=lambda _:None);colab.files=types.SimpleNamespace(download=downloads.append)
        commands=[];restored=[];ns=namespace()
        class FakeProcess:
            def __init__(self,command,**kwargs):
                commands.append(command)
                cfg,_,work=ns['restore'](Path(command[command.index('--root')+1]),content)
                restored.append((cfg,work));run_root=Path(cfg['output_root'])
                archives=[]
                for i in (1,2):
                    archive=run_root/f'fixture_part{i:02d}.zip';archive.write_bytes(b'fixture archive');archives.append(str(archive))
                (run_root/'latest_results.json').write_text(json.dumps(dict(status='complete',phase=command[command.index('--phase')+1],archives=archives)))
                self.pid=98765;self.stdout=io.StringIO('visible inner stdout and stderr\n')
            def wait(self,timeout=None):return 0
        def run(cell,phase='all',added=30):
            source=cell.replace("PHASE = 'all'",'PHASE = '+repr(phase)).replace('RUN_UNTIL_ADDED = 30 ',f'RUN_UNTIL_ADDED = {added} ')
            output=io.StringIO()
            with patch.dict(sys.modules,{'google':google,'google.colab':colab}),patch('subprocess.Popen',FakeProcess),redirect_stdout(output):
                exec(compile(source,'bbox-colab-fixture','exec'),{})
            return output.getvalue()
        return dict(first=first,resume=resume,root=root,content=content,sources=sources,commands=commands,restored=restored,downloads=downloads,run=run)

    def test_all_adds_only30_detr_epochs_and_resume_restores_after_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));f['run'](f['first'],added=15)
            original=(f['root']/'experiment_config.json').read_bytes();shutil.rmtree(f['content'])
            output=f['run'](f['resume'],added=30)
            self.assertEqual(original,(f['root']/'experiment_config.json').read_bytes())
            self.assertEqual([c[c.index('--run-until-added')+1] for c in f['commands']],['15','30'])
            self.assertEqual([c[c.index('--phase')+1] for c in f['commands']],['all','all'])
            for cfg,work in f['restored']:
                self.assertEqual(cfg['added_epochs'],30);self.assertNotIn('roi_epochs',cfg);self.assertNotIn('yolo_epochs',cfg)
                self.assertEqual((cfg['tile_size'],cfg['tile_stride'],cfg['tile_max_det']),(1024,768,200))
                self.assertEqual((cfg['continuation_image_size'],cfg['continuation_batch'],cfg['continuation_accumulate'],cfg['continuation_precision']),(1280,2,2,'bf16'))
                self.assertNotEqual(cfg['base_root'],cfg['output_root'])
                for name,source in f['sources'].items():self.assertEqual((work/name).read_text(),source)
            self.assertEqual(len(f['downloads']),4)
            self.assertIn('visible inner stdout and stderr',output)
            self.assertIn('visible inner stdout and stderr',(f['root']/'colab_bootstrap.log').read_text())

    def test_phase_changes_keep_config_and_base_folder_unmodified(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));f['run'](f['first'],phase='probes')
            saved=(f['root']/'experiment_config.json').read_bytes()
            f['run'](f['resume'],phase='extend',added=15);f['run'](f['first'],phase='all',added=30)
            self.assertEqual(saved,(f['root']/'experiment_config.json').read_bytes())
            self.assertFalse(Path(f['restored'][0][0]['base_root']).exists())
            self.assertEqual([c[c.index('--phase')+1] for c in f['commands']],['probes','extend','all'])

    def test_manifest_and_config_tampering_stop_before_child(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));f['run'](f['first'])
            config=f['root']/'experiment_config.json';old=config.read_bytes();config.write_bytes(old+b' ')
            with self.assertRaisesRegex(ValueError,'설정'):f['run'](f['resume'])
            config.write_bytes(old)
            pointer=json.loads((f['root']/'current_code.json').read_text())
            source=f['root']/'code'/pointer['version']/builder.MODULES[0];source.write_text('changed')
            with self.assertRaisesRegex(ValueError,'코드'):f['run'](f['resume'])
            self.assertEqual(len(f['commands']),1)

    def test_original_checkpoint_and_source_guards_before_dependency_setup(self):
        ns=namespace()
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);source=base/'code'/('a'*16);source.mkdir(parents=True)
            module=source/'fixture.py';module.write_text('# source\n')
            manifest=source/'source_manifest.json';manifest.write_text(json.dumps({'fixture.py':ns['digest'](module)}))
            cfg={}
            for key in ['annotations','train_split','val_split','yolo']:
                p=base/key;p.write_text('fixture');cfg[key]=str(p)
            (base/'images').mkdir();cfg['images']=str(base/'images')
            config=base/'experiment_config.json';config.write_text(json.dumps(cfg))
            pointer=dict(version='a'*16,config_sha256=ns['digest'](config),manifest_sha256=ns['digest'](manifest))
            (base/'current_code.json').write_text(json.dumps(pointer))
            ns['EXPECTED_BASE_CONFIG']=pointer['config_sha256'];ns['EXPECTED_BASE_MANIFEST']=pointer['manifest_sha256']
            stamps={}
            for name in ['detr/epoch030.pt','roi/epoch030.pt']:
                p=base/name;p.parent.mkdir(exist_ok=True);p.write_bytes(b'checkpoint '+name.encode());stamps[name]=ns['digest'](p)
            ns['EXPECTED_CHECKPOINTS']=stamps
            ns['verify_base']({'base_root':str(base)},'probes')
            with self.assertRaisesRegex(FileNotFoundError,'optimizer'):ns['verify_base']({'base_root':str(base)},'all')
            (base/'detr/last_full.pt').write_bytes(b'full checkpoint')
            ns['verify_base']({'base_root':str(base)},'extend')
            (base/'roi/epoch030.pt').write_bytes(b'other model')
            with self.assertRaisesRegex(ValueError,'가중치'):ns['verify_base']({'base_root':str(base)},'probes')

    def test_initial_bootstrap_error_is_visible_and_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);durable=root/'code';durable.mkdir()
            (durable/builder.BOOTSTRAP_NAME).write_text("import sys\nprint('fixture missing original checkpoint',file=sys.stderr,flush=True)\nraise SystemExit(1)\n")
            ns=dict(sys=sys,os=os,signal=signal,subprocess=subprocess,root=root,durable=durable,
                    PHASE='all',RUN_UNTIL_ADDED=30,datetime=datetime,timezone=timezone)
            output=io.StringIO()
            with redirect_stdout(output),self.assertRaises(subprocess.CalledProcessError):exec(compile(builder.EXECUTE,'bbox-execute','exec'),ns)
            self.assertIn('fixture missing original checkpoint',output.getvalue())
            self.assertIn('fixture missing original checkpoint',(root/'colab_bootstrap.log').read_text())
            self.assertEqual(ns['process'].poll(),1)

    def test_dependency_setup_preserves_colab_cuda_pair(self):
        ns=namespace()
        def version(name):
            if name=='torch':return '2.11.0+cu130'
            if name=='torchvision':return '0.26.0+cu130'
            from importlib.metadata import PackageNotFoundError
            raise PackageNotFoundError(name)
        with tempfile.TemporaryDirectory() as directory,patch('importlib.metadata.version',side_effect=version),patch('importlib.util.find_spec',return_value=object()),patch('subprocess.run') as run:
            ns['ensure_dependencies'](Path(directory));command=run.call_args.args[0]
            self.assertIn('transformers==4.57.1',command);self.assertIn('ultralytics==8.4.182',command)
            self.assertFalse(any(s.startswith(('torch==','torchvision==')) for s in command))
            self.assertEqual(Path(command[command.index('-c')+1]).read_text(),'torch==2.11.0+cu130\ntorchvision==0.26.0+cu130\n')

    @unittest.skipUnless(os.name=='posix','requires POSIX process groups')
    def test_stopping_bootstrap_stops_pipeline_child(self):
        ns=namespace()
        with tempfile.TemporaryDirectory() as directory:
            tmp=Path(directory);root=tmp/'run';version='a'*16;durable=root/'code'/version;durable.mkdir(parents=True)
            pidfile=tmp/'worker_pid'
            worker="from pathlib import Path\nimport os,time\nPath("+repr(str(pidfile))+").write_text(str(os.getpid()))\ntime.sleep(60)\n"
            (durable/'bbox_probe_pipeline.py').write_text(worker)
            manifest={'bbox_probe_pipeline.py':hashlib.sha256(worker.encode()).hexdigest()}
            mpath=durable/'source_manifest.json';mpath.write_text(json.dumps(manifest))
            cfg=dict(base_root=str(tmp/'base'),output_root=str(root),added_epochs=30)
            cpath=root/'experiment_config.json';cpath.write_text(json.dumps(cfg))
            (root/'current_code.json').write_text(json.dumps(dict(version=version,manifest_sha256=ns['digest'](mpath),config_sha256=ns['digest'](cpath))))
            module=tmp/'bootstrap.py';module.write_text(builder.BOOTSTRAP)
            wrapper=tmp/'wrapper.py';wrapper.write_text('import importlib.util,signal\nfrom pathlib import Path\n'
                +'spec=importlib.util.spec_from_file_location("b",'+repr(str(module))+')\n'
                +'b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)\n'
                +'old=b.restore\nb.restore=lambda root:old(root,Path('+repr(str(tmp/'content'))+'))\n'
                +'b.ensure_dependencies=lambda work:None\nb.check_gpu=lambda *args:None\nb.verify_base=lambda *args:None\n'
                +'signal.signal(signal.SIGTERM,b.interrupted)\nb.run(Path('+repr(str(root))+'),"probes",30)\n')
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

    def test_actual_bundle_payload_notebook_and_budget(self):
        if not all((builder.ROOT/'tools'/name).is_file() for name in builder.MODULES):self.skipTest('Engines still being authored')
        metadata=builder.build();cell=Path(metadata['copy_cell']).read_text();tree=ast.parse(cell)
        payload=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in n.targets))
        raw=zlib.decompress(base64.b64decode(payload));sources=json.loads(raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),metadata['source_bundle_sha256'])
        self.assertEqual(set(sources),set(builder.MODULES)|{builder.BOOTSTRAP_NAME})
        for name in builder.MODULES:self.assertEqual(sources[name],(builder.ROOT/'tools'/name).read_text())
        self.assertEqual((metadata['default_added_detr_epochs'],metadata['final_cumulative_detr_epochs'],metadata['added_roi_epochs'],metadata['added_yolo_epochs']),(30,60,0,0))
        self.assertEqual(metadata['default_phase'],'all')
        self.assertEqual(metadata['default_continuation_precision'],'bf16')
        self.assertIn('_A100_',metadata['copy_cell'])
        notebook=json.loads(Path(metadata['notebook']).read_text())
        self.assertEqual(cell,''.join(notebook['cells'][1]['source']))
        self.assertEqual(Path(metadata['resume_cell']).read_text(),''.join(notebook['cells'][3]['source']))


if __name__=='__main__':unittest.main()
