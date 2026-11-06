"""Actual copy-cell reset, immutable source, total-epoch and visible-error contracts."""
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
import zipfile
import zlib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_detr_longrun_colab as builder


def namespace():
    ns={'__name__':'detr150_bootstrap_fixture'}
    exec(compile(builder.BOOTSTRAP,'detr150-bootstrap','exec'),ns)
    return ns


class DetrLongrunColabTests(unittest.TestCase):
    def fixture(self,tmp):
        sources={name:'# fixture '+name+'\n' for name in builder.MODULES}
        sources[builder.BOOTSTRAP_NAME]=builder.BOOTSTRAP
        with patch.object(builder,'ROOT',tmp/'repo'),patch.object(builder,'build_sources',return_value=sources):
            metadata=builder.build(downloads=tmp/'downloads')
        drive=tmp/'drive';content=tmp/'content'
        first=Path(metadata['copy_cell']).read_text().replace('/content/drive/MyDrive',str(drive))
        resume=Path(metadata['resume_cell']).read_text().replace('/content/drive/MyDrive',str(drive))
        parent=drive/'filament-segmentation-2026/fixed_split_experiments'
        root=parent/'detr_a100_e150_v1';source=parent/'bbox_improvement_v1';base=parent/'detection_roi_detr_v1'
        source.mkdir(parents=True);base.mkdir()
        (source/'experiment_config.json').write_text(json.dumps({'base_root':str(base)}))
        (source/'sentinel').write_text('source must stay unchanged')
        google=types.ModuleType('google');colab=types.ModuleType('google.colab');google.colab=colab
        downloads=[];colab.drive=types.SimpleNamespace(mount=lambda _:None);colab.files=types.SimpleNamespace(download=downloads.append)
        commands=[];restored=[];ns=namespace()
        class FakeProcess:
            def __init__(self,command,**kwargs):
                commands.append(command)
                cfg,_,work=ns['restore'](Path(command[command.index('--root')+1]),content)
                restored.append((cfg,work));run_root=Path(cfg['output_root'])
                archive=run_root/'fixture_part01.zip';archive.write_bytes(b'fixture archive')
                (run_root/'latest_results.json').write_text(json.dumps(dict(status='paused',completed_epoch=int(command[-1]),best_epoch=60,best_pq=.49,archives=[str(archive)])))
                self.pid=98765;self.stdout=io.StringIO('visible inner stdout and stderr\n')
            def wait(self,timeout=None):return 0
        def run(cell,until=150):
            text=cell.replace('RUN_UNTIL_EPOCH = 150 ',f'RUN_UNTIL_EPOCH = {until} ')
            output=io.StringIO()
            with patch.dict(sys.modules,{'google':google,'google.colab':colab}),patch('subprocess.Popen',FakeProcess),redirect_stdout(output):
                exec(compile(text,'detr150-colab-fixture','exec'),{})
            return output.getvalue()
        return dict(first=first,resume=resume,root=root,source=source,base=base,content=content,sources=sources,
                    commands=commands,restored=restored,downloads=downloads,run=run)

    def test_absolute_budget_and_resume_after_complete_content_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));f['run'](f['first'],until=80)
            original=(f['root']/'experiment_config.json').read_bytes();shutil.rmtree(f['content'])
            output=f['run'](f['resume'],until=150)
            self.assertEqual(original,(f['root']/'experiment_config.json').read_bytes())
            self.assertEqual([c[c.index('--run-until')+1] for c in f['commands']],['80','150'])
            self.assertTrue(all('--run-until-added' not in c for c in f['commands']))
            for cfg,work in f['restored']:
                self.assertEqual((cfg['max_epoch'],cfg['patience'],cfg['min_delta']),(150,10,.001))
                self.assertEqual((cfg['continuation_lr'],cfg['continuation_backbone_lr'],cfg['continuation_lr_floor']),(5e-6,5e-7,.2))
                self.assertEqual((cfg['continuation_image_size'],cfg['continuation_batch'],cfg['continuation_accumulate'],cfg['continuation_precision']),(1280,2,2,'bf16'))
                self.assertEqual((cfg['roi_eval_batch'],cfg['eval_box_thresholds'],cfg['eval_min_area']),(8,[.6,.8,.9],300))
                self.assertNotIn('added_epochs',cfg);self.assertNotIn('roi_epochs',cfg);self.assertNotIn('yolo_epochs',cfg)
                for name,source in f['sources'].items():self.assertEqual((work/name).read_text(),source)
            self.assertEqual(len(f['downloads']),2)
            self.assertIn('visible inner stdout and stderr',output)
            self.assertIn('visible inner stdout and stderr',(f['root']/'colab_bootstrap.log').read_text())
            self.assertEqual((f['source']/'sentinel').read_text(),'source must stay unchanged')

    def test_first_cell_is_idempotent_and_rejects_invalid_total_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));f['run'](f['first'],until=60)
            saved=(f['root']/'experiment_config.json').read_bytes()
            f['run'](f['first'],until=150)
            self.assertEqual(saved,(f['root']/'experiment_config.json').read_bytes())
            for until in (0,30,151,210):
                with self.assertRaisesRegex(ValueError,'60~150'):f['run'](f['resume'],until=until)
            self.assertEqual(len(f['commands']),2)

    def test_manifest_config_and_output_overlap_fail_before_child(self):
        with tempfile.TemporaryDirectory() as directory:
            f=self.fixture(Path(directory));f['run'](f['first'])
            config=f['root']/'experiment_config.json';old=config.read_bytes();config.write_bytes(old+b' ')
            with self.assertRaisesRegex(ValueError,'설정'):f['run'](f['resume'])
            config.write_bytes(old)
            pointer=json.loads((f['root']/'current_code.json').read_text())
            source=f['root']/'code'/pointer['version']/builder.MODULES[0];source.write_text('changed')
            with self.assertRaisesRegex(ValueError,'코드'):f['run'](f['resume'])
            for protected in (f['source'],f['base']):
                for bad in (protected,protected/'nested',protected.parent):
                    altered=f['first'].replace("OUTPUT = "+repr(str(f['root'])),"OUTPUT = "+repr(str(bad)))
                    with self.assertRaisesRegex(ValueError,'OUTPUT'):f['run'](altered)
            self.assertEqual(len(f['commands']),1)

    def test_original60_weights_code_and_frozen_roi_guard(self):
        ns=namespace()
        with tempfile.TemporaryDirectory() as directory:
            tmp=Path(directory);source=tmp/'source';base=tmp/'original';output=tmp/'new'
            code=source/'code'/('a'*16);code.mkdir(parents=True);base.mkdir()
            module=code/'fixture.py';module.write_text('# original source\n')
            manifest=code/'source_manifest.json';manifest.write_text(json.dumps({'fixture.py':ns['digest'](module)}))
            config=source/'experiment_config.json';config.write_text(json.dumps(dict(base_root=str(base))))
            pointer=dict(version='a'*16,config_sha256=ns['digest'](config),manifest_sha256=ns['digest'](manifest))
            (source/'current_code.json').write_text(json.dumps(pointer))
            ns['EXPECTED_SOURCE_CONFIG']=pointer['config_sha256'];ns['EXPECTED_SOURCE_MANIFEST']=pointer['manifest_sha256']
            extension=source/'detr_extension';extension.mkdir()
            (extension/'training_identity.json').write_text(json.dumps(dict(identity='reviewed60')))
            ns['EXPECTED_SOURCE_IDENTITY']='reviewed60'
            compact=extension/'epoch060.pt';compact.write_bytes(b'compact60');ns['EXPECTED_SOURCE_CHECKPOINT']=ns['digest'](compact)
            cfg=dict(source_root=str(source),output_root=str(output))
            with self.assertRaisesRegex(FileNotFoundError,'optimizer'):ns['verify_source'](cfg)
            (extension/'last_full.pt').write_bytes(b'full60')
            roi=base/'roi/epoch030.pt';roi.parent.mkdir();roi.write_bytes(b'fixed roi30');ns['EXPECTED_ROI_CHECKPOINT']=ns['digest'](roi)
            original={}
            for key in ('annotations','train_split','val_split'):
                p=base/key;p.write_text('fixture');original[key]=str(p)
            (base/'images').mkdir();original['images']=str(base/'images')
            (base/'experiment_config.json').write_text(json.dumps(original))
            ns['verify_source'](cfg)
            roi.write_bytes(b'wrong roi')
            with self.assertRaisesRegex(ValueError,'ROI30'):ns['verify_source'](cfg)
            compact.write_bytes(b'wrong detector')
            with self.assertRaisesRegex(ValueError,'DETR60 가중치'):ns['verify_source'](cfg)

    def test_early_inner_error_streamed_and_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);durable=root/'code';durable.mkdir()
            (durable/builder.BOOTSTRAP_NAME).write_text("import sys\nprint('fixture missing original checkpoint',file=sys.stderr,flush=True)\nraise SystemExit(1)\n")
            ns=dict(sys=sys,os=os,signal=signal,subprocess=subprocess,root=root,durable=durable,
                    RUN_UNTIL_EPOCH=150,datetime=datetime,timezone=timezone)
            output=io.StringIO()
            with redirect_stdout(output),self.assertRaisesRegex(RuntimeError,'전체 로그'):
                exec(compile(builder.EXECUTE,'detr150-execute','exec'),ns)
            self.assertIn('fixture missing original checkpoint',output.getvalue())
            self.assertIn('fixture missing original checkpoint',(root/'colab_bootstrap.log').read_text())
            self.assertEqual(ns['process'].poll(),1)

    def test_dependency_setup_preserves_colab_torch_cuda_pair(self):
        ns=namespace()
        def version(name):
            if name=='torch':return '2.11.0+cu130'
            if name=='torchvision':return '0.26.0+cu130'
            from importlib.metadata import PackageNotFoundError
            raise PackageNotFoundError(name)
        with tempfile.TemporaryDirectory() as directory,patch('importlib.metadata.version',side_effect=version),patch('importlib.util.find_spec',return_value=object()),patch('subprocess.run') as run:
            ns['ensure_dependencies'](Path(directory));command=run.call_args.args[0]
            self.assertIn('transformers==4.57.1',command)
            self.assertFalse(any(s.startswith(('torch==','torchvision==')) for s in command))
            self.assertEqual(Path(command[command.index('-c')+1]).read_text(),'torch==2.11.0+cu130\ntorchvision==0.26.0+cu130\n')

    @unittest.skipUnless(os.name=='posix','requires POSIX process groups')
    def test_stopping_bootstrap_stops_training_child(self):
        ns=namespace()
        with tempfile.TemporaryDirectory() as directory:
            tmp=Path(directory);root=tmp/'run';version='a'*16;durable=root/'code'/version;durable.mkdir(parents=True)
            pidfile=tmp/'worker_pid'
            worker="from pathlib import Path\nimport os,time\nPath("+repr(str(pidfile))+").write_text(str(os.getpid()))\ntime.sleep(60)\n"
            (durable/'detr_longrun.py').write_text(worker)
            manifest={'detr_longrun.py':hashlib.sha256(worker.encode()).hexdigest()}
            mpath=durable/'source_manifest.json';mpath.write_text(json.dumps(manifest))
            cfg=dict(source_root=str(tmp/'source'),output_root=str(root),max_epoch=150)
            cpath=root/'experiment_config.json';cpath.write_text(json.dumps(cfg))
            (root/'current_code.json').write_text(json.dumps(dict(version=version,manifest_sha256=ns['digest'](mpath),config_sha256=ns['digest'](cpath))))
            module=tmp/'bootstrap.py';module.write_text(builder.BOOTSTRAP)
            wrapper=tmp/'wrapper.py';wrapper.write_text('import importlib.util,signal\nfrom pathlib import Path\n'
                +'spec=importlib.util.spec_from_file_location("b",'+repr(str(module))+')\n'
                +'b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)\n'
                +'old=b.restore\nb.restore=lambda root:old(root,Path('+repr(str(tmp/'content'))+'))\n'
                +'b.ensure_dependencies=lambda work:None\nb.check_gpu=lambda *args:None\nb.verify_source=lambda *args:None\n'
                +'signal.signal(signal.SIGTERM,b.interrupted)\nb.run(Path('+repr(str(root))+'),150)\n')
            process=subprocess.Popen([sys.executable,str(wrapper)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            try:
                deadline=time.monotonic()+8
                while not pidfile.exists() and time.monotonic()<deadline:
                    if process.poll() is not None:break
                    time.sleep(.05)
                self.assertTrue(pidfile.exists(),'training worker did not start')
                worker_pid=int(pidfile.read_text());os.killpg(process.pid,signal.SIGTERM);process.communicate(timeout=12)
                with self.assertRaises(ProcessLookupError):os.kill(worker_pid,0)
            finally:
                if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.communicate(timeout=5)
                if pidfile.exists():
                    try:os.kill(int(pidfile.read_text()),signal.SIGKILL)
                    except ProcessLookupError:pass

    def test_actual_bundle_payload_notebook_and_budget(self):
        if not all((builder.ROOT/'tools'/name).is_file() for name in builder.MODULES):self.skipTest('Engines still being authored')
        with tempfile.TemporaryDirectory() as directory:
            metadata=builder.build(downloads=directory)
            cell=Path(metadata['copy_cell']).read_text();tree=ast.parse(cell)
            payload=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in n.targets))
            raw=zlib.decompress(base64.b64decode(payload));sources=json.loads(raw)
            self.assertEqual(hashlib.sha256(raw).hexdigest(),metadata['source_bundle_sha256'])
            self.assertEqual(set(sources),set(builder.MODULES)|{builder.BOOTSTRAP_NAME})
            for name in builder.MODULES:self.assertEqual(sources[name],(builder.ROOT/'tools'/name).read_text())
            self.assertEqual((metadata['existing_detr_epochs'],metadata['max_cumulative_detr_epochs'],metadata['max_additional_detr_epochs']),(60,150,90))
            self.assertEqual((metadata['added_roi_epochs'],metadata['added_yolo_epochs']),(0,0))
            self.assertEqual((metadata['validation_images'],metadata['validation_interval_epochs'],metadata['patience_epochs']),(91,1,10))
            notebook=json.loads(Path(metadata['notebook']).read_text())
            self.assertEqual(cell,''.join(notebook['cells'][1]['source']))
            self.assertEqual(Path(metadata['resume_cell']).read_text(),''.join(notebook['cells'][3]['source']))
            with zipfile.ZipFile(metadata['package']) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(json.loads(archive.read('source/source_manifest.json')),metadata['source_sha256'])

    def test_bundle_imports_without_repository_or_prior_colab_content(self):
        if not all((builder.ROOT/'tools'/name).is_file() for name in builder.MODULES):self.skipTest('Engines still being authored')
        with tempfile.TemporaryDirectory() as directory:
            isolated=Path(directory)
            for name,source in builder.build_sources().items():(isolated/name).write_text(source)
            script=('import sys; sys.path.insert(0,sys.argv[1]); '
                    'import detr_longrun, detr_longrun_eval; '
                    'assert callable(detr_longrun.run); '
                    'assert callable(detr_longrun_eval.evaluate_checkpoint); '
                    'assert callable(detr_longrun_eval.export_results); print("isolated import ok")')
            result=subprocess.run([sys.executable,'-I','-c',script,str(isolated)],cwd=isolated,
                                  stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout)
            self.assertIn('isolated import ok',result.stdout)

    def test_real_engine_preserves_cell_config_digest_across_resume(self):
        """The real worker must not normalize/rewrite the immutable cell input."""
        import torch
        sys.path.insert(0,str(Path(__file__).resolve().parent))
        from test_detr_longrun import source_fixture, evaluation
        import detr_longrun
        torch.set_num_threads(2)
        ns=namespace()
        with tempfile.TemporaryDirectory() as directory:
            tmp=Path(directory);cfg=source_fixture(tmp,added=2)
            root=Path(cfg['output_root']);version='c'*16;durable=root/'code'/version
            durable.mkdir(parents=True)
            manifest={}
            for name,source in builder.build_sources().items():
                path=durable/name;path.write_text(source)
                manifest[name]=ns['digest'](path)
            manifest_path=durable/'source_manifest.json'
            manifest_path.write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
            cfg.update(code_dir=str(durable),code_manifest_sha256=ns['digest'](manifest_path))
            config_path=root/'experiment_config.json'
            config_path.write_text(json.dumps(cfg,sort_keys=True,indent=2)+'\n')
            original=config_path.read_bytes()
            pointer=dict(version=version,manifest_sha256=ns['digest'](manifest_path),
                         config_sha256=ns['digest'](config_path))
            (root/'current_code.json').write_text(json.dumps(pointer))
            scores=evaluation({2:.4,3:.42,4:.43})
            with redirect_stdout(io.StringIO()):detr_longrun.run(cfg,3,scores)
            self.assertEqual(config_path.read_bytes(),original)
            _,_,work=ns['restore'](root,tmp/'content')
            shutil.rmtree(tmp/'content')
            restored,_,work=ns['restore'](root,tmp/'content')
            with redirect_stdout(io.StringIO()):result=detr_longrun.run(restored,4,scores)
            self.assertEqual(result['completed_epoch'],4)
            self.assertEqual(config_path.read_bytes(),original)
            ns['restore'](root,tmp/'content')


if __name__=='__main__':unittest.main()
