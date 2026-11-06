"""Fresh-runtime restore, immutable identity and cancellation of Colab workers."""
from contextlib import redirect_stdout
import ast
import base64
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
import zlib
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_targeted_synthetic_colab as builder


def bootstrap_namespace():
    value={'__name__':'bootstrap_fixture'}
    exec(compile(builder.BOOTSTRAP,'bootstrap-fixture','exec'),value)
    return value


class TargetedColabTests(unittest.TestCase):
    def test_real_bundle_embeds_current_sources_and_gpu_safety_probe(self):
        metadata=builder.build()
        cell=Path(metadata['copy_cell']).read_text()
        tree=ast.parse(cell)
        payload=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in n.targets))
        raw=zlib.decompress(base64.b64decode(payload));sources=json.loads(raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),metadata['source_bundle_sha256'])
        self.assertEqual(set(sources),set(builder.MODULES)|{builder.BOOTSTRAP_NAME})
        for name in builder.MODULES:
            self.assertEqual(sources[name],(builder.ROOT/'tools'/name).read_text())
        self.assertEqual(sources[builder.BOOTSTRAP_NAME],builder.BOOTSTRAP)
        self.assertIn('configure_cuda()',builder.BOOTSTRAP)
        self.assertIn('for groups in (1,3):',builder.BOOTSTRAP)
        self.assertIn("environment['TORCH_CUDNN_V8_API_DISABLED'] = '1'",builder.BOOTSTRAP)
        self.assertIn("'--run-until', str(run_until)",builder.BOOTSTRAP)
        self.assertNotIn('epochs=100',builder.BOOTSTRAP)
        notebook=json.loads(Path(metadata['notebook']).read_text())
        self.assertEqual(cell,''.join(notebook['cells'][1]['source']))
        self.assertEqual(Path(metadata['resume_cell']).read_text(),''.join(notebook['cells'][3]['source']))

    def fixture(self, temporary, with_previews=False):
        sources={name:('# fixture '+name+'\n' if name.endswith('.py') else '{}') for name in builder.MODULES}
        sources[builder.BOOTSTRAP_NAME]=builder.BOOTSTRAP
        with patch.object(builder,'ROOT',temporary/'repo'),patch.object(builder,'build_sources',return_value=sources):
            metadata=builder.build()
        drive=temporary/'drive'; content=temporary/'content'
        first=Path(metadata['copy_cell']).read_text().replace('/content/drive/MyDrive',str(drive))
        resume=Path(metadata['resume_cell']).read_text().replace('/content/drive/MyDrive',str(drive))
        root=drive/'filament-segmentation-2026/fixed_split_experiments/targeted_synthetic_v3_yolo_t4_v1'
        google=types.ModuleType('google'); colab=types.ModuleType('google.colab');google.colab=colab
        downloads=[];colab.drive=types.SimpleNamespace(mount=lambda _:None);colab.files=types.SimpleNamespace(download=downloads.append)
        commands=[]; restored=[]
        ns=bootstrap_namespace()
        class FakeProcess:
            def __init__(self,command,**kwargs):
                commands.append(command)
                cfg,path,work=ns['restore'](Path(command[command.index('--root')+1]),content)
                restored.append((cfg,work))
                archive=root/'fixture_results.zip';archive.write_bytes(b'fixture archive')
                result={'status':'fixture_complete','archives':[str(archive)]}
                if with_previews:
                    data=root/'data';data.mkdir(exist_ok=True)
                    for name in ('synthetic_preview.png','synthetic_native_preview.png'):
                        (data/name).write_bytes(b'fixture png read by mocked IPython Image')
                    result['dataset']={'preview':str(data/'synthetic_preview.png'),'manifest':str(data/'dataset_manifest.json')}
                (root/'result.json').write_text(json.dumps(result))
                self.pid=98765
                self.stdout=io.StringIO('fixture bootstrap output\n')
            def wait(self,timeout=None):return 0
        def run(cell,until=30):
            text=cell.replace('RUN_UNTIL = 30 ',f'RUN_UNTIL = {until} ')
            with patch.dict(sys.modules,{'google':google,'google.colab':colab}),patch('subprocess.Popen',FakeProcess),redirect_stdout(io.StringIO()):
                exec(compile(text,'colab-fixture','exec'),{})
        return first,resume,root,content,sources,commands,restored,downloads,run

    def test_fresh_resume_recovers_deleted_content_and_keeps_final30(self):
        with tempfile.TemporaryDirectory() as directory:
            first,resume,root,content,sources,commands,restored,downloads,run=self.fixture(Path(directory))
            run(first,10)
            original_cfg=(root/'experiment_config.json').read_bytes()
            shutil.rmtree(content)
            run(resume,30)
            self.assertEqual(original_cfg,(root/'experiment_config.json').read_bytes())
            self.assertEqual([c[c.index('--run-until')+1] for c in commands],['10','30'])
            self.assertEqual([cfg['epochs'] for cfg,_ in restored],[30,30])
            self.assertTrue(all(cfg['batch']==cfg['nbs']==1 for cfg,_ in restored))
            self.assertTrue(all(cfg['imgsz']==2048 and cfg['cudnn_enabled'] for cfg,_ in restored))
            self.assertTrue(all(cfg['full_ensemble'] for cfg,_ in restored))
            for name,source in sources.items():self.assertEqual((restored[-1][1]/name).read_text(),source)
            self.assertEqual(len(downloads),2)

    def test_changed_config_or_durable_code_blocks_before_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            first,resume,root,_,_,commands,_,_,run=self.fixture(Path(directory))
            run(first)
            config=root/'experiment_config.json';original=config.read_bytes();config.write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError,'설정'):
                run(resume)
            config.write_bytes(original)
            pointer=json.loads((root/'current_code.json').read_text())
            source=root/'code'/pointer['version']/'targeted_synthetic_training.py';source.write_text('changed')
            with self.assertRaisesRegex(ValueError,'코드'):
                run(resume)
            self.assertEqual(len(commands),1)

    def test_new_training_settings_cannot_silently_resume_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            first,_,_,_,_,commands,_,_,run=self.fixture(Path(directory))
            run(first)
            with self.assertRaisesRegex(ValueError,'기존 실험 설정'):
                run(first.replace('FINAL_EPOCHS = 30 ','FINAL_EPOCHS = 40 '))
            self.assertEqual(len(commands),1)

    def test_prepare_allows_missing_native_and_evaluate_needs_no_synthetic_inputs(self):
        ns=bootstrap_namespace()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg={'full_ensemble':True}
            for key in ('annotations','train_split','val_split','yolo'):
                path=root/key;path.write_text('fixture');cfg[key]=str(path)
            (root/'images').mkdir();cfg['images']=str(root/'images')
            cfg['native']=str(root/'native')
            Path(cfg['native']).write_text('fixture')
            ns['check_paths'](cfg,'evaluate')
            Path(cfg['native']).unlink()
            with self.assertRaisesRegex(FileNotFoundError,'native'):
                ns['check_paths'](cfg,'evaluate')
            for key in ('synthetic_annotations','synthetic_split','hard_donors'):
                path=root/key;path.write_text('fixture');cfg[key]=str(path)
            (root/'synthetic_images').mkdir();cfg['synthetic_images']=str(root/'synthetic_images')
            ns['check_paths'](cfg,'prepare')
            with self.assertRaisesRegex(FileNotFoundError,'native'):
                ns['check_paths'](cfg,'train')

    def test_completion_displays_both_previews_with_manifest_sibling_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            first,_,_,_,_,_,_,_,run=self.fixture(Path(directory),with_previews=True)
            display_module=types.ModuleType('IPython.display');shown=[]
            display_module.Image=lambda filename:filename
            display_module.display=shown.append
            with patch.dict(sys.modules,{'IPython.display':display_module}):
                run(first)
            self.assertEqual([Path(x).name for x in shown],['synthetic_preview.png','synthetic_native_preview.png'])

    def test_dependency_bootstrap_preserves_cuda_torch_pair_with_constraints(self):
        ns=bootstrap_namespace()
        def version(name):
            if name=='torch':return '2.11.0+cu130'
            if name=='torchvision':return '0.26.0+cu130'
            raise builder_import_error(name)
        with tempfile.TemporaryDirectory() as directory,patch('importlib.metadata.version',side_effect=version), \
             patch('importlib.util.find_spec',return_value=object()),patch('subprocess.run') as run:
            ns['ensure_dependencies'](Path(directory))
            command=run.call_args.args[0]
            self.assertIn('ultralytics==8.4.182',command)
            self.assertIn('segmentation-models-pytorch==0.5.0',command)
            constraints=Path(command[command.index('-c')+1]).read_text()
            self.assertEqual(constraints,'torch==2.11.0+cu130\ntorchvision==0.26.0+cu130\n')
            self.assertFalse(any(x.startswith('torch==') or x.startswith('torchvision==') for x in command))

    def test_first_cell_surfaces_early_bootstrap_stderr_before_failure(self):
        # Real child fails before the inner training log exists. Its stderr
        # must reach Python print(), which is what Colab displays in the cell.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);durable=root/'code';durable.mkdir()
            (durable/builder.BOOTSTRAP_NAME).write_text(
                "import sys\nprint('fixture early missing input',file=sys.stderr,flush=True)\nraise SystemExit(1)\n")
            output=io.StringIO()
            namespace=dict(sys=sys,os=os,signal=signal,subprocess=subprocess,
                           durable=durable,root=root,PHASE='all',RUN_UNTIL=30,
                           ARMS=['control_v2','targeted_v3'])
            try:
                with redirect_stdout(output),self.assertRaises(subprocess.CalledProcessError):
                    exec(compile(builder.EXECUTE,'first-cell-execute','exec'),namespace)
                self.assertIn('fixture early missing input',output.getvalue())
                self.assertFalse((root/'colab_all.log').exists())
                self.assertEqual(namespace['process'].poll(),1)
            finally:
                if 'process' in namespace:namespace['process'].stdout.close()

    @unittest.skipUnless(os.name=='posix','process-group signal test requires POSIX')
    def test_stop_bootstrap_also_stops_separate_worker_process_group(self):
        ns=bootstrap_namespace()
        with tempfile.TemporaryDirectory() as directory:
            tmp=Path(directory);root=tmp/'run';durable=root/'code'/'fixture';durable.mkdir(parents=True)
            pidfile=tmp/'worker_pid'
            launcher="from pathlib import Path\nimport os,time\nPath("+repr(str(pidfile))+").write_text(str(os.getpid()))\ntime.sleep(60)\n"
            (durable/'targeted_synthetic_launcher.py').write_text(launcher)
            manifest={'targeted_synthetic_launcher.py':hashlib.sha256(launcher.encode()).hexdigest()}
            manifest_path=durable/'source_manifest.json';manifest_path.write_text(json.dumps(manifest))
            config={'output_root':str(root),'epochs':30,'cache_root':str(tmp/'cache'),'full_ensemble':False}
            for key in ('annotations','train_split','val_split','yolo','synthetic_annotations','synthetic_split','hard_donors'):
                path=tmp/key;path.write_text('fixture');config[key]=str(path)
            for key in ('images','synthetic_images'):
                path=tmp/key;path.mkdir();config[key]=str(path)
            config_path=root/'experiment_config.json';config_path.write_text(json.dumps(config))
            pointer=dict(version='fixture',manifest_sha256=ns['digest'](manifest_path),config_sha256=ns['digest'](config_path))
            (root/'current_code.json').write_text(json.dumps(pointer))
            module=tmp/'bootstrap.py';module.write_text(builder.BOOTSTRAP)
            wrapper=tmp/'wrapper.py'
            wrapper.write_text('import importlib.util,signal\nfrom pathlib import Path\n'
                +'spec=importlib.util.spec_from_file_location("fixture_bootstrap",'+repr(str(module))+')\n'
                +'b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)\n'
                +'original_restore=b.restore\nb.restore=lambda root: original_restore(root,Path('+repr(str(tmp/'content'))+'))\n'
                +'b.ensure_dependencies=lambda work: None\nb.check_gpu=lambda *args: None\n'
                +'signal.signal(signal.SIGTERM,b.interrupted)\n'
                +'b.run(Path('+repr(str(root))+'),"train",1,["control_v2"])\n')
            process=subprocess.Popen([sys.executable,str(wrapper)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,text=True)
            try:
                deadline=time.monotonic()+8
                while not pidfile.exists() and time.monotonic()<deadline:
                    if process.poll() is not None:break
                    time.sleep(.05)
                self.assertTrue(pidfile.exists(),'Worker failed to start')
                worker=int(pidfile.read_text())
                os.killpg(process.pid,signal.SIGTERM)
                process.communicate(timeout=10)
                with self.assertRaises(ProcessLookupError):os.kill(worker,0)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGKILL);process.communicate(timeout=5)
                if pidfile.exists():
                    try:os.kill(int(pidfile.read_text()),signal.SIGKILL)
                    except ProcessLookupError:pass


def builder_import_error(name):
    from importlib.metadata import PackageNotFoundError
    return PackageNotFoundError(name)


if __name__=='__main__':unittest.main()
