"""Plain inference notebook, durable snapshot, and process-boundary checks."""
import ast
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT/'notebooks/SolarFilament_CHASE_Test_T4.ipynb'
spec = importlib.util.spec_from_file_location('chase_test_colab',ROOT/'tools/chase_shape_test_colab.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def git(path,*args):
    return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()


@pytest.fixture
def fixture(tmp_path):
    repository=tmp_path/'remote';repository.mkdir()
    for name in launcher.SOURCE_FILES:
        path=repository/name;path.parent.mkdir(parents=True,exist_ok=True)
        if name=='tools/chase_shape_test_colab.py':
            shutil.copyfile(ROOT/name,path)
        else:
            path.write_text('# Fixture source.\n')
    subprocess.run(['git','init','--initial-branch=main',str(repository)],check=True,capture_output=True)
    git(repository,'config','user.name','CPU test')
    git(repository,'config','user.email','cpu@example.invalid')
    git(repository,'add','.')
    git(repository,'commit','-m','Inference fixture')
    base=tmp_path/'drive/MyDrive/filament-segmentation-2026'
    training=base/'fixed_split_experiments/chase_shape_best84_recipe_t4_v1'
    images=base/'MAGFiLO_1.0_Kaggle_2026/test/test_images'
    output=base/'submissions'/launcher.SCOPE
    for name in ('train/state.json','train/config.json','train/identity.json'):
        path=training/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    images.mkdir(parents=True)
    return types.SimpleNamespace(repository=repository,base=base,training=training,
        images=images,output=output,commit=git(repository,'rev-parse','HEAD'))


def install(item):
    return launcher.install(item.repository,item.training,item.images,item.output)


def write_result(output,**changes):
    csv=output/'submission.csv';csv.write_text('filament_id,segmentation_rle\n')
    archive=output/'review.zip';archive.write_bytes(b'review fixture')
    result=dict(status='complete',images_completed=180,checkpoint_epoch=82,
        output_root=str(output),csv=str(csv),csv_sha256=launcher.sha(csv),
        archive=str(archive),archive_sha256=launcher.sha(archive))
    result.update(changes)
    (output/'latest_result.json').write_text(json.dumps(result))
    return result


def test_source_bundle_survives_checkout_loss_without_touching_training(fixture):
    before={p.name:p.read_bytes() for p in (fixture.training/'train').iterdir()}
    cfg,code,state=install(fixture)
    assert state['source_commit']==fixture.commit and not state['source_worktree_dirty']
    assert cfg['inference_only'] and cfg['native_only'] and cfg['expected_epoch']==82
    assert len(launcher.SOURCE_FILES)==5
    assert not any('synthesis' in name or 'chase_shape_training.py' in name for name in launcher.SOURCE_FILES)
    for name in launcher.SOURCE_FILES:
        assert (code/name).read_bytes()==(fixture.repository/name).read_bytes()
    shutil.rmtree(fixture.repository)
    assert launcher.restore(fixture.output,fixture.training,fixture.images)==(cfg,code,state)
    assert before=={p.name:p.read_bytes() for p in (fixture.training/'train').iterdir()}


@pytest.mark.parametrize('name',['tools/chase_shape_test_inference.py','tools/solar_a100_train_standalone.py'])
def test_saved_source_tampering_is_rejected_and_preserved(fixture,name):
    _,code,_=install(fixture);path=code/name
    path.write_text('# Changed source\n')
    with pytest.raises(ValueError,match='checksum differs'):
        install(fixture)
    assert path.read_text()=='# Changed source\n'


def test_changed_settings_and_unidentified_outputs_fail_closed(fixture):
    install(fixture)
    with pytest.raises(ValueError,match='paths/settings changed'):
        launcher.restore(fixture.output,fixture.training/'different',fixture.images)
    path=fixture.output/'inference_launcher_config.json'
    cfg=json.loads(path.read_text());cfg['expected_epoch']=100;path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='settings checksum'):
        launcher.restore(fixture.output)
    (fixture.output/launcher.STATE_NAME).unlink()
    write_result(fixture.output)
    with pytest.raises(ValueError,match='lack an inference bootstrap'):
        install(fixture)


def test_input_output_boundaries_and_source_symlinks(fixture,tmp_path):
    with pytest.raises(ValueError,match='separate'):
        launcher.settings(fixture.training,fixture.images,fixture.training/'predictions')
    with pytest.raises(ValueError,match='separate'):
        launcher.settings(fixture.training,fixture.images,fixture.base)
    _,code,_=install(fixture)
    saved=code/'tools/chase_shape_test_inference.py'
    external=tmp_path/'external.py';external.write_bytes(saved.read_bytes())
    saved.unlink();saved.symlink_to(external)
    with pytest.raises(ValueError,match='Unsafe saved source'):
        launcher.restore(fixture.output)


def test_installer_preserves_cuda_and_numpy_abi(fixture,monkeypatch):
    install(fixture)
    versions={'torch':'2.11.0+cu130','torchvision':'0.26.0+cu130','numpy':'2.4.6',
        'segmentation-models-pytorch':'0.4.0','timm':'1.0.20','pycocotools':'2.0.11'}
    monkeypatch.setattr(importlib.metadata,'version',lambda name:versions[name])
    monkeypatch.setattr(importlib.util,'find_spec',lambda name:object())
    calls=[]
    monkeypatch.setattr(subprocess,'run',lambda command,**kw:calls.append((command,kw)))
    launcher.ensure_dependencies(fixture.output)
    command,kw=calls.pop()
    assert kw['check'] and 'segmentation-models-pytorch==0.5.0' in command
    assert not any(value.startswith(('torch==','torchvision==','ultralytics')) for value in command)
    assert (fixture.output/'keep_colab_runtime.txt').read_text().splitlines()==[
        'torch==2.11.0+cu130','torchvision==0.26.0+cu130','numpy==2.4.6']


def test_worker_traceback_reaches_console_and_drive_log(tmp_path,capsys):
    logfile=tmp_path/'inference.log'
    command=[sys.executable,'-u','-c','print("image progress",flush=True); raise RuntimeError("fixture failure")']
    with pytest.raises(subprocess.CalledProcessError):
        launcher.monitored(command,logfile,tmp_path,{})
    assert 'RuntimeError: fixture failure' in logfile.read_text()
    assert 'image progress' in capsys.readouterr().out


def test_interrupt_stops_only_owned_process_group(tmp_path,monkeypatch):
    killed=[]
    class Process:
        pid=99999999
        calls=0
        def poll(self):return None if self.calls<2 else -15
        def wait(self,**kwargs):
            self.calls+=1
            if self.calls==1:raise KeyboardInterrupt()
            return -15
    process=Process()
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**kw:process)
    monkeypatch.setattr(launcher.os,'killpg',lambda pid,sig:killed.append((pid,sig)))
    with pytest.raises(KeyboardInterrupt):
        launcher.monitored(['fixture'],tmp_path/'log',tmp_path,{})
    assert killed==[(process.pid,signal.SIGTERM)]


def test_run_launches_only_inference_with_saved_sources(fixture,monkeypatch):
    cfg,code,_=install(fixture)
    monkeypatch.setattr(launcher,'ensure_dependencies',lambda output:None)
    calls=[]
    def child(command,log,cwd,env):
        calls.append(command)
        assert cwd==code/'tools' and env['TORCH_CUDNN_V8_API_DISABLED']=='1'
        assert env['PYTHONUNBUFFERED']=='1' and log==fixture.output/'inference.log'
        write_result(fixture.output)
    monkeypatch.setattr(launcher,'monitored',child)
    result=launcher.run(fixture.training,fixture.images,fixture.output)
    assert result['images_completed']==180 and len(calls)==1
    assert calls[0][2]==str(code/'tools/chase_shape_test_inference.py')
    assert '--training-root' in calls[0] and '--test-images' in calls[0]
    assert '--phase' not in calls[0] and '--run-until' not in calls[0]


@pytest.mark.parametrize('changes',[{'status':'partial'},{'images_completed':179},{'checkpoint_epoch':100}])
def test_partial_or_wrong_checkpoint_cannot_be_downloaded(fixture,changes):
    install(fixture);write_result(fixture.output,**changes)
    with pytest.raises((RuntimeError,ValueError)):
        launcher.checked_result(fixture.output)


def test_csv_archive_hashes_and_output_paths_are_verified(fixture,tmp_path):
    install(fixture);result=write_result(fixture.output)
    assert launcher.checked_result(fixture.output)==result
    Path(result['csv']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        launcher.checked_result(fixture.output)
    outside=tmp_path/'other.zip';outside.write_bytes(b'outside')
    write_result(fixture.output,archive=str(outside),archive_sha256=launcher.sha(outside))
    with pytest.raises(ValueError,match='unsafe'):
        launcher.checked_result(fixture.output)


@pytest.fixture
def notebook(fixture,tmp_path,monkeypatch):
    document=json.loads(NOTEBOOK.read_text())
    cells=[c for c in document['cells'] if c['cell_type']=='code']
    assert len(cells)==1 and not cells[0]['outputs'] and cells[0]['execution_count'] is None
    source=''.join(cells[0]['source']);compile(source,str(NOTEBOOK),'exec')
    assert 'base64' not in source.lower() and len(source)<9000
    checkout=tmp_path/'checkout'
    source=(source.replace('/content/drive/MyDrive/filament-segmentation-2026',str(fixture.base))
        .replace('/content/solar_chase_test_github',str(checkout))
        .replace('https://github.com/ikjoo5/solar_filament.git',str(fixture.repository)))
    downloads=[];commands=[]
    fake=types.ModuleType('google.colab')
    fake.drive=types.SimpleNamespace(mount=lambda *a:None)
    fake.files=types.SimpleNamespace(download=lambda path:downloads.append(path))
    monkeypatch.setitem(sys.modules,'google.colab',fake)
    real_popen=subprocess.Popen
    def popen(command,**kwargs):
        if len(command)>2 and command[2].endswith('chase_shape_test_colab.py'):
            commands.append(command)
            assert kwargs['start_new_session'] and kwargs['stdout']==subprocess.PIPE
            assert kwargs['stderr']==subprocess.STDOUT and kwargs['text']
            repo=Path(command[2]).parents[1]
            launcher.install(repo,fixture.training,fixture.images,fixture.output)
            write_result(fixture.output)
            return types.SimpleNamespace(wait=lambda **kw:0,poll=lambda:0,pid=999999,
                                         stdout=io.StringIO('native image1/180\n'))
        return real_popen(command,**kwargs)
    monkeypatch.setattr(subprocess,'Popen',popen)
    return types.SimpleNamespace(fixture=fixture,source=source,checkout=checkout,downloads=downloads,
        commands=commands,execute=lambda:exec(compile(source,'test_prediction_cell','exec'),{}))


def test_one_cell_first_execution_and_offline_code_resume(notebook,capsys):
    item=notebook;item.execute()
    assert len(item.downloads)==2 and 'native image1/180' in capsys.readouterr().out
    shutil.rmtree(item.checkout);shutil.rmtree(item.fixture.repository)
    item.execute()
    assert not item.checkout.exists() and len(item.downloads)==4
    assert str(item.fixture.output/'code') in item.commands[-1][2]


def test_notebook_rejects_modified_launcher_before_execution(notebook):
    item=notebook;item.execute()
    _,code,_=launcher.restore(item.fixture.output)
    (code/'tools/chase_shape_test_colab.py').write_text('raise RuntimeError("must not run")')
    with pytest.raises(ValueError,match='checksum changed'):
        item.execute()
    assert len(item.commands)==1


@pytest.mark.parametrize('exit_code',[0,7])
def test_notebook_relays_real_child_stdout_stderr_and_exit_status(capsys,exit_code):
    source=''.join(next(c['source'] for c in json.loads(NOTEBOOK.read_text())['cells'] if c['cell_type']=='code'))
    tree=ast.parse(source)
    index=next(i for i,node in enumerate(tree.body) if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='process' for t in node.targets))
    block=ast.Module(body=tree.body[index:index+2],type_ignores=[])
    namespace=dict(subprocess=subprocess,os=launcher.os,signal=signal,
        command=[sys.executable,'-u','-c',
            'import sys; print("image1/180",flush=True); print("visible stderr",file=sys.stderr,flush=True); '
            'sys.exit('+str(exit_code)+')'])
    if exit_code:
        with pytest.raises(subprocess.CalledProcessError) as error:
            exec(compile(block,'output_bridge','exec'),namespace)
        assert error.value.returncode==exit_code
    else:
        exec(compile(block,'output_bridge','exec'),namespace)
    output=capsys.readouterr().out
    assert 'image1/180' in output and 'visible stderr' in output
    assert namespace['process'].stdout.closed
