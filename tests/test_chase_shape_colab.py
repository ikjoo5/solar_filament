"""Exercise source pinning, fresh-runtime resume and subprocess boundaries on CPU."""
from contextlib import contextmanager
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import io
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import types
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('chase_shape_colab', ROOT/'tools/chase_shape_colab.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)
NOTEBOOK = ROOT/'notebooks/SolarFilament_CHASE_Shape_T4.ipynb'


def git(path, *args):
    return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()


@pytest.fixture
def fixture(tmp_path):
    repository=tmp_path/'remote'; repository.mkdir()
    for name in launcher.SOURCE_FILES:
        path=repository/name; path.parent.mkdir(parents=True,exist_ok=True)
        if name=='tools/chase_shape_colab.py':
            shutil.copyfile(ROOT/name,path)
        else:
            path.write_text('{}\n' if name.endswith('.json') else '# Fixture source, never trained.\n')
    subprocess.run(['git','init','--initial-branch=main',str(repository)],check=True,capture_output=True)
    git(repository,'config','user.name','Colab test')
    git(repository,'config','user.email','test@example.invalid')
    git(repository,'add','.')
    git(repository,'commit','-m','CPU fixture')
    base=tmp_path/'drive/MyDrive/filament-segmentation-2026'
    output=base/'fixed_split_experiments'/launcher.SCOPE
    cache=tmp_path/'local-cache'
    return types.SimpleNamespace(repository=repository,base=base,output=output,cache=cache,
                                 commit=git(repository,'rev-parse','HEAD'))


def install(item):
    return launcher.install(item.repository,item.base,item.output,item.cache)


def populate_inputs(config):
    for key in ('annotations','train_split','val_split','clean_background','baseline_checkpoint'):
        path=Path(config[key]); path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'fixture')
    Path(config['images']).mkdir(parents=True,exist_ok=True)


def write_result(output,status='paused',epoch=30):
    archive=output/'fixture_results.zip'; archive.write_bytes(b'fixture result archive')
    result=dict(status=status,epochs_completed=epoch,output_root=str(output),archive=str(archive),
                archive_sha256=launcher.sha(archive))
    (output/'latest_result.json').write_text(json.dumps(result))
    return result


def test_saved_code_restores_after_checkout_disappears(fixture):
    config,path,code,state=install(fixture)
    assert state['source_commit']==fixture.commit and not state['source_worktree_dirty']
    assert config['clean_background_provenance']=='external_user_supplied'
    assert config['clean_background']==str(fixture.base/'clean_solar.jpg')
    assert 'URL not provided' in config['clean_background_source_note']
    assert 'best.pt' in config['baseline_checkpoint']
    assert set(json.loads((code/'source_manifest.json').read_text()))==set(launcher.SOURCE_FILES)
    for name in launcher.SOURCE_FILES:
        assert (code/name).read_bytes()==(fixture.repository/name).read_bytes()
    initial=(fixture.output/launcher.STATE_NAME).read_bytes()
    shutil.rmtree(fixture.repository)
    restored=launcher.restore(fixture.output,fixture.base,fixture.cache)
    assert restored[:3]==(config,path,code)
    assert (fixture.output/launcher.STATE_NAME).read_bytes()==initial


@pytest.mark.parametrize('target', ['tools/chase_shape_training.py','configs/chase_ahas2024_2025_manifest.json'])
def test_modified_snapshot_never_overwritten(fixture,target):
    _,_,code,_=install(fixture)
    path=code/target; path.write_text('user modified content')
    with pytest.raises(ValueError,match='checksum changed'):
        install(fixture)
    assert path.read_text()=='user modified content'


def test_changed_config_or_background_path_cannot_resume(fixture):
    config,path,_,_=install(fixture)
    with pytest.raises(ValueError,match='differs from the requested'):
        launcher.install(fixture.repository,fixture.base,fixture.output,fixture.cache,
                         clean_background=fixture.base/'a_different_background.jpg')
    config['seed']=999; path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='configuration checksum changed'):
        launcher.restore(fixture.output)
    assert json.loads(path.read_text())['seed']==999


def test_snapshot_preflight_preserves_unidentified_outputs(fixture):
    fixture.output.mkdir(parents=True)
    (fixture.output/'latest_result.json').write_text('{}')
    with pytest.raises(ValueError,match='no bootstrap identity'):
        install(fixture)
    assert not (fixture.output/'code').exists()
    assert (fixture.output/'latest_result.json').read_text()=='{}'


def test_snapshot_rejects_symlink_escape(fixture,tmp_path):
    _,_,code,_=install(fixture)
    target=code/'tools/chase_shape_training.py'
    external=tmp_path/'external.py'; external.write_bytes(target.read_bytes())
    target.unlink(); target.symlink_to(external)
    with pytest.raises(ValueError,match='escapes'):
        launcher.restore(fixture.output)


def test_outputs_and_cache_cannot_overlap_original_data(fixture):
    with pytest.raises(ValueError,match='separate'):
        launcher.settings(fixture.base,fixture.base/'MAGFiLO_1.0_Kaggle_2026/train/results',
                          fixture.repository,fixture.cache)
    with pytest.raises(ValueError,match='local Colab disk'):
        launcher.settings(fixture.base,fixture.output,fixture.repository,'/content/drive/cache')


def test_dependency_install_preserves_colab_torch_and_numpy(fixture,monkeypatch):
    fixture.output.mkdir(parents=True)
    versions={'torch':'2.11.0+cu130','torchvision':'0.26.0+cu130','numpy':'2.1.0',
              'segmentation-models-pytorch':'0.4.0','timm':'1.0.20','pycocotools':'2.0.11'}
    monkeypatch.setattr(importlib.metadata,'version',lambda name:versions[name])
    monkeypatch.setattr(importlib.util,'find_spec',lambda name:object())
    calls=[]
    monkeypatch.setattr(subprocess,'run',lambda command,**kwargs:calls.append((command,kwargs)))
    launcher.ensure_dependencies(fixture.output)
    command,options=calls.pop()
    assert options['check'] and 'segmentation-models-pytorch==0.5.0' in command
    assert '--upgrade-strategy' in command and 'only-if-needed' in command
    assert not any(value.startswith(('torch==','torchvision==','ultralytics')) for value in command)
    assert (fixture.output/'keep_colab_runtime.txt').read_text().splitlines()==[
        'torch==2.11.0+cu130','torchvision==0.26.0+cu130','numpy==2.1.0']


def test_inner_worker_failure_keeps_traceback(tmp_path,capsys):
    logfile=tmp_path/'training.log'
    command=[sys.executable,'-u','-c','print("inner failure detail", flush=True); raise RuntimeError("fixture failure")']
    with pytest.raises(subprocess.CalledProcessError):
        launcher.monitored(command,logfile,tmp_path,{})
    assert 'RuntimeError: fixture failure' in logfile.read_text()
    assert 'inner failure detail' in capsys.readouterr().out


def test_interrupt_terminates_only_owned_child_group(tmp_path,monkeypatch):
    killed=[]
    class Process:
        pid=99999999
        calls=0
        def poll(self): return None if self.calls<2 else -15
        def wait(self,**kwargs):
            self.calls+=1
            if self.calls==1: raise KeyboardInterrupt()
            return -15
    process=Process()
    monkeypatch.setattr(subprocess,'Popen',lambda *args,**kwargs:process)
    monkeypatch.setattr(launcher.os,'killpg',lambda pid,sig:killed.append((pid,sig)))
    with pytest.raises(KeyboardInterrupt):
        launcher.monitored(['fixture'],tmp_path/'log',tmp_path,{})
    assert killed==[(process.pid,signal.SIGTERM)]


@pytest.mark.parametrize('phase,expected_calls',[('prepare',1),('all',2),('train',2)])
def test_gpu_preflight_only_for_training_and_saved_config_is_forwarded(fixture,monkeypatch,phase,expected_calls):
    config,path,code,_=install(fixture); populate_inputs(config)
    calls=[]
    monkeypatch.setattr(launcher,'ensure_dependencies',lambda output:None)
    def child(command,log,cwd,env):
        calls.append(command)
        assert cwd==code/'tools' and env['TORCH_CUDNN_V8_API_DISABLED']=='1'
        assert env['PYTHONUNBUFFERED']=='1'
        write_result(fixture.output)
    monkeypatch.setattr(launcher,'monitored',child)
    result=launcher.run(fixture.base,fixture.output,phase,30,fixture.cache)
    assert len(calls)==expected_calls and result['epochs_completed']==30
    assert calls[-1][-6:]==['--config',str(path),'--phase',phase,'--run-until','30']
    assert 'chase_shape_training.py' in calls[-1][2]
    if phase!='prepare': assert 'torch.float16' in calls[0][-1]


def test_result_zip_checksum_and_output_boundary(fixture,tmp_path):
    install(fixture); result=write_result(fixture.output)
    assert launcher.checked_result(fixture.output)==result
    Path(result['archive']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        launcher.checked_result(fixture.output)
    outside=tmp_path/'outside.zip'; outside.write_bytes(b'outside')
    result.update(archive=str(outside),archive_sha256=launcher.sha(outside))
    (fixture.output/'latest_result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='outside'):
        launcher.checked_result(fixture.output)


@pytest.fixture
def notebook(fixture,tmp_path,monkeypatch):
    document=json.loads(NOTEBOOK.read_text())
    cells=[c for c in document['cells'] if c['cell_type']=='code']
    assert len(cells)==1 and not cells[0]['outputs'] and cells[0]['execution_count'] is None
    source=''.join(cells[0]['source'])
    assert 'base64' not in source.lower() and 'PAYLOAD' not in source
    assert 'RUN_UNTIL_EPOCH = 100' in source and len(source)<9000
    compile(source,str(NOTEBOOK),'exec')
    checkout=tmp_path/'checkout'
    source=(source.replace('/content/drive/MyDrive/filament-segmentation-2026',str(fixture.base))
        .replace('/content/chase_shape_best84_recipe_t4_cache',str(fixture.cache))
        .replace('https://github.com/ikjoo5/solar_filament.git',str(fixture.repository))
        .replace('/content/solar_chase_shape_github',str(checkout)))
    downloads=[]; commands=[]
    fake=types.ModuleType('google.colab')
    fake.drive=types.SimpleNamespace(mount=lambda *args:None)
    fake.files=types.SimpleNamespace(download=lambda path:downloads.append(path))
    monkeypatch.setitem(sys.modules,'google.colab',fake)
    real_popen=subprocess.Popen
    def popen(command,**kwargs):
        if len(command)>2 and command[2].endswith('chase_shape_colab.py'):
            commands.append(command)
            assert kwargs['start_new_session']
            bootstrap=Path(command[2]); repo=bootstrap.parents[1]
            launcher.install(repo,fixture.base,fixture.output,fixture.cache)
            write_result(fixture.output)
            assert kwargs['stdout']==subprocess.PIPE and kwargs['stderr']==subprocess.STDOUT
            assert kwargs['text'] and kwargs['bufsize']==1
            return types.SimpleNamespace(wait=lambda **kw:0,poll=lambda:0,pid=999999,
                                         stdout=io.StringIO('Worker progress from saved launcher\n'))
        return real_popen(command,**kwargs)
    monkeypatch.setattr(subprocess,'Popen',popen)
    return types.SimpleNamespace(fixture=fixture,source=source,checkout=checkout,
                                 downloads=downloads,commands=commands,
                                 execute=lambda:exec(compile(source,'one_colab_cell','exec'),{}))


def test_one_cell_first_launch_and_network_free_runtime_reset(notebook):
    item=notebook; item.execute()
    state=json.loads((item.fixture.output/launcher.STATE_NAME).read_text())
    assert state['source_commit']==item.fixture.commit and len(item.downloads)==1
    shutil.rmtree(item.checkout)
    shutil.rmtree(item.fixture.repository)
    item.execute()
    assert not item.checkout.exists() and len(item.downloads)==2
    assert str(item.fixture.output/'code') in item.commands[-1][2]
    assert '--clean-background' in item.commands[-1]


def test_one_cell_rejects_changed_bootstrap_before_execution(notebook):
    item=notebook; item.execute()
    _,_,code,_=launcher.restore(item.fixture.output)
    (code/'tools/chase_shape_colab.py').write_text('raise RuntimeError("must not execute")')
    with pytest.raises(ValueError,match='Saved launcher changed'):
        item.execute()
    assert len(item.commands)==1 and len(item.downloads)==1


def test_one_cell_preserves_dirty_checkout(notebook):
    item=notebook; item.execute()
    shutil.rmtree(item.fixture.output)
    changed=item.checkout/'tools/chase_shape_colab.py'; changed.write_text('# user edit\n')
    with pytest.raises(ValueError,match='local edits'):
        item.execute()
    assert changed.read_text()=='# user edit\n' and len(item.commands)==1


@pytest.mark.parametrize('exit_code',[0,7])
def test_notebook_forwards_real_child_stdout_and_stderr_to_kernel(capsys,exit_code):
    """Kernel capture does not receive inherited child FDs; relay both streams."""
    document=json.loads(NOTEBOOK.read_text())
    source=''.join(next(c['source'] for c in document['cells'] if c['cell_type']=='code'))
    tree=ast.parse(source)
    start=next(i for i,node in enumerate(tree.body) if isinstance(node,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='process' for t in node.targets))
    assert isinstance(tree.body[start+1],ast.Try)
    block=ast.Module(body=tree.body[start:start+2],type_ignores=[])
    namespace=dict(subprocess=subprocess,os=launcher.os,signal=signal,
        command=[sys.executable,'-u','-c',
                 'import sys; print("epoch 22/100",flush=True); '
                 'print("stderr visible",file=sys.stderr,flush=True); sys.exit('+str(exit_code)+')'])
    if exit_code:
        with pytest.raises(subprocess.CalledProcessError) as error:
            exec(compile(block,'notebook_output_bridge','exec'),namespace)
        assert error.value.returncode==exit_code
    else:
        exec(compile(block,'notebook_output_bridge','exec'),namespace)
    captured=capsys.readouterr()
    assert 'epoch 22/100' in captured.out and 'stderr visible' in captured.out
    assert namespace['process'].stdout.closed


def test_cache_helper_is_pinned_and_survives_network_free_resume(tmp_path,monkeypatch):
    source=''.join(next(c['source'] for c in json.loads(NOTEBOOK.read_text())['cells']
                        if c['cell_type']=='code'))
    tree=ast.parse(source)
    definitions=[node for node in tree.body if
                 (isinstance(node,ast.FunctionDef) and node.name=='cache_tools') or
                 (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and
                    t.id in ('CACHE_HELPER_REV','CACHE_HELPER_SHA') for t in node.targets))]
    namespace=dict(OUTPUT=tmp_path,hashlib=hashlib,importlib=importlib,urllib=urllib)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),'cache_loader','exec'),namespace)
    payload=(ROOT/'tools/chase_shape_cache.py').read_bytes()
    assert hashlib.sha256(payload).hexdigest()==namespace['CACHE_HELPER_SHA']
    calls=[]
    def download(url,**kwargs):
        calls.append(url)
        assert '/'+namespace['CACHE_HELPER_REV']+'/' in url
        return io.BytesIO(payload)
    monkeypatch.setattr(urllib.request,'urlopen',download)
    helper=namespace['cache_tools']()
    assert callable(helper.backup) and callable(helper.restore) and len(calls)==1
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**kw:pytest.fail('resume requires no download'))
    assert namespace['cache_tools']().restore(tmp_path)['status']=='not_prepared'
    saved=next((tmp_path/'cache_support').glob('*.py'))
    saved.write_bytes(b'raise RuntimeError("must never execute changed code")')
    with pytest.raises(ValueError,match='checksum differs'):
        namespace['cache_tools']()


def test_backup_notebook_is_single_plain_cell_and_uses_same_pinned_helper():
    document=json.loads((ROOT/'notebooks/SolarFilament_CHASE_Cache_Backup.ipynb').read_text())
    cells=[c for c in document['cells'] if c['cell_type']=='code']
    assert len(cells)==1 and not cells[0]['outputs'] and cells[0]['execution_count'] is None
    source=''.join(cells[0]['source'])
    compile(source,'backup_notebook','exec')
    assert 'base64' not in source.lower() and 'cache_tools().backup(OUTPUT)' in source
    assert hashlib.sha256((ROOT/'tools/chase_shape_cache.py').read_bytes()).hexdigest() in source
    assert 'BACKUP COMPLETE' in source
