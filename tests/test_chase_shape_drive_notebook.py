"""Execute the plain Drive-recovery cell with local fixtures and fake Google APIs."""
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import types
import urllib.request

import pytest

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'notebooks/SolarFilament_CHASE_Drive_Recovery.ipynb'
HELPER=ROOT/'tools/chase_shape_drive_recovery.py'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source():
    cells=[c for c in json.loads(NOTEBOOK.read_text())['cells'] if c['cell_type']=='code']
    assert len(cells)==1 and cells[0]['outputs']==[] and cells[0]['execution_count'] is None
    text=''.join(cells[0]['source']);compile(text,str(NOTEBOOK),'exec')
    assert 'base64' not in text.lower() and len(text)<15000
    return text


def test_plain_cell_pins_the_published_helper_bytes():
    tree=ast.parse(source())
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
        and n.targets[0].id in ('HELPER_SHA','HELPER_COMMIT')}
    assert constants['HELPER_SHA']==digest(HELPER)
    assert constants['HELPER_COMMIT']=='5f87e09fc51dbe91f599450c3c7cab20d343f723'


@pytest.fixture
def cell(tmp_path,monkeypatch):
    base=tmp_path/'project';helper=tmp_path/'cached_helper.py';helper.write_bytes(HELPER.read_bytes())
    downloads=[];queries=[];mounts=[];auths=[]
    class Service:
        def files(self):return self
        def list(self,**kwargs):
            queries.append(kwargs)
            return types.SimpleNamespace(execute=lambda **kw:{'files':[]})
    service=Service()
    google=types.ModuleType('google');google.__path__=[]
    google_auth=types.ModuleType('google.auth');google_auth.default=lambda:('credentials',None)
    google.auth=google_auth
    colab=types.ModuleType('google.colab')
    colab.auth=types.SimpleNamespace(authenticate_user=lambda:auths.append(True))
    colab.drive=types.SimpleNamespace(mount=lambda *a,**kw:mounts.append((a,kw)))
    colab.files=types.SimpleNamespace(download=lambda path:downloads.append(path))
    api=types.ModuleType('googleapiclient');api.__path__=[]
    discovery=types.ModuleType('googleapiclient.discovery')
    def build(name,version,**kwargs):
        assert (name,version)==('drive','v3') and kwargs['credentials']=='credentials'
        return service
    discovery.build=build
    for name,module in [('google',google),('google.auth',google_auth),('google.colab',colab),
                        ('googleapiclient',api),('googleapiclient.discovery',discovery)]:
        monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**kw:pytest.fail('Unexpected network access'))
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**kw:pytest.fail('Unexpected inference process'))
    text=(source().replace('/content/drive/MyDrive/filament-segmentation-2026',str(base))
          .replace('/content/solar_chase_drive_recovery.py',str(helper))
          .replace('/content/solar_chase_epoch082_recovered',str(tmp_path/'runtime/recovered')))
    return types.SimpleNamespace(base=base,helper=helper,text=text,downloads=downloads,
                                 queries=queries,mounts=mounts,auths=auths)


def test_not_found_is_a_normal_report_without_gpu_or_subprocess(cell,capsys):
    namespace={};exec(cell.text,namespace)
    result=namespace['result']
    assert result['status']=='not_found' and result['search_complete']
    assert len(cell.queries)==2 and len(cell.auths)==1 and len(cell.mounts)==1
    assert 'trashed = false' in cell.queries[0]['q'] and 'trashed = true' in cell.queries[1]['q']
    assert cell.downloads==[result['report_path']] and Path(result['report_path']).is_file()
    assert not Path(result['report_path']).is_relative_to(cell.base)
    assert 'No training or model inference was started' in capsys.readouterr().out


@pytest.mark.parametrize('completed,corrupt',[(180,False),(179,False),(180,True)])
def test_recovered_file_reuses_launcher_and_requires_complete_valid_outputs(cell,monkeypatch,completed,corrupt):
    recovery=cell.base/'recovered_checkpoints/chase_shape_epoch082';recovery.mkdir(parents=True)
    checkpoint=recovery/'best_epoch082.pt';checkpoint.write_bytes(b'reviewed fixture weights')
    weight_sha=digest(checkpoint);report=recovery/'report.json';report.write_text('{}')
    cell.helper.write_text('CHECKPOINT_SHA = '+repr(weight_sha)+'\n'
        'def recover(service, output):\n    return '+repr(dict(status='recovered',
         checkpoint=str(checkpoint),report_path=str(report)))+'\n')
    text=cell.text.replace(digest(HELPER),digest(cell.helper))
    output=cell.base/'submissions/chase_shape_epoch082_native_t4_recovery_v2'
    manifest={'tools/chase_shape_test_colab.py':hashlib.sha256(b'# Fixture launcher\n').hexdigest()}
    canonical=json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    version=hashlib.sha256(canonical).hexdigest()[:16];snapshot=output/'code'/version
    (snapshot/'tools').mkdir(parents=True)
    (snapshot/'tools/chase_shape_test_colab.py').write_bytes(b'# Fixture launcher\n')
    manifest_path=snapshot/'source_manifest.json';manifest_path.write_text(json.dumps(manifest))
    (output/'test_bootstrap_state.json').write_text(json.dumps(dict(output_root=str(output),
        version=version,manifest_sha256=digest(manifest_path))))
    calls=[]
    class Process:
        stdout=io.StringIO('Fixture inference complete\n')
        def poll(self):return 0
        def wait(self,**kwargs):return 0
    def popen(command,**kwargs):
        calls.append(command)
        assert command[command.index('--checkpoint')+1]==str(checkpoint)
        assert kwargs['stdout']==subprocess.PIPE and kwargs['stderr']==subprocess.STDOUT
        assert '--run-until' not in command and '--phase' not in command
        csv=output/'submission.csv';csv.write_text('filament_id,segmentation_rle\n')
        archive=output/'result.zip';archive.write_bytes(b'ZIP fixture')
        result=dict(status='complete',images_completed=completed,checkpoint_epoch=82,
            checkpoint_sha256=weight_sha,output_root=str(output),csv=str(csv),archive=str(archive),
            csv_sha256=digest(csv),archive_sha256=digest(archive))
        (output/'latest_result.json').write_text(json.dumps(result))
        if corrupt:csv.write_text('changed')
        return Process()
    monkeypatch.setattr(subprocess,'Popen',popen)
    if completed!=180 or corrupt:
        with pytest.raises(ValueError):exec(text,{})
        assert cell.downloads==[]
    else:
        namespace={};exec(text,namespace)
        assert namespace['result']['images_completed']==180
        assert cell.downloads==[str(output/'submission.csv'),str(output/'result.zip')]
    assert len(calls)==1 and checkpoint.read_bytes()==b'reviewed fixture weights'
