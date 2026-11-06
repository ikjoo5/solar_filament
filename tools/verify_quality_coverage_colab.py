"""Validate packaged Colab bytes and launcher orchestration without a GPU."""
from __future__ import annotations

import ast
import base64
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch
import zipfile
import zlib

import build_quality_coverage_colab as builder


def verify():
    manifest=json.loads((builder.DEST/(builder.STEM+'_manifest.json')).read_text())
    first=Path(manifest['copy_cell']).read_text();resume=Path(manifest['resume_cell']).read_text()
    compile(first,'first','exec');compile(resume,'resume','exec')
    tree=ast.parse(first)
    payload=next(node.args[0].value for node in ast.walk(tree)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
        and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant))
    raw=zlib.decompress(base64.b64decode(payload));bundle=json.loads(raw)
    assert hashlib.sha256(raw).hexdigest()==manifest['bundle_sha256']
    for name,source in bundle['sources'].items():
        assert source==(builder.REPO/'tools'/name).read_text(),name
        assert hashlib.sha256(source.encode()).hexdigest()==manifest['source_sha256'][name]
    for name,encoded in bundle['assets'].items():
        value=base64.b64decode(encoded)
        assert value==builder.ASSETS[name].read_bytes(),name
        assert hashlib.sha256(value).hexdigest()==manifest['asset_sha256'][name]
    notebook=json.loads(Path(manifest['notebook']).read_text())
    assert ''.join(notebook['cells'][1]['source'])==first
    assert ''.join(notebook['cells'][3]['source'])==resume
    with zipfile.ZipFile(manifest['package']) as archive:
        assert archive.testzip() is None
        for name,source in bundle['sources'].items():assert archive.read('tools/'+name)==source.encode()
        for name,encoded in bundle['assets'].items():assert archive.read('assets/'+name)==base64.b64decode(encoded)
    assert hashlib.sha256(Path(manifest['package']).read_bytes()).hexdigest()==manifest['package_sha256']

    checks=['cell syntax','bundle identity','source exact match','asset exact match','notebook parity','zip CRC']
    with tempfile.TemporaryDirectory(prefix='coverage_colab_',dir='/tmp') as directory:
        base=Path(directory)/'drive';cache=Path(directory)/'cache'
        root=base/'fixed_split_experiments/quality_coverage_t4_v1'
        old=base/'fixed_split_experiments/mask_quality_v1';old.mkdir(parents=True)
        (old/'best.pt').write_bytes(b'stub checkpoint');(old/'corpus.json').write_text('{}')
        (old/'proposals').mkdir();(base/'MAGFiLO_1.0_Kaggle_2026/train/train_images').mkdir(parents=True)
        phases=[];downloads=[];failure=[False];preflight_failure=[False];preflight_calls=[]
        colab=types.ModuleType('google.colab')
        colab.drive=types.SimpleNamespace(mount=lambda path:None)
        colab.files=types.SimpleNamespace(download=lambda path:downloads.append(path))
        google=types.ModuleType('google');google.colab=colab

        class Process:
            pid=987654321
            def __init__(self,command,**kwargs):
                phase=command[command.index('--phase')+1];device=command[command.index('--device')+1]
                phases.append((phase,device))
                assert kwargs['start_new_session']
                assert kwargs['cwd']==Path(command[2]).parent
                assert kwargs['env']['TORCH_CUDNN_V8_API_DISABLED']=='1'
                assert command[command.index('--epochs')+1]=='30'
                self.bad=phase=='train' and failure[0]
                if phase=='prepare':
                    assert device=='cpu';cache.mkdir(parents=True,exist_ok=True)
                    (cache/'restored').write_text('ready')
                else:
                    assert phase=='train' and device=='cuda' and (cache/'restored').is_file()
                    archive=root/'results.zip';archive.write_bytes(b'stub downloadable result')
                    (root/'result.json').write_text(json.dumps(dict(archive=str(archive),status='paused',epochs_completed=5)))
                self.stdout=io.StringIO('Traceback: deliberately simulated child failure\n' if self.bad else phase+' complete\n')
            def wait(self,timeout=None):return int(self.bad)

        def rewrite(text):
            return text.replace('/content/drive/MyDrive/filament-segmentation-2026',str(base)).replace('/content/solar_quality_coverage_v1/roi',str(cache))
        def execute(text):
            with contextlib.redirect_stdout(io.StringIO()):exec(compile(rewrite(text),'smoke-colab','exec'),{})
        def run_preflight(command,**kwargs):
            assert command[1:3]==['-u','-c'] and 'torch.cuda.is_available()' in command[3]
            assert kwargs['env']['TORCH_CUDNN_V8_API_DISABLED']=='1'
            preflight_calls.append(command)
            return types.SimpleNamespace(returncode=int(preflight_failure[0]))
        with patch.dict(sys.modules,{'google':google,'google.colab':colab}), \
             patch.object(importlib.util,'find_spec',return_value=object()), \
             patch.object(subprocess,'Popen',Process), \
             patch.object(subprocess,'run',side_effect=run_preflight), \
             patch('os.killpg',return_value=None):
            execute(first)
            assert phases==[('prepare','cpu'),('train','cuda')]
            assert len(downloads)==1
            identity=(root/'current_code.json').read_bytes()
            (cache/'restored').unlink();cache.rmdir()
            execute(resume)
            assert (cache/'restored').is_file() and len(phases)==4 and len(downloads)==2
            execute(first)
            assert (root/'current_code.json').read_bytes()==identity and len(phases)==6
            assert len(preflight_calls)==3
            checks+=['first launcher order/environment','resume after cache deletion','same first cell idempotence','result download']
            preflight_failure[0]=True
            try:execute(resume)
            except RuntimeError as error:assert 'GPU 사전 확인' in str(error)
            else:raise AssertionError('CUDA preflight failure was ignored')
            assert len(phases)==6
            preflight_failure[0]=False
            checks+=['CUDA preflight failure stops before ROI prepare']
            code=root/'code'/manifest['bundle_sha256'][:16]/'quality_coverage_finetune.py'
            original=code.read_bytes();code.write_bytes(original+b'\n# changed\n')
            try:execute(resume)
            except ValueError:pass
            else:raise AssertionError('Changed source was accepted')
            code.write_bytes(original)
            try:execute(resume.replace('BATCH_SIZE = 16','BATCH_SIZE = 8'))
            except ValueError:pass
            else:raise AssertionError('Changed launch settings were accepted')
            assert len(phases)==6
            failure[0]=True
            try:execute(resume)
            except RuntimeError as error:assert '종료 코드 1' in str(error)
            else:raise AssertionError('Child failure was suppressed')
            assert 'deliberately simulated' in (root/'colab_train.log').read_text()
            checks+=['tampered source rejected','changed schedule rejected','child error retained and raised']
    cli=subprocess.run([sys.executable,str(builder.REPO/'tools/quality_coverage_finetune.py'),'--help'],
        cwd=builder.REPO,capture_output=True,text=True,check=True)
    assert '--assets' in cli.stdout and '--phase' in cli.stdout and '--run-until' in cli.stdout
    subprocess.run([sys.executable,'-c',
        'import sys;sys.path.insert(0,"tools");import quality_coverage_finetune,quality_coverage_sampling,quality_append_api,filament_eval,submission_inference'],
        cwd=builder.REPO,capture_output=True,text=True,check=True)
    subprocess.run([sys.executable,'-m','compileall','-q',*[str(builder.REPO/'tools'/name) for name in builder.MODULES]],
        cwd=builder.REPO,capture_output=True,text=True,check=True)
    checks+=['trainer CLI smoke','all five module imports','compileall']
    output=dict(status='passed',checks=checks,source_sha256=manifest['source_sha256'],
        bundle_sha256=manifest['bundle_sha256'],gpu_training_run=False,
        note='Launcher lifecycle uses simulated children; module imports and CLI checks run actual Python. No GPU training was run.')
    target=builder.ART/'quality_coverage_colab_validation_28a5be';target.mkdir(exist_ok=True)
    (target/'package_verification.json').write_text(json.dumps(output,indent=2)+'\n')
    return output


if __name__=='__main__':print(json.dumps(verify(),indent=2))
