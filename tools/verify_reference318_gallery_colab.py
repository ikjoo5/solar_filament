"""Check the shipped one-cell bytes, orchestration and viewer filter logic."""
from __future__ import annotations
import ast,base64,contextlib,csv,hashlib,importlib.util,io,json,os
from pathlib import Path
import subprocess,sys,tempfile,types,zipfile,zlib
from unittest.mock import patch
from PIL import Image
import build_reference318_gallery_colab as builder


def verify():
    result=json.loads((builder.DEST/(builder.STEM+'_manifest.json')).read_text())
    cell=Path(result['copy_cell']).read_text();compile(cell,'copycell','exec')
    payload=next(node.args[0].value for node in ast.walk(ast.parse(cell))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
        and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant))
    raw=zlib.decompress(base64.b64decode(payload));bundle=json.loads(raw)
    assert hashlib.sha256(raw).hexdigest()==result['bundle_sha256']
    for name,source in bundle['sources'].items():
        assert source==(builder.REPO/'tools'/name).read_text()
        assert hashlib.sha256(source.encode()).hexdigest()==result['source_sha256'][name]
    for name,encoded in bundle['assets'].items():
        value=base64.b64decode(encoded)
        assert value==builder.ASSETS[name].read_bytes()
        assert hashlib.sha256(value).hexdigest()==result['asset_sha256'][name]
    nb=json.loads(Path(result['notebook']).read_text())
    assert len(nb['cells'])==1 and nb['cells'][0]['cell_type']=='code'
    assert ''.join(nb['cells'][0]['source'])==cell
    with zipfile.ZipFile(result['package']) as z:
        assert z.testzip() is None
        for name,source in bundle['sources'].items():assert z.read('tools/'+name)==source.encode()
        for name,encoded in bundle['assets'].items():assert z.read('assets/'+name)==base64.b64decode(encoded)
    assert hashlib.sha256(Path(result['package']).read_bytes()).hexdigest()==result['package_sha256']
    checks=['cell syntax','bundle digest','source exact match','asset exact match','one-cell notebook parity','package CRC']

    # Widget model stub exercises Python filter/event logic, not a live frontend.
    class Widget:
        def __init__(self,*args,**kwargs):
            object.__setattr__(self,'_observers',[]);object.__setattr__(self,'_clicks',[])
            for key,value in kwargs.items():object.__setattr__(self,key,value)
            if not hasattr(self,'value'):
                options=kwargs.get('options',[])
                object.__setattr__(self,'value',options[0][1] if options else '')
            object.__setattr__(self,'disabled',False)
        def __setattr__(self,key,value):
            old=getattr(self,key,None);object.__setattr__(self,key,value)
            for name,func in list(self._observers):
                if key==name and old!=value:func({'name':key,'old':old,'new':value})
        def observe(self,func,names):self._observers.append((names,func))
        def on_click(self,func):self._clicks.append(func)
        def click(self):
            if not self.disabled:
                for func in self._clicks:func(self)
    widgets=types.ModuleType('ipywidgets')
    for name in ['Dropdown','Checkbox','ToggleButtons','IntSlider','Button','HTML','Image','Layout','VBox','HBox']:
        setattr(widgets,name,Widget)

    with tempfile.TemporaryDirectory(dir='/tmp',prefix='ref318_colab_') as tmp:
        temp=Path(tmp);base=temp/'drive';content=temp/'content';content.mkdir()
        image_dir=base/'MAGFiLO_1.0_Kaggle_2026/test/test_images';image_dir.mkdir(parents=True)
        root=base/'diagnostics/reference318_visual_review_v1'
        downloads=[];calls=[];failure=[False];displayed=[]
        colab=types.ModuleType('google.colab')
        colab.drive=types.SimpleNamespace(mount=lambda path:None)
        colab.files=types.SimpleNamespace(download=lambda path:downloads.append(path))
        google=types.ModuleType('google');google.colab=colab
        fixture_rows=[dict(image='photoA',reference_id='photoA_1',panel='216',matched='false',rendered='true'),
                      dict(image='photoA',reference_id='photoA_2',panel='216',matched='true',rendered='true'),
                      dict(image='photoB',reference_id='photoB_1',panel='102',matched='false',rendered='true'),
                      dict(image='photoC',reference_id='photoC_1',panel='216',matched='false',rendered='false')]
        class Process:
            pid=987654321
            def __init__(self,command,**kwargs):
                calls.append(command)
                assert command[command.index('--images')+1]==str(image_dir)
                assert kwargs['start_new_session'] and kwargs['cwd']==Path(command[2]).parent
                assert kwargs['env']['OMP_NUM_THREADS']=='1'
                out=Path(command[command.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
                for directory in ['cases','full']:(out/directory).mkdir(exist_ok=True)
                rows=[]
                for row in fixture_rows:
                    row=dict(row,baseline_max_iou='.8' if row['matched']=='true' else '.1',
                             case_path='cases/'+row['reference_id']+'.jpg',full_path='full/'+row['image']+'.jpg')
                    rows.append(row)
                    if row['rendered']=='true':
                        Image.new('RGB',(8,8),(20,40,60)).save(out/row['case_path'])
                        Image.new('RGB',(8,8),(60,40,20)).save(out/row['full_path'])
                with (out/'case_manifest.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                (out/'result.json').write_text(json.dumps(dict(status='complete',images_completed=2,cases_total=3)))
                (out/'index.html').write_text('<p>fixture</p>')
                if not (out/'human_review.csv').exists():(out/'human_review.csv').write_text('reference_id,note\n')
                self.stdout=io.StringIO('Traceback: fixture failure\n' if failure[0] else 'fixture completed\n')
            def wait(self,timeout=None):return int(failure[0])
        def rewrite(source):
            return source.replace('/content/drive/MyDrive/filament-segmentation-2026',str(base)).replace("Path('/content')",repr(content).replace('PosixPath','Path'))
        def execute(source=cell):
            ns={}
            with contextlib.redirect_stdout(io.StringIO()):exec(compile(rewrite(source),'launcherfixture','exec'),ns)
            return ns
        import IPython.display
        with patch.dict(sys.modules,{'google':google,'google.colab':colab,'ipywidgets':widgets}), \
             patch.object(importlib.util,'find_spec',return_value=object()), \
             patch.object(subprocess,'Popen',Process), \
             patch.object(IPython.display,'display',side_effect=lambda *items:displayed.extend(items)), \
             patch('os.killpg',return_value=None):
            ns=execute()
            assert len(calls)==1 and len(downloads)==1
            assert content in Path(downloads[0]).parents and root not in Path(downloads[0]).parents
            with zipfile.ZipFile(downloads[0]) as z:assert 'index.html' in z.namelist() and z.testzip() is None
            assert len(ns['_select_cases']())==2 and len(ns['_select_cases'](include_matched=True))==3
            assert len(ns['all_cases'])==4 and not ns['_select_cases'](image='photoC',include_matched=True)
            assert len(ns['_select_cases'](panel='102'))==1
            ns['panel'].value='102';assert len(ns['state']['rows'])==1
            ns['image_filter'].value='photoA';assert not ns['state']['rows'] and ns['slider'].disabled
            ns['image_filter'].value='all';ns['panel'].value='all';ns['controls'].value=True
            assert len(ns['state']['rows'])==3 and ns['slider'].max==3
            ns['following'].click();assert ns['slider'].value==2
            ns['following'].click();assert ns['slider'].value==3 and ns['following'].disabled
            ns['previous'].click();assert ns['slider'].value==2
            ns['mode'].value='full_path';assert ns['picture'].value
            ns['controls'].value=False;assert len(ns['state']['rows'])==2 and ns['slider'].value==1
            try:ns['_gallery_path']('../../outside.jpg')
            except ValueError:pass
            else:raise AssertionError('Unsafe image path accepted')
            checks+=['CPU child orchestration','ZIP outside Drive','offline HTML packaged','default unmatched filter',
                     'all rendered cases accessible','partial unrendered cases excluded','panel/photo empty filters','viewer next previous and full view','gallery path confinement']
            code=root/'code'/result['bundle_sha256'][:16]/'reference318_gallery.py'
            source=code.read_bytes();execute();assert code.read_bytes()==source and len(calls)==2
            checks+=['same first cell rerun source stability']
            code.write_bytes(source+b'\n# tamper')
            try:execute()
            except ValueError:pass
            else:raise AssertionError('Changed embedded source accepted')
            code.write_bytes(source)
            failure[0]=True
            try:execute()
            except RuntimeError as exc:assert '종료 코드 1' in str(exc)
            else:raise AssertionError('Child failure suppressed')
            assert 'fixture failure' in (root/('visual_review_'+result['bundle_sha256'][:16]+'.log')).read_text()
            failure[0]=False
            checks+=['tampered source rejected','child traceback retained and failure raised']
            # Missing ipywidgets takes the numbered-view fallback, with every row still reachable.
            with patch.dict(sys.modules,{'ipywidgets':None}):
                fallback=execute()
                with contextlib.redirect_stdout(io.StringIO()):
                    selected=fallback['show_reference_case'](3,include_matched=True)
                assert selected['reference_id']=='photoB_1'
            checks+=['numbered fallback reaches last case']
    cli=subprocess.run([sys.executable,str(builder.REPO/'tools/reference318_gallery.py'),'--help'],capture_output=True,text=True,check=True)
    assert '--images' in cli.stdout and '--assets' in cli.stdout
    checks+=['real runtime CLI import smoke']
    output=builder.ART/'reference318_gallery_verification_20261103';output.mkdir(exist_ok=True)
    report=dict(status='passed',checks=checks,bundle_sha256=result['bundle_sha256'],source_sha256=result['source_sha256'],
                notes='Launcher subprocess and widget model are simulated; runtime CLI uses real Python. Full126-photo rendering happens in user Colab. Independent runtime fixtures are reported separately.')
    (output/'package_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':verify()
