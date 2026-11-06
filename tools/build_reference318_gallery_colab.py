"""Build a single-code-cell, CPU-only, resumable reference318 visual review."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import zipfile
import zlib

REPO=Path(__file__).resolve().parents[1]
ART=REPO.parent/'artifacts'
DEST=ART/'downloads'
ASSET_ROOT=ART/'reference318_gallery_assets_20261103'
ASSETS={name:ASSET_ROOT/name for name in ('targets.json','own_baseline.csv','manifest.json')}
MODULES=('reference318_gallery.py',)
STEM='SolarFilament_Reference318_VisualReview'

CELL='''# CPU 시각화 전용: 학습·추론·제출 파일 변경 없이 원본과 마스크를 비교합니다.
# 같은 셀을 다시 실행하면 완료된 그림은 검증하고 재사용합니다.
# 공개 참조318개 / 원본126장 / 현재 불일치315개 + 대응된 대조3개입니다.
# 공개 참조는 정답이 아니며, BEST RAW는 참조와 가장 유사한 원시 후보를 사후 선택한 진단입니다.
CREATE_ZIP = True       # 전체 그림+오프라인 뷰어를 /content에 압축; Drive 용량을 중복 사용하지 않습니다.
AUTO_DOWNLOAD = True
SHOW_VIEWER = True
LIMIT_IMAGES = None     # None이면 전체126장. 테스트용 숫자를 넣으면 부분 결과라고 표시합니다.

from pathlib import Path
from google.colab import drive, files
from datetime import datetime, timezone
from collections import deque
import base64, csv, hashlib, html, importlib.util, json, os, signal, subprocess, sys, uuid, zipfile, zlib
drive.mount('/content/drive')
BASE = Path('/content/drive/MyDrive/filament-segmentation-2026')
IMAGE_DIR = BASE/'MAGFiLO_1.0_Kaggle_2026/test/test_images'
ROOT = BASE/'diagnostics/reference318_visual_review_v1'
if LIMIT_IMAGES is not None and (type(LIMIT_IMAGES) is not int or LIMIT_IMAGES < 1):
    raise ValueError('LIMIT_IMAGES는 None 또는 양의 정수여야 합니다.')
if not IMAGE_DIR.is_dir():
    raise FileNotFoundError('원본 test 이미지 폴더를 찾지 못했습니다. IMAGE_DIR만 실제 경로로 수정하세요: '+str(IMAGE_DIR))
ROOT.mkdir(parents=True,exist_ok=True)
raw=zlib.decompress(base64.b64decode("""__PAYLOAD__"""))
if hashlib.sha256(raw).hexdigest()!='__SHA__':
    raise ValueError('복사된 셀 내용이 완전하지 않습니다. 셀 전체를 다시 복사해 주세요.')
bundle=json.loads(raw);version='__VERSION__'
CODE=ROOT/'code'/version;ASSETS=ROOT/'assets'/version;OUTPUT=ROOT/'gallery'/version
def save_exact(path,data):
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('보관된 코드/참조가 변경됐습니다: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    pending.write_bytes(data);pending.replace(path)
for name,source in bundle['sources'].items():
    if Path(name).name!=name:raise ValueError('잘못된 코드 이름')
    compile(source,name,'exec');save_exact(CODE/name,source.encode())
for name,value in bundle['assets'].items():
    if Path(name).name!=name:raise ValueError('잘못된 입력 이름')
    save_exact(ASSETS/name,base64.b64decode(value))
manifest=dict(sources={name:hashlib.sha256(source.encode()).hexdigest() for name,source in bundle['sources'].items()},
              assets={name:hashlib.sha256(base64.b64decode(value)).hexdigest() for name,value in bundle['assets'].items()})
save_exact(CODE/'source_manifest.json',json.dumps(manifest,sort_keys=True,indent=2).encode())
for folder,key in ((CODE,'sources'),(ASSETS,'assets')):
    for name,expected in manifest[key].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('코드 또는 참조 해시가 다릅니다: '+name)
missing=[package for module,package in [('PIL','Pillow'),('numpy','numpy'),('pycocotools','pycocotools')]
         if importlib.util.find_spec(module) is None]
if missing:subprocess.run([sys.executable,'-m','pip','install',*missing],check=True)
command=[sys.executable,'-u',str(CODE/'reference318_gallery.py'),
         '--images',str(IMAGE_DIR),'--assets',str(ASSETS),'--output',str(OUTPUT)]
if LIMIT_IMAGES is not None:command+=['--limit-images',str(LIMIT_IMAGES)]
environment=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
log_path=ROOT/('visual_review_'+version+'.log');tail=deque(maxlen=50)
print('GPU 불필요. 원본 test 이미지와 저장된 마스크를 CPU로 그립니다.',flush=True)
print('결과 폴더:',OUTPUT,flush=True)
print('같은 셀 재실행 시 완료 그림을 재사용합니다. 로그:',log_path,flush=True)
with log_path.open('a',encoding='utf-8',buffering=1) as log:
    log.write('\\n=== '+datetime.now(timezone.utc).isoformat()+' ===\\n')
    process=subprocess.Popen(command,cwd=CODE,env=environment,start_new_session=True,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,errors='replace')
    try:
        for line in process.stdout:
            print(line,end='',flush=True);log.write(line);tail.append(line)
        status=process.wait()
        if status:
            print('\\n===== 실제 오류 마지막 출력 =====\\n'+''.join(tail),flush=True)
            raise RuntimeError('시각화 종료 코드 '+str(status)+'; 위 traceback과 로그를 확인하세요: '+str(log_path))
    except BaseException:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
        raise
    finally:
        process.stdout.close()
result=json.loads((OUTPUT/'result.json').read_text())
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
with (OUTPUT/'case_manifest.csv').open(newline='',encoding='utf-8-sig') as stream:
    all_cases=list(csv.DictReader(stream))
all_cases.sort(key=lambda row:(row['image'],row['reference_id']))
def _matched(row):
    return str(row['matched']).strip().lower() in {'true','1','yes'}
def _select_cases(panel='all',include_matched=False,image='all'):
    return [row for row in all_cases
            if str(row.get('rendered','true')).strip().lower() in {'true','1','yes'}
            and (panel=='all' or str(row['panel'])==str(panel))
            and (include_matched or not _matched(row))
            and (image=='all' or row['image']==image)]
def _gallery_path(relative):
    path=(OUTPUT/relative).resolve()
    if OUTPUT.resolve() not in path.parents or not path.is_file():
        raise ValueError('잘못된 그림 경로: '+str(path))
    return path
def show_reference_case(number=1,panel='all',include_matched=False,full=False,image='all'):
    """위젯이 없는 환경에서도 모든 항목을 번호로 볼 수 있습니다(1부터 시작)."""
    from IPython.display import display, Image as DisplayImage
    selected=_select_cases(panel,include_matched,image)
    if not selected:
        print('현재 필터에 해당하는 항목이 없습니다.');return
    if type(number) is not int or not 1<=number<=len(selected):
        raise ValueError('number는 1~'+str(len(selected))+' 범위입니다.')
    row=selected[number-1]
    print(str(number)+'/'+str(len(selected)),row['reference_id'],'참조 묶음',row['panel'],
          '현재 자체 마스크와 최대 IoU',row['baseline_max_iou'])
    display(DisplayImage(filename=str(_gallery_path(row['full_path'] if full else row['case_path']))))
    return row
if SHOW_VIEWER and all_cases:
    from IPython.display import display, HTML
    try:
        import ipywidgets as widgets
        panel=widgets.Dropdown(options=[('전체318 참조','all'),('누락 후보216','216'),('형상 교체102','102')],description='참조 묶음')
        controls=widgets.Checkbox(value=False,description='이미 대응된 대조3개 포함',indent=False)
        image_filter=widgets.Dropdown(options=[('모든 원본 이미지','all')]+[(name,name) for name in sorted({r['image'] for r in all_cases})],description='사진')
        mode=widgets.ToggleButtons(options=[('객체 확대','case_path'),('원본 전체 위치','full_path')])
        slider=widgets.IntSlider(value=1,min=1,max=max(1,len(_select_cases())),description='항목',continuous_update=False,
                                 layout=widgets.Layout(width='90%'))
        previous=widgets.Button(description='◀ 이전');following=widgets.Button(description='다음 ▶')
        heading=widgets.HTML();picture=widgets.Image(format='jpeg',layout=widgets.Layout(width='100%'))
        state={'rows':_select_cases(),'changing':False}
        def redraw(change=None):
            if state['changing']:return
            selected=state['rows']
            if not selected:
                picture.value=b'';heading.value='<b>현재 필터에 해당하는 항목이 없습니다.</b>'
                previous.disabled=following.disabled=True;return
            position=min(max(slider.value,1),len(selected));row=selected[position-1]
            picture.value=_gallery_path(row[mode.value]).read_bytes()
            status='대응됨(대조)' if _matched(row) else '현재 자체 출력과 IoU≤0.5'
            heading.value='<b>'+str(position)+' / '+str(len(selected))+' — '+html.escape(row['reference_id'])+'</b><br>'+\
                '참조 묶음 '+html.escape(str(row['panel']))+' · '+status+' · 최대 IoU '+html.escape(str(row['baseline_max_iou']))
            previous.disabled=position<=1;following.disabled=position>=len(selected)
        def reset_view(change=None):
            state['changing']=True
            state['rows']=_select_cases(panel.value,controls.value,image_filter.value)
            slider.value=1;slider.max=max(1,len(state['rows']));slider.disabled=not state['rows']
            state['changing']=False;redraw()
        panel.observe(reset_view,names='value');controls.observe(reset_view,names='value');image_filter.observe(reset_view,names='value')
        slider.observe(redraw,names='value');mode.observe(redraw,names='value')
        previous.on_click(lambda button:setattr(slider,'value',max(1,slider.value-1)))
        following.on_click(lambda button:setattr(slider,'value',min(slider.max,slider.value+1)))
        display(HTML('<p><b>공개 참조는 정답이 아닙니다.</b> 파랑=공개 참조, 주황=자체 최종 출력, 보라=참조와 가장 유사한 자체 원시 후보. '
                     'BEST RAW는 실제 제출 결과가 아닌 사후 진단입니다. 원본126장의 위치 그림도 위 버튼에서 볼 수 있습니다.</p>'))
        display(widgets.VBox([widgets.HBox([panel,controls]),image_filter,mode,
                             widgets.HBox([previous,following]),slider,heading,picture]))
        redraw()
    except ImportError:
        print('위젯이 없어 번호로 보는 함수를 준비했습니다. show_reference_case(번호) 또는 ZIP의 index.html을 사용하세요.')
        show_reference_case(1)
print('모든 그림의 오프라인 뷰어:',OUTPUT/'index.html')
print('검토 메모용 CSV:',OUTPUT/'human_review.csv')
print('흑점 의심 여부 등의 메모는 사람이 검토할 자료입니다. 이 셀은 test 메모로 학습하지 않습니다.')
if CREATE_ZIP:
    archive=Path('/content')/('SolarFilament_Reference318_VisualReview_'+version+'.zip')
    pending=archive.with_suffix('.pending')
    with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as bundle_zip:
        for path in sorted(OUTPUT.rglob('*')):
            if path.is_file() and not path.name.endswith('.pending'):
                bundle_zip.write(path,str(path.relative_to(OUTPUT)))
    pending.replace(archive)
    print('전체 시각화 ZIP:',archive,'/',round(archive.stat().st_size/1024**2,1),'MiB')
    print('ZIP을 해제한 뒤 index.html을 열면 모든 항목을 순서대로 볼 수 있습니다.')
    if result.get('status')!='complete':print('주의: 이 ZIP은 LIMIT_IMAGES에 따른 부분 결과입니다.')
    if AUTO_DOWNLOAD:
        try:files.download(str(archive))
        except Exception as exc:print('자동 다운로드가 중단됐습니다. Colab 파일 목록에서 위 ZIP을 내려받으세요:',exc)
'''

README='''# test 참조318개 전체 시각화

- 원본 test126장에 공개 참조318개(누락 후보216+형상 교체102)가 있습니다.
- 현재 자체 출력과 IoU>0.5로 대응하는 대조3개를 제외한315개를 기본 표시합니다. 대조군 포함 버튼으로318개 전부 볼 수 있습니다.
- 공개 참조는 숨겨진 test 정답이 아닙니다. BEST RAW는 참조와 가장 유사한 원시 후보를 사후 선택한 진단이며 실제 모델의 최종 선택이 아닙니다.

## 실행

1. OneCell.ipynb의 유일한 코드 셀 전체를 Colab에 복사해 실행합니다. CPU 런타임으로 충분합니다.
2. 기본 원본 폴더는 `MyDrive/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images`입니다.
3. 코드·참조·시각화는 Drive `diagnostics/reference318_visual_review_v1`에 저장됩니다. 같은 셀을 재실행하면 완료된 그림을 검증하고 재사용합니다.
4. 실행 후 Colab의 이전/다음·항목 슬라이더·사진 선택으로 전체315개를 볼 수 있습니다. 위젯이 없으면 `show_reference_case(번호)` 함수와 오프라인 HTML을 사용할 수 있습니다.
5. 전체 ZIP은 Drive 대신 `/content`에 만들고 다운로드합니다. ZIP을 해제해 `index.html`을 열면 오프라인에서 모든 그림을 볼 수 있습니다.

## 그림

- 원본126장 각각의 전체 위치 그림: 원본 / 공개 참조 / 자체 최종 마스크.
- 객체318개 각각의 확대 그림: 원본 확대 / 공개 참조 / 자체 최종 마스크 / 가장 유사한 자체 원시 후보.
- 그림 파일은 총444개입니다. 기본 뷰어는 불일치315개를 표시하며, 같은 사진에 여러 객체가 있을 수 있습니다.
- 파랑=공개 참조, 주황=자체 최종 출력, 보라=원시 후보. 원본 밝기는 유지하며 마스크는 반투명으로 겹칩니다.
- `case_manifest.csv`에는 원본 파일명·참조ID·crop좌표·현재 대응 여부·마스크 비교값이 있습니다.
- `human_review.csv`는 흑점 의심/필라멘트 의심 등의 사람 검토 메모용입니다. 재실행해도 작성한 메모를 덮어쓰지 않습니다. 이 프로그램은 test 검토 메모를 학습에 사용하지 않습니다.

GPU 추론·재학습·제출CSV 변경은 없습니다. 클라우드에서 실제 test JPEG 전체를 보유하지 않아 원본126장의 실제 렌더링은 사용자의 Colab/Drive에서 실행됩니다. 코드는 별도의 합성 fixture로 좌표·재개·출력 무결성을 검증합니다.
'''


def build():
    sources={name:(REPO/'tools'/name).read_text() for name in MODULES}
    for name,source in sources.items():compile(source,name,'exec')
    assets={name:path.read_bytes() for name,path in ASSETS.items()}
    targets=json.loads(assets['targets.json'])
    assert len(targets['images'])==126
    cases=[t for image in targets['images'] for t in image['targets']]
    assert len(cases)==318 and sum(bool(t['matched']) for t in cases)==3
    raw=json.dumps(dict(sources=sources,assets={name:base64.b64encode(data).decode() for name,data in assets.items()}),
                   sort_keys=True,separators=(',',':')).encode()
    sha=hashlib.sha256(raw).hexdigest()
    cell=CELL.replace('__PAYLOAD__',base64.b64encode(zlib.compress(raw,9)).decode()).replace('__SHA__',sha).replace('__VERSION__',sha[:16])
    compile(cell,'reference318_visual_review_colab','exec')
    DEST.mkdir(parents=True,exist_ok=True)
    copy=DEST/(STEM+'_CopyCell.txt');copy.write_text(cell)
    notebook=DEST/(STEM+'_OneCell.ipynb')
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3')),
            cells=[dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=cell.splitlines(keepends=True))])
    notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1))
    readme=DEST/(STEM+'_README.ko.md');readme.write_text(README)
    package=DEST/(STEM+'_Package.zip')
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in [copy,notebook,readme]:archive.write(path,path.name)
        for name,source in sources.items():archive.writestr('tools/'+name,source)
        for name,data in assets.items():archive.writestr('assets/'+name,data)
        assert archive.testzip() is None
    result=dict(bundle_sha256=sha,source_sha256={name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()},
                asset_sha256={name:hashlib.sha256(data).hexdigest() for name,data in assets.items()},
                copy_cell=str(copy),notebook=str(notebook),package=str(package),readme=str(readme),
                package_sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
                counts=dict(original_images=126,reference_targets=318,unmatched=315,matched_controls=3))
    (DEST/(STEM+'_manifest.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':build()
