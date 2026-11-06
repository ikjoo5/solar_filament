# Paste into the SAME Colab notebook. No GPU inference or model training.
import ast, subprocess, sys, tempfile, json
from pathlib import Path
prefix='''
import csv, hashlib, json, sys, uuid, zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import cv2, numpy as np
from pycocotools import mask as mu
from scipy.interpolate import UnivariateSpline
'''
capture=r'''
cv2.setNumThreads(2)
cache_root=RUN/'official_pq_validation'
def latest(name):
    candidates=sorted(cache_root.glob('*/'+name))
    assert candidates, '원시 캐시를 찾을 수 없습니다: '+name
    return candidates[-1]
old=latest('raw_views.jsonl');new=latest('raw_preprocess_views.jsonl')
old_meta=json.loads((old.parent/'summary.json').read_text())
new_meta=json.loads((new.parent/'summary.json').read_text())
assert old_meta['source_sha256']==new_meta['source_sha256'], '두 실험의 입력 파일이 다릅니다.'
for path in (ANN,BEST,TRAIN,VAL):
    with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    assert actual==old_meta['source_sha256'].get(path.name), '기존 캐시와 현재 파일이 다릅니다: '+str(path)
with VAL.open(newline='',encoding='utf-8-sig') as f:
    selection=list(csv.DictReader(f))
filenames={Path(r['file_name']).stem:r['file_name'] for r in selection}
tile_size=128;pad=24;selected={stem:set() for stem in filenames}
for path,threshold in [(old,.15),(new,.25)]:
    seen=set()
    for line in path.open(encoding='utf-8'):
        row=json.loads(line);stem=row['image']
        assert stem in filenames and row['shape']==[2048,2048]
        assert row.get('coordinate_space')=='original', '원본 좌표 캐시가 필요합니다.'
        key=(stem,row['view']);assert key not in seen, '캐시 영상/뷰 중복'
        seen.add(key)
        for instance in row['instances']:
            if instance['confidence']<threshold:continue
            rle={'size':[2048,2048],'counts':instance['rle'].encode('ascii')}
            x,y,w,h=map(int,mu.toBbox(rle))
            if w==0 or h==0:continue
            assert 0<=x<x+w<=2048 and 0<=y<y+h<=2048, '마스크 좌표 오류'
            for top in range(max(0,(y-pad)//tile_size)*tile_size,
                             min(2048,y+h+pad),tile_size):
                for left in range(max(0,(x-pad)//tile_size)*tile_size,
                                  min(2048,x+w+pad),tile_size):
                    selected[stem].add((top,left))
    assert {stem for stem,view in seen}==set(filenames), '캐시에 검증 영상이 누락되었습니다.'
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
out=RUN/'pixel_tile_exports'/stamp;out.mkdir(parents=True,exist_ok=False)
manifest=dict(schema_version=1,image_size=[2048,2048],tile_size=tile_size,
    preprocessing='baseline limb only; cx1024 cy1024 radius900 bins100 spline400 gain4',
    selection='Old cache conf>=.15 and new preprocessing conf>=.25 bounding boxes, padded24; no GT mask selection',
    source_sha256=old_meta['source_sha256'],images={})
# Reserve space for the repeated complete manifest, ZIP headers and filenames.
manifest_reserve=65536+sum(1024+len(filenames[stem].encode('utf-8'))*6+
    len(tiles)*(400+len(stem.encode('utf-8'))*6) for stem,tiles in selected.items())
payload_limit=min(22*1024*1024,24*1024*1024-manifest_reserve)
assert payload_limit>65536, '타일 목록이 너무 큽니다.'
parts=[];archive=None;used=0
def open_part():
    path=out/f'pixel_tiles_{stamp}_part{len(parts)+1:02d}.zip'
    parts.append(path)
    return zipfile.ZipFile(path,'x',compression=zipfile.ZIP_STORED)
for index,stem in enumerate(sorted(filenames),1):
    filename=filenames[stem];assert Path(filename).name==filename
    source=IMAGES/filename
    image=limb(cv2.imread(str(source),cv2.IMREAD_GRAYSCALE))
    with source.open('rb') as f:raw_hash=hashlib.file_digest(f,'sha256').hexdigest()
    info=dict(file_name=filename,raw_sha256=raw_hash,
              limb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),tiles=[])
    for top,left in sorted(selected[stem]):
        tile=image[top:top+tile_size,left:left+tile_size]
        ok,encoded=cv2.imencode('.png',tile,[cv2.IMWRITE_PNG_COMPRESSION,3])
        assert ok
        payload=encoded.tobytes()
        name=f'tiles/{stem}/{top}_{left}.png'
        entry_size=len(payload)+2*len(name.encode('utf-8'))+98
        if archive is None or used+entry_size>payload_limit:
            if archive is not None:archive.close()
            archive=open_part();used=0
        archive.writestr(name,payload);used+=entry_size
        info['tiles'].append(dict(path=name,x=left,y=top,width=tile.shape[1],height=tile.shape[0],
                                  sha256=hashlib.sha256(payload).hexdigest()))
    manifest['images'][stem]=info
    print(f'[{index}/{len(filenames)}] {filename}: {len(info["tiles"])} tiles',flush=True)
if archive is not None:archive.close()
if not parts:
    archive=open_part();archive.close()
manifest['parts']=[p.name for p in parts]
raw_manifest=json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8')
assert len(raw_manifest)+1024<manifest_reserve, '타일 목록 크기 검증 실패'
for path in parts:
    with zipfile.ZipFile(path,'a') as z:z.writestr('manifest.json',raw_manifest)
    assert path.stat().st_size<24*1024*1024, '한 묶음이 큽니다. 오류 내용을 보내주세요.'
    print(path,round(path.stat().st_size/1024/1024,2),'MiB')
Path(sys.argv[1]).write_text(json.dumps([str(p) for p in parts]),encoding='utf-8')
'''
def build_worker(source):
    tree=ast.parse(source)
    names={'DATA','RUN','IMAGES','ANN','BEST','TRAIN','VAL','yy','xx','rr','disk','radii','bins'}
    nodes=[];seen=set()
    for node in tree.body:
        if isinstance(node,ast.Assign):
            targets={n.id for t in node.targets for n in ast.walk(t) if isinstance(n,ast.Name)}
            if targets and targets<=names:
                nodes.append(node);seen|=targets
        elif isinstance(node,ast.FunctionDef) and node.name=='limb':
            nodes.append(node);seen.add('limb')
    assert {'IMAGES','RUN','rr','disk','radii','bins','limb'}<=seen, '기존 SCRIPT 구조를 확인해주세요.'
    worker=prefix+ast.unparse(ast.Module(body=nodes,type_ignores=[]))+'\n'+capture
    compile(worker,'pixel-export','exec')
    return worker

from google.colab import files
assert isinstance(globals().get('SCRIPT'),str), '기존 평가를 실행했던 Colab의 새 셀에서 실행하세요.'
worker=build_worker(SCRIPT)
work=Path(tempfile.mkdtemp(prefix='solar_pixels_'))
script=work/'export.py';script.write_text(worker,encoding='utf-8')
result=work/'archives.json'
subprocess.run([sys.executable,str(script),str(result)],check=True)
for archive in json.loads(result.read_text(encoding='utf-8')):
    files.download(archive)
