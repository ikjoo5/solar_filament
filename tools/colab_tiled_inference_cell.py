# Paste into a NEW cell after the current restored SCRIPT configuration.
# GPU runtime (T4/A100) is required. This runs new tiled inference only.
import ast, subprocess, sys, tempfile
from pathlib import Path

PREFIX = r'''
import csv, hashlib, importlib.metadata, json, math, sys, uuid, zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import cv2, numpy as np, torch
from pycocotools import mask as mu
from scipy.interpolate import UnivariateSpline
from ultralytics import YOLO
'''
HELPERS = r'''
def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def encode_native_roi(mask,top,left):
    # Tight native ROI, column-major COCO RLE; no resizing or scale transform.
    assert mask.ndim==2 and 0<=top and 0<=left
    assert top+mask.shape[0]<=2048 and left+mask.shape[1]<=2048
    counts=[];previous=0
    for column in range(mask.shape[1]):
        transitions=np.flatnonzero(np.diff(np.r_[0,mask[:,column].astype(np.int8),0]))
        for start,end in transitions.reshape(-1,2):
            first=(left+column)*2048+top+int(start)
            last=(left+column)*2048+top+int(end)
            counts.extend([first-previous,last-first]);previous=last
    counts.append(2048*2048-previous)
    return mu.frPyObjects({'size':[2048,2048],'counts':counts},2048,2048)

def edge_sides(mask,top,left):
    sides=[]
    if top>0 and mask[:4,:].any():sides.append('top')
    if left>0 and mask[:,:4].any():sides.append('left')
    if top+1024<2048 and mask[-4:,:].any():sides.append('bottom')
    if left+1024<2048 and mask[:,-4:].any():sides.append('right')
    return sides

def encode_result(result,stem,size,top,left):
    confidences=(np.asarray(result.boxes.conf.cpu().numpy())
                 if result.boxes is not None else np.empty(0))
    count=len(confidences)
    assert confidences.ndim==1 and np.isfinite(confidences).all()
    assert ((confidences>=0)&(confidences<=1)).all()
    instances=[];empty=0
    if result.masks is None:
        assert count==0,'검출 상자는 있지만 인스턴스 마스크가 없습니다.'
    else:
        masks=result.masks.data.float().cpu().numpy()
        assert masks.shape==(count,1024,1024), ('crop 원본 좌표 마스크가 필요합니다.',masks.shape)
        for pixels,confidence in zip(masks,confidences):
            mask=pixels>=.5
            ys,xs=np.nonzero(mask)
            if len(ys)==0:
                empty+=1;continue
            sides=edge_sides(mask,top,left)
            y1,y2=int(ys.min()),int(ys.max())+1
            x1,x2=int(xs.min()),int(xs.max())+1
            rle=encode_native_roi(mask[y1:y2,x1:x2],top+y1,left+x1)
            instances.append({'confidence':float(confidence),'rle':rle['counts'].decode('ascii'),
                'edge_clipped':bool(sides),'edge_sides':sides})
    return {'schema_version':1,'image':stem,'view':f'tile{size}_y{top}_x{left}',
        'input_size':size,'crop_bbox':[left,top,1024,1024],'shape':[2048,2048],
        'coordinate_space':'original','raw_min_confidence':.05,'max_det':100,
        'max_det_reached':count>=100,'raw_box_count':count,'empty_mask_count':empty,
        'instances':instances}

def infer_image(image,stem,model):
    assert image.shape==(2048,2048) and image.dtype==np.uint8
    for top in (0,512,1024):
        for left in (0,512,1024):
            crop=np.ascontiguousarray(image[top:top+1024,left:left+1024])
            bgr=cv2.cvtColor(crop,cv2.COLOR_GRAY2BGR)
            for size in (1536,2048):
                result=model.predict(bgr,imgsz=size,conf=.05,iou=.7,retina_masks=True,
                    max_det=100,device=0,verbose=False)[0]
                yield encode_result(result,stem,size,top,left)
'''
RUN = r'''
assert torch.cuda.is_available(),'Colab 런타임에서 T4/A100 GPU를 선택하세요.'
cv2.setNumThreads(2)
source_sha256={path.name:digest(path) for path in (ANN,BEST,TRAIN,VAL)}
assert source_sha256[BEST.name]=='81c2a27d95821b5ede283169c0da8c08d360ecc4c2a60a1b9c6baf96d76bcccc','기존 best.pt와 다릅니다.'
cache_root=RUN/'official_pq_validation'
previous=sorted(path for path in cache_root.glob('*/summary.json')
    if (path.parent/'raw_views.jsonl').is_file() or (path.parent/'raw_preprocess_views.jsonl').is_file())
assert previous,'이전 원시 캐시 실험의 summary.json을 찾지 못했습니다.'
previous=previous[-1]
assert json.loads(previous.read_text())['source_sha256']==source_sha256,'이전 실험과 입력 파일이 다릅니다.'
with VAL.open(newline='',encoding='utf-8-sig') as handle:selection=list(csv.DictReader(handle))
val_names=sorted({row['file_name'] for row in selection})
assert val_names and len({row['image_id'] for row in selection})==len(selection),'validation ID를 확인하세요.'
assert all(Path(name).name==name and (IMAGES/name).is_file() for name in val_names),'validation 이미지 경로를 확인하세요.'
assert len({Path(name).stem for name in val_names})==len(val_names),'이미지 stem 중복'
model=YOLO(str(BEST),task='segment')
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
output=cache_root/stamp;output.mkdir(parents=True,exist_ok=False)
(output/'worker_source.py').write_text(Path(__file__).read_text(),encoding='utf-8')
raw_path=output/'raw_tiled_views.jsonl'
counters=Counter();image_sha256={};limb_sha256={};per_image={}
with raw_path.open('x',encoding='utf-8') as cache:
    for index,name in enumerate(val_names,1):
        stem=Path(name).stem;source=IMAGES/name
        image=limb(cv2.imread(str(source),cv2.IMREAD_GRAYSCALE))
        image_sha256[stem]=digest(source)
        limb_sha256[stem]=hashlib.sha256(image.tobytes()).hexdigest()
        views=set();counts=Counter()
        for row in infer_image(image,stem,model):
            assert row['view'] not in views;views.add(row['view'])
            cache.write(json.dumps(row,ensure_ascii=False)+'\n');cache.flush()
            counters['records']+=1;counters['instances']+=len(row['instances'])
            counters['max_det_records']+=int(row['max_det_reached'])
            counters['empty_masks']+=row['empty_mask_count']
            counters['edge_clipped_instances']+=sum(int(i['edge_clipped']) for i in row['instances'])
            counts[str(row['input_size'])]+=len(row['instances'])
        assert len(views)==18
        per_image[stem]=dict(counts)
        print(f'[{index}/{len(val_names)}] {name}: 18 views, masks={dict(counts)}',flush=True)
assert counters['records']==18*len(val_names)
versions={package:importlib.metadata.version(package) for package in
    ('torch','ultralytics','numpy','scipy','opencv-python','pycocotools')}
summary={'schema_version':1,'experiment':'overlapping_native1024_tiled_inference_v1',
    'scope':'New GPU candidate generation on the same development images; no GT mask parsing or scoring',
    'physical_images':len(val_names),'annotation_groups':len(selection),'views_per_image':18,
    'crop_size':1024,'crop_top_left_positions':[0,512,1024],'input_sizes':[1536,2048],
    'raw_min_confidence':.05,'nms_iou':.7,'retina_masks':True,'max_det':100,
    'edge_rule':'Record only: touches first/last4 crop pixels on a border inside the full image; never discarded',
    'source_sha256':source_sha256,'previous_summary':str(previous),'versions':versions,
    'gpu':torch.cuda.get_device_name(0),'counts':dict(counters),'per_image_mask_counts':per_image,
    'image_sha256':image_sha256,'limb_sha256':limb_sha256,
    'raw_tiled_views_sha256':digest(raw_path),'worker_source_sha256':digest(output/'worker_source.py')}
(output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
archive=output.parent/f'{stamp}_tiled_results.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as handle:
    for path in (raw_path,output/'summary.json',output/'worker_source.py'):
        handle.write(path,arcname=path.name)
Path(sys.argv[1]).write_text(str(archive),encoding='utf-8')
print(json.dumps({'archive':str(archive),'archive_mib':round(archive.stat().st_size/1024/1024,2),
    'images':len(val_names),'records':counters['records'],'instances':counters['instances'],
    'max_det_records':counters['max_det_records']},ensure_ascii=False,indent=2),flush=True)
'''

def build_worker(source):
    tree=ast.parse(source)
    names={'DATA','RUN','IMAGES','ANN','BEST','TRAIN','VAL','yy','xx','rr','disk','radii','bins'}
    nodes=[];seen=set()
    for node in tree.body:
        if isinstance(node,ast.Assign):
            targets={n.id for target in node.targets for n in ast.walk(target) if isinstance(n,ast.Name)}
            if targets and targets<=names:nodes.append(node);seen|=targets
        elif isinstance(node,ast.FunctionDef) and node.name=='limb':
            nodes.append(node);seen.add('limb')
    assert names|{'limb'}<=seen,'기존 SCRIPT 경로/limb 설정을 확인하세요.'
    worker=PREFIX+ast.unparse(ast.Module(body=nodes,type_ignores=[]))+'\n'+HELPERS+'\n'+RUN
    compile(worker,'tiled-inference','exec')
    return worker

assert isinstance(globals().get('SCRIPT'),str),'현재 설정 복구 셀을 먼저 실행하세요.'
subprocess.run([sys.executable,'-m','pip','install','-q','ultralytics==8.4.182',
    'pycocotools==2.0.11','scipy==1.17.0','opencv-python==5.0.0.93'],check=True)
from google.colab import files
worker=build_worker(SCRIPT)
work=Path(tempfile.mkdtemp(prefix='solar_tiled_'))
script=work/'infer.py';script.write_text(worker,encoding='utf-8')
result=work/'archive.txt'
subprocess.run([sys.executable,str(script),str(result)],check=True)
files.download(result.read_text(encoding='utf-8'))
