"""CPU pixel/context descriptors for all frozen616/91 proposals, with resumable exports.

Only numpy, OpenCV and pycocotools are required. Targets are copied as labels in
a separate field; neither descriptor computation nor candidate inclusion uses them.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import zipfile

import cv2
import numpy as np
from pycocotools import mask as mu

INPUTS = dict(annotations='5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1',
    train_split='aa0733424815731c5c108ada91e45525eb17ca3a42b377bf19e3089d86aede7e',
    val_split='54c6a9aac5a3b70b3ecc6721109a8b0fe8ec020e90cb56e3848fccc92eadc61f',
    yolo='81c2a27d95821b5ede283169c0da8c08d360ecc4c2a60a1b9c6baf96d76bcccc')
CODE = {'fixed_split_mask_quality.py':'0cdf246a8ff545398038db64f602e50a643d10382be8e9b68b69f846ee8d5cdd',
    'submission_inference.py':'7c6a9d829025801482350b2f009a51a7ffb1eaa3d270cdce80a6a0691f815227',
    'evaluate_grouped_oof.py':'90f8ffc63d37ba326c990bd0723c865efb60cbf76fb2291ad918cfe6c757a7ad',
    'grouped_cv_training.py':'55a14f4dd7eaac8307463c40b8ad338bc868e98a7c6be8740ce62ebc443991c3'}
SIGMAS = (0, 1, 2)
RINGS = ('near', 'mid', 'far')
GEOMETRY = ('confidence','log1p_area','area_fraction','bbox_width_fraction','bbox_height_fraction',
    'log_aspect_ratio','extent','compactness','components','centroid_x_norm','centroid_y_norm',
    'radius_mean','radius_max','outside_disk_fraction','valid_inside_fraction',
    'near_valid_pixels','mid_valid_pixels','far_valid_pixels',
    'near_valid_fraction','mid_valid_fraction','far_valid_fraction')
SCALE_COLUMNS = ('inside_mean','inside_median','inside_p10','inside_p90','inside_std') + tuple(
    f'{ring}_{stat}' for ring in RINGS for stat in ('mean','median','mad','std')) + tuple(
    f'{ring}_dark_z' for ring in RINGS) + ('gradient_inside_mean','gradient_inside_p90') + tuple(
    f'gradient_{ring}_mean' for ring in RINGS) + ('texture_std_ratio','texture_gradient_ratio')
COLUMNS = list(GEOMETRY) + [f'sigma{sigma}_{name}' for sigma in SIGMAS for name in SCALE_COLUMNS]
SPEC = dict(schema_version=1, sigmas=list(SIGMAS), rings_pixels=[[0,3],[3,9],[9,25]],
    distance='OpenCV L2 distanceTransform maskSize5; rings exclude candidate mask',
    normalization='Per-image in-disk median and max(1.4826*MAD,1 grayscale DN); no across-image fitting',
    dark_contrast='(ring median - mask median)/max(1.4826 ring MAD,1 normalized grayscale DN)',
    gradients='Sobel3x3 magnitude at each Gaussian scale, scale1/8',
    solar_geometry='image center; radius width*900/2048; production2048 square',
    statistics_validity='In-disk pixels only; empty statistics are null with explicit pixel counts',
    rings_exclude_other_candidates=False, feature_rounding_decimals=7,
    candidate_selection='All original raw proposals, no train48 cap, no GT-based candidate selection',
    target_columns=['mean_matched_IoU','mean_match_probability'], targets_are_features=False,
    columns=COLUMNS, training=False, model_inference=False)


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(data): return hashlib.sha256(data).hexdigest()
def fingerprint(value): return sha(json.dumps(value, sort_keys=True).encode())
def file_sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    pending.replace(path)


def save_gzip(path, value):
    path=Path(path);pending=path.with_name(path.name+'.pending')
    payload=dict(value,checksum=fingerprint(value))
    data=json.dumps(payload,separators=(',',':'),allow_nan=False).encode()
    pending.write_bytes(gzip.compress(data,compresslevel=6,mtime=0));pending.replace(path)


def read_gzip(path):
    value=json.loads(gzip.decompress(Path(path).read_bytes()));checksum=value.pop('checksum',None)
    require(checksum==fingerprint(value),'Gzip payload checksum changed: '+str(path))
    return value


def checked_rle(counts, shape):
    """Bound COCO runs before entering the native decoder, including tiny masks."""
    require(isinstance(counts,str) and bool(counts),'Missing compressed RLE')
    previous=[];total=area=index=position=0;pixels=int(np.prod(shape))
    while position<len(counts):
        number=shift=0
        while True:
            require(position<len(counts),'Truncated RLE')
            digit=ord(counts[position])-48;position+=1
            require(0<=digit<=63,'Invalid RLE character')
            number|=(digit&31)<<shift;shift+=5
            require(shift<=65,'RLE overflow')
            if not digit&32:
                if digit&16:number|=-1<<shift
                break
        if index>2:number+=previous[-2]
        require(number>=0 and total+number<=pixels,'RLE outside image')
        total+=number;area+=number if index%2 else 0
        previous=(previous+[number])[-2:];index+=1
    require(total==pixels and area>0,'Invalid or empty proposal RLE')
    value=dict(size=list(shape),counts=counts.encode('ascii'))
    require(int(mu.area(value))==area,'RLE area disagreement')
    return value


def make_maps(image):
    require(image.ndim==2 and image.shape[0]==image.shape[1] and image.dtype==np.uint8,'Expected square grayscale uint8 image')
    size=image.shape[0];center=size/2;radius=size*900/2048
    yy,xx=np.ogrid[:size,:size];disk=(xx-center)**2+(yy-center)**2<=radius**2
    values=image[disk].astype(np.float32)/255
    median=float(np.median(values));mad=float(np.median(np.abs(values-median)))
    scale=max(1.4826*mad,1/255)
    normalized=(image.astype(np.float32)/255-median)/scale
    scales=[]
    for sigma in SIGMAS:
        smooth=cv2.GaussianBlur(normalized,(0,0),sigma) if sigma else normalized
        dx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3,scale=1/8)
        dy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3,scale=1/8)
        scales.append((smooth,cv2.magnitude(dx,dy)))
    return dict(shape=image.shape,center=center,radius=radius,disk=disk,scales=scales,
                one_DN=1/(255*scale),normalization=dict(median=median,mad=mad,scale=scale))


def statistics(values, extended=False):
    if not len(values):return dict(mean=None,median=None,mad=None,std=None,p10=None,p90=None)
    median=float(np.median(values))
    result=dict(mean=float(values.mean()),median=median,mad=float(np.median(np.abs(values-median))),std=float(values.std()))
    result.update(p10=float(np.percentile(values,10)) if extended else None,
                  p90=float(np.percentile(values,90)) if extended else None)
    return result


def features(maps, value, confidence):
    """Descriptor-only function: no target, role, split or annotation inputs."""
    h,w=maps['shape'];area=int(mu.area(value));x,y,bw,bh=map(int,mu.toBbox(value))
    x0,y0=max(0,x-27),max(0,y-27);x1,y1=min(w,x+bw+27),min(h,y+bh+27)
    decoded=mu.decode(value);mask=decoded[y0:y1,x0:x1].astype(bool)
    require(int(mask.sum())==area,'Mask coordinates do not match bbox')
    disk=maps['disk'][y0:y1,x0:x1];inside=mask&disk
    distance=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,5)
    annuli=[(distance>low)&(distance<=high)&~mask for low,high in SPEC['rings_pixels']]
    rings=[annulus&disk for annulus in annuli]
    yy,xx=np.nonzero(mask);rr=np.sqrt((xx+x0-maps['center'])**2+(yy+y0-maps['center'])**2)/maps['radius']
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    perimeter=sum(cv2.arcLength(contour,True) for contour in contours)
    result=dict(confidence=float(confidence),log1p_area=math.log1p(area),area_fraction=area/(h*w),
        bbox_width_fraction=bw/w,bbox_height_fraction=bh/h,log_aspect_ratio=math.log(bw/bh),
        extent=area/(bw*bh),compactness=4*math.pi*area/perimeter**2 if perimeter else None,
        components=cv2.connectedComponents(mask.astype(np.uint8))[0]-1,
        centroid_x_norm=(float(xx.mean())+x0-maps['center'])/maps['radius'],
        centroid_y_norm=(float(yy.mean())+y0-maps['center'])/maps['radius'],
        radius_mean=float(rr.mean()),radius_max=float(rr.max()),outside_disk_fraction=float((~disk[mask]).mean()),
        valid_inside_fraction=float(inside.sum()/area))
    for name,annulus,ring in zip(RINGS,annuli,rings):
        result[name+'_valid_pixels']=int(ring.sum())
        result[name+'_valid_fraction']=float(ring.sum()/annulus.sum()) if annulus.any() else 0.
    for sigma,(gray,gradient) in zip(SIGMAS,maps['scales']):
        gray=gray[y0:y1,x0:x1];gradient=gradient[y0:y1,x0:x1]
        ins=statistics(gray[inside],True);around=[statistics(gray[ring]) for ring in rings]
        prefix=f'sigma{sigma}_'
        for stat in ('mean','median','p10','p90','std'):result[prefix+'inside_'+stat]=ins[stat]
        for name,ring in zip(RINGS,around):
            for stat in ('mean','median','mad','std'):result[prefix+name+'_'+stat]=ring[stat]
            result[prefix+name+'_dark_z']=((ring['median']-ins['median'])/max(1.4826*ring['mad'],maps['one_DN'])
                if ring['median'] is not None and ins['median'] is not None else None)
        grad=gradient[inside]
        gm=float(grad.mean()) if len(grad) else None
        result[prefix+'gradient_inside_mean']=gm
        result[prefix+'gradient_inside_p90']=float(np.percentile(grad,90)) if len(grad) else None
        gradient_rings=[float(gradient[ring].mean()) if ring.any() else None for ring in rings]
        for name,v in zip(RINGS,gradient_rings):result[prefix+'gradient_'+name+'_mean']=v
        result[prefix+'texture_std_ratio']=ins['std']/max(around[-1]['std'],maps['one_DN']) if ins['std'] is not None and around[-1]['std'] is not None else None
        result[prefix+'texture_gradient_ratio']=gm/max(gradient_rings[-1],maps['one_DN']) if gm is not None and gradient_rings[-1] is not None else None
    vector=[None if result[name] is None else round(float(result[name]),7) for name in COLUMNS]
    require(all(v is None or math.isfinite(v) for v in vector),'Nonfinite descriptor')
    return vector


def load_corpus(root, expected_counts=(616,91), expected_totals=(53535,7172)):
    identity_bytes=(root/'corpus_identity.json').read_bytes();corpus_bytes=(root/'corpus.json').read_bytes()
    identity=json.loads(identity_bytes);corpus=json.loads(corpus_bytes)
    require(identity['inputs']==INPUTS and identity['code']==CODE,'Use the reviewed original fixed616/91 corpus')
    require(fingerprint(identity)==corpus['identity'],'Corpus fingerprint changed')
    counts=Counter();totals=Counter();names=set();stems=set()
    for row in corpus['records']:
        name=row['name'];stem=Path(name).stem;role=row['role']
        require(role in ('train','validation') and name==Path(name).name and '\\' not in name
                and name.endswith('.jpeg') and name not in names and stem not in stems,'Invalid or overlapping physical image roles')
        require(isinstance(row['proposal_count'],int) and row['proposal_count']>=0,'Invalid raw proposal count')
        names.add(name);stems.add(stem);counts[role]+=1;totals[role]+=row['proposal_count']
    require(tuple(counts[r] for r in ('train','validation'))==expected_counts,'Expected original616/91 physical images')
    require((corpus['training_images'],corpus['validation_images'])==expected_counts,'Corpus image totals differ')
    require(tuple(totals[r] for r in ('train','validation'))==expected_totals,'Expected all53535 train/7172 validation raw proposals')
    return corpus,identity,dict(counts),dict(totals),{'corpus_identity.json':identity_bytes,'corpus.json':corpus_bytes}


def checked_raw(path, row, corpus_id, image_sha):
    require(file_sha(path)==row['raw_sha'],'Original proposal bytes changed: '+row['name'])
    raw=read_gzip(path);origin=raw['identity']
    require(origin==dict(corpus=corpus_id,image_sha=image_sha,name=row['name'],role=row['role']),
            'Proposal/image/split provenance mismatch: '+row['name'])
    require(len(raw['instances'])==row['proposal_count'],'Raw proposal order/count changed')
    for instance in raw['instances']:
        confidence=instance['confidence'];target=instance['target']
        require(isinstance(confidence,(int,float)) and not isinstance(confidence,bool) and math.isfinite(confidence)
                and .05<=confidence<=1,'Invalid original confidence')
        require(isinstance(target,list) and len(target)==2 and all(isinstance(v,(int,float))
                and not isinstance(v,bool) and math.isfinite(v) and 0<=v<=1 for v in target), 'Invalid separate target label')
    return raw


def write_parts(output,common,files,max_bytes,export_id):
    entries={name:dict(bytes=len(data),sha256=sha(data)) for name,data in common.items()}
    entries.update({name:dict(bytes=path.stat().st_size,sha256=file_sha(path)) for name,path in files})
    base=dict(export_id=export_id,entries=entries)
    reserve=sum(len(v)+512 for v in common.values())+len(json.dumps(base).encode())+sum(len(n)+8 for n,_ in files)+65536
    groups=[];group=[];size=reserve
    for name,path in files:
        cost=path.stat().st_size+512
        require(reserve+cost<max_bytes,'One image cache exceeds archive part budget')
        if group and size+cost>max_bytes:groups.append(group);group=[];size=reserve
        group.append((name,path));size+=cost
    if group or not groups:groups.append(group)
    destination=output/'exports'/export_id[:16];destination.mkdir(parents=True,exist_ok=True);archives=[]
    for index,group in enumerate(groups,1):
        path=destination/f'candidate_context_{export_id[:12]}_part{index:02d}.zip';pending=path.with_suffix('.pending')
        manifest=dict(base,part_number=index,parts_total=len(groups),files_in_this_part=list(common)+[n for n,_ in group])
        with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED) as bundle:
            for name,data in common.items():bundle.writestr(name,data)
            for name,source in group:bundle.write(source,name,compress_type=zipfile.ZIP_STORED)
            bundle.writestr('export_manifest.json',json.dumps(manifest,indent=2)+'\n')
            require(bundle.testzip() is None,'ZIP CRC check failed')
        require(pending.stat().st_size<=max_bytes,'Archive part budget exceeded')
        pending.replace(path);archives.append(str(path))
    return archives


def run(root,images,output,limit_images=None,max_part_bytes=28*1024**2,*,expected_counts=(616,91),expected_totals=(53535,7172),expected_shape=(2048,2048)):
    root,images,output=map(lambda p:Path(p).resolve(),(root,images,output))
    require(root.is_dir() and images.is_dir(),'Known corpus/images directory missing; update ROOT/IMAGES explicitly, no automatic search')
    require(output!=root and output not in root.parents,'Use a separate descriptor output folder')
    corpus,origin,counts,totals,metadata=load_corpus(root,expected_counts,expected_totals)
    source=Path(__file__).read_bytes();source_sha=sha(source)
    identity=dict(corpus=corpus['identity'],corpus_bytes_sha256=sha(metadata['corpus.json']),
        source_sha256=source_sha,spec=SPEC,shape=list(expected_shape),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__,pycocotools=importlib.metadata.version('pycocotools')))
    identity_id=fingerprint(identity);request_path=output/'request.json'
    if request_path.exists():require(json.loads(request_path.read_text())['identity']==identity,'Descriptor source/config/runtime changed; use a new output folder')
    output.mkdir(parents=True,exist_ok=True);cache=output/'images';cache.mkdir(exist_ok=True)
    source_path=output/'code'/source_sha/'export_candidate_context_features.py';source_path.parent.mkdir(parents=True,exist_ok=True)
    if source_path.exists():require(source_path.read_bytes()==source,'Persisted exporter source changed')
    else:source_path.write_bytes(source)
    request=dict(identity=identity,root=str(root),images=str(images),output=str(output),expected_counts=list(expected_counts),expected_totals=list(expected_totals))
    atomic_json(request_path,request);atomic_json(output/'current_code.json',dict(path=str(source_path),sha256=source_sha))
    files=[];role_images=Counter();role_rows=Counter();reused=0
    records=corpus['records'][:limit_images] if limit_images is not None else corpus['records']
    for number,row in enumerate(records,1):
        name=row['name'];stem=Path(name).stem;path=images/name;image_sha=file_sha(path)
        raw=checked_raw(root/'proposals'/f'{stem}.json.gz',row,corpus['identity'],image_sha)
        image_identity=dict(dataset=identity_id,name=name,role=row['role'],image_sha256=image_sha,raw_sha256=row['raw_sha'])
        saved=cache/(stem+'.json.gz')
        if saved.exists():
            payload=read_gzip(saved);require(payload['identity']==image_identity,'Descriptor image identity changed: '+name)
            require(len(payload['rows'])==len(raw['instances']) and [r['index'] for r in payload['rows']]==list(range(len(raw['instances']))),'Cached candidate order/count differs')
            require(payload['feature_columns']==COLUMNS and payload['target_columns']==SPEC['target_columns'],'Cached descriptor schema differs')
            for cached,instance in zip(payload['rows'],raw['instances']):
                require(cached['rle_counts']==instance['rle'] and cached['rle_sha256']==sha(instance['rle'].encode())
                        and cached['confidence']==instance['confidence'] and cached['targets']==instance['target'],'Cached raw proposal/target join differs')
                require(len(cached['features'])==len(COLUMNS) and all(v is None or isinstance(v,(int,float)) and math.isfinite(v)
                        for v in cached['features']),'Cached descriptor vector differs')
            reused+=1
        else:
            image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            require(image is not None and image.shape==expected_shape and image.dtype==np.uint8,'Expected original grayscale image: '+name)
            maps=make_maps(image);rows=[]
            for index,instance in enumerate(raw['instances']):
                value=checked_rle(instance['rle'],expected_shape)
                vector=features(maps,value,instance['confidence'])
                rows.append(dict(index=index,rle_counts=instance['rle'],rle_sha256=sha(instance['rle'].encode()),
                    confidence=instance['confidence'],features=vector,targets=list(instance['target'])))
                if (index+1)%100==0:print(f'  Candidates {name}: {index+1}/{len(raw["instances"])}',flush=True)
            payload=dict(identity=image_identity,feature_columns=COLUMNS,target_columns=SPEC['target_columns'],
                         image_normalization=maps['normalization'],rows=rows)
            save_gzip(saved,payload)
        files.append(('images/'+saved.name,saved));role_images[row['role']]+=1;role_rows[row['role']]+=len(payload['rows'])
        print(f'Context [{number}/{len(records)}] {name}: all_raw={len(payload["rows"])} cached={reused} role={row["role"]}',flush=True)
    complete=len(records)==len(corpus['records'])
    if complete:require(dict(role_images)==counts and dict(role_rows)==totals,'Incomplete all-proposal export')
    summary=dict(status='complete' if complete else 'paused',completed_physical_images=len(records),expected_physical_images=sum(counts.values()),
        completed_images_by_role=dict(role_images),rows_by_role=dict(role_rows),expected_rows_by_role=totals,
        feature_count=len(COLUMNS),targets_are_features=False,all_original_raw_candidates=True,
        original_source_inputs=origin['inputs'],original_source_code=origin['code'],
        photos_included=False,weights_included=False,model_training=False,new_neural_inference=False,
        original_GT_file_read=False,labels='Separate original mean-matched-IoU and match-probability labels; train-only fitting, validation development evaluation')
    common=dict(metadata);common.update({'request.json':request_path.read_bytes(),
        'summary.json':(json.dumps(summary,indent=2)+'\n').encode(),
        'feature_schema.json':(json.dumps(SPEC,indent=2)+'\n').encode(),
        'execution_source/export_candidate_context_features.py':source})
    export_id=fingerprint(dict(dataset=identity_id,files={n:file_sha(p) for n,p in files},summary=summary))
    archives=write_parts(output,common,files,max_part_bytes,export_id)
    result=dict(summary,export_id=export_id,archives=archives,archive_bytes=[Path(p).stat().st_size for p in archives],
        archive_sha256=[file_sha(p) for p in archives],source_sha256=source_sha,image_cache_reused=reused)
    atomic_json(output/'result.json',result)
    return result


def main():
    base=Path('/content/drive/MyDrive/filament-segmentation-2026')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(base/'fixed_split_experiments/mask_quality_v1'))
    parser.add_argument('--images',default=str(base/'MAGFiLO_1.0_Kaggle_2026/train/train_images'))
    parser.add_argument('--output');parser.add_argument('--limit-images',type=int)
    parser.add_argument('--max-part-mib',type=float,default=28.)
    args=parser.parse_args()
    require(args.limit_images is None or args.limit_images>0,'--limit-images must be positive')
    require(args.max_part_mib>0,'--max-part-mib must be positive')
    output=args.output or str(Path(args.root)/'candidate_context_features/v1')
    print(json.dumps(run(args.root,args.images,output,args.limit_images,int(args.max_part_mib*1024**2)),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
