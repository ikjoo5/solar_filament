"""CPU-only, label-free image features for cached solar masks.

Nothing in this module imports a model or reads annotation masks. The exporter
uses an existing current-checkpoint OOF snapshot and writes scalar features,
never source images or weights. Ridge features describe image evidence; they
are not calibrated mask-IoU estimates or a claimed PQ improvement.
"""
import argparse
import csv
from functools import lru_cache
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import zipfile

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy.interpolate import UnivariateSpline

SIZE = 2048
PROVIDER_SHA = '7c6a9d829025801482350b2f009a51a7ffb1eaa3d270cdce80a6a0691f815227'
PARAMETERS = dict(schema=1, sigmas=[1., 2., 4.], hessian_beta=.5,
    ring_radius_pixels=7, ring_kernel='ellipse15', ring_excludes_baseline_foreground=True,
    ridge_disk_boundary_margin_pixels=14, photometric_mad_scale=1.4826,
    raw_minimum_confidence=.15, raw_minimum_area=200,
    novel_maximum_instance_iou=.1, novel_maximum_union_overlap=.25,
    excluded_clipped_tiles=True, excluded_preprocessing_limb=True,
    feature_input='Exact frozen limb-corrected grayscale/255',
    scope='Label-free physical image descriptors, not calibrated quality probabilities')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    pending.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def checked_rle(value):
    if value['size'] != [SIZE, SIZE]:
        raise ValueError('Native2048 coordinates required')
    counts = value['counts']
    rle = dict(size=[SIZE, SIZE], counts=counts.encode('ascii') if isinstance(counts, str) else counts)
    if int(mu.area(rle)) <= 0:
        raise ValueError('Empty mask in target pool')
    return rle


@lru_cache(maxsize=1)
def limb_geometry():
    yy, xx = np.ogrid[:2048, :2048]
    rr = (np.sqrt((xx-1024)**2+(yy-1024)**2)/900).astype(np.float32)
    disk = rr <= 1
    radii = (np.arange(100)+.5)/100
    bins = np.minimum((rr[disk]*100).astype(np.int32),99)
    return rr, disk, radii, bins


def limb(image):
    """Exact non-Torch copy of the frozen submission preprocessing."""
    if image is None or image.shape != (2048,2048) or image.dtype != np.uint8:
        raise ValueError('Original grayscale uint8 2048-square image required')
    rr, disk, radii, bins = limb_geometry()
    hist = np.bincount(bins*256+image[disk].astype(np.int32),minlength=25600).reshape(100,256)
    count = hist.sum(axis=1)
    if np.any(count <= 0):
        raise ValueError('Empty radial bin')
    med = np.argmax(hist.cumsum(axis=1)>=((count+1)//2)[:,None],axis=1).astype(float)
    spline = UnivariateSpline(radii,med,k=3,s=400)
    reference = float(np.median(spline(radii)[radii<=.15]))
    gain = np.clip(reference/np.maximum(spline(rr[disk]),1),.25,4)
    result = np.zeros_like(image)
    result[disk] = np.clip(image[disk].astype(np.float32)*gain,0,255).astype(np.uint8)
    return result


def ridge_maps(image, sigmas=(1., 2., 4.), beta=.5):
    """Scale-normalized dark-ridge response and unoriented tangent evidence."""
    gray = np.asarray(image, dtype=np.float32)
    if gray.ndim != 2 or not np.isfinite(gray).all():
        raise ValueError('Finite grayscale array required')
    best = np.zeros_like(gray);cosine = np.zeros_like(gray);sine = np.zeros_like(gray)
    anisotropy = np.zeros_like(gray);selected_scale = np.zeros_like(gray)
    for sigma in sigmas:
        blurred = cv2.GaussianBlur(gray, (0,0), float(sigma), borderType=cv2.BORDER_REFLECT101)
        xx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3, borderType=cv2.BORDER_REFLECT101)*(sigma*sigma)
        yy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3, borderType=cv2.BORDER_REFLECT101)*(sigma*sigma)
        xy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3, borderType=cv2.BORDER_REFLECT101)*(sigma*sigma)
        disc = np.sqrt(np.maximum((xx-yy)**2+4*xy*xy, 0))
        plus = .5*(xx+yy+disc);minus = .5*(xx+yy-disc)
        swap = np.abs(plus) < np.abs(minus)
        large = np.where(swap, minus, plus);small = np.where(swap, plus, minus)
        ratio = small/(beta*np.abs(large)+1e-8)
        response = np.maximum(large, 0)*np.exp(-.5*ratio*ratio)
        aniso = np.maximum(np.abs(large)-np.abs(small),0)/(np.abs(large)+np.abs(small)+1e-8)
        c = np.divide(-(xx-yy), disc, out=np.zeros_like(gray), where=disc>1e-8)
        s = np.divide(-2*xy, disc, out=np.zeros_like(gray), where=disc>1e-8)
        # A negative dominant eigenvalue is a bright ridge and has zero response.
        better = response > best
        best[better] = response[better];cosine[better] = c[better];sine[better] = s[better]
        anisotropy[better] = aniso[better];selected_scale[better] = sigma
    return dict(strength=best, cos2=cosine, sin2=sine,
                anisotropy=anisotropy, scale=selected_scale)


def distribution(values, prefix):
    if not len(values):
        return {prefix+'_mean':None, prefix+'_median':None, prefix+'_p90':None}
    return {prefix+'_mean':float(np.mean(values)), prefix+'_median':float(np.median(values)),
            prefix+'_p90':float(np.quantile(values,.9))}


def mask_features(gray, maps, mask, photometric_valid=None, ridge_valid=None, ring_radius=7,
                  ring_exclusion=None, bbox=None, pixel_area=None):
    """Describe one full-coordinate mask; missing valid pixels stay explicit."""
    if mask.shape != gray.shape or mask.dtype != np.bool_:
        raise ValueError('Boolean mask must have the same image coordinates')
    valid = np.ones(mask.shape, bool) if photometric_valid is None else photometric_valid
    ridge_ok = valid if ridge_valid is None else ridge_valid
    if bbox is None:
        y, x = np.nonzero(mask)
        if not len(y):raise ValueError('Nonempty mask required')
        box=(int(x.min()),int(y.min()),int(x.max()-x.min()+1),int(y.max()-y.min()+1))
    else:
        if len(bbox)!=4 or any(int(value)!=value for value in bbox):
            raise ValueError('Integer xywh bounding box required')
        box=tuple(map(int,bbox))
    x,y,width,height=box
    if min(x,y)<0 or min(width,height)<=0 or x+width>mask.shape[1] or y+height>mask.shape[0]:
        raise ValueError('Bounding box is outside image coordinates')
    left=max(0,x-ring_radius);top=max(0,y-ring_radius)
    right=min(mask.shape[1],x+width+ring_radius);bottom=min(mask.shape[0],y+height+ring_radius)
    region=np.s_[top:bottom,left:right];local=mask[region]
    count=int(local.sum())
    if count<=0 or (pixel_area is not None and (int(pixel_area)!=pixel_area or count!=pixel_area)):
        raise ValueError('Mask area differs from bounding-box source')
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*ring_radius+1,2*ring_radius+1))
    ring=cv2.dilate(local.astype(np.uint8),kernel).astype(bool)&~local
    if ring_exclusion is not None:
        if ring_exclusion.shape!=mask.shape:raise ValueError('Ring exclusion coordinates differ')
        ring &= ~ring_exclusion[region]
    inside=local&valid[region];outside=ring&valid[region]
    ri=local&ridge_ok[region];ro=ring&ridge_ok[region]
    result=dict(area=count,photometric_inside_pixels=int(inside.sum()),
        photometric_ring_pixels=int(outside.sum()),ridge_inside_pixels=int(ri.sum()),ridge_ring_pixels=int(ro.sum()),
        outside_disk_fraction=1-float(inside.sum())/count,
        ridge_valid_fraction=float(ri.sum())/count)
    result.update(distribution(gray[region][inside],'intensity_inside'))
    result.update(distribution(gray[region][outside],'intensity_ring'))
    if outside.any() and inside.any():
        ring_values=gray[region][outside];median=float(np.median(ring_values))
        mad=float(np.median(np.abs(ring_values-median)))
        result.update(ring_mad=mad,dark_z=(median-float(np.median(gray[region][inside])))/max(1.4826*mad,1/255))
    else:result.update(ring_mad=None,dark_z=None)
    result.update(distribution(maps['strength'][region][ri],'ridge_inside'))
    result.update(distribution(maps['strength'][region][ro],'ridge_ring'))
    if ri.any():
        strength=maps['strength'][region][ri];weights=strength*maps['anisotropy'][region][ri]
        mass=float(weights.sum());total=float(strength.sum())
        if mass>1e-10:
            c=float(np.sum(weights*maps['cos2'][region][ri]));s=float(np.sum(weights*maps['sin2'][region][ri]))
            coherence=float(np.clip(math.hypot(c,s)/mass,0,1))
        else:coherence=0.
        result.update(ridge_orientation_coherence=coherence,ridge_orientation_valid=mass>1e-10,
            ridge_anisotropic_mass_fraction=float(np.clip(mass/total,0,1)) if total>1e-10 else 0.,
            ridge_scale_mean=float(maps['scale'][region][ri].mean()))
    else:result.update(ridge_orientation_coherence=None,ridge_orientation_valid=False,
        ridge_anisotropic_mass_fraction=None,ridge_scale_mean=None)
    if not all(value is None or isinstance(value,bool) or math.isfinite(value) for value in result.values()):
        raise ValueError('Nonfinite pixel descriptor')
    return result


def target_masks(full, raw):
    original=[checked_rle(v) for v in full['masks']['full_selected']]
    covered=mu.merge(original) if original else None
    unique={}
    for family in ('geometric','preprocessing','tiled'):
        for view in raw[family]:
            if view['image'] != Path(full['file_name']).stem or view['shape'] != [SIZE,SIZE] or view['coordinate_space'] != 'original':
                raise ValueError('Raw image/coordinate mismatch')
            if family=='preprocessing' and view['view']=='limb':continue
            for candidate in view['instances']:
                if family=='tiled' and candidate.get('edge_clipped',False):continue
                conf=float(candidate['confidence'])
                if not math.isfinite(conf) or not 0<=conf<=1:raise ValueError('Invalid raw confidence')
                if conf<PARAMETERS['raw_minimum_confidence']:continue
                rle=checked_rle(dict(size=[SIZE,SIZE],counts=candidate['rle']))
                if int(mu.area(rle))<PARAMETERS['raw_minimum_area']:continue
                key=rle['counts']
                if key not in unique:unique[key]=dict(rle=rle,confidence=conf)
                else:unique[key]['confidence']=max(unique[key]['confidence'],conf)
    candidates=list(unique.values())
    overlap=np.asarray(mu.iou([r['rle'] for r in candidates],original,[0]*len(original))) if original and candidates else None
    targets=[dict(role='baseline',index=i,rle=rle,confidence=None,max_original_iou=1.,original_overlap=1.)
             for i,rle in enumerate(original)]
    for index,row in enumerate(candidates):
        rle=row['rle'];area=int(mu.area(rle));iou=float(overlap[index].max()) if overlap is not None else 0.
        fraction=int(mu.area(mu.merge([rle,covered],intersect=True)))/area if covered is not None else 0.
        if iou>.1 or fraction>.25:continue
        targets.append(dict(role='novel_raw',index=index,rle=rle,confidence=row['confidence'],max_original_iou=iou,original_overlap=fraction))
    return targets


def runtime_versions():
    result=dict(numpy=np.__version__,opencv=cv2.__version__)
    for name in ('scipy','pycocotools'):
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]='unknown'
    return result


def load_fold(root,snapshot,fold):
    folder=snapshot/f'fold{fold}';summary=read(folder/'summary.json');identity=read(folder/'snapshot_identity.json')
    if fingerprint(identity['payload'])!=identity['signature'] or summary['identity']!=identity['signature']:
        raise ValueError('Full snapshot identity changed')
    full_path=folder/'full_records.jsonl'
    if sha(full_path)!=summary['records_sha256']:raise ValueError('Full record checksum changed')
    native_folder=root/f'fold{fold}'/'stage030';native_summary=read(native_folder/'stage_summary.json')
    native_path=native_folder/'rich_records.jsonl'
    if sha(native_path)!=summary['native_rich_sha256'] or sha(native_path)!=native_summary['rich_predictions_sha256']:
        raise ValueError('Native records changed')
    cfg=native_summary['training_config'];yolo=read(root/f'fold{fold}'/'yolo/config.json')
    for key,filename in [('train_split','train_split.csv'),('val_split','val_split.csv')]:
        expected=cfg['source_sha256'][key]
        if sha(root/f'fold{fold}'/filename)!=expected or yolo['settings']['sources'][key]!=expected:
            raise ValueError('Fold split source changed')
    with native_path.open() as stream:
        rich={r['file_name']:r for r in map(json.loads,stream)}
    with full_path.open() as stream:
        full=[json.loads(line) for line in stream]
    if len(full)!=len(rich) or {r['file_name'] for r in full}!=set(rich) or set(rich)!=set(native_summary['image_names']):
        raise ValueError('Native/full validation coverage differs')
    for record in full:
        native=rich[record['file_name']]
        native_unsigned={k:v for k,v in native.items() if k!='record_sha256'}
        native_digest=hashlib.sha256(json.dumps(native_unsigned,ensure_ascii=False,sort_keys=True,
            separators=(',',':')).encode()).hexdigest()
        if (fingerprint({k:v for k,v in record.items() if k!='record_sha256'})!=record['record_sha256']
                or record['identity']!=identity['signature'] or record['fold']!=fold
                or record['native_epoch']!=30 or record['yolo_epoch']!=summary['yolo_epoch']
                or record['image_sha256']!=native['image_sha256']
                or native['record_sha256']!=native_digest or native['fold']!=fold or native['stage_epoch']!=30
                or native['checkpoint_sha256']!=identity['payload']['native_sha256']
                or summary['yolo']['checkpoint_sha256']!=identity['payload']['yolo_sha256']):
            raise ValueError('Image/model/fold provenance changed')
    return sorted(full,key=lambda r:r['file_name']),rich,summary


def image_record(root,snapshot,images,fold,full,native,summary,identity,cache_path,_extract=None):
    name=full['file_name'];image_path=images/name
    raw_path=(snapshot/f'fold{fold}'/'yolo_records'/(Path(name).stem+'.json.gz')) if fold==4 else root/f'fold{fold}'/'stage030/yolo_records'/(Path(name).stem+'.json.gz')
    source=dict(full_record_sha256=full['record_sha256'],native_record_sha256=native['record_sha256'],
        raw_sha256=sha(raw_path),image_sha256=sha(image_path))
    if source['image_sha256']!=full['image_sha256']:raise ValueError('Original image bytes changed: '+name)
    raw=json.loads(gzip.decompress(raw_path.read_bytes()));checksum=raw.pop('checksum')
    if fingerprint(raw)!=checksum or raw['identity']!=dict(image_sha256=full['image_sha256'],
        yolo_sha256=summary['yolo']['checkpoint_sha256'],provider_sha256=PROVIDER_SHA):
        raise ValueError('Raw mask model/image/provider changed')
    expected=fingerprint(dict(export=identity,file_name=name,fold=fold,sources=source))
    if cache_path.exists():
        old=json.loads(gzip.decompress(cache_path.read_bytes()))
        unsigned={k:v for k,v in old.items() if k!='record_sha256'}
        if fingerprint(unsigned)!=old['record_sha256'] or old['identity']!=expected:
            raise ValueError('Existing pixel feature cache changed; preserved')
        return old,True
    pixels=cv2.imread(str(image_path),cv2.IMREAD_GRAYSCALE)
    if pixels is None or pixels.shape!=(SIZE,SIZE) or pixels.dtype!=np.uint8:
        raise ValueError('Original2048 grayscale image required')
    corrected=limb(pixels).astype(np.float32)/255.
    maps=_extract(corrected) if _extract is not None else ridge_maps(corrected,tuple(PARAMETERS['sigmas']),PARAMETERS['hessian_beta'])
    rr,disk,_,_=limb_geometry()
    ridge_valid=rr <= 1-PARAMETERS['ridge_disk_boundary_margin_pixels']/900
    weak=dict(native['tta']['unions']['foreground']['0.2'])
    weak['counts']=weak['counts'].encode('ascii') if isinstance(weak['counts'],str) else weak['counts']
    rows=[];targets=target_masks(full,raw)
    baseline=[r['rle'] for r in targets if r['role']=='baseline']
    occupied=mu.decode(mu.merge(baseline)).astype(bool) if baseline else np.zeros((SIZE,SIZE),bool)
    for target in targets:
        rle=target['rle'];mask=mu.decode(rle).astype(bool)
        x,y,w,h=map(int,mu.toBbox(rle))
        pixel_area=int(mu.area(rle))
        descriptor=mask_features(corrected,maps,mask,disk,ridge_valid,PARAMETERS['ring_radius_pixels'],
            occupied,bbox=(x,y,w,h),pixel_area=pixel_area)
        descriptor.update(file_name=name,fold=fold,native_epoch=30,yolo_epoch=summary['yolo_epoch'],
            role=target['role'],source_index=target['index'],rle_counts_sha256=hashlib.sha256(rle['counts']).hexdigest(),
            raw_confidence=target['confidence'],bbox_x=x,bbox_y=y,bbox_width=w,bbox_height=h,
            maximum_original_iou=target['max_original_iou'],original_overlap_fraction=target['original_overlap'],
            native_weak_coverage=int(mu.area(mu.merge([rle,weak],intersect=True)))/int(mu.area(rle)))
        rows.append(descriptor)
    record=dict(file_name=name,fold=fold,identity=expected,sources=source,rows=rows,
                target_count=len(rows),ground_truth_read=False,neural_inference=False)
    record['record_sha256']=fingerprint(record)
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    pending=cache_path.with_name(cache_path.name+'.pending')
    pending.write_bytes(gzip.compress(json.dumps(record,ensure_ascii=False,allow_nan=False).encode(),mtime=0));pending.replace(cache_path)
    return record,False


def package(output,files,limit=20*1024*1024):
    folder=output/'bundles';folder.mkdir(exist_ok=True)
    archives=[];handle=None;used=0;entries=[]
    try:
        for path in files:
            relative=path.relative_to(output).as_posix();digest=sha(path);pieces=[]
            with path.open('rb') as stream:
                number=0
                while data:=stream.read(limit):
                    number+=1
                    if handle is None or used+len(data)>limit:
                        if handle is not None:handle.close()
                        archive=folder/f'pixel_quality_part{len(archives)+1:02d}.zip';archives.append(archive)
                        handle=zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED);used=0
                    member=relative if path.stat().st_size<=limit else f'_chunks/{digest[:20]}.part{number:04d}'
                    handle.writestr(member,data);used+=len(data)
                    pieces.append(dict(archive=archives[-1].name,member=member))
            entries.append(dict(path=relative,bytes=path.stat().st_size,sha256=digest,pieces=pieces))
    finally:
        if handle is not None:handle.close()
    manifest=dict(files=entries,archives=[p.name for p in archives],images_and_weights_included=False)
    for path in archives:
        with zipfile.ZipFile(path,'a',zipfile.ZIP_DEFLATED) as z:z.writestr('archive_manifest.json',json.dumps(manifest,indent=2))
        if path.stat().st_size>=24*1024*1024:raise ValueError('Feature archive exceeds24MiB')
    atomic_json(folder/'archive_manifest.json',manifest)
    return [str(p) for p in archives]


def run(root,images,output,snapshot=None,limit_images=None,_extract=None):
    cv2.setNumThreads(1)
    root,images,output=map(Path,(root,images,output))
    snapshot=Path(snapshot) if snapshot is not None else root/'current_epoch_evaluation/native030_yolo_current_v1'
    if not output.resolve().is_relative_to((root/'pixel_quality_exports').resolve()):
        raise ValueError('Output must be a separate study/pixel_quality_exports directory')
    provider=root/'code/5c1fb752ac88/submission_inference.py'
    if sha(provider)!=PROVIDER_SHA:raise ValueError('Frozen inference source changed')
    request=dict(parameters=PARAMETERS,source_sha256=sha(__file__),provider_sha256=PROVIDER_SHA,
        runtime_versions=runtime_versions(),root=str(root),snapshot=str(snapshot),images=str(images),
        ground_truth_read=False,model_training=False,neural_inference=False)
    identity=fingerprint(request);output.mkdir(parents=True,exist_ok=True)
    if (output/'request.json').exists() and read(output/'request.json')!=dict(identity=identity,request=request):
        raise ValueError('Feature configuration or runtime changed; use a new output folder')
    atomic_json(output/'request.json',dict(identity=identity,request=request))
    source_dir=output/'execution_source';source_dir.mkdir(exist_ok=True)
    (source_dir/'export_solar_pixel_quality.py').write_bytes(Path(__file__).read_bytes())
    all_names=set();records=[];expected=0;reused=0;epochs={}
    for fold in range(5):
        full,rich,summary=load_fold(root,snapshot,fold);expected+=len(full)
        epochs[str(fold)]=summary['yolo_epoch']
        names={r['file_name'] for r in full}
        if all_names&names:raise ValueError('Duplicate physical image across folds')
        all_names.update(names)
        for row in full:
            if limit_images is not None and len(records)>=limit_images:continue
            cache=output/'records'/f'fold{fold}'/(Path(row['file_name']).stem+'.json.gz')
            record,was_reused=image_record(root,snapshot,images,fold,row,rich[row['file_name']],summary,identity,cache,_extract)
            records.append(record);reused+=int(was_reused)
            print(f'Pixel quality CPU [{len(records)}] fold{fold} {row["file_name"]} '
                  f'targets={record["target_count"]} cached={was_reused}',flush=True)
    if expected!=707 or len(all_names)!=707:raise ValueError('Full707-image snapshot expected')
    columns=sorted({k for r in records for row in r['rows'] for k in row})
    csv_path=output/'features.csv';pending=csv_path.with_name(csv_path.name+'.pending')
    with pending.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for r in records:writer.writerows(r['rows'])
    pending.replace(csv_path)
    combined=output/'features.jsonl.gz';pending=combined.with_name(combined.name+'.pending')
    with gzip.open(pending,'wt',encoding='utf-8') as stream:
        for record in records:
            for row in record['rows']:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    pending.replace(combined)
    summary=dict(status='complete_pixel_features' if len(records)==expected else 'partial_pixel_features',
        expected_physical_images=expected,completed_physical_images=len(records),
        targets=sum(r['target_count'] for r in records),reused_images=reused,
        image_records=[dict(file_name=r['file_name'],fold=r['fold'],targets=r['target_count'],
                            record_sha256=r['record_sha256'],sources=r['sources']) for r in records],
        ground_truth_read=False,model_training=False,neural_inference=False,
        native_epochs=[30]*5,yolo_epochs=epochs,
        feature_identity=identity,scope=PARAMETERS['scope'])
    atomic_json(output/'summary.json',summary)
    archives=package(output,[output/'request.json',output/'summary.json',csv_path,combined,source_dir/'export_solar_pixel_quality.py'])
    result=dict(status=summary['status'],archives=archives,summary=str(output/'summary.json'),
        completed_physical_images=len(records),expected_physical_images=expected,
        targets=summary['targets'],ground_truth_read=False,model_training=False,neural_inference=False)
    atomic_json(output/'result.json',result);print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--images',required=True)
    p.add_argument('--output',required=True);p.add_argument('--snapshot');p.add_argument('--limit-images',type=int)
    args=p.parse_args()
    run(args.root,args.images,args.output,args.snapshot,args.limit_images)
