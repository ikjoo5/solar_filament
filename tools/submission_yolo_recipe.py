"""Portable frozen YOLO half of the PQ .472593 submission pipeline.

This module consumes predictions only. There are no annotation, validation-set,
checkpoint, or workspace dependencies. Every returned COCO RLE is in the native
2048-square coordinate system and has byte ``counts``.
"""
from collections.abc import Mapping

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage

SIZE = 2048
GEOMETRIC_VIEWS = ('base', 'scale', 'hflip', 'vflip')
PREPROCESSING_VIEWS = ('limb', 'clahe16', 'clahe32', 'percentile',
                       'unsharp', 'gaussian', 'bilateral', 'blackhat')


def native_rle(value):
    """Accept either raw-view proposals or ordinary COCO RLE dictionaries."""
    if 'rle' in value:
        value = value['rle']
    if isinstance(value, (str, bytes)):
        return dict(size=[SIZE, SIZE], counts=value.encode('ascii') if isinstance(value,str) else value)
    if list(value['size']) != [SIZE,SIZE]:
        raise ValueError('All input masks must use native 2048-square coordinates')
    return dict(size=[SIZE,SIZE],counts=value['counts'].encode('ascii')
                if isinstance(value['counts'],str) else value['counts'])


def encode_roi(mask, top, left, height=SIZE, width=SIZE):
    counts, previous = [], 0
    for col in range(mask.shape[1]):
        changes = np.flatnonzero(np.diff(np.r_[0,mask[:,col].astype(np.int8),0]))
        for start,end in changes.reshape(-1,2):
            first=(left+col)*height+top+int(start)
            last=(left+col)*height+top+int(end)
            counts.extend([first-previous,last-first]);previous=last
    counts.append(height*width-previous)
    return mu.frPyObjects(dict(size=[height,width],counts=counts),height,width)


def resolve_rois(candidates, originals, regions, shape=(SIZE,SIZE)):
    """Reserve original pixels and give new pixels to the nearest original."""
    if not len(candidates)==len(originals)==len(regions):
        raise ValueError('Mismatched per-instance ROI lists')
    owner=np.full(shape,-1,np.int32)
    for index,(mask,region) in enumerate(zip(originals,regions)):
        if mask.shape!=owner[region].shape or np.any(mask & (owner[region]>=0)):
            raise ValueError('Original masks overlap or have inconsistent coordinates')
        owner[region][mask]=index
    winner=np.full(shape,-1,np.int32);best=np.full(shape,np.inf,np.float64)
    for index,(candidate,original,region) in enumerate(zip(candidates,originals,regions)):
        if candidate.shape!=original.shape:
            raise ValueError('Candidate ROI shape differs from original')
        permitted=candidate & ((owner[region]<0)|(owner[region]==index))
        distance=ndimage.distance_transform_edt(~original)
        better=permitted & (distance<best[region])
        winner[region][better]=index;best[region][better]=distance[better]
    return [candidate & (winner[region]==index)
            for index,(candidate,region) in enumerate(zip(candidates,regions))]


def _view_map(records, expected):
    rows=list(records.values()) if isinstance(records,Mapping) else list(records)
    result={record['view']:record for record in rows}
    if len(rows)!=len(result) or set(result)!=set(expected):
        raise ValueError(f'Expected exactly these views: {expected}')
    # Canonical insertion order is a frozen confidence-tie rule.
    return {name:result[name] for name in expected}


def _component_rles(foreground, minimum=300):
    labels,count=ndimage.label(foreground,structure=np.ones((3,3),np.uint8))
    areas=np.bincount(labels.ravel());regions=ndimage.find_objects(labels)
    result=[]
    for label in sorted((i for i in range(1,count+1) if areas[i]>=minimum),key=lambda i:-int(areas[i])):
        ys,xs=regions[label-1]
        result.append(encode_roi(labels[ys,xs]==label,ys.start,xs.start))
    return result


def _guard(candidate, original, minimum_iou):
    ratio=int(candidate.sum())/int(original.sum())
    union=int((candidate|original).sum())
    iou=int((candidate&original).sum())/union if union else 0.
    return .5<=ratio<=1.5 and iou>=minimum_iou


def _encode_resolved(candidates, originals, regions):
    output=[]
    for mask,region in zip(resolve_rois(candidates,originals,regions),regions):
        yy,xx=np.nonzero(mask)
        if not len(yy):
            raise ValueError('Frozen instance-preserving transformation removed a mask')
        top,left=int(yy.min()),int(xx.min())
        output.append(encode_roi(mask[top:int(yy.max())+1,left:int(xx.max())+1],
                                 region[0].start+top,region[1].start+left))
    return output


def _consensus(geometric):
    votes=np.zeros((SIZE,SIZE),np.uint8)
    for view in ('base','scale','vflip'):
        encoded=[native_rle(p) for p in geometric[view]['instances'] if p['confidence']>=.2]
        if encoded:votes+=mu.decode(mu.merge(encoded))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    closed=cv2.morphologyEx((votes>=2).astype(np.uint8),cv2.MORPH_CLOSE,kernel,
                            borderType=cv2.BORDER_CONSTANT,borderValue=0)
    return _component_rles(closed,300)


def _preprocessing_rescue(originals, geometric, preprocessing):
    output=list(originals)
    covered=mu.decode(mu.merge(output)).astype(bool) if output else np.zeros((SIZE,SIZE),bool)
    candidates=[p for view,record in preprocessing.items() if view!='limb'
                for p in record['instances'] if p['confidence']>=.2]
    candidates.sort(key=lambda p:-p['confidence'])
    cr=[native_rle(p) for p in candidates]
    old=[native_rle(p) for record in geometric.values() for p in record['instances'] if p['confidence']>=.1]
    agreement=(np.asarray(mu.iou(cr,old,[0]*len(old))).max(axis=1)
               if cr and old else np.zeros(len(cr)))
    for index,(proposal,rle) in enumerate(zip(candidates,cr)):
        area=int(mu.area(rle))
        # prep1 is automatically satisfied by this proposal's own view.
        if proposal['confidence']<.3 or area<300 or agreement[index]<.3:continue
        x,y,w,h=map(int,mu.toBbox(rle))
        mask=mu.decode(rle)[y:y+h,x:x+w].astype(bool);occupied=covered[y:y+h,x:x+w]
        if int((mask&occupied).sum())/area>=.1:continue
        remaining=mask & ~occupied
        if int(remaining.sum())<300:continue
        occupied|=remaining
        output.append(encode_roi(remaining,y,x))
    return output


def _smooth(originals):
    local_originals=[];regions=[];candidates=[]
    for encoded in originals:
        x,y,w,h=map(int,mu.toBbox(encoded))
        region=np.s_[max(0,y-9):min(SIZE,y+h+9),max(0,x-9):min(SIZE,x+w+9)]
        original=mu.decode(encoded)[region].astype(bool)
        changed=cv2.GaussianBlur(original.astype(np.float32),(0,0),1.2,
                                  borderType=cv2.BORDER_CONSTANT)>=.6
        if not _guard(changed,original,.5):changed=original.copy()
        local_originals.append(original);regions.append(region);candidates.append(changed)
    return _encode_resolved(candidates,local_originals,regions)


def _tiled_majority(originals, records):
    records=list(records.values()) if isinstance(records,Mapping) else list(records)
    # The source exports y,x,scale order. Reconstruct it even for dict callers.
    records.sort(key=lambda r:(r['crop_bbox'][1],r['crop_bbox'][0],r['input_size']))
    proposals=[]
    for record in records:
        for p in record['instances']:
            if p['edge_clipped'] or p['confidence']<.15:continue
            rle=native_rle(p)
            proposals.append(dict(rle=rle,confidence=p['confidence'],size=record['input_size'],
                                  bbox=list(map(int,mu.toBbox(rle)))))
    tr=[p['rle'] for p in proposals]
    matrix=(np.asarray(mu.iou(originals,tr,[0]*len(tr))) if originals and tr
            else np.zeros((len(originals),len(tr))))
    assignment=matrix.argmax(axis=0) if originals and tr else np.empty(0,int)
    candidates=[];local_originals=[];regions=[]
    for index,rle in enumerate(originals):
        selected=[]
        for scale in (1536,2048):
            valid=[j for j,p in enumerate(proposals) if assignment[j]==index and p['size']==scale
                   and p['confidence']>=.35 and matrix[index,j]>=.3]
            selected.append(max(valid,key=lambda j:proposals[j]['confidence']) if valid else None)
        boxes=[list(map(int,mu.toBbox(rle)))]+[proposals[j]['bbox'] for j in selected if j is not None]
        x=min(b[0] for b in boxes);y=min(b[1] for b in boxes)
        right=max(b[0]+b[2] for b in boxes);bottom=max(b[1]+b[3] for b in boxes)
        region=np.s_[y:bottom,x:right];original=mu.decode(rle)[region].astype(bool)
        candidate=original.copy()
        if None not in selected:
            votes=original.astype(np.uint8)
            for j in selected:
                bx,by,bw,bh=proposals[j]['bbox']
                votes[by-y:by-y+bh,bx-x:bx-x+bw]+=mu.decode(tr[j])[by:by+bh,bx:bx+bw]
            candidate=votes>=2
            if not _guard(candidate,original,.85):candidate=original.copy()
        candidates.append(candidate);local_originals.append(original);regions.append(region)
    return _encode_resolved(candidates,local_originals,regions)


def build_yolo_reference(image, geometric_records, preprocessing_records, tiled_records,
                         return_stages=False):
    """Reproduce the frozen pre-U-Net reference for one physical image.

    Records use the existing raw export schemas. Their ``image`` must match
    ``image``; sequence or view-keyed dictionary inputs are both accepted.
    With ``return_stages=True`` return a dict of the four ordered RLE lists.
    """
    geometric=_view_map(geometric_records,GEOMETRIC_VIEWS)
    preprocessing=_view_map(preprocessing_records,PREPROCESSING_VIEWS)
    tiles=list(tiled_records.values()) if isinstance(tiled_records,Mapping) else list(tiled_records)
    if len(tiles)!=18 or len({r['view'] for r in tiles})!=18:
        raise ValueError('Exactly eighteen distinct native tiled records are required')
    for record in list(geometric.values())+list(preprocessing.values())+tiles:
        if record.get('image')!=image or record.get('shape')!=[SIZE,SIZE] or record.get('coordinate_space')!='original':
            raise ValueError('Prediction record image or native coordinate contract differs')
    stages={}
    stages['consensus']=_consensus(geometric)
    stages['preprocessing_rescue']=_preprocessing_rescue(stages['consensus'],geometric,preprocessing)
    stages['smooth']=_smooth(stages['preprocessing_rescue'])
    stages['tiled']=_tiled_majority(stages['smooth'],tiles)
    return stages if return_stages else stages['tiled']
