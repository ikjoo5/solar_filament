"""Portable inference used by the frozen PQ0.472593 submission pipeline.

All public image methods accept the ORIGINAL grayscale uint8 2048-square image.
No annotations, validation selections, or workspace-specific paths are read.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage
from scipy.interpolate import UnivariateSpline
import torch

YOLO_SHA256 = '81c2a27d95821b5ede283169c0da8c08d360ecc4c2a60a1b9c6baf96d76bcccc'
NATIVE_SHA256 = 'ae6b209f72355fd52d2cb07182e69028e97c703a1977ad1e7b38b27cf8cebe69'
FG_THRESHOLDS = [.2, .35, .5, .65, .8]
SPINE_THRESHOLDS = [.3, .5, .7]
VIEWS = [('base', ()), ('hflip', (-1,)), ('vflip', (-2,)), ('hvflip', (-2, -1))]
PREPROCESS_PARAMS = {
    'limb': {'operation': 'identity after original limb correction'},
    'clahe16': {'operation': 'CLAHE blend', 'grid': [16,16], 'clip_limit': 1.5, 'blend': .5},
    'clahe32': {'operation': 'CLAHE blend', 'grid': [32,32], 'clip_limit': 2., 'blend': .5},
    'percentile': {'operation': 'inside-disk percentile stretch blend', 'percentiles': [1,99], 'blend': .25},
    'unsharp': {'operation': 'unsharp', 'gaussian_sigma': 1., 'gain': .5},
    'gaussian': {'operation': 'Gaussian blur', 'sigma': .5},
    'bilateral': {'operation': 'bilateral', 'diameter': 5, 'sigma_color': 10, 'sigma_space': 3},
    'blackhat': {'operation': 'subtract blackhat response', 'ellipse': [15,15], 'gain': .5},
}


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


@lru_cache(maxsize=1)
def limb_geometry():
    yy, xx = np.ogrid[:2048, :2048]
    rr = (np.sqrt((xx-1024)**2+(yy-1024)**2)/900).astype(np.float32)
    disk = rr <= 1
    radii = (np.arange(100)+.5)/100
    bins = np.minimum((rr[disk]*100).astype(np.int32),99)
    return rr, disk, radii, bins


def limb(image):
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


def preprocess_variant(image, name, valid_mask=None):
    valid = limb_geometry()[1] if valid_mask is None else np.asarray(valid_mask,dtype=bool)
    if image.ndim != 2 or image.dtype != np.uint8 or valid.shape != image.shape:
        raise ValueError('Preprocessing needs matching uint8 image and disk mask')
    if name not in PREPROCESS_PARAMS:
        raise ValueError(f'Unknown preprocessing view: {name}')
    base = np.ascontiguousarray(image)
    values = base.astype(np.float32)
    if name == 'limb':
        result = base.copy()
    elif name in ('clahe16','clahe32'):
        params = PREPROCESS_PARAMS[name]
        enhanced = cv2.createCLAHE(clipLimit=params['clip_limit'],tileGridSize=tuple(params['grid'])).apply(base)
        result = .5*values+.5*enhanced.astype(np.float32)
    elif name == 'percentile':
        if not valid.any():
            raise ValueError('Disk mask has no pixels')
        low,high = np.percentile(base[valid],[1,99])
        stretched = np.clip((values-low)*(255./(high-low)),0,255) if high>low else values
        result = .75*values+.25*stretched
    elif name == 'unsharp':
        smooth = cv2.GaussianBlur(values,(0,0),1.)
        result = values+.5*(values-smooth)
    elif name == 'gaussian':
        result = cv2.GaussianBlur(base,(0,0),.5)
    elif name == 'bilateral':
        result = cv2.bilateralFilter(base,5,10,3)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        response = cv2.morphologyEx(base,cv2.MORPH_BLACKHAT,kernel)
        result = values-.5*response.astype(np.float32)
    result = np.clip(result,0,255).astype(np.uint8)
    result[~valid] = 0
    return np.ascontiguousarray(result)


def rle(mask):
    encoded = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return dict(size=list(encoded['size']),counts=encoded['counts'].decode('ascii'))


def encode_native_roi(mask,top,left):
    if (mask.ndim != 2 or top < 0 or left < 0
            or top+mask.shape[0] > 2048 or left+mask.shape[1] > 2048):
        raise ValueError('ROI outside original 2048 image')
    counts,previous = [],0
    for column in range(mask.shape[1]):
        transitions = np.flatnonzero(np.diff(np.r_[0,mask[:,column].astype(np.int8),0]))
        for start,end in transitions.reshape(-1,2):
            first = (left+column)*2048+top+int(start)
            last = (left+column)*2048+top+int(end)
            counts.extend([first-previous,last-first]);previous=last
    counts.append(2048**2-previous)
    return mu.frPyObjects({'size':[2048,2048],'counts':counts},2048,2048)


def edge_sides(mask,top,left):
    sides = []
    if top > 0 and mask[:4,:].any(): sides.append('top')
    if left > 0 and mask[:,:4].any(): sides.append('left')
    if top+1024 < 2048 and mask[-4:,:].any(): sides.append('bottom')
    if left+1024 < 2048 and mask[:,-4:].any(): sides.append('right')
    return sides


def model_input(image):
    x = np.repeat(image[...,None],3,axis=2).astype(np.float32)/255.
    return ((x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1)


def tile_positions(size,tile,overlap=.25):
    if tile <= 0 or tile > size or not 0 <= overlap < 1:
        raise ValueError('Invalid native tile geometry')
    stride = max(1,int(round(tile*(1-overlap))))
    positions = list(range(0,size-tile+1,stride))
    if positions[-1] != size-tile:
        positions.append(size-tile)
    return positions


def blend_window(tile):
    one = np.hanning(tile).astype(np.float32)
    return np.maximum(one[:,None]*one[None,:],.001)


@torch.inference_mode()
def predict_heads(model,image,device,tile=1024,overlap=.25,batch=4,amp_dtype=None):
    model.eval()
    height,width = image.shape
    coordinates = [(y,x) for y in tile_positions(height,tile,overlap) for x in tile_positions(width,tile,overlap)]
    weight = blend_window(tile)
    numerator = np.zeros((2,height,width),np.float32)
    denominator = np.zeros((height,width),np.float32)
    for start in range(0,len(coordinates),batch):
        selected = coordinates[start:start+batch]
        inputs = torch.from_numpy(np.stack([model_input(image[y:y+tile,x:x+tile]) for y,x in selected])).to(device)
        with torch.autocast(device_type=device.type,dtype=amp_dtype or torch.float32,enabled=amp_dtype is not None):
            output = model(inputs)
        if tuple(output.shape) != (len(selected),2,tile,tile):
            raise ValueError(f'Expected native two-head output, received {tuple(output.shape)}')
        probabilities = output.float().sigmoid().cpu().numpy()
        for (y,x),probability in zip(selected,probabilities):
            numerator[:,y:y+tile,x:x+tile] += probability*weight
            denominator[y:y+tile,x:x+tile] += weight
    if np.any(denominator <= 0):
        raise ValueError('Native tile stitching left uncovered pixels')
    result = numerator/denominator
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite native probabilities')
    return result


def predict_views(model,image,device,tile=1024,overlap=.25,batch=4,amp_dtype=None,progress=None):
    base,total,unions = None,None,{}
    for view,axes in VIEWS:
        transformed = np.ascontiguousarray(np.flip(image,axes)) if axes else image
        heads = predict_heads(model,transformed,device,tile,overlap,batch,amp_dtype)
        restored = np.flip(heads,axes) if axes else heads
        if base is None:
            base,total = restored,restored.copy()
        else:
            total += restored
        unions[view] = rle(restored[0]>=.5)
        if progress is not None:
            progress(view)
    return base,total/np.float32(len(VIEWS)),unions


def prediction_record(heads,name,image_sha='',identity='submission',min_area=100):
    foreground,spine = heads
    result = dict(schema_version=1,identity=identity,file_name=name,image_sha256=image_sha,
                  size=list(foreground.shape),unions={},components={})
    for head_name,probability,thresholds in (
            ('foreground',foreground,FG_THRESHOLDS),('spine',spine,SPINE_THRESHOLDS)):
        result['unions'][head_name] = {f'{t:g}':rle(probability>=t) for t in thresholds}
    for threshold in FG_THRESHOLDS:
        labels,count = ndimage.label(foreground>=threshold,structure=np.ones((3,3),np.uint8))
        areas = np.bincount(labels.ravel())
        components = []
        for label,region in enumerate(ndimage.find_objects(labels),1):
            if region is None or areas[label] < min_area:
                continue
            local = labels[region] == label
            full = np.zeros_like(foreground,dtype=np.uint8)
            full[region] = local
            stats = {}
            for head_name,probability in [('foreground',foreground),('spine',spine)]:
                values = probability[region][local]
                stats[head_name] = dict(mean=float(values.mean()),max=float(values.max()),p90=float(np.percentile(values,90)))
            components.append(dict(rle=rle(full),area=int(areas[label]),bbox=[region[1].start,region[0].start,
                region[1].stop-region[1].start,region[0].stop-region[0].start],stats=stats))
        result['components'][f'{threshold:g}'] = components
    return result


class InferenceProvider:
    """Lazy model loading plus per-image stages suitable for durable resume."""

    def __init__(self,yolo_checkpoint,native_checkpoint,device='cuda',native_batch=4,
                 progress=None,verify_weights=True,*,yolo_model=None,native_model=None):
        self.yolo_checkpoint = Path(yolo_checkpoint) if yolo_checkpoint is not None else None
        self.native_checkpoint = Path(native_checkpoint) if native_checkpoint is not None else None
        self.device = torch.device(device)
        self.native_batch = int(native_batch)
        if self.native_batch <= 0:
            raise ValueError('native_batch must be positive')
        self.progress = progress
        self.verify_weights = verify_weights
        self._yolo_model,self._native_model = yolo_model,native_model

    def _notify(self,stage,view):
        if self.progress is not None:
            self.progress(stage,view)

    def _yolo(self):
        if self._yolo_model is None:
            if self.yolo_checkpoint is None:
                raise ValueError('YOLO checkpoint is required')
            if self.verify_weights and digest(self.yolo_checkpoint) != YOLO_SHA256:
                raise ValueError('YOLO checkpoint differs from the selected validation pipeline')
            from ultralytics import YOLO
            self._yolo_model = YOLO(str(self.yolo_checkpoint),task='segment')
        return self._yolo_model

    def _native(self):
        if self._native_model is None:
            if self.native_checkpoint is None:
                raise ValueError('Native checkpoint is required')
            if self.verify_weights and digest(self.native_checkpoint) != NATIVE_SHA256:
                raise ValueError('Native checkpoint differs from the selected validation pipeline')
            checkpoint = torch.load(self.native_checkpoint,map_location='cpu',weights_only=True)
            cfg = checkpoint['config']
            for key,value in [('encoder','tu-convnext_tiny'),('crop',1024),('overlap',.25),('preprocessing','limb_only')]:
                if cfg.get(key) != value:
                    raise ValueError(f'Unexpected native checkpoint configuration: {key}')
            if checkpoint['epoch'] != 84:
                raise ValueError('The selected native checkpoint is epoch84')
            import segmentation_models_pytorch as smp
            model = smp.Unet(encoder_name='tu-convnext_tiny',encoder_weights=None,in_channels=3,classes=2,activation=None)
            model.load_state_dict(checkpoint['model_state_dict'])
            self._native_model = model.to(self.device).eval()
        return self._native_model

    def _result(self,pixels,size):
        return self._yolo().predict(cv2.cvtColor(np.ascontiguousarray(pixels),cv2.COLOR_GRAY2BGR),
            imgsz=size,conf=.05,iou=.7,retina_masks=True,device=str(self.device),
            max_det=100,verbose=False)[0]

    @staticmethod
    def _whole_record(result,stem,view,size,parameters=None):
        raw = []
        if result.masks is not None and result.boxes is not None:
            for mask,confidence in zip(result.masks.data.float().cpu().numpy(),result.boxes.conf.cpu().numpy()):
                if mask.shape != (2048,2048):
                    mask = cv2.resize(mask,(2048,2048),interpolation=cv2.INTER_NEAREST)
                mask = mask >= .5
                if view == 'hflip': mask = np.fliplr(mask)
                if view == 'vflip': mask = np.flipud(mask)
                raw.append(dict(confidence=float(confidence),rle=rle(np.ascontiguousarray(mask))['counts']))
        record = dict(schema_version=1,image=stem,view=view,input_size=size,shape=[2048,2048],
            coordinate_space='original',raw_min_confidence=.05,max_det_reached=len(raw)>=100,instances=raw)
        if parameters is not None:
            record['parameters'] = parameters
        return record

    @staticmethod
    def _tile_record(result,stem,size,top,left):
        confidences = (np.asarray(result.boxes.conf.cpu().numpy()) if result.boxes is not None else np.empty(0))
        count = len(confidences)
        if confidences.ndim != 1 or not np.isfinite(confidences).all() or not ((confidences>=0)&(confidences<=1)).all():
            raise ValueError('Invalid tiled detection confidence')
        instances,empty = [],0
        if result.masks is None:
            if count:
                raise ValueError('Tiled detection boxes have no masks')
        else:
            masks = result.masks.data.float().cpu().numpy()
            if masks.shape != (count,1024,1024):
                raise ValueError(f'Expected original crop1024 masks, received {masks.shape}')
            for pixels,confidence in zip(masks,confidences):
                mask = pixels >= .5
                yy,xx = np.nonzero(mask)
                if len(yy) == 0:
                    empty += 1
                    continue
                sides = edge_sides(mask,top,left)
                y1,y2,x1,x2 = int(yy.min()),int(yy.max())+1,int(xx.min()),int(xx.max())+1
                encoded = encode_native_roi(mask[y1:y2,x1:x2],top+y1,left+x1)
                instances.append(dict(confidence=float(confidence),rle=encoded['counts'].decode('ascii'),
                    edge_clipped=bool(sides),edge_sides=sides))
        return dict(schema_version=1,image=stem,view=f'tile{size}_y{top}_x{left}',input_size=size,
            crop_bbox=[left,top,1024,1024],shape=[2048,2048],coordinate_space='original',
            raw_min_confidence=.05,max_det=100,max_det_reached=count>=100,
            raw_box_count=count,empty_mask_count=empty,instances=instances)

    def geometric(self,image_id,image):
        corrected = limb(image)
        records = []
        for view,size in [('base',2048),('scale',2560),('hflip',2048),('vflip',2048)]:
            pixels = np.fliplr(corrected) if view == 'hflip' else np.flipud(corrected) if view == 'vflip' else corrected
            records.append(self._whole_record(self._result(pixels,size),image_id,view,size))
            self._notify('geometric',view)
        return records

    def preprocessing(self,image_id,image):
        corrected = limb(image)
        records = []
        for view,params in PREPROCESS_PARAMS.items():
            pixels = preprocess_variant(corrected,view)
            records.append(self._whole_record(self._result(pixels,2048),image_id,view,2048,params))
            self._notify('preprocessing',view)
        return records

    def tiled(self,image_id,image):
        corrected = limb(image)
        records = []
        for top in (0,512,1024):
            for left in (0,512,1024):
                crop = np.ascontiguousarray(corrected[top:top+1024,left:left+1024])
                for size in (1536,2048):
                    records.append(self._tile_record(self._result(crop,size),image_id,size,top,left))
                    self._notify('tiled',records[-1]['view'])
        return records

    def native(self,image_id,image,*,file_name=None,image_sha256='',identity='submission'):
        corrected = limb(image)
        amp = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if self.device.type == 'cuda' else None
        heads,average,unions = predict_views(self._native(),corrected,self.device,batch=self.native_batch,
            amp_dtype=amp,progress=lambda view:self._notify('native',view))
        name = file_name or f'{image_id}.jpeg'
        record = prediction_record(heads,name,image_sha256,identity)
        record['tta'] = prediction_record(average,name,image_sha256,identity)
        record['view_unions'] = unions
        return record


def native_server_main(argv=None):
    """Pipe/file worker; never constructs or imports an Ultralytics model."""
    import argparse
    import json
    import sys

    protocol = 'solar-native-v1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-server', action='store_true', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args(argv)
    provider = InferenceProvider(None,args.checkpoint,device=args.device,
        native_batch=args.batch,verify_weights=True,
        progress=lambda stage,view: print(f'Native {view}',file=sys.stderr,flush=True))
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get('id')
            if request.get('protocol') != protocol:
                raise ValueError('Unknown native worker protocol')
            if request.get('command') == 'close':
                break
            if request.get('command') != 'native':
                raise ValueError('Unknown native worker command')
            pixels = np.load(request['input_path'],allow_pickle=False)
            record = provider.native(request['image_id'],pixels,
                file_name=request['file_name'],image_sha256=request['image_sha256'],identity=request['identity'])
            target = Path(request['output_path'])
            pending = target.with_name(target.name+'.pending')
            pending.write_text(json.dumps(record,ensure_ascii=False,separators=(',', ':'),allow_nan=False),encoding='utf-8')
            pending.replace(target)
            response = dict(protocol=protocol,status='ok',id=request_id,output_path=str(target))
        except Exception as error:
            response = dict(protocol=protocol,status='error',id=request_id,
                error_type=type(error).__name__,message=str(error))
        print(json.dumps(response,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    native_server_main()
