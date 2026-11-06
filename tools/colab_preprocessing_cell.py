# Paste in a NEW cell of the Colab notebook that ran the previous evaluation.
import ast, subprocess, sys, tempfile
from pathlib import Path
from google.colab import files

assert isinstance(globals().get('SCRIPT'), str), '기존 평가를 실행했던 Colab의 새 셀에서 실행하세요.'

INITIALIZE = '''
PREPROCESS_PARAMS = {
    'limb': {'operation':'identity after original limb correction'},
    'clahe16': {'operation':'CLAHE blend','grid':[16,16],'clip_limit':1.5,'blend':.5},
    'clahe32': {'operation':'CLAHE blend','grid':[32,32],'clip_limit':2.0,'blend':.5},
    'percentile': {'operation':'inside-disk percentile stretch blend','percentiles':[1,99],'blend':.25},
    'unsharp': {'operation':'unsharp','gaussian_sigma':1.0,'gain':.5},
    'gaussian': {'operation':'Gaussian blur','sigma':.5},
    'bilateral': {'operation':'bilateral','diameter':5,'sigma_color':10,'sigma_space':3},
    'blackhat': {'operation':'subtract blackhat response','ellipse':[15,15],'gain':.5}
}

def preprocess_variant(image, name, valid_mask=None):
    valid = disk if valid_mask is None else np.asarray(valid_mask,dtype=bool)
    if image.ndim != 2 or image.dtype != np.uint8 or valid.shape != image.shape:
        raise ValueError('preprocessing requires grayscale uint8 and matching disk mask')
    if name not in PREPROCESS_PARAMS:
        raise ValueError('unknown preprocessing view: '+str(name))
    base = np.ascontiguousarray(image)
    values = base.astype(np.float32)
    if name == 'limb':
        result = base.copy()
    elif name in ('clahe16','clahe32'):
        p = PREPROCESS_PARAMS[name]
        enhanced = cv2.createCLAHE(clipLimit=p['clip_limit'],tileGridSize=tuple(p['grid'])).apply(base)
        result = .5*values+.5*enhanced.astype(np.float32)
    elif name == 'percentile':
        if not np.any(valid):
            raise ValueError('disk mask has no pixels')
        low,high = np.percentile(base[valid],[1,99])
        stretched = np.clip((values-low)*(255./(high-low)),0,255) if high>low else values
        result = .75*values+.25*stretched
    elif name == 'unsharp':
        smooth = cv2.GaussianBlur(values,(0,0),1.0)
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

def encode_native_roi(mask,top,left):
    height,width = 2048,2048
    runs,previous = [],0
    for col in range(mask.shape[1]):
        edges = np.flatnonzero(np.diff(np.r_[0,mask[:,col].astype(np.int8),0]))
        for start,end in edges.reshape(-1,2):
            first = (left+col)*height+top+int(start)
            last = (left+col)*height+top+int(end)
            runs.extend([first-previous,last-first]); previous=last
    runs.append(height*width-previous)
    return mu.frPyObjects({'size':[height,width],'counts':runs},height,width)

PREP_SPECS = []
for view in PREPROCESS_PARAMS:
    for confidence in (.15,.20,.25):
        for area in (100,200):
            PREP_SPECS.append(dict(name=view+'_'+str(area),confidence=confidence,
                views=[view],required=1,minimum_area=area))
for prefix,members,required in [
    ('contrast3of4',['limb','clahe16','clahe32','unsharp'],3),
    ('smooth2of3',['limb','gaussian','bilateral'],2)
]:
    for confidence in (.15,.20,.25):
        for area in (200,300):
            PREP_SPECS.append(dict(name=prefix+'_'+str(area),confidence=confidence,
                views=members,required=required,minimum_area=area))
for required in (4,6):
    for confidence in (.15,.20):
        for area in (200,300):
            PREP_SPECS.append(dict(name='all8votes'+str(required)+'_'+str(area),confidence=confidence,
                views=list(PREPROCESS_PARAMS),required=required,minimum_area=area))
assert len(PREP_SPECS)==68
predictions = {(s['name'],s['confidence']):{} for s in PREP_SPECS}
assert len(predictions)==68
'''

POSTPROCESS = '''
def postprocess(foreground,minimum_area):
    n,labels,stats,_ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8),connectivity=8)
    encoded = []
    for label in range(1,n):
        x,y,w,h,area = map(int,stats[label])
        if area>=minimum_area:
            rle = encode_native_roi(labels[y:y+h,x:x+w]==label,y,x)
            assert int(mu.area(rle))==area
            encoded.append((rle,area))
    return encoded
'''

INFERENCE = '''
for index,name in enumerate(val_names,1):
    image = limb(cv2.imread(str(IMAGES/name),cv2.IMREAD_GRAYSCALE))
    raw_views = {}
    for view,parameters in PREPROCESS_PARAMS.items():
        pixels = preprocess_variant(image,view)
        result = model.predict(cv2.cvtColor(pixels,cv2.COLOR_GRAY2BGR),
            imgsz=2048,conf=.05,iou=.7,retina_masks=True,device=0,
            max_det=100,verbose=False)[0]
        raw = []
        if result.masks is not None and result.boxes is not None:
            for mask,confidence in zip(result.masks.data.float().cpu().numpy(),result.boxes.conf.cpu().numpy()):
                if mask.shape != (2048,2048):
                    mask = cv2.resize(mask,(2048,2048),interpolation=cv2.INTER_NEAREST)
                raw.append((np.ascontiguousarray(mask>=.5),float(confidence)))
        raw_views[view] = raw
        records = [{'confidence':c,'rle':mu.encode(np.asfortranarray(m.astype(np.uint8)))['counts'].decode('ascii')}
                   for m,c in raw]
        with (output/'raw_preprocess_views.jsonl').open('a',encoding='utf-8') as cache:
            cache.write(json.dumps({'schema_version':1,'image':Path(name).stem,'view':view,
                'parameters':parameters,'input_size':2048,'shape':[2048,2048],
                'coordinate_space':'original','raw_min_confidence':.05,
                'max_det_reached':len(raw)>=100,'instances':records})+'\\n')
    unions = {}
    for view,raw in raw_views.items():
        for confidence in (.15,.20,.25):
            foreground = np.zeros((2048,2048),dtype=bool)
            for mask,c in raw:
                if c>=confidence:foreground |= mask
            unions[view,confidence] = foreground
    encoded_cache = {}
    for spec in PREP_SPECS:
        key = (tuple(spec['views']),spec['confidence'],spec['required'])
        if key not in encoded_cache:
            count = np.zeros((2048,2048),dtype=np.uint8)
            for view in spec['views']:count += unions[view,spec['confidence']]
            lowest_area = 100 if len(spec['views'])==1 else 200
            encoded_cache[key] = postprocess(count>=spec['required'],lowest_area)
        predictions[spec['name'],spec['confidence']][Path(name).stem] = [
            rle for rle,area in encoded_cache[key] if area>=spec['minimum_area']]
    print(f'[{index}/{len(val_names)}] {name}: '+str({v:len(r) for v,r in raw_views.items()}),flush=True)
'''


class Patch(ast.NodeTransformer):
    def __init__(self):
        self.counts = dict(init=0,post=0,loop=0,summary=0)

    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id=='predictions' for t in node.targets):
            self.counts['init'] += 1
            return ast.parse(INITIALIZE).body
        if any(isinstance(t,ast.Name) and t.id=='summary' for t in node.targets):
            self.counts['summary'] += 1
            assert isinstance(node.value,ast.Call)
            # Replace experiment keys if a previously patched worker is passed.
            added={'experiment','views','preprocessing_parameters','candidate_plan','target_exceeded'}
            node.value.keywords=[k for k in node.value.keywords if k.arg not in added]
            for item in node.value.keywords:
                if item.arg=='note':
                    item.value=ast.Constant('Eight bounded preprocessing variants on original limb input; fixed global settings; same-development tuning, not learned preprocessing or independent CV.')
            expressions={
                'experiment':"'preprocessing_8view_v1'",
                'views':'list(PREPROCESS_PARAMS)',
                'preprocessing_parameters':'PREPROCESS_PARAMS',
                'candidate_plan':'PREP_SPECS',
                'target_exceeded':"max(r['PQ'] for r in comparisons) > .5"
            }
            node.value.keywords.extend(ast.keyword(arg=k,value=ast.parse(v,mode='eval').body)
                                       for k,v in expressions.items())
        return node

    def visit_FunctionDef(self,node):
        if node.name=='postprocess':
            self.counts['post'] += 1
            return ast.parse(POSTPROCESS).body[0]
        return node

    def visit_For(self,node):
        names={n.id for n in ast.walk(node.target) if isinstance(n,ast.Name)}
        if names=={'index','name'}:
            self.counts['loop'] += 1
            return ast.parse(INFERENCE).body[0]
        return node


patch=Patch()
tree=patch.visit(ast.parse(SCRIPT))
assert patch.counts==dict(init=1,post=1,loop=1,summary=1), '기존 SCRIPT 구조가 다릅니다. 오류 내용을 보내주세요.'
worker=ast.unparse(ast.fix_missing_locations(tree))
compile(worker,'preprocessing-experiment','exec')
work3=Path(tempfile.mkdtemp(prefix='solar_preprocessing_'))
script3=work3/'evaluate.py';script3.write_text(worker,encoding='utf-8')
result_path3=work3/'result_zip.txt'
subprocess.run([sys.executable,str(script3),str(result_path3)],check=True)
files.download(result_path3.read_text(encoding='utf-8'))
