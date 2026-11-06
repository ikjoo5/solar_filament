# Run in a NEW cell in the same Colab notebook as the previous evaluation.
import ast, subprocess, sys, tempfile
from pathlib import Path
from google.colab import files

assert isinstance(globals().get('SCRIPT'), str), '기존 평가 코드를 실행했던 Colab에서 새 셀로 실행하세요.'

INITIALIZE = '''
THRESHOLDS = [.05, .10, .15, .20, .25]
predictions = {(f'base_{a}', t): {} for a in (20, 100) for t in THRESHOLDS}
predictions.update({(f'vote{v}_{a}', t): {} for v in (50, 75)
                    for a in (100, 200) for t in (.10, .15, .20)})
'''
POSTPROCESS = '''
def postprocess(masks, mode):
    if not masks:
        return []
    union = np.logical_or.reduce(masks).astype(np.uint8)
    minimum_area = int(mode.rsplit('_', 1)[1])
    n, labels, stats, _ = cv2.connectedComponentsWithStats(union, connectivity=8)
    ids = sorted((i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= minimum_area),
                 key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
    return [labels == i for i in ids]
'''
INFERENCE = '''
for index, name in enumerate(val_names, 1):
    image = limb(cv2.imread(str(IMAGES/name), cv2.IMREAD_GRAYSCALE))
    views = {}
    for view, size in [('base',2048), ('scale',2560), ('hflip',2048), ('vflip',2048)]:
        pixels = np.fliplr(image) if view == 'hflip' else np.flipud(image) if view == 'vflip' else image
        result = model.predict(cv2.cvtColor(np.ascontiguousarray(pixels), cv2.COLOR_GRAY2BGR),
            imgsz=size, conf=.05, iou=.7, retina_masks=True, device=0,
            max_det=100, verbose=False)[0]
        raw = []
        if result.masks is not None and result.boxes is not None:
            for mask, confidence in zip(result.masks.data.float().cpu().numpy(), result.boxes.conf.cpu().numpy()):
                if mask.shape != (2048,2048):
                    mask = cv2.resize(mask, (2048,2048), interpolation=cv2.INTER_NEAREST)
                mask = mask >= .5
                if view == 'hflip': mask = np.fliplr(mask)
                if view == 'vflip': mask = np.flipud(mask)
                raw.append((np.ascontiguousarray(mask),float(confidence)))
        views[view] = raw
        records = [{'confidence': c, 'rle': mu.encode(np.asfortranarray(m.astype(np.uint8)))['counts'].decode('ascii')}
                   for m,c in raw]
        with (output/'raw_views.jsonl').open('a', encoding='utf-8') as cache:
            cache.write(json.dumps({'schema_version':1,'image':Path(name).stem,'view':view,'input_size':size,
                'shape':[2048,2048],'coordinate_space':'original','raw_min_confidence':.05,
                'max_det_reached':len(raw)>=100,'instances':records})+'\\n')
    votes = {}
    for t in (.10,.15,.20):
        count = np.zeros((2048,2048), dtype=np.uint8)
        for raw in views.values():
            union = np.zeros((2048,2048),dtype=bool)
            for mask, confidence in raw:
                if confidence >= t: union |= mask
            count += union
        votes[t] = count
    for (mode,t), predicted in predictions.items():
        if mode.startswith('base_'):
            masks = [m for m,c in views['base'] if c >= t]
        else:
            fraction = int(mode.split('_')[0][4:])/100
            masks = [votes[t] >= int(np.ceil(4*fraction))]
        final_masks = postprocess(masks,mode)
        predicted[Path(name).stem] = [mu.encode(np.asfortranarray(m.astype(np.uint8))) for m in final_masks]
    print(f'[{index}/{len(val_names)}] {name}: '+str({v:len(r) for v,r in views.items()}),flush=True)
'''

class Patch(ast.NodeTransformer):
    counts = None
    def __init__(self):
        self.counts = dict(init=0,post=0,loop=0,summary=0)
    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id=='predictions' for t in node.targets):
            self.counts['init'] += 1
            return ast.parse(INITIALIZE).body
        if any(isinstance(t,ast.Name) and t.id=='summary' for t in node.targets):
            self.counts['summary'] += 1
            assert isinstance(node.value,ast.Call)
            for key in node.value.keywords:
                if key.arg=='note':key.value=ast.Constant('Four-view binary votes, not probability maps. Same-set tuning. Raw cache stores actual confidence.')
            node.value.keywords.append(ast.keyword(arg='experiment',value=ast.Constant('low_conf_4view_v1')))
            node.value.keywords.append(ast.keyword(arg='views',value=ast.parse("['base2048','scale2560','hflip2048','vflip2048']",mode='eval').body))
            node.value.keywords.append(ast.keyword(arg='target_exceeded',value=ast.parse("max(r['PQ'] for r in comparisons) > .5",mode='eval').body))
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

patch = Patch()
tree = patch.visit(ast.parse(SCRIPT))
assert patch.counts == dict(init=1,post=1,loop=1,summary=1), '기존 SCRIPT 구조가 다릅니다. 오류 내용을 보내주세요.'
worker = ast.unparse(ast.fix_missing_locations(tree))
assert "raw_views.jsonl" in worker
# Keep the raw JSONL in the result ZIP as well as all CSV/JSON outputs.
compile(worker,'next-inference','exec')
work2=Path(tempfile.mkdtemp(prefix='solar_next_'))
script2=work2/'evaluate.py'; script2.write_text(worker,encoding='utf-8')
result_path2=work2/'result_zip.txt'
subprocess.run([sys.executable,str(script2),str(result_path2)],check=True)
files.download(result_path2.read_text(encoding='utf-8'))
