"""CPU-only, resumable visual review of fixed external-reference test masks.

Reference masks and reference-selected raw candidates are audit aids, not test
ground truth or inference outputs. This module performs no training/inference.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import html
import io
import json
import math
from pathlib import Path
import re
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pycocotools import mask as mu

SHAPE = (2048, 2048)
COLORS = {'reference': (0, 205, 245), 'own': (255, 145, 35), 'raw': (226, 65, 218)}
OPTIONS = dict(version=1, case_panel=512, full_panel=600, alpha=.25,
               min_crop=160, context_fraction=.3, context_pixels=24,
               jpeg_quality=90, raw_selection='maximum reference IoU for visualization only')
CSV_FIELDS = ['image', 'file_name', 'reference_id', 'panel', 'matched', 'baseline_max_iou',
              'best_baseline_id', 'case_path', 'full_path', 'rendered',
              'crop_x0', 'crop_y0', 'crop_x1', 'crop_y1', 'reference_area', 'reference_rle_sha256',
              'best_raw_source', 'best_raw_view', 'best_raw_confidence', 'best_raw_reference_iou']
SAFE = re.compile(r'^[A-Za-z0-9_.-]+$')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.'+uuid.uuid4().hex+'.pending')
    pending.write_bytes(data)
    pending.replace(path)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_rle(counts):
    """Bound compressed COCO runs before passing untrusted counts to C."""
    require(isinstance(counts, str) and bool(counts), 'Missing compressed RLE')
    previous, total, foreground, index, position = [], 0, 0, 0, 0
    while position < len(counts):
        number = shift = 0
        while True:
            require(position < len(counts), 'Truncated RLE')
            digit = ord(counts[position])-48
            position += 1
            require(0 <= digit <= 63, 'Invalid RLE character')
            number |= (digit & 31) << shift
            shift += 5
            require(shift <= 65, 'RLE count overflow')
            if not digit & 32:
                if digit & 16:
                    number |= -1 << shift
                break
        if index > 2:
            number += previous[-2]
        require(number >= 0 and total+number <= SHAPE[0]*SHAPE[1], 'RLE exceeds native image')
        total += number
        if index % 2:
            foreground += number
        previous = (previous+[number])[-2:]
        index += 1
    require(total == SHAPE[0]*SHAPE[1] and foreground > 0, 'Empty or incomplete RLE')
    result = dict(size=list(SHAPE), counts=counts.encode('ascii'))
    require(int(mu.area(result)) == foreground, 'RLE area mismatch')
    return result


def safe_name(value, label):
    require(isinstance(value, str) and bool(SAFE.fullmatch(value)) and value not in {'.', '..'}, 'Invalid '+label)
    return value


def read_assets(assets):
    assets = Path(assets)
    manifest_path = assets/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get('schema_version') == 1, 'Unsupported assets manifest')
    files = manifest['files']
    require({'targets.json', 'own_baseline.csv'} <= set(files), 'Incomplete assets manifest')
    for name, expected in files.items():
        safe_name(name, 'asset filename')
        require(sha(assets/name) == expected, 'Bundled asset changed: '+name)
    targets = json.loads((assets/'targets.json').read_text())
    require(targets.get('schema_version') == 1, 'Unsupported targets schema')
    images, references = set(), set()
    by_image = defaultdict(list)
    own_ids = set()
    csv.field_size_limit(2**31-1)
    with (assets/'own_baseline.csv').open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ['filament_id', 'segmentation_rle'], 'Unexpected baseline CSV columns')
        for row in reader:
            key = safe_name(row['filament_id'], 'baseline ID')
            require(key not in own_ids and '_' in key, 'Duplicate or invalid baseline ID')
            own_ids.add(key)
            image, suffix = key.rsplit('_', 1)
            require(suffix.isdigit(), 'Invalid baseline instance suffix')
            by_image[image].append(dict(id=key, rle=checked_rle(row['segmentation_rle'])))
    for image in by_image:
        masks = [row['rle'] for row in by_image[image]]
        require(int(mu.area(mu.merge(masks))) == sum(int(mu.area(m)) for m in masks), 'Baseline masks overlap: '+image)
    unmatched = 0
    for photo in targets['images']:
        stem = safe_name(photo['image'], 'image stem')
        name = safe_name(photo['file_name'], 'image filename')
        require(Path(name).stem == stem and Path(name).suffix.lower() in {'.jpeg', '.jpg', '.png'}, 'Image filename/stem mismatch')
        require(stem not in images and photo['targets'], 'Duplicate image or empty image target list')
        require(re.fullmatch('[0-9a-f]{64}', photo['image_sha256']) is not None, 'Missing source image SHA256')
        images.add(stem)
        originals = by_image[stem]
        for target in photo['targets']:
            key = safe_name(target['reference_id'], 'reference ID')
            require(key not in references and key.rsplit('_', 1)[0] == stem, 'Duplicate reference or wrong image')
            require(target['panel'] in {216, 102}, 'Unknown fixed panel')
            references.add(key)
            ref = checked_rle(target['reference_rle'])
            ious = np.asarray(mu.iou([ref], [row['rle'] for row in originals], [0]*len(originals))).reshape(-1) if originals else np.zeros(0)
            actual = float(ious.max()) if len(ious) else 0.
            require(abs(actual-float(target['baseline_max_iou'])) < 1e-9, 'Baseline/reference IoU does not reproduce: '+key)
            require(type(target['matched']) is bool and target['matched'] == (actual > .5), 'Incorrect strict-IoU status: '+key)
            if not target['matched']:
                unmatched += 1
            best_id = target.get('best_baseline_id')
            if best_id is not None:
                matches = [i for i, row in enumerate(originals) if row['id'] == best_id]
                require(len(matches) == 1 and abs(float(ious[matches[0]])-actual) < 1e-9, 'Incorrect best baseline ID')
            best = target.get('best_raw')
            if best:
                raw = checked_rle(best['mask_rle'])
                measured = float(mu.iou([ref], [raw], [0])[0, 0])
                require(abs(measured-float(best['reference_iou'])) < 1e-9, 'Raw/reference IoU does not reproduce: '+key)
                require(math.isfinite(best['confidence']) and 0 <= best['confidence'] <= 1, 'Invalid raw confidence')
    expected = manifest['expected']
    require((len(images), len(references), unmatched) == (expected['images'], expected['targets'], expected['unmatched']), 'Frozen panel counts differ')
    if 'baseline_masks' in expected:
        require(len(own_ids) == expected['baseline_masks'], 'Baseline count differs')
    return targets, by_image, manifest, sha(manifest_path)


def crop_box(reference):
    x, y, w, h = map(float, mu.toBbox(reference))
    side = min(2048, max(OPTIONS['min_crop'], math.ceil(max(w, h)*(1+2*OPTIONS['context_fraction'])+2*OPTIONS['context_pixels'])))
    x0 = min(2048-side, max(0, math.floor(x+w/2-side/2)))
    y0 = min(2048-side, max(0, math.floor(y+h/2-side/2)))
    return (x0, y0, x0+side, y0+side)


def font(size):
    for path in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def panel(original, masks, color, size, title):
    """Render in native coordinates, then resize both image/mask consistently."""
    original = original.convert('RGB')
    scale = size/max(original.size)
    width, height = max(1, round(original.width*scale)), max(1, round(original.height*scale))
    rgb = np.asarray(original.resize((width, height), Image.Resampling.LANCZOS)).copy()
    if masks:
        union = np.logical_or.reduce(masks)
        require(union.shape == (original.height, original.width), 'Crop mask/image coordinates differ')
        mask = Image.fromarray(union.astype(np.uint8)*255).resize((width, height), Image.Resampling.NEAREST)
        inside = np.asarray(mask) > 0
        outline = np.asarray(mask.filter(ImageFilter.MaxFilter(3))) != np.asarray(mask.filter(ImageFilter.MinFilter(3)))
        rgb[inside] = np.round(rgb[inside]*(1-OPTIONS['alpha'])+np.asarray(color)*OPTIONS['alpha']).astype(np.uint8)
        rgb[outline] = color
    result = Image.new('RGB', (size, size+38), (19, 23, 28))
    result.paste(Image.fromarray(rgb), ((size-width)//2, 38+(size-height)//2))
    draw = ImageDraw.Draw(result)
    draw.text((9, 10), title, font=font(16), fill=(color if masks else (240, 240, 240)))
    return result


def montage(panels, header, footer):
    gap, margin = 10, 16
    width = sum(p.width for p in panels)+gap*(len(panels)-1)+margin*2
    height = max(p.height for p in panels)+128
    result = Image.new('RGB', (width, height), (10, 14, 18))
    draw = ImageDraw.Draw(result)
    draw.text((margin, 12), header, font=font(19), fill=(245, 245, 245))
    x = margin
    for p in panels:
        result.paste(p, (x, 44))
        x += p.width+gap
    for index, line in enumerate(footer):
        draw.text((margin, height-65+index*23), line, font=font(16), fill=(214, 221, 230))
    return result


def save_jpeg(path, image):
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=OPTIONS['jpeg_quality'], optimize=True, subsampling=0)
    atomic_bytes(path, stream.getvalue())


def render_photo(photo, original, originals, output):
    stem = photo['image']
    own_masks = [mu.decode(row['rle']).astype(bool) for row in originals]
    refs = [mu.decode(checked_rle(target['reference_rle'])).astype(bool) for target in photo['targets']]
    full = montage([
        panel(original, [], COLORS['reference'], OPTIONS['full_panel'], 'ORIGINAL'),
        panel(original, refs, COLORS['reference'], OPTIONS['full_panel'], 'PUBLIC REFERENCE / selected cases'),
        panel(original, own_masks, COLORS['own'], OPTIONS['full_panel'], 'OWN FINAL / all predictions')],
        f'{photo["file_name"]} | {len(photo["targets"])} selected reference masks',
        ['Reference masks are public predictions, NOT test ground truth.', 'Cyan: selected public references. Orange: frozen own final predictions.'])
    full_path = Path('full')/(stem+'.jpg')
    save_jpeg(output/full_path, full)
    paths = [str(full_path)]
    for target, reference in zip(photo['targets'], refs):
        ref_rle = checked_rle(target['reference_rle'])
        box = crop_box(ref_rle)
        x0, y0, x1, y1 = box
        crop = original.crop(box)
        ref_crop = reference[y0:y1, x0:x1]
        own_crops = [mask[y0:y1, x0:x1] for mask in own_masks if mask[y0:y1, x0:x1].any()]
        raw = target.get('best_raw')
        raw_crop = mu.decode(checked_rle(raw['mask_rle']))[y0:y1, x0:x1].astype(bool) if raw else None
        raw_title = 'BEST RAW / reference-selected' if raw else 'NO OVERLAPPING RAW CANDIDATE'
        raw_info = (f'Raw: {raw["source"]}/{raw["view"]} | confidence={raw["confidence"]:.3f} | reference IoU={raw["reference_iou"]:.3f}'
                    if raw else 'No own raw candidate overlaps this reference mask in the available audit pool.')
        status = 'ALREADY MATCHED (>0.5)' if target['matched'] else 'NOT MATCHED (<=0.5)'
        figure = montage([
            panel(crop, [], COLORS['reference'], OPTIONS['case_panel'], 'ORIGINAL CROP'),
            panel(crop, [ref_crop], COLORS['reference'], OPTIONS['case_panel'], 'PUBLIC REFERENCE / not GT'),
            panel(crop, own_crops, COLORS['own'], OPTIONS['case_panel'], 'OWN FINAL / crop contents'),
            panel(crop, [raw_crop] if raw_crop is not None else [], COLORS['raw'], OPTIONS['case_panel'], raw_title)],
            f'{target["reference_id"]} | panel {target["panel"]} | own max IoU={target["baseline_max_iou"]:.3f} | {status}',
            [raw_info, 'Same native crop; masks may extend beyond this view. Reference-selected raw is diagnostic only, not a model selection.'])
        case_path = Path('cases')/(target['reference_id']+'.jpg')
        save_jpeg(output/case_path, figure)
        paths.append(str(case_path))
    return paths


def case_rows(targets, rendered):
    result = []
    for photo in sorted(targets['images'], key=lambda row: row['image']):
        for target in sorted(photo['targets'], key=lambda row: row['reference_id']):
            value = checked_rle(target['reference_rle'])
            box = crop_box(value)
            raw = target.get('best_raw') or {}
            result.append(dict(image=photo['image'], file_name=photo['file_name'], reference_id=target['reference_id'], panel=target['panel'],
                matched=target['matched'], baseline_max_iou=target['baseline_max_iou'], best_baseline_id=target.get('best_baseline_id') or '',
                case_path='cases/'+target['reference_id']+'.jpg', full_path='full/'+photo['image']+'.jpg', rendered=photo['image'] in rendered,
                crop_x0=box[0], crop_y0=box[1], crop_x1=box[2], crop_y1=box[3], reference_area=int(mu.area(value)),
                reference_rle_sha256=hashlib.sha256(target['reference_rle'].encode()).hexdigest(),
                best_raw_source=raw.get('source', ''), best_raw_view=raw.get('view', ''),
                best_raw_confidence=raw.get('confidence', ''), best_raw_reference_iou=raw.get('reference_iou', '')))
    return result


def write_csv(path, rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode('utf-8-sig'))


HTML = r'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>태양 필라멘트 참조 시각 검토</title><style>
body{font:16px system-ui,sans-serif;background:#10151b;color:#e7edf4;margin:20px}button,select,input,textarea{font:inherit;padding:7px;margin:4px;background:#23303d;color:#fff;border:1px solid #586b7e;border-radius:4px}a{color:#71d4ff}#viewer{display:block;max-width:100%;height:auto;margin-top:12px}small{color:#b9c5d0}#notes{width:min(800px,90%);height:65px}h1{font-size:23px}.bar{position:sticky;top:0;background:#10151bf2;padding:8px 0}#number{width:85px}
</style><h1>원본 / 공개 참조 / 자체 최종 출력 / 자체 원시 후보</h1>
<p id="summary"></p><p><b>공개 참조는 정답지가 아닙니다.</b> 원시 후보는 비교를 위해 참조와 가장 비슷한 후보를 골랐습니다. 새 모델의 추론 결과가 아닙니다.</p>
<div class="bar"><select id="group"><option value="all">216개 + 102개</option><option value="216">누락 참조216개</option><option value="102">형상 교체 참조102개</option></select>
<label><input id="controls" type="checkbox">이미 대응하는 대조 사례 포함</label>
<select id="mode"><option value="case">객체별 확대</option><option value="photo">원본 사진별 전체 보기</option></select><br>
<button id="previous">← 이전</button><input id="number" type="number" min="1" value="1"><span id="counter"></span><button id="next">다음 →</button>
<a id="full" target="_blank">전체 사진 비교 열기</a> · <a href="case_manifest.csv">전체 목록 CSV</a></div>
<p id="caption"></p><a id="large" target="_blank"><img id="viewer" alt="선택 사례의 원본과 마스크 비교"></a>
<div id="annotation"><p>시각 검토 메모: <select id="visual"><option value="">미분류</option><option value="sunspot_suspected">흑점 의심</option><option value="filament_like">필라멘트처럼 보임</option><option value="artifact_or_noise">영상 흔적·잡음 의심</option><option value="boundary_or_extent">경계·범위 문제</option><option value="unclear">판단 어려움</option><option value="other">기타</option></select></p><textarea id="notes" placeholder="관찰한 특징을 기록하세요."></textarea><br><button id="export">메모 CSV 내려받기</button>
<p><small>메모는 이 브라우저에 저장됩니다(가능한 경우). CSV로 내보내 보관하세요. test 메모는 시각 진단용이며 학습 정답으로 자동 사용하지 않습니다. Drive의 human_review.csv는 별도 수동 기록용 원본입니다.</small></p></div>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent), all=data.cases;
const el=id=>document.getElementById(id);let current=0,list=[],memo={};
try{memo=JSON.parse(localStorage.getItem(data.storage_key)||'{}')}catch(e){}
el('summary').textContent=`원본 사진 ${data.expected.images}장 / 고정 참조 ${data.expected.targets}개 / 최종 불일치 ${data.expected.unmatched}개. 렌더링 완료: ${data.rendered_images}장. ${data.partial?'부분 실행 결과입니다. 전체 셀을 재실행해 완료하세요.':''}`;
function filter(){const g=el('group').value,show=el('controls').checked;list=all.filter(x=>x.rendered&&(g==='all'||String(x.panel)===g)&&(show||!x.matched));if(el('mode').value==='photo'){const seen=new Set();list=list.filter(x=>!seen.has(x.image)&&seen.add(x.image))}current=0;render()}
function render(){const row=list[current];el('counter').textContent=` / ${list.length}`;el('number').max=Math.max(1,list.length);el('number').value=row?current+1:0;el('previous').disabled=current<=0;el('next').disabled=current>=list.length-1;if(!row){el('caption').textContent='현재 조건의 렌더링 완료 사례가 없습니다.';el('viewer').removeAttribute('src');el('annotation').hidden=true;return}const photo=el('mode').value==='photo';el('viewer').src=photo?row.full_path:row.case_path;el('large').href=el('viewer').getAttribute('src');el('full').href=row.full_path;el('caption').textContent=photo?row.file_name:`${row.reference_id} | panel${row.panel} | 자체 최종과 최대 IoU ${Number(row.baseline_max_iou).toFixed(3)} | ${row.matched?'이미 대응':'불일치'}`;el('annotation').hidden=photo;const note=memo[row.reference_id]||{};el('visual').value=note.visual_class||'';el('notes').value=note.comment||''}
function save(){const row=list[current];if(!row||el('mode').value==='photo')return;memo[row.reference_id]={visual_class:el('visual').value,comment:el('notes').value};try{localStorage.setItem(data.storage_key,JSON.stringify(memo))}catch(e){}}
for(const name of ['group','controls','mode'])el(name).addEventListener('change',filter);
el('previous').onclick=()=>{save();current=Math.max(0,current-1);render()};el('next').onclick=()=>{save();current=Math.min(list.length-1,current+1);render()};el('number').onchange=()=>{save();current=Math.max(0,Math.min(list.length-1,Number(el('number').value)-1));render()};el('visual').onchange=save;el('notes').oninput=save;
el('export').onclick=()=>{save();const fields=['reference_id','image','panel','visual_class','comment'];const quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';const lines=[fields.join(',')];for(const row of all){const n=memo[row.reference_id]||{};lines.push([row.reference_id,row.image,row.panel,n.visual_class||'',n.comment||''].map(quote).join(','))}const url=URL.createObjectURL(new Blob(['\ufeff'+lines.join('\r\n')],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='reference318_visual_notes.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
filter();</script></html>'''


def progress_valid(output, stem, identity, expected_paths):
    path = output/'.progress'/(stem+'.json')
    if not path.is_file():
        return False
    try:
        saved = json.loads(path.read_text())
        if saved['identity'] != identity or set(saved['outputs']) != set(expected_paths):
            return False
        for relative, expected in saved['outputs'].items():
            target = (output/relative).resolve()
            if output.resolve() not in target.parents or not target.is_file() or sha(target) != expected:
                return False
        return bool(saved['outputs'])
    except (KeyError, ValueError, OSError):
        return False


def run(images, assets, output, limit_images=None):
    images, assets, output = map(lambda p: Path(p).resolve(), (images, assets, output))
    require(images.is_dir(), 'Test images directory is missing: '+str(images))
    require(output != images and output not in images.parents and images not in output.parents, 'Output must be separate from source images')
    require(output != assets and output not in assets.parents and assets not in output.parents, 'Output must be separate from frozen assets')
    require(limit_images is None or type(limit_images) is int and limit_images > 0, 'limit-images must be positive')
    targets, baseline, manifest, manifest_sha = read_assets(assets)
    photos = sorted(targets['images'], key=lambda row: row['image'])
    missing = [p['file_name'] for p in photos if not (images/p['file_name']).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} source test images under {images}: {missing[:5]}')
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(schema_version=1, manifest_sha256=manifest_sha, source_sha256=sha(__file__), images=str(images), options=OPTIONS)
    contract = output/'run_identity.json'
    if contract.exists():
        require(json.loads(contract.read_text()) == identity, 'Saved gallery code/assets/settings differ; use a new output folder')
    else:
        atomic_json(contract, identity)
    rendered, reused = set(), 0
    for index, photo in enumerate(photos, 1):
        if limit_images is not None and index > limit_images:
            break
        source = images/photo['file_name']
        require(sha(source) == photo['image_sha256'], 'Source test JPEG hash differs from frozen inference: '+str(source))
        picture_identity = dict(run=fingerprint(identity), image_sha256=photo['image_sha256'], photo_sha256=fingerprint(photo))
        expected_paths = ['full/'+photo['image']+'.jpg']+['cases/'+t['reference_id']+'.jpg' for t in photo['targets']]
        if progress_valid(output, photo['image'], picture_identity, expected_paths):
            reused += 1
        else:
            with Image.open(source) as handle:
                require(handle.size == (SHAPE[1], SHAPE[0]), 'Expected original2048x2048 source image: '+str(source))
                original = handle.convert('RGB')
            paths = render_photo(photo, original, baseline[photo['image']], output)
            atomic_json(output/'.progress'/(photo['image']+'.json'), dict(identity=picture_identity, outputs={name: sha(output/name) for name in paths}))
        rendered.add(photo['image'])
        if index == 1 or index % 10 == 0 or index == len(photos):
            print(f'Visual review [{index}/{len(photos)}] {photo["file_name"]}; reused={reused}', flush=True)
    rows = case_rows(targets, rendered)
    write_csv(output/'case_manifest.csv', rows, CSV_FIELDS)
    review = output/'human_review.csv'
    fields = ['reference_id', 'image', 'panel', 'visual_class', 'comment']
    if not review.exists():
        write_csv(review, [dict(reference_id=r['reference_id'], image=r['image'], panel=r['panel'], visual_class='', comment='') for r in rows], fields)
    data = dict(cases=rows, expected=manifest['expected'], rendered_images=len(rendered), partial=len(rendered) != len(photos), storage_key='filament_reference_review_'+manifest_sha)
    escaped = json.dumps(data, ensure_ascii=False).replace('<', chr(92)+'u003c')
    atomic_bytes(output/'index.html', HTML.replace('__DATA__', escaped).encode())
    result = dict(status='complete' if len(rendered) == len(photos) else 'partial', images_completed=len(rendered), images_expected=len(photos),
        cases_total=len(rows), cases_rendered=sum(r['rendered'] for r in rows), unmatched=sum(not r['matched'] for r in rows),
        matched_controls=sum(r['matched'] for r in rows), images_reused=reused,
        index_html=str(output/'index.html'), case_manifest=str(output/'case_manifest.csv'), human_review=str(review),
        source_identity=identity, generated_jpegs=sum(1+len(p['targets']) for p in photos if p['image'] in rendered),
        scope='Public-reference visual audit only. Not test GT/PQ; no training or new model inference.',
        raw_selection='Reference-max-IoU raw masks for diagnosis only, not model-selected predictions',
        review_notes='Manual test observations are never consumed by training code')
    atomic_json(output/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_identity'}, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['images', 'assets', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--limit-images', type=int)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == '__main__':
    main()
