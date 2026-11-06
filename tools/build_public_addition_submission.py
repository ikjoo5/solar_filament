"""Frozen CSV ablation: historical own LB0.39 plus216 disjoint public instances.

Select natural public masks with max IoU <0.1 against our original instances.
Discard any candidate sharing even one pixel with an original mask. Append the
remaining exact masks under fresh IDs. Actual Kaggle rejects overlaps, even
though the supplied self-evaluation notebook does not enforce this constraint.
No model, mask editing, GT, or remote submission.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from pycocotools import mask as mu
from filament_eval import read_csv

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT/'attachments/39d1023a-a635-4f96-90ec-d273fa62693e/submission_pq0472593.csv'
REFERENCE = ROOT/'attachments/d03f8d87-a646-4f04-89dd-4ad6ab534cec/submission.csv'
NAMES = ROOT/'attachments/14e0ea8c-23c5-4cd5-9344-a783251831af/official_image_names.json'
BASELINE_SHA = '13dd9cc48bf2e444792254611a86e4e8a1c7412717de073f61310ba98cb54db0'
REFERENCE_SHA = '0ab3932122a563ba96f4d243934d5c00680dd49b9d1165912a4f180907d2c603'
OUTPUT = ROOT/'artifacts/downloads/submission_own039_plus_public216_no_overlap.csv'
REPORT = ROOT/'artifacts/public055_addition_to039_v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_exact(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError('Existing artifact differs; refusing overwrite: '+str(path))
    path.write_bytes(data)


def known_placeholder(row):
    return (row.image_id in {'20160513133714Th', '20180507140210Th'} and row.area == 2304
            and np.array_equal(mu.toBbox(row.rle(2048, 2048)), [1000,1000,48,48]))


def csv_bytes(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode()


def require_disjoint(rows):
    """Enforce the actual submission constraint, including one-pixel overlaps."""
    groups = defaultdict(list)
    for row in rows:
        groups[row.image_id].append(row)
    for image, members in groups.items():
        masks = [row.rle(2048,2048) for row in members]
        excess = sum(int(mu.area(mask)) for mask in masks)-int(mu.area(mu.merge(masks)))
        if excess != 0:
            raise ValueError(f'Overlapping submission masks: {image}, {excess} excess pixels')


def main():
    assert sha(BASELINE) == BASELINE_SHA
    assert sha(REFERENCE) == REFERENCE_SHA
    expected = {Path(name).stem for name in json.loads(NAMES.read_text())['test_images']}
    assert len(expected) == 180
    ours, public = read_csv(BASELINE), read_csv(REFERENCE)
    assert len(ours) == 1347 and len(public) == 1344
    groups = defaultdict(list); next_id = Counter()
    for row in ours:
        groups[row.image_id].append(row)
        next_id[row.image_id] = max(next_id[row.image_id], int(row.filament_id.rsplit('_',1)[1]))
    for rows in groups.values():
        assert int(mu.area(mu.merge([r.rle(2048,2048) for r in rows]))) == sum(r.area for r in rows)
    additions, mapping, excluded, overlapping_candidates = [], [], [], []
    for row in public:
        if known_placeholder(row):
            excluded.append(row.filament_id)
            continue
        original = groups[row.image_id]
        other = [r.rle(2048,2048) for r in original]
        mask = row.rle(2048,2048)
        max_iou = float(mu.iou([mask],other,[0]*len(other)).max()) if other else 0.
        if max_iou >= .1:
            continue
        overlap = int(mu.area(mu.merge([mask,mu.merge(other)],intersect=True))) if other else 0
        if overlap:
            overlapping_candidates.append(dict(image=row.image_id,reference_id=row.filament_id,
                area=row.area,max_iou_against_original=max_iou,overlap_pixels_with_original=overlap,
                overlap_fraction=overlap/row.area,reason='Pixel overlap prohibited by actual Kaggle validator'))
            continue
        next_id[row.image_id] += 1
        identifier = f'{row.image_id}_{next_id[row.image_id]}'
        additions.append(dict(filament_id=identifier,segmentation_rle=row.counts))
        mapping.append(dict(image=row.image_id,added_id=identifier,reference_id=row.filament_id,
            area=row.area,max_iou_against_original=max_iou,overlap_pixels_with_original=overlap,
            overlap_fraction=overlap/row.area,mask_edited=False))
    assert len(additions) == 216 and len(overlapping_candidates) == 9 and len(excluded) == 2
    original_bytes = BASELINE.read_bytes()
    assert original_bytes.endswith(b'\n')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream,fieldnames=['filament_id','segmentation_rle'],
                            lineterminator='\r\n' if original_bytes.endswith(b'\r\n') else '\n')
    writer.writerows(additions)
    output_bytes = original_bytes+stream.getvalue().encode()
    REPORT.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=REPORT,suffix='.pending.csv') as candidate:
        candidate.write(output_bytes); candidate.flush()
        result = read_csv(candidate.name)
    assert len(result) == 1563 and len({r.filament_id for r in result}) == 1563
    assert result[:1347] == ours
    assert output_bytes.startswith(original_bytes)
    assert all(r.image_id in expected for r in result)
    assert len({r.image_id for r in result}) == 180
    public_by_id = {r.filament_id:r for r in public}
    assert all(r.counts == public_by_id[m['reference_id']].counts for r,m in zip(result[1347:],mapping))
    require_disjoint(result)
    grouped = defaultdict(list)
    for row in result: grouped[row.image_id].append(row)
    overlap_images, overlap_pixels = [], 0
    for stem, rows in grouped.items():
        assert len({r.counts for r in rows}) == len(rows), 'Exact duplicate mask: '+stem
        rles = [r.rle(2048,2048) for r in rows]
        excess = sum(r.area for r in rows)-int(mu.area(mu.merge(rles)))
        if excess:
            overlap_images.append(stem); overlap_pixels += excess
    assert sha(BASELINE) == BASELINE_SHA and sha(REFERENCE) == REFERENCE_SHA
    save_exact(OUTPUT,output_bytes)
    manifest = dict(experiment='own039_plus_public216_no_overlap',
        hypothesis='Net public-LB effect of adding216 pixel-disjoint natural public predictions to unchanged own039 baseline',
        baseline_public_lb_user_reported=.39,reference_public_lb_user_reported=.55,
        experiment_public_lb=None,submitted=False,training=False,neural_inference=False,ground_truth_used=False,
        selection='Public instances with max IoU <0.1 AND zero pixel intersection against original baseline instances; exact two known placeholder squares excluded',
        selection_scope='All official180 test images, no use of hidden public/private subset membership',
        source_sha256=dict(baseline=BASELINE_SHA,reference=REFERENCE_SHA,official_names=sha(NAMES),builder=sha(Path(__file__))),
        input_baseline_rows=1347,reference_rows=1344,weak_overlap_candidates=225,
        excluded_overlap_candidates=9,added_rows=216,output_rows=len(result),
        baseline_images=178,output_images=180,images_with_additions=len({r['image'] for r in mapping}),
        excluded_placeholder_ids=excluded,output=str(OUTPUT),output_sha256=sha(OUTPUT),
        original_csv_bytes_preserved_as_prefix=True,all_original_ids_and_masks_preserved=True,
        all_added_masks_pixel_exact_reference=True,new_ids_unique=True,exact_duplicate_masks=0,
        added_masks_with_any_overlap=sum(r['overlap_pixels_with_original']>0 for r in mapping),
        overlap_pixels_sum=overlap_pixels,images_with_overlap=overlap_images,
        overlap_policy='Actual Kaggle rejects overlaps (user reported). Exclude every intersecting addition, retain baseline and remaining additions exactly, then enforce zero overlap across the entire final CSV before publishing.',
        supersedes_rejected_sha256='1787d5edb1acf00e5e13bcd62e5bb46b3564896e104f4e6f1e9a28152b67a4e6',
        interpretation='Leaderboard difference measures the net effect of216 additions on its public subset; it does not identify individual TP/FP/FN or establish all216 are correct. This is a CSV hybrid, not a newly improved reproducible model.')
    save_exact(REPORT/'manifest.json',(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode())
    save_exact(REPORT/'added_instances.csv',csv_bytes(mapping,list(mapping[0])))
    save_exact(REPORT/'excluded_overlap_instances.csv',csv_bytes(overlapping_candidates,list(overlapping_candidates[0])))
    save_exact(REPORT/'build_public_addition_submission.py',Path(__file__).read_bytes())
    text = f'''# 기존 LB0.39 + 공개 예측216개 추가 실험: 겹침 오류 수정

- 제출 파일: `{OUTPUT.name}`
- 실제 Kaggle 제출기가 이전225개 추가 파일의 겹침을 거부했다. self-evaluation 계산 코드와 제출 유효성 검사는 서로 다르다. 이전 파일은 다시 제출하지 않는다.
- 기존1,347개 마스크의 ID·RLE·순서와 CSV 원본 바이트를 그대로 보존하고 공개216개 마스크를 새로운 ID로 추가했다. 총1,563개,180장이다.
- 추가 조건: 원래0.39 마스크들과 최대 mask IoU <0.1이며 교집합 픽셀이0. 공개 임시 사각형2개와 기존 마스크와 겹치는 추가후보9개를 제외했다.
- 216개 추가 마스크의 픽셀을 바꾸지 않았다. 전체 출력의 겹침은0픽셀이다.9개는 잘라내면 일부 마스크가 거의 소실되므로 원본 유지 실험에서는 제외했다. 기존 마스크를 깎거나 조각난 신규 마스크를 넣지 않았다.
- 기존0.39 파일과 동일한 대회에 이 파일을 제출하고 표시 가능한 최대 자릿수의 점수를 기록한다. 자동 제출은 실행하지 않았다.
- 상승은 추가 묶음의 public 평가 순효과가 양수라는 뜻이다.216개 전부가 정답이라는 뜻은 아니다. 점수가 같아도 반올림·public 부분집합·상쇄 때문에 추가 후보 전부의 정답 여부는 확정되지 않는다.
- 이 실험은 CSV 혼합 실험이다. 현재 실행 중인 자체 best84+R1 모델 실험과 별도로 기록한다. 실제 test PQ는 로컬에서 계산하지 않았다.

## 검증

- 공식180개 이미지명에 속하는1,563행, 중복ID·동일마스크 중복·겹침 없음.
- 기존1,347행 원본과 같음; 신규216행은 선택된 공개 마스크와 픽셀 단위로 동일.
- `added_instances.csv`: 공개 ID → 새 ID, 면적, 원래 결과와의 IoU·겹침 기록.
- `manifest.json`: 입력·출력SHA, 고정 선택 규칙, 점수 미측정 상태.
- `excluded_overlap_instances.csv`: 제외된9개와 겹침 면적.
'''
    save_exact(REPORT/'README.ko.md',text.encode())
    package = OUTPUT.parent/'SolarFilament_Own039_PlusPublic216_NoOverlap_Analysis.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.write(OUTPUT,OUTPUT.name)
        for path in sorted(REPORT.iterdir()):
            if path.is_file():archive.write(path,path.name)
    print(json.dumps({key:manifest[key] for key in ('output','output_rows','added_rows','images_with_additions',
        'added_masks_with_any_overlap','overlap_pixels_sum','output_sha256')},indent=2))


if __name__ == '__main__': main()
