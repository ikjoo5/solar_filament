"""Frozen PQ.472593 inference: resumable per-stage caches and one final CSV.

Only image filenames from the original split CSVs are read to prevent accidental
train/validation export. No ground-truth masks are read or used at inference.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import zlib

import cv2
import numpy as np
from pycocotools import mask as mu

from submission_yolo_recipe import build_yolo_reference
from submission_native_recipe import build_native_submission

PIPELINE = 'solar_filament_pq047259304445299505_v1'
CHECKPOINT_SHA256 = {
    'yolo': '81c2a27d95821b5ede283169c0da8c08d360ecc4c2a60a1b9c6baf96d76bcccc',
    'native': 'ae6b209f72355fd52d2cb07182e69028e97c703a1977ad1e7b38b27cf8cebe69',
}
SPLIT_SHA256 = {
    'train_split': 'aa0733424815731c5c108ada91e45525eb17ca3a42b377bf19e3089d86aede7e',
    'val_split': '54c6a9aac5a3b70b3ecc6721109a8b0fe8ec020e90cb56e3848fccc92eadc61f',
}
CODE_FILES = ('submission_driver.py', 'submission_inference.py',
              'submission_yolo_recipe.py', 'submission_native_recipe.py',
              'submission_native_process.py')
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
CSV_COLUMNS = ['filament_id', 'segmentation_rle']


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def json_ready(value):
    if isinstance(value, bytes):
        return value.decode('ascii')
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def compact(value):
    return json.dumps(json_ready(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.pending')
    temporary.write_text(compact(value)+'\n', encoding='utf-8')
    temporary.replace(path)


def write_envelope(path, payload, identity, image_sha):
    """One atomic CRC-protected file; envelope works without a global manifest."""
    data = json_ready(payload)
    encoded = compact(data).encode()
    envelope = dict(schema_version=1, identity=identity, image_sha256=image_sha,
        payload_sha256=hashlib.sha256(encoded).hexdigest(), payload=data)
    path = Path(path)
    temporary = path.with_name(path.name+'.pending')
    try:
        with temporary.open('wb') as f:
            f.write(gzip.compress(compact(envelope).encode(), compresslevel=6, mtime=0))
        temporary.replace(path)
    finally:
        if temporary.is_file():
            temporary.unlink()


def read_envelope(path, identity, image_sha):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        envelope = json.loads(gzip.decompress(path.read_bytes()))
        if not isinstance(envelope,dict):
            return None
        if (envelope.get('schema_version'),envelope.get('identity'),envelope.get('image_sha256')) != (1,identity,image_sha):
            return None
        data = envelope['payload']
        if hashlib.sha256(compact(data).encode()).hexdigest() != envelope['payload_sha256']:
            return None
        return data
    except (OSError, EOFError, ValueError, KeyError, TypeError, zlib.error):
        return None


def checked_rle(value):
    """Validate compressed runs before calling the C decoder, including zeros."""
    if not isinstance(value,dict) or value.get('size') != [2048,2048]:
        raise ValueError('마스크는 원본 2048×2048 좌표여야 합니다.')
    counts = value['counts']
    if isinstance(counts, bytes):
        counts = counts.decode('ascii')
    if not isinstance(counts,str) or not counts:
        raise ValueError('빈 RLE 문자열')
    previous = []
    total = area = index = position = 0
    while position < len(counts):
        number = shift = 0
        while True:
            if position >= len(counts):
                raise ValueError('잘린 RLE')
            digit = ord(counts[position])-48
            position += 1
            if not 0 <= digit <= 63:
                raise ValueError('RLE 문자 오류')
            number |= (digit & 31) << shift
            shift += 5
            if shift > 65:
                raise ValueError('RLE 숫자 범위 오류')
            if not digit & 32:
                if digit & 16:
                    number |= -1 << shift
                break
        if index > 2:
            number += previous[-2]
        if number < 0:
            raise ValueError('음수 RLE 길이')
        total += number
        if total > 2048**2:
            raise ValueError('RLE 좌표 범위 초과')
        if index % 2:
            area += number
        previous.append(number)
        if len(previous)>2:
            previous.pop(0)
        index += 1
    if total != 2048**2 or area <= 0:
        raise ValueError('잘못된 크기 또는 빈 인스턴스 RLE')
    result = dict(size=[2048,2048], counts=counts.encode('ascii'))
    if int(mu.area(result)) != area:
        raise ValueError('RLE 면적 불일치')
    return result


def validate_predictions(rles):
    values = [checked_rle(value) for value in rles]
    if values and int(mu.area(mu.merge(values))) != sum(int(mu.area(value)) for value in values):
        raise ValueError('최종 인스턴스 마스크가 겹칩니다.')
    return values


def list_test_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'테스트 이미지 폴더를 찾지 못했습니다: {directory}')
    if any(part.lower() in ('train','train_images') for part in directory.parts):
        raise ValueError('제출은 test_images 폴더에서 생성해야 합니다.')
    paths = sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),key=lambda p:p.name)
    if not paths:
        raise ValueError(f'테스트 이미지가 없습니다: {directory}')
    stems = [p.stem for p in paths]
    if len(stems) != len(set(stems)) or any('_' in stem or not stem or any(c.isspace() for c in stem) for stem in stems):
        raise ValueError('테스트 파일 stem에 중복·공백·밑줄이 있습니다. 공식 ID 구분 규칙과 맞지 않습니다.')
    return paths


def check_splits(config, paths):
    excluded = set()
    for key, expected in SPLIT_SHA256.items():
        source = Path(config[key])
        if digest(source) != expected:
            raise ValueError(f'기존 분할 CSV가 달라졌습니다: {source}')
        with source.open(newline='',encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or 'file_name' not in reader.fieldnames:
                raise ValueError(f'split CSV 형식 오류: {source}')
            excluded.update(Path(row['file_name']).stem for row in reader)
    overlap = excluded & {p.stem for p in paths}
    if overlap:
        raise ValueError(f'train/validation 이미지가 테스트 폴더에 있습니다: {sorted(overlap)[:5]}')


def finalize_frame(stem, stages):
    old = build_yolo_reference(stem,stages['geometric'],stages['preprocessing'],stages['tiled'])
    return build_native_submission(stem,old,stages['geometric'],stages['preprocessing'],stages['tiled'],stages['native'])


class IsolatedInference:
    """YOLO and native keep the original independent CUDA/cuBLAS environments."""
    def __init__(self,yolo,native):
        self.yolo,self.native_process=yolo,native

    def __getattr__(self,name):
        return getattr(self.yolo,name)

    def native(self,*args,**kwargs):
        return self.native_process.native(*args,**kwargs)


def valid_stage(name,data,stem):
    if name=='native':
        return (isinstance(data,dict) and Path(data.get('file_name','')).stem==stem
            and all(isinstance(data.get(key),dict) for key in ('unions','components','tta','view_unions')))
    expected={'geometric':4,'preprocessing':8,'tiled':18}[name]
    return (isinstance(data,list) and len(data)==expected
        and all(isinstance(row,dict) and row.get('image')==stem and isinstance(row.get('view'),str)
            and isinstance(row.get('instances'),list) for row in data)
        and len({row['view'] for row in data})==expected)


def load_manifest(path,identity):
    path=Path(path)
    if not path.is_file():
        return None
    try:
        saved=json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):
        saved=None
    if not isinstance(saved,dict) or not isinstance(saved.get('identity'),str):
        print('진행 기록을 개별 이미지·단계 해시에서 복구합니다.',flush=True)
        return None
    if saved['identity'] != identity:
        raise ValueError('기존 작업과 코드·GPU/라이브러리·모델·테스트 이미지가 다릅니다. 같은 환경으로 재개하거나 OUTPUT 경로를 새 폴더로 바꾸세요.')
    return saved


def process_frame(path, image_sha, identity, output, provider, finalizer=finalize_frame, keep_stages=False):
    path, output = Path(path), Path(output)
    completed = output/'completed'/f'{path.stem}.json.gz'
    prior = read_envelope(completed,identity,image_sha)
    if isinstance(prior,dict):
        try:
            if (prior.get('file_name')!=path.name or not isinstance(prior.get('rles'),list)
                    or prior.get('instances')!=len(prior['rles'])
                    or not isinstance(prior.get('capped_yolo_views'),int) or prior['capped_yolo_views']<0):
                raise ValueError('완료 기록 형식 오류')
            validate_predictions(prior['rles'])
            return prior, True
        except (ValueError,KeyError,TypeError):
            pass
    source_bytes=path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != image_sha:
        raise ValueError(f'추론 중 원본 이미지가 변경됐습니다: {path}')
    pixels = cv2.imdecode(np.frombuffer(source_bytes,np.uint8),cv2.IMREAD_GRAYSCALE)
    if pixels is None or pixels.shape != (2048,2048) or pixels.dtype != np.uint8:
        raise ValueError(f'원본 uint8 2048×2048 테스트 이미지가 필요합니다: {path}')
    stages = {}
    stage_paths = []
    for name in ('geometric','preprocessing','tiled','native'):
        cache = output/'stages'/f'{path.stem}.{name}.json.gz'
        stage_paths.append(cache)
        data = read_envelope(cache,identity,image_sha)
        if not valid_stage(name,data,path.stem):
            data=None
        if data is None:
            print(f'  {path.name}: {name} 추론',flush=True)
            if name == 'native':
                data = provider.native(path.stem,pixels,file_name=path.name,image_sha256=image_sha,identity=identity)
            else:
                data = getattr(provider,name)(path.stem,pixels)
            if not valid_stage(name,data,path.stem):
                raise ValueError(f'추론 단계 출력 형식 오류: {path.name} {name}')
            write_envelope(cache,data,identity,image_sha)
        else:
            print(f'  {path.name}: {name} 저장 결과 재사용',flush=True)
        stages[name] = data
    predictions = validate_predictions(finalizer(path.stem,stages))
    capped = sum(int(record.get('max_det_reached',False)) for stage in ('geometric','preprocessing','tiled') for record in stages[stage])
    payload = dict(file_name=path.name,rles=json_ready(predictions),instances=len(predictions),capped_yolo_views=capped)
    write_envelope(completed,payload,identity,image_sha)
    if keep_stages is False:
        for cache in stage_paths:
            if cache.is_file():
                cache.unlink()  # Only this run's completed frame caches; checkpoints are untouched.
    return payload, False


def export_csv(path, records):
    path = Path(path)
    temporary = path.with_name(path.name+'.pending')
    count = 0
    try:
        with temporary.open('w',newline='',encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
            seen = set()
            for record in records:
                stem = Path(record['file_name']).stem
                for index,rle in enumerate(validate_predictions(record['rles']),1):
                    identifier = f'{stem}_{index}'
                    if identifier in seen:
                        raise ValueError('중복 제출 ID')
                    seen.add(identifier)
                    writer.writerow([identifier,rle['counts'].decode('ascii')])
                    count += 1
        temporary.replace(path)
    finally:
        if temporary.is_file():
            temporary.unlink()
    return count


def run_submission(config):
    import torch
    from submission_inference import InferenceProvider
    from submission_native_process import NativeProcessClient
    if not torch.cuda.is_available():
        raise ValueError('Colab 런타임에서 GPU를 선택하세요. 검증과 동일한 A100을 권장합니다.')
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    paths = list_test_images(config['test_images'])
    check_splits(config,paths)
    weights = {}
    for key in CHECKPOINT_SHA256:
        source = Path(config[key+'_checkpoint'])
        if digest(source) != CHECKPOINT_SHA256[key]:
            raise ValueError(f'검증에 사용한 {key} 체크포인트와 다릅니다: {source}')
        weights[key] = CHECKPOINT_SHA256[key]
    print(f'테스트 이미지 {len(paths)}장을 확인했습니다. 입력 해시를 계산합니다.',flush=True)
    inputs = [dict(file_name=p.name,sha256=digest(p)) for p in paths]
    code = {name:digest(Path(__file__).parent/name) for name in CODE_FILES}
    versions = {name:importlib.metadata.version(name) for name in ('torch','ultralytics','segmentation-models-pytorch','timm','numpy','scipy','pycocotools')}
    runtime = dict(versions=versions,opencv=cv2.__version__,gpu=torch.cuda.get_device_name(0),
        amp='bfloat16' if torch.cuda.is_bf16_supported() else 'float16',native_batch=4,
        native_separate_process=True,native_cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        yolo_cublas_workspace_config=':4096:8')
    provenance = dict(schema_version=1,pipeline=PIPELINE,code_sha256=code,weights=weights,inputs=inputs,runtime=runtime)
    identity = hashlib.sha256(compact(provenance).encode()).hexdigest()
    output = Path(config['output'])
    output.mkdir(parents=True,exist_ok=True)
    manifest_path = output/'manifest.json'
    load_manifest(manifest_path,identity)
    for directory in ('stages','completed'):
        (output/directory).mkdir(exist_ok=True)
    manifest = dict(identity=identity,provenance=provenance,status='running',completed={})
    atomic_json(manifest_path,manifest)
    native_process=NativeProcessClient(config['native_checkpoint'],device='cuda:0',batch=4)
    provider = IsolatedInference(InferenceProvider(config['yolo_checkpoint'],config['native_checkpoint'],
        device='cuda:0',native_batch=4,verify_weights=True),native_process)
    records = []
    started = time.monotonic()
    try:
        for index,(path,item) in enumerate(zip(paths,inputs),1):
            record,reused = process_frame(path,item['sha256'],identity,output,provider,
                keep_stages=bool(config.get('keep_stages',False)))
            records.append(record)
            manifest['completed'][path.name] = dict(record_sha256=digest(output/'completed'/f'{path.stem}.json.gz'),
                instances=record['instances'],image_sha256=item['sha256'])
            atomic_json(manifest_path,manifest)
            print(f'[{index}/{len(paths)}] {path.name}: {record["instances"]}개'+(' (재사용)' if reused else ' (저장 완료)'),flush=True)
    finally:
        native_process.close()
    # CSV is published only when every actual test image, including blank predictions, is complete.
    if len(records) != len(paths) or set(manifest['completed']) != {p.name for p in paths}:
        raise RuntimeError('완료 이미지 수가 다릅니다. 제출 CSV를 생성하지 않습니다.')
    destination = output/'submission_pq0472593.csv'
    rows = export_csv(destination,records)
    manifest['status'] = 'complete'
    manifest['submission_sha256'] = digest(destination)
    atomic_json(manifest_path,manifest)
    summary = dict(status='complete',pipeline=PIPELINE,csv=str(destination),sha256=digest(destination),
        physical_test_images=len(paths),prediction_rows=rows,images_without_predictions=[r['file_name'] for r in records if not r['rles']],
        capped_yolo_views=sum(r['capped_yolo_views'] for r in records),
        columns=CSV_COLUMNS,validation_pq_reference=0.47259304445299505,
        leaderboard_score=None,extra_training=False,elapsed_this_session_seconds=time.monotonic()-started,
        resume='Verified per-stage and per-image results on Drive; same frozen code/runtime/weights/input bytes')
    atomic_json(output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--result',required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    result = run_submission(config)
    atomic_json(Path(args.result),result)


if __name__ == '__main__':
    main()
