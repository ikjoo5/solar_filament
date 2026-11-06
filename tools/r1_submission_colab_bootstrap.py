"""Restore the full best84 pipeline plus frozen R1 from Drive in a fresh process."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require_separate(output, protected):
    output, protected = Path(output).resolve(), Path(protected).resolve()
    if output == protected or output in protected.parents or protected in output.parents:
        raise ValueError('OUTPUT은 원본 이미지·가중치 폴더와 분리된 새 폴더여야 합니다: '+str(protected))


def restore(root, work_root=Path('/content/solar_best84_pipeline_r1_code')):
    root = Path(root)
    pointer_path = root/'current_code.json'
    if not pointer_path.is_file():
        raise FileNotFoundError('최초 실행 셀을 먼저 실행하세요: '+str(pointer_path))
    pointer = json.loads(pointer_path.read_text())
    if pointer.get('scope') != 'best84_pipeline_r1_test_v1':
        raise ValueError('이 폴더는 best84 전체 파이프라인 + R1 제출 실험 폴더가 아닙니다. OUTPUT을 확인하세요.')
    version = pointer['version']
    if not re.fullmatch('[0-9a-f]{16}', version):
        raise ValueError('Invalid saved code version')
    durable = root/'code'/version
    manifests = [('source_manifest.json',durable,pointer['source_manifest_sha256']),
                 ('assets_manifest.json',root/'assets',pointer['assets_manifest_sha256'])]
    parsed = {}
    for name, folder, expected in manifests:
        path = folder/name
        if digest(path) != expected:
            raise ValueError('저장된 코드/입력 목록이 변경되었습니다: '+str(path))
        entries = json.loads(path.read_text())
        if not entries or any(Path(n).name != n for n in entries):
            raise ValueError('Invalid manifest: '+name)
        for filename, checksum in entries.items():
            item = folder/filename
            if not item.is_file() or digest(item) != checksum:
                raise ValueError('저장된 코드/입력이 변경되었습니다. 원본을 복원하거나 새 OUTPUT을 사용하세요: '+str(item))
        parsed[name] = entries
    if set(parsed['assets_manifest.json']) != {'official_test_names.json'}:
        raise ValueError('이 파이프라인의 입력 assets에는 공식 test 파일명 목록만 있어야 합니다.')
    config_path = root/'experiment_config.json'
    if digest(config_path) != pointer['config_sha256']:
        raise ValueError('저장된 설정이 변경되었습니다. 원래 설정을 복원하거나 새 OUTPUT을 사용하세요.')
    config = json.loads(config_path.read_text())
    if 'baseline_csv' in config:
        raise ValueError('기존 CSV 입력 설정은 지원하지 않습니다. 원본 이미지부터 실행하는 새 OUTPUT을 사용하세요.')
    if Path(config['output_root']).resolve() != root.resolve():
        raise ValueError('다른 OUTPUT 폴더의 설정입니다. 원래 OUTPUT을 사용하세요.')
    for key, asset in [('expected_names_path','official_test_names.json')]:
        if Path(config[key]).resolve() != (root/'assets'/asset).resolve():
            raise ValueError('다른 실험의 입력 경로가 섞였습니다: '+key)
    for protected in [config['images'],Path(config['yolo_checkpoint']).parent,
                      Path(config['native_checkpoint']).parent]:
        require_separate(root,protected)
    work = Path(work_root)/version
    work.mkdir(parents=True,exist_ok=True)
    for name in parsed['source_manifest.json']:
        content = (durable/name).read_bytes()
        compile(content,name,'exec')
        pending = work/(name+'.pending')
        pending.write_bytes(content)
        pending.replace(work/name)
    return config,config_path,work


def verify_inputs(config):
    if config.get('expected_images') != 180:
        raise ValueError('이 제출 실험은 공식 test 이미지180장 전체로 고정합니다.')
    if config.get('device') != 'cuda' or config.get('precision') != 'fp16' or config.get('native_batch') != 1:
        raise ValueError('이번 T4 제출 추론은 CUDA·FP16·native batch1로 고정합니다.')
    for key in ('yolo_checkpoint','native_checkpoint','expected_names_path'):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(key+' 파일을 찾지 못했습니다: '+str(config[key]))
    images = Path(config['images'])
    if not images.is_dir():
        raise FileNotFoundError('test_images 폴더를 찾지 못했습니다: '+str(images))
    names = json.loads(Path(config['expected_names_path']).read_text())
    if not isinstance(names,list) or len(names) != 180 or len(set(names)) != 180:
        raise ValueError('공식 test 파일명 목록이 올바르지 않습니다.')
    if any(Path(n).name != n for n in names):
        raise ValueError('공식 test 파일명은 파일 이름만 포함해야 합니다.')
    missing = [n for n in names if not (images/n).is_file()]
    if missing:
        raise FileNotFoundError('공식 test 이미지 '+str(len(missing))+'장이 없습니다. 예시: '+', '.join(missing[:5]))


def ensure_dependencies(work):
    preserved = {}
    for package in ('torch','torchvision'):
        try:
            preserved[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError('Colab 기본 GPU 런타임의 '+package+'가 필요합니다.') from exc
    required = []
    for package, expected in [('ultralytics','8.4.182'),('segmentation-models-pytorch','0.5.0'),
                              ('timm','1.0.20'),('pycocotools','2.0.11')]:
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != expected:
            required.append(package+'=='+expected)
    for module, package in [('numpy','numpy'),('cv2','opencv-python'),('scipy','scipy')]:
        if importlib.util.find_spec(module) is None:
            required.append(package)
    if required:
        constraints = Path(work)/'keep_colab_torch.txt'
        constraints.write_text(''.join(k+'=='+v+'\n' for k,v in preserved.items()))
        print('Colab torch/torchvision 유지; 필요한 패키지만 설치:',', '.join(required),flush=True)
        subprocess.run([sys.executable,'-m','pip','install','--upgrade-strategy','only-if-needed',
                        '-c',str(constraints),*required],check=True)
    if any(importlib.metadata.version(k) != v for k,v in preserved.items()):
        raise RuntimeError('설치 중 Colab torch/torchvision 버전이 변경되었습니다.')


def interrupted(signum,frame):
    raise KeyboardInterrupt('Interrupted by signal '+str(signum))


def stop_child(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run(root,max_images=None):
    if max_images is not None and (type(max_images) is not int or not 1 <= max_images <= 180):
        raise ValueError('MAX_IMAGES는 None 또는1~180 정수입니다. 에폭 수가 아닙니다.')
    config,config_path,work = restore(root)
    verify_inputs(config)
    ensure_dependencies(work)
    env = dict(os.environ,PYTHONUNBUFFERED='1',TORCH_CUDNN_V8_API_DISABLED='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    probe = """import torch,torchvision
assert torch.cuda.is_available(), 'Colab 런타임을 T4 GPU로 설정하세요.'
print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda,'GPU:',torch.cuda.get_device_name(0),flush=True)
layer=torch.nn.Conv2d(3,2,3,padding=1).cuda().eval()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
    value=layer(torch.randn(1,3,64,64,device='cuda'))
assert value.dtype==torch.float16 and bool(torch.isfinite(value).all()), 'CUDA FP16 검사 실패'
torch.cuda.synchronize()
print('CUDA FP16 검사 통과',flush=True)
"""
    names = json.loads(Path(config['expected_names_path']).read_text())
    all_final_files_present = all((Path(root)/'cache'/Path(name).stem/'final.json.gz').is_file() for name in names)
    if all_final_files_present:
        # Presence is only a hint to avoid the probe. The runner still verifies
        # every image, source, model, package, cache checksum and stage binding
        # before export; untrusted/missing data cannot bypass those checks.
        print('전체 final 캐시가 있습니다. GPU 사전 검사를 생략하고 원본·환경·캐시를 다시 검증합니다.',flush=True)
    else:
        subprocess.run([sys.executable,'-u','-c',probe],env=env,check=True)
    print('원본 test 이미지 → 기존 YOLO+best84 전체 파이프라인 → R1 순서로 새 추론합니다.',flush=True)
    print('기존 제출 CSV를 입력하지 않습니다. 같은 실행에서 R1 적용 전/후 CSV 두 개를 생성합니다.',flush=True)
    print('학습0epoch; test180장 T4 FP16 추론 후 동결된 R1을 한 번 적용합니다.',flush=True)
    print('이번 실행:', '전체180장' if max_images is None else str(max_images)+'장까지; 부분 결과는 제출용으로 내려받지 않습니다.',flush=True)
    print('Drive 재개 폴더:',root,flush=True)
    command = [sys.executable,'-u',str(work/'r1_test_submission.py'),'--config',str(config_path)]
    if max_images is not None:
        command += ['--max-images',str(max_images)]
    # Inherit the outer launcher's process group so interrupting the cell also
    # terminates pip, probes, the runner, and any worker descendants.
    process = subprocess.Popen(command,env=env,cwd=work)
    try:
        code = process.wait()
        if code:
            raise RuntimeError('추론 종료 코드 '+str(code)+'. 위 실제 traceback과 Drive colab_bootstrap.log를 확인하세요.')
    finally:
        stop_child(process)
    result_path = Path(root)/'latest_results.json'
    if not result_path.is_file():
        raise FileNotFoundError('추론 상태 파일이 생성되지 않았습니다: '+str(result_path))
    result = json.loads(result_path.read_text())
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


if __name__ == '__main__':
    signal.signal(signal.SIGTERM,interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--max-images',type=int,default=None)
    args = parser.parse_args()
    run(Path(args.root),args.max_images)
