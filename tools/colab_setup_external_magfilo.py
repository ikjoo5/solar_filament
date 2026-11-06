# Colab 새 셀에 전체 복사하세요. 외부 MAGFiLO는 평가 전용입니다.
from pathlib import Path
import json
import urllib.parse
import urllib.request

BASE = Path('/content/drive/MyDrive/filament-segmentation-2026')
DOWNLOAD_ANNOTATIONS = True
ANNOTATION_FILE_ID = None  # 공개 파일이 여러 개라면 명시하거나 annotations.json을 직접 보관합니다.
DATASET_DOI = 'doi:10.7910/DVN/J6JNVK'
API = 'https://dataverse.harvard.edu/api'
MAX_JSON_BYTES = 64 * 1024**2
ROLE = dict(data_role='external_evaluation_only', allow_training=False,
    allow_pseudo_label_training=False, allow_test_time_parameter_updates=False,
    public_dataset='https://doi.org/10.7910/DVN/J6JNVK',
    eligibility='Exclude all official train/validation/test physical observations before evaluation')


def new_file(path, content):
    """Never overwrite an existing file; clean up only our failed new write."""
    path = Path(path)
    try:
        handle = path.open('xb')
    except FileExistsError:
        return False
    try:
        with handle:
            handle.write(content)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return True


def validate_annotations(raw):
    value = json.loads(raw)
    if not isinstance(value, dict) or not isinstance(value.get('images'), list) or not value['images']:
        raise ValueError('외부 주석 JSON의 COCO images 목록을 확인하세요.')
    if not isinstance(value.get('annotations'), list):
        raise ValueError('외부 주석 JSON의 COCO annotations 목록을 확인하세요.')
    ids = [str(row['id']) for row in value['images']]
    known = set(ids)
    if len(ids) != len(known) or any(str(row['image_id']) not in known for row in value['annotations']):
        raise ValueError('외부 주석 JSON의 이미지 ID 또는 참조가 잘못됐습니다.')
    return dict(annotation_image_records=len(ids), annotated_instances=len(value['annotations']))


def fetch(url, opener=urllib.request.urlopen):
    request = urllib.request.Request(url, headers={'User-Agent': 'SolarFilamentEvaluation/1.0'})
    with opener(request, timeout=20) as response:
        raw = response.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError('주석 JSON 크기가 64MiB를 넘습니다. 파일을 직접 확인하세요.')
    return raw


def setup_external(base=BASE, download=DOWNLOAD_ANNOTATIONS, file_id=ANNOTATION_FILE_ID,
                   opener=urllib.request.urlopen):
    base = Path(base)
    root = base/'external_MAGFiLO'
    for directory in (root, root/'images', root/'manifests', root/'evaluations'):
        directory.mkdir(parents=True, exist_ok=True)
    role_path = root/'DATA_ROLE.json'
    if role_path.is_file():
        existing = json.loads(role_path.read_text(encoding='utf-8'))
        if not isinstance(existing, dict) or any(existing.get(k) != ROLE[k] for k in ('data_role', 'allow_training',
                'allow_pseudo_label_training', 'allow_test_time_parameter_updates')):
            raise ValueError('기존 DATA_ROLE.json이 평가 전용 설정과 다릅니다. 원본은 변경하지 않았습니다.')
    new_file(role_path, (json.dumps(ROLE, ensure_ascii=False, indent=2)+'\n').encode())
    annotation_path = root/'annotations.json'
    result = dict(root=str(root), annotations=str(annotation_path), images=str(root/'images'),
        download_status='not_requested', training_performed=False, images_downloaded=0)
    if annotation_path.is_file():
        result.update(validate_annotations(annotation_path.read_bytes()), download_status='existing_annotations_preserved')
    elif download:
        try:
            if file_id is None:
                url = API+'/datasets/:persistentId/?'+urllib.parse.urlencode({'persistentId': DATASET_DOI})
                metadata = json.loads(fetch(url, opener))
                files = [row['dataFile'] for row in metadata['data']['latestVersion']['files']]
                candidates = [row for row in files if row['filename'].lower().endswith('.json')]
                named = [row for row in candidates if 'annotation' in row['filename'].lower()]
                candidates = named if named else candidates
                if len(candidates) != 1:
                    result.update(download_status='annotation_file_selection_needed',
                        candidates=[dict(id=row['id'], filename=row['filename']) for row in candidates])
                    file_id = None
                else:
                    file_id = candidates[0]['id']
            if file_id is not None:
                url = API+'/access/datafile/'+str(int(file_id))
                raw = fetch(url, opener)
                counts = validate_annotations(raw)
                created = new_file(annotation_path, raw)
                if not created:
                    counts = validate_annotations(annotation_path.read_bytes())
                result.update(counts, source_url=url,
                    download_status='annotations_downloaded' if created else 'existing_annotations_preserved')
        except Exception as error:
            result.update(download_status='download_unavailable', error=f'{type(error).__name__}: {error}',
                next_step=f'공개 주석 JSON을 {annotation_path}에 보관하세요. 폴더와 기존 파일은 유지됩니다.')
    config = dict(external_annotations=str(annotation_path), external_images=str(root/'images'),
        official_train_images=str(base/'MAGFiLO_1.0_Kaggle_2026/train/train_images'),
        official_test_images=str(base/'MAGFiLO_1.0_Kaggle_2026/test/test_images'),
        yolo_checkpoint='/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/best.pt',
        native_checkpoint='/content/drive/MyDrive/SolarFilament_NativeUNet_A100/20261028T030808Z_bedf3e21/best.pt',
        output=str(root/'evaluations/frozen_lb039_v1'), split='external_dev')
    config_path = root/'manifests/evaluation_config.json'
    new_file(config_path, (json.dumps(config, ensure_ascii=False, indent=2)+'\n').encode())
    result['evaluation_config'] = str(config_path)
    return result


if __name__ == '__main__':
    from google.colab import drive
    drive.mount('/content/drive')
    print(json.dumps(setup_external(), ensure_ascii=False, indent=2))
