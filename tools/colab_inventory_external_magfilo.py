"""Colab CPU cell: obtain public annotation metadata, exclude all official IDs.

No labels are scored, no image is downloaded and no model is imported/trained.
The official test directory must include images with zero predicted filaments.
"""
import hashlib
import json
from pathlib import Path
import re
import urllib.request

BASE = Path('/content/drive/MyDrive/filament-segmentation-2026')
PUBLIC_URL = 'https://dataverse.harvard.edu/api/access/datafile/10407987'
PUBLIC_MD5 = 'bfa414567b4dfa2a6b4a9eccedb6aa20'
PUBLIC_BYTES = 60602626
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
IDENTIFIER = re.compile(r'(\d{14}[a-z]h)', re.I)


def physical_key(value):
    match = IDENTIFIER.search(str(value))
    if not match:
        raise ValueError(f'Unknown GONG observation identifier: {value}')
    return match.group(1).lower()


def load_verified_annotations(path, *, opener=urllib.request.urlopen,
                              expected_md5=PUBLIC_MD5, expected_bytes=PUBLIC_BYTES):
    """Preserve existing files; atomic checksum-verified single public JSON fetch."""
    path = Path(path)
    if path.is_file():
        raw = path.read_bytes()
        downloaded = False
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(PUBLIC_URL, headers={'User-Agent': 'SolarFilamentEvaluation/1.0'})
        with opener(request, timeout=40) as response:
            raw = response.read(expected_bytes + 1)
        downloaded = True
    if len(raw) != expected_bytes or hashlib.md5(raw).hexdigest() != expected_md5:
        raise ValueError(f'Public annotation checksum/size differs; existing file preserved: {path}')
    data = json.loads(raw)
    if not isinstance(data.get('images'), list) or not isinstance(data.get('annotations'), list):
        raise ValueError('Public MAGFiLO must contain COCO images and annotations')
    if downloaded:
        temporary = path.with_name(path.name+'.pending')
        temporary.write_bytes(raw)
        if path.exists():
            temporary.unlink()
            raise FileExistsError(f'Another process created {path}; no existing file overwritten')
        temporary.replace(path)
    return data, dict(downloaded=downloaded, bytes=len(raw), md5=expected_md5,
                      sha256=hashlib.sha256(raw).hexdigest(), source_url=PUBLIC_URL)


def directory_inventory(directory):
    directory = Path(directory)
    files = sorted(p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise ValueError(f'Complete official image directory is required: {directory}')
    return dict(directory=str(directory), image_files=len(files),
                physical_identifiers=sorted({physical_key(path.name) for path in files}))


def inventory_annotations(data, train_inventory, test_inventory):
    """Identifier-only screen; never call unmatched observations independent data."""
    public = {physical_key(row['file_name']) for row in data['images']}
    train = set(train_inventory['physical_identifiers'])
    test = set(test_inventory['physical_identifiers'])
    if train & test:
        raise ValueError('Official train/validation and test directories overlap in physical identifiers')
    potential = sorted(public-train-test)
    return dict(data_role='external_evaluation_only', allow_training=False,
        allow_pseudo_label_training=False, allow_test_time_parameter_updates=False,
        label_masks_used_for_selection=False, predictions_generated=False, scores_computed=False,
        public_physical_images=len(public), public_annotation_groups=len(data['images']),
        public_annotation_instances=len(data['annotations']),
        overlap_official_train_validation=len(public & train), overlap_official_test=len(public & test),
        potential_external_before_byte_pixel_audit=len(potential),
        actual_eligible_external_N=None if potential else 0,
        status=('requires_file_and_decoded_pixel_duplicate_audit' if potential else 'no_additional_external_observations'),
        potential_physical_identifiers=potential, official_train_validation=train_inventory, official_test=test_inventory,
        completeness_note='Uses every image in the supplied official directories, including zero-prediction test images; directory completeness must match the official competition download.',
        next_step=('Do not tune on these labels yet. Exclude file/pixel duplicates and conflicting versions before external scoring.'
                   if potential else 'This public release provides no additional independent observations after official-ID exclusion.'))


def run_inventory(base=BASE):
    base = Path(base)
    external = base/'external_MAGFiLO'
    manifest_dir = external/'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    # Keep this pinned public release separate from any user's existing JSON.
    annotation_path = manifest_dir/'magfilo_2024_v1.0.json'
    data, provenance = load_verified_annotations(annotation_path)
    official = base/'MAGFiLO_1.0_Kaggle_2026'
    train = directory_inventory(official/'train/train_images')
    test = directory_inventory(official/'test/test_images')
    result = inventory_annotations(data, train, test)
    result.update(public_annotations=str(annotation_path), public_provenance=provenance,
                  image_downloads=0, gpu_used=False)
    destination = manifest_dir/'external_observation_inventory.json'
    pending = destination.with_name(destination.name+'.pending')
    pending.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    pending.replace(destination)
    brief = {key: value for key, value in result.items()
             if key not in {'official_train_validation', 'official_test', 'potential_physical_identifiers'}}
    print(json.dumps(brief, ensure_ascii=False, indent=2), flush=True)
    print('Upload this small metadata-only manifest:', destination, flush=True)
    return destination


if __name__ == '__main__':
    from google.colab import drive, files
    drive.mount('/content/drive')
    files.download(str(run_inventory()))
