"""Read-only inventory of the existing synthetic COCO data and source split.

This reports what can be proved from metadata. A source-pool split is not
per-generated-image donor/background provenance and does not certify labels.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_split(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    ids = [str(row['image_id']) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate split image_id')
    return set(ids), {row['file_name'] for row in rows}


def inspect(root, original_annotations, train_split, val_split):
    root = Path(root)
    paths = dict(images=root/'images', synthetic_annotations=root/'synthetic_annotations.json',
                 split_info=root/'split_info.json', original_annotations=Path(original_annotations),
                 train_split=Path(train_split), val_split=Path(val_split))
    result = dict(scope='Metadata audit only; does not certify background labels or actual donor use',
                  root=str(root), paths={k: str(v) for k, v in paths.items()},
                  present={k: v.exists() for k, v in paths.items()})
    if not all(result['present'].values()):
        result['status'] = 'missing_input'
        return result
    original = json.loads(paths['original_annotations'].read_text(encoding='utf-8-sig'))
    source = json.loads(paths['split_info'].read_text(encoding='utf-8-sig'))
    synthetic = json.loads(paths['synthetic_annotations'].read_text(encoding='utf-8-sig'))
    result['sha256'] = {k: digest(v) for k, v in paths.items() if v.is_file()}
    by_id = {str(row['id']): row for row in original['images']}
    train_ids, train_files = read_split(paths['train_split'])
    val_ids, val_files = read_split(paths['val_split'])
    if len(by_id) != len(original['images']) or not train_ids.isdisjoint(val_ids) or train_files & val_files:
        raise ValueError('Original image IDs or physical train/val split are invalid')
    if not ((train_ids | val_ids) <= set(by_id)):
        raise ValueError('Current split contains unknown original image IDs')
    result['current_split'] = dict(train_groups=len(train_ids), val_groups=len(val_ids),
                                  train_physical_images=len(train_files), val_physical_images=len(val_files))
    required = {'train_image_ids', 'val_image_ids'}
    result['split_info_keys'] = sorted(source)
    if required <= set(source):
        eligible = {str(v) for v in source['train_image_ids']}
        heldout = {str(v) for v in source['val_image_ids']}
        pool_valid = not eligible & heldout and eligible | heldout == set(by_id)
        source_files = {by_id[i]['file_name'] for i in eligible if i in by_id}
        promoted_train = {i for i, row in by_id.items() if row['file_name'] in source_files}
        result['source_pool'] = dict(
            source_train_ids=len(eligible), source_val_ids=len(heldout),
            ids_partition_original=pool_valid,
            eligible_source_files=len(source_files),
            eligible_files_overlapping_current_val=sorted(source_files & val_files),
            physical_filename_promoted_train_ids_match_current=(promoted_train == train_ids),
            source_pool_consistent_with_current_train=(pool_valid and source_files <= train_files
                                                       and not (source_files & val_files)),
            interpretation='Eligible original source pool only; actual donor/background provenance still needs generated-image metadata')
    else:
        result['source_pool'] = dict(source_pool_consistent_with_current_train=False,
                                    reason='Required train_image_ids/val_image_ids missing')
    images = synthetic.get('images', [])
    annotations = synthetic.get('annotations', [])
    ids = [str(row['id']) for row in images]
    filenames = [Path(row['file_name']).name for row in images]
    known = set(ids)
    counts = Counter(str(row['image_id']) for row in annotations)
    if len(ids) != len(known) or len(filenames) != len(set(filenames)) or not set(counts) <= known:
        raise ValueError('Synthetic IDs/physical filenames are duplicate or annotations reference unknown IDs')
    missing = [name for name in filenames if not (paths['images']/name).is_file()]
    polygon = sum(isinstance(row.get('segmentation'), list) for row in annotations)
    def numeric_spine(row):
        values = row.get('spine')
        if not isinstance(values, list) or len(values) < 4 or len(values) % 2:
            return False
        try:
            return all(math.isfinite(float(value)) for value in values)
        except (TypeError, ValueError):
            return False
    spines = sum(numeric_spine(row) for row in annotations)
    image_fields = sorted({key for row in images for key in row})
    annotation_fields = sorted({key for row in annotations for key in row})
    provenance_fields = [key for key in image_fields + annotation_fields
                         if any(word in key.lower() for word in ('source', 'donor', 'background', 'parent'))]
    result['synthetic'] = dict(images=len(images), annotations=len(annotations),
        empty_label_images=sum(counts[i] == 0 for i in ids), missing_images=missing,
        polygon_list_annotations=polygon, numeric_even_spine_lists=spines,
        image_fields=image_fields, annotation_fields=annotation_fields,
        potential_provenance_fields=sorted(set(provenance_fields)),
        original_image_shapes_from_metadata=sorted({str((r.get('height'), r.get('width'))) for r in images}),
        image_record_examples=images[:2], annotation_record_keys_examples=[sorted(r) for r in annotations[:2]])
    result['status'] = 'metadata_complete' if not missing else 'missing_synthetic_images'
    result['independent_provenance_verified'] = False
    result['provenance_interpretation'] = 'User reports existing train/validation separation complete; this metadata-only tool does not independently certify per-image provenance and does not require additional approval.'
    result['quality_notes'] = [
        'Source-pool membership does not prove every generated image used only those donors.',
        'A clean_solar.jpg filename does not prove an empty or validation-independent background.',
        'Missing synthetic spine targets must mask auxiliary spine loss, not become negative spine labels.',
        'Validation must retain the original physical-image split and use real images only.']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/content/drive/MyDrive/filament-segmentation-2026/synthetic_copy_paste_v2')
    parser.add_argument('--original-annotations', default='/content/drive/MyDrive/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    parser.add_argument('--train-split', default='/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/train_split.csv')
    parser.add_argument('--val-split', default='/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/val_split.csv')
    parser.add_argument('--output', default='/content/synthetic_data_audit.json')
    args = parser.parse_args()
    report = inspect(args.root, args.original_annotations, args.train_split, args.val_split)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
