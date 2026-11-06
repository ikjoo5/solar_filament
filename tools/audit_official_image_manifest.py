"""Screen a complete official filename manifest against public MAGFiLO metadata.

Uses image identities only. A remaining identity is a candidate, not yet an
independent evaluation image: official byte/pixel exclusion still follows.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import date
import hashlib
import json
from pathlib import Path

from colab_inventory_external_magfilo import load_verified_annotations, physical_key


def manifest_inventory(manifest):
    if not isinstance(manifest, dict) or set(manifest) != {'train_images', 'test_images'}:
        raise ValueError('Manifest must contain train_images and test_images lists')
    result = {}
    for role, names in manifest.items():
        if not isinstance(names, list) or not names:
            raise ValueError(f'Complete nonempty {role} list required')
        if any(not isinstance(name, str) or Path(name).name != name or '\\' in name
               for name in names):
            raise ValueError('Expected image basenames, not paths')
        keys = [physical_key(name) for name in names]
        if len(set(names)) != len(names) or len(set(keys)) != len(keys):
            raise ValueError(f'Duplicate filename or physical identifier in {role}')
        result[role] = set(keys)
    if result['train_images'] & result['test_images']:
        raise ValueError('Official train/test physical identifiers overlap')
    return result


def observation_day(key):
    return date(int(key[:4]), int(key[4:6]), int(key[6:8]))


def audit_manifest(manifest, public, official, previous=None):
    inventories = manifest_inventory(manifest)
    train, test = inventories['train_images'], inventories['test_images']
    expected_train = {physical_key(row['file_name']) for row in official['images']}
    if train != expected_train:
        raise ValueError('Manifest train identities differ from official train annotations')
    by_key = defaultdict(list)
    for row in public['images']:
        by_key[physical_key(row['file_name'])].append(row)
    public_keys = set(by_key)
    candidates = sorted(public_keys - train - test)
    train_days = {observation_day(key) for key in train}
    official_days = train_days | {observation_day(key) for key in test}
    records = []
    for key in candidates:
        members = by_key[key]
        day = observation_day(key)
        records.append(dict(physical_identifier=key,
            file_names=sorted({row['file_name'] for row in members}),
            source_image_ids=sorted(str(row['id']) for row in members),
            source_urls=sorted({row.get('url', '') for row in members}),
            capture_day=day.isoformat(), observatory=key[-2:],
            nearest_train_day_distance=min(abs((day-d).days) for d in train_days),
            nearest_official_day_distance=min(abs((day-d).days) for d in official_days)))
    result = dict(schema_version=1,
        status='requires_file_and_decoded_pixel_duplicate_audit' if candidates
               else 'no_additional_external_observations',
        data_role='external_evaluation_only', allow_training=False,
        label_masks_used_for_selection_or_scoring=False, scores_computed=False,
        official_train_physical=len(train), official_test_physical=len(test),
        official_duplicate_identifiers=0, official_train_test_overlap=0,
        official_train_matches_annotation_ids=True,
        public_physical=len(public_keys), public_annotation_groups=len(public['images']),
        public_overlap_official_train=len(public_keys & train),
        public_overlap_official_test=len(public_keys & test),
        official_train_missing_from_public=sorted(train-public_keys),
        official_test_missing_from_public=sorted(test-public_keys),
        candidate_external_physical=len(candidates),
        candidate_external_annotation_groups=sum(len(by_key[key]) for key in candidates),
        actual_eligible_external_N=None if candidates else 0,
        candidate_observatories=dict(sorted(Counter(row['observatory'] for row in records).items())),
        candidates_more_than_3days_from_train=sum(row['nearest_train_day_distance'] > 3 for row in records),
        candidates_more_than_3days_from_all_official=sum(row['nearest_official_day_distance'] > 3 for row in records),
        candidates_within_3days_of_train=sum(row['nearest_train_day_distance'] <= 3 for row in records),
        candidate_records=records,
        completeness_note='All entries of the supplied official-directory manifest are excluded; completeness is based on the user-provided directory inventory.',
        interpretation='Different observation IDs establish candidate eligibility only. Byte/pixel duplicate exclusion is pending, and nearby dates can remain temporally correlated.')
    if previous is not None:
        old = set(previous['potentialkeys'])
        result['previous_unresolved_physical'] = len(old)
        result['previous_unresolved_now_official_test'] = sorted(old & test)
        result['previous_unresolved_still_candidates'] = len(old & set(candidates))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--public-annotations', type=Path, required=True)
    parser.add_argument('--official-annotations', type=Path, required=True)
    parser.add_argument('--previous-audit', type=Path)
    parser.add_argument('--model-train-split', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    public, provenance = load_verified_annotations(args.public_annotations)
    result = audit_manifest(json.loads(args.manifest.read_text()), public,
        json.loads(args.official_annotations.read_text()),
        json.loads(args.previous_audit.read_text()) if args.previous_audit else None)
    result['inputs'] = {label: dict(path=str(path.resolve()),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for label, path in [('official_manifest', args.manifest),
                            ('official_annotations', args.official_annotations)]}
    result['public_provenance'] = provenance
    if args.model_train_split:
        with args.model_train_split.open() as stream:
            training = {physical_key(row['file_name']) for row in csv.DictReader(stream)}
        official_train = manifest_inventory(json.loads(args.manifest.read_text()))['train_images']
        if not training or training - official_train:
            raise ValueError('Model-training split must be a nonempty subset of official train')
        days = {observation_day(key) for key in training}
        for record in result['candidate_records']:
            record['nearest_model_training_day_distance'] = min(
                abs((observation_day(record['physical_identifier'])-day).days) for day in days)
        result['actual_model_training_physical'] = len(training)
        result['candidates_within_3days_of_model_training'] = sum(
            r['nearest_model_training_day_distance'] <= 3 for r in result['candidate_records'])
        result['candidates_more_than_3days_from_model_training'] = sum(
            r['nearest_model_training_day_distance'] > 3 for r in result['candidate_records'])
        result['inputs']['model_train_split'] = dict(path=str(args.model_train_split.resolve()),
            sha256=hashlib.sha256(args.model_train_split.read_bytes()).hexdigest())
        result['training_distance_note'] = 'Unqualified train fields refer to the official707-image train/validation pool; model_training fields refer to the616-image optimizer training split.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items()
                     if key not in {'candidate_records', 'inputs', 'public_provenance'}}, indent=2))


if __name__ == '__main__':
    main()
