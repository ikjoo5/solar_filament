"""Package frozen validation proposals for CPU score replay; standard library only."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import zipfile

FILES = ('summary.json', 'history.json', 'training_config.json', 'corpus_identity.json',
         'corpus.json', 'baseline.json', 'best_validation_predictions.csv', 'best_validation_scores.json')
INPUTS = dict(annotations='5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1',
    train_split='aa0733424815731c5c108ada91e45525eb17ca3a42b377bf19e3089d86aede7e',
    val_split='54c6a9aac5a3b70b3ecc6721109a8b0fe8ec020e90cb56e3848fccc92eadc61f',
    yolo='81c2a27d95821b5ede283169c0da8c08d360ecc4c2a60a1b9c6baf96d76bcccc')
CODE = {'fixed_split_mask_quality.py': '0cdf246a8ff545398038db64f602e50a643d10382be8e9b68b69f846ee8d5cdd',
    'submission_inference.py': '7c6a9d829025801482350b2f009a51a7ffb1eaa3d270cdce80a6a0691f815227',
    'evaluate_grouped_oof.py': '90f8ffc63d37ba326c990bd0723c865efb60cbf76fb2291ad918cfe6c757a7ad',
    'grouped_cv_training.py': '55a14f4dd7eaac8307463c40b8ad338bc868e98a7c6be8740ce62ebc443991c3'}
PROTOCOL = dict(version=1, controls=['confidence', 'quality'],
    mixes={'mix10': '0.9*c+0.1*q', 'mix25': '0.75*c+0.25*q', 'veto25': 'c*(0.75+0.25*q)'},
    thresholds=[.1, .15, .2, .25, .3], mask_nms_iou=.5, minimum_area=300,
    ownership='score-order nonoverlapping masks', checkpoint='frozen uploaded best epoch',
    primary_ranking_only=dict(methods=['mix10', 'mix25', 'veto25'], admission='original confidence >= 0.3',
        score_use='NMS and ownership order only; no additional quality/mix threshold',
        comparison='confidence0.3 with identical admitted proposal pool'),
    secondary_whole_rule='Three mixes times five thresholds; includes calibration changes, not causal proof of quality ranking',
    planned_prediction_CSVs=28, checkpoint_tuning=False, GT_fit=False, retraining=False, new_inference=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return sha(json.dumps(value, sort_keys=True).encode())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probability(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def collect(root, *, expected_counts=(616, 91), expected_proposals=7172):
    """Check provenance and proposal ordering without reading images or GT files."""
    root = Path(root)
    content = {name: (root/name).read_bytes() for name in FILES}
    parsed = {name: json.loads(value) for name, value in content.items() if name.endswith('.json')}
    summary, corpus = parsed['summary.json'], parsed['corpus.json']
    identity, config = parsed['corpus_identity.json'], parsed['training_config.json']
    scores, history = parsed['best_validation_scores.json'], parsed['history.json']
    require(identity['inputs'] == INPUTS and identity['code'] == CODE, 'Unexpected original inputs or frozen source hashes')
    identity_sha = fingerprint(identity)
    require(corpus['identity'] == identity_sha == config['corpus'], 'Corpus identity mismatch')
    require(summary['baseline'] == parsed['baseline.json'], 'Baseline summary differs from saved baseline')
    completed, final = summary['epochs_completed'], summary['final_epoch']
    require(config['epochs'] == final and 1 <= completed <= final, 'Invalid completed/final epoch')
    require([row['epoch'] for row in history] == list(range(1, completed+1)), 'Missing or reordered epoch history')
    require(summary['status'] == ('complete' if completed == final else 'paused'), 'Inconsistent completion status')
    measured = [row for row in history if 'validation' in row]
    require(bool(measured), 'No recorded validation checkpoint')
    best = max(measured, key=lambda row: row['validation']['selected']['pq'])
    require(scores['epoch'] == best['epoch'] and 1 <= scores['epoch'] <= completed, 'Best score epoch mismatch')
    require(abs(summary['best_development_pq']-best['validation']['selected']['pq']) < 1e-12, 'Best PQ mismatch')
    roles = {'train': [], 'validation': []}
    names, stems = set(), set()
    for record in corpus['records']:
        name = record['name']; stem = Path(name).stem
        require(record['role'] in roles and name == Path(name).name and name.endswith('.jpeg')
                and '\\' not in name and name not in names and stem not in stems, 'Invalid or duplicate corpus image')
        count, selected = record['proposal_count'], record['selected']
        require(isinstance(count, int) and count >= 0 and record['roi_count'] == len(selected)
                and selected == sorted(set(selected)) and all(isinstance(i, int) and 0 <= i < count for i in selected),
                'Invalid corpus proposal selection')
        names.add(name); stems.add(stem); roles[record['role']].append(record)
    counts = tuple(len(roles[role]) for role in roles)
    require(counts == expected_counts, 'Expected original616/91 physical-image roles')
    require((corpus['training_images'], corpus['validation_images']) == counts
            == (summary['training_images'], summary['validation_images']), 'Image counts disagree')
    require(sum(row['roi_count'] for row in roles['train']) == summary['training_proposals'], 'Training proposal metadata differs')
    validation = roles['validation']
    require(set(scores['scores']) == {Path(row['name']).stem for row in validation}, 'Validation score image coverage differs')
    total = 0
    for number, record in enumerate(validation, 1):
        name = record['name']; stem = Path(name).stem; n = record['proposal_count']
        require(isinstance(n, int) and n >= 0 and record['roi_count'] == n
                and record['selected'] == list(range(n)), 'Validation must retain all proposals in original order')
        q = scores['scores'][stem]
        require(len(q) == n and all(map(probability, q)), 'Invalid or misaligned quality scores')
        relative = 'proposals/'+stem+'.json.gz'; data = (root/relative).read_bytes()
        require(sha(data) == record['raw_sha'], 'Proposal file SHA mismatch: '+name)
        raw = json.loads(gzip.decompress(data)); checksum = raw.pop('checksum', None)
        require(checksum == fingerprint(raw), 'Proposal payload checksum mismatch: '+name)
        origin = raw['identity']
        require(origin['corpus'] == identity_sha and origin['role'] == 'validation' and origin['name'] == name,
                'Proposal identity mismatch: '+name)
        require(isinstance(origin['image_sha'], str) and len(origin['image_sha']) == 64
                and all(c in '0123456789abcdef' for c in origin['image_sha']), 'Invalid source image hash')
        require(len(raw['instances']) == n, 'Proposal count mismatch: '+name)
        require(all(probability(row['confidence']) and isinstance(row['rle'], str) and row['rle']
                    for row in raw['instances']), 'Invalid confidence or RLE: '+name)
        content[relative] = data  # Preserve exact bytes, including existing target fields; never use target values.
        total += n
        if number % 20 == 0 or number == len(validation):
            print(f'Checked validation [{number}/{len(validation)}], proposals={total}', flush=True)
    require(total == expected_proposals == summary['validation_proposals'], 'Expected all7172 validation proposals')
    manifest = dict(schema_version=1, corpus_identity=identity_sha, selected_epoch=scores['epoch'],
        epochs_completed=completed, final_epoch=final, training_images=counts[0], validation_images=counts[1],
        validation_proposals=total, protocol=PROTOCOL, original_input_hashes=identity['inputs'],
        frozen_source_hashes=identity['code'], exporter_sha256=sha(Path(__file__).read_bytes()),
        entries={name: dict(bytes=len(data), sha256=sha(data)) for name, data in content.items()},
        original_gzip_bytes_preserved=True, existing_target_fields='Carried unchanged; not read as decision features',
        images_included=False, weights_included=False, original_GT_file_read=False,
        models_loaded=False, training_or_inference_performed=False,
        provenance_limit='Input hashes are checked against recorded provenance; large original files are not reopened.')
    return content, manifest


def export(root, output, **validation_options):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output != root and output not in root.parents, 'Use a separate export directory')
    content, manifest = collect(root, **validation_options)
    output.mkdir(parents=True, exist_ok=True)
    archive = output/'mask_quality_replay_validation.zip'
    pending = archive.with_suffix('.pending')
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode()
    with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for name, data in content.items():
            bundle.writestr(name, data)
        bundle.writestr('replay_manifest.json', manifest_bytes)
        require(bundle.testzip() is None, 'ZIP CRC verification failed')
    pending.replace(archive)
    result = dict(status='complete', archive=str(archive), archive_bytes=archive.stat().st_size,
        archive_sha256=sha(archive.read_bytes()), validation_images=manifest['validation_images'],
        manifest_sha256=sha(manifest_bytes),
        validation_proposals=manifest['validation_proposals'], selected_epoch=manifest['selected_epoch'],
        final_epoch=manifest['final_epoch'], source_sha256=manifest['exporter_sha256'],
        model_training=False, new_inference=False, original_images_required=False)
    pending_json = output/'result.json.pending'
    pending_json.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    pending_json.replace(output/'result.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/mask_quality_v1')
    parser.add_argument('--output')
    args = parser.parse_args()
    output = args.output or str(Path(args.root)/'quality_replay_exports/v1')
    print(json.dumps(export(args.root, output), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
