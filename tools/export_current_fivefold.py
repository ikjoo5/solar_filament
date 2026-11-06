"""Evaluate stopped fold checkpoints without ever entering a training runner.

The native stage remains fixed at30. A fold without the matching YOLO stage
uses its actual last completed checkpoint, whose epoch is reported separately.
Original study/config/checkpoint files are read-only; output is a new snapshot.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
from pycocotools import mask as mu
import torch

from evaluate_fivefold_study import metadata_fold, selected_stage_folders
from evaluate_grouped_oof import aggregate, checked_rle, truth_for
from fivefold_full_pipeline import compose, fingerprint, write_csv, local_artifact
from grouped_cv_training import atomic_json, configure_cuda, digest, metric
from submission_inference import InferenceProvider

SOURCE_NAMES = ('export_current_fivefold.py', 'fivefold_full_pipeline.py',
                'submission_inference.py', 'submission_yolo_recipe.py',
                'submission_native_recipe.py')


def read(path):
    return json.loads(Path(path).read_text())


def preserve_json(path, value):
    path = Path(path)
    if path.exists() and read(path) != value:
        raise ValueError('Existing different snapshot preserved: '+str(path))
    if not path.exists():
        atomic_json(path, value)


def audit_yolo_settings(root, native, fold):
    folder = Path(root)/f'fold{fold}'/'yolo'
    cfg = read(folder/'config.json')
    if (cfg['identity'] != fingerprint(cfg['settings'])
            or cfg['settings']['fold'] != fold
            or cfg['settings']['architecture'] != 'yolo26s-seg.pt'):
        raise ValueError('YOLO fold/initialization identity changed')
    for key in ('annotations', 'fold_plan', 'train_split', 'val_split', 'base_runtime'):
        if cfg['settings']['sources'][key] != native['training_config']['source_sha256'][key]:
            raise ValueError('Native/YOLO fold sources differ: '+key)
    return cfg


def compact_payload(payload, config, history):
    """Keep inference EMA/live model exactly; never rewrite completed epochs."""
    completed = payload.get('cv_epoch')
    if (not isinstance(completed, int) or isinstance(completed, bool)
            or not 1 <= completed <= 100
            or payload.get('cv_identity') != config['identity']):
        raise ValueError('Unverifiable completed YOLO checkpoint')
    if (not history or history[-1].get('epoch') != completed
            or payload.get('cv_history') != history
            or [row.get('epoch') for row in history] != list(range(1, completed+1))):
        raise ValueError('YOLO checkpoint/history epochs differ')
    if payload.get('ema') is None and payload.get('model') is None:
        raise ValueError('YOLO inference model is missing')
    compact = dict(payload)
    for key in ('optimizer', 'scaler', 'cv_live_state_dict', 'cv_rng', 'cv_scheduler'):
        compact[key] = None
    return compact, completed


def select_yolo(root, native, fold, native_stage, folder):
    cfg = audit_yolo_settings(root, native, fold)
    canonical = Path(root)/f'fold{fold}'/'yolo'/f'stage{native_stage:03d}.json'
    if canonical.exists():
        old = read(canonical)
        path = local_artifact(Path(root), old['checkpoint'], fold)
        if (old['identity'] != cfg['identity'] or old['fold'] != fold
                or old['stage_epoch'] != native_stage
                or digest(path) != old['checkpoint_sha256']):
            raise ValueError('Completed YOLO stage identity/hash changed')
        return dict(fold=fold, epoch=native_stage, identity=cfg['identity'],
                    checkpoint=str(path), checkpoint_sha256=old['checkpoint_sha256'],
                    selection='Existing declared stage, not validation-selected')
    source = Path(root)/f'fold{fold}'/'yolo'/'last_full.pt'
    source_sha = digest(source)
    saved_manifest = folder/'yolo_snapshot.json'
    target = folder/'yolo_snapshot.pt'
    if saved_manifest.is_file():
        old = read(saved_manifest)
        if (old['source_checkpoint_sha256'] != source_sha or old['identity'] != cfg['identity']
                or old['fold'] != fold or digest(target) != old['checkpoint_sha256']):
            raise ValueError('Stopped checkpoint changed since snapshot; choose a new output folder')
        return old
    # These are the user's own Drive training checkpoints. No source module or
    # training main is imported or invoked to load/compact their saved EMA.
    print(f'Fold{fold}: loading saved YOLO checkpoint for inference snapshot; training remains stopped.', flush=True)
    payload = torch.load(source, map_location='cpu', weights_only=False)
    compact, completed = compact_payload(payload, cfg, read(source.parent/'history.json'))
    if digest(source) != source_sha:
        raise ValueError('Training checkpoint changed while reading; stop training first')
    pending = target.with_name(target.name+'.pending')
    torch.save(compact, pending)
    pending.replace(target)
    result = dict(fold=fold, epoch=completed, identity=cfg['identity'],
                  checkpoint=str(target), checkpoint_sha256=digest(target),
                  source_checkpoint=str(source), source_checkpoint_sha256=source_sha,
                  selection='Last fully saved completed epoch; no extra training')
    atomic_json(saved_manifest, result)
    del payload, compact
    return result


def checked_full_records(root, native, yolo, fold, stage, rich):
    """Return existing full recipes only after immutable provenance checks."""
    choice = Path(root)/f'fold{fold}'/f'stage{stage:03d}'/'full_export_result.json'
    if not choice.exists() or yolo['epoch'] != stage:
        return None
    chosen = read(choice)
    folder = local_artifact(Path(root), chosen['output_directory'], fold)
    summary = read(folder/'summary.json')
    payload = summary['payload']
    if (summary != chosen or summary['fold'] != fold or summary['stage_epoch'] != stage
            or summary['study_identity'] != native['study_identity']
            or fingerprint(payload) != summary['identity']
            or payload['yolo_sha256'] != yolo['checkpoint_sha256']
            or payload['native_sha256'] != native['model_sha256']
            or payload['native_records_sha256'] != native['rich_predictions_sha256']):
        raise ValueError('Existing full-pipeline model/stage identity differs')
    expected_sources = {name: digest(Path(__file__).with_name(name)) for name in
                        ('fivefold_full_pipeline.py', 'submission_inference.py',
                         'submission_yolo_recipe.py', 'submission_native_recipe.py')}
    if payload['sources'] != expected_sources:
        raise ValueError('Existing full-pipeline composition sources differ')
    path = folder/'full_records.jsonl'
    if digest(path) != summary['records_sha256']:
        raise ValueError('Existing full records changed')
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if len(records) != len(rich) or {r['file_name'] for r in records} != set(rich):
        raise ValueError('Existing full records miss validation images')
    for method in summary['methods']:
        if digest(folder/(method+'.csv')) != summary['csv_sha256'][method]:
            raise ValueError('Existing prediction CSV changed')
    for row in records:
        if (fingerprint({k: v for k, v in row.items() if k != 'record_sha256'}) != row['record_sha256']
                or row['identity'] != summary['identity'] or row['fold'] != fold
                or row['stage_epoch'] != stage
                or row['image_sha256'] != rich[row['file_name']]['image_sha256']):
            raise ValueError('Existing per-image full provenance changed')
    return records


def audit_masks(records):
    for row in records:
        for values in row['masks'].values():
            rles = [checked_rle(value) for value in values]
            if rles and int(mu.area(mu.merge(rles))) != sum(int(mu.area(value)) for value in rles):
                raise ValueError('Snapshot predictions overlap or contain duplicates')


def export_fold(root, images, native, yolo, rich_rows, output, stage, device, sources, _provider=None):
    fold = native['fold'];folder = output/f'fold{fold}'
    folder.mkdir(exist_ok=True)
    rich = {r['file_name']: r for r in rich_rows}
    identity = dict(study_identity=native['study_identity'], fold=fold,
                    native_epoch=stage, yolo_epoch=yolo['epoch'],
                    native_sha256=native['model_sha256'], yolo_sha256=yolo['checkpoint_sha256'],
                    native_records_sha256=native['rich_predictions_sha256'], sources=sources)
    signature = fingerprint(identity)
    preserve_json(folder/'snapshot_identity.json', dict(signature=signature, payload=identity))
    original = checked_full_records(root, native, yolo, fold, stage, rich)
    reused = original is not None
    records = []
    cache = folder/'records';cache.mkdir(exist_ok=True)
    raw_dir = folder/'yolo_records';raw_dir.mkdir(exist_ok=True)
    original_by_name = {r['file_name']: r for r in original} if reused else {}
    provider = _provider
    for index, name in enumerate(sorted(rich), 1):
        path = cache/(Path(name).stem+'.json')
        image_sha = rich[name]['image_sha256']
        if not reused and digest(images/name) != image_sha:
            raise ValueError('Native/YOLO original image bytes differ')
        if path.exists():
            row = read(path)
            if (fingerprint({k: v for k, v in row.items() if k != 'record_sha256'}) != row['record_sha256']
                    or row['identity'] != signature or row['image_sha256'] != image_sha
                    or row['native_epoch'] != stage or row['yolo_epoch'] != yolo['epoch']):
                raise ValueError('Current snapshot image cache changed')
        else:
            original_row = original_by_name.get(name)
            if original_row is not None:
                masks = original_row['masks']
                source_record = original_row['record_sha256']
            else:
                image_path = images/name
                stem = Path(name).stem
                raw_path = raw_dir/(stem+'.json.gz')
                raw_identity = dict(image_sha256=image_sha, yolo_sha256=yolo['checkpoint_sha256'],
                                    provider_sha256=sources['submission_inference.py'])
                if raw_path.exists():
                    raw = json.loads(gzip.decompress(raw_path.read_bytes()))
                    checksum = raw.pop('checksum')
                    if fingerprint(raw) != checksum or raw['identity'] != raw_identity:
                        raise ValueError('Current YOLO raw inference cache changed')
                else:
                    pixels = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                    if pixels is None or pixels.shape != (2048, 2048):
                        raise ValueError('Original2048 grayscale image required')
                    if provider is None:
                        provider = InferenceProvider(yolo['checkpoint'], None, device=device, verify_weights=False)
                    print(f'Fold{fold} YOLO inference [{index}/{len(rich)}] {name}: geometric/preprocessing/tiled views', flush=True)
                    raw = dict(identity=raw_identity, geometric=provider.geometric(stem, pixels),
                               preprocessing=provider.preprocessing(stem, pixels), tiled=provider.tiled(stem, pixels))
                    raw['checksum'] = fingerprint(raw)
                    pending = raw_path.with_name(raw_path.name+'.pending')
                    pending.write_bytes(gzip.compress(json.dumps(raw).encode(), mtime=0))
                    pending.replace(raw_path)
                masks = compose(stem, raw['geometric'], raw['preprocessing'], raw['tiled'], rich[name])
                source_record = None
            row = dict(file_name=name, image_sha256=image_sha, identity=signature, fold=fold,
                       native_epoch=stage, yolo_epoch=yolo['epoch'], masks=masks,
                       reused_original_record_sha256=source_record)
            row['record_sha256'] = fingerprint(row)
            atomic_json(path, row)
        records.append(row)
        print(f'Evaluation only fold{fold} native{stage}/YOLO{yolo["epoch"]} [{index}/{len(rich)}] {name}', flush=True)
    audit_masks(records)
    combined = folder/'full_records.jsonl'
    pending = combined.with_name(combined.name+'.pending')
    pending.write_text(''.join(json.dumps(r)+'\n' for r in records))
    pending.replace(combined)
    methods = set.intersection(*(set(row['masks']) for row in records))
    for method in sorted(methods):
        write_csv(folder/(method+'.csv'), records, method)
    summary = dict(fold=fold, native_epoch=stage, yolo_epoch=yolo['epoch'],
                   reused_completed_full_pipeline=reused, identity=signature,
                   records_sha256=digest(combined), methods=sorted(methods),
                   csv_sha256={method: digest(folder/(method+'.csv')) for method in sorted(methods)},
                   native_rich_sha256=native['rich_predictions_sha256'], yolo=yolo)
    atomic_json(folder/'summary.json', summary)
    return records, summary


def evaluate_records(data, records_by_fold, summaries, study_identity, native_stage):
    """Pool sufficient statistics; mixed budgets never masquerade as equal30."""
    seen = set()
    for records in records_by_fold.values():
        names = {r['file_name'] for r in records}
        if len(names) != len(records) or names & seen:
            raise ValueError('An OOF image appears in multiple folds')
        seen.update(names)
    if set(records_by_fold) != set(range(5)) or seen != {r['file_name'] for r in data['images']}:
        raise ValueError('Current snapshot must cover all five folds and all official physical images')
    methods = set.intersection(*(set(row['masks']) for rows in records_by_fold.values() for row in rows))
    result = {}
    for method in sorted(methods):
        all_groups = [];per_fold = {}
        for fold, rows in sorted(records_by_fold.items()):
            truth, mapping = truth_for(data, {r['file_name'] for r in rows})
            prediction = {Path(r['file_name']).stem: [checked_rle(v) for v in r['masks'][method]] for r in rows}
            measured = metric(truth, prediction, mapping)
            all_groups.extend(measured['per_group'])
            per_fold[str(fold)] = aggregate(measured['per_group'])
            per_fold[str(fold)].update(blank_gt_groups=measured['blank_gt_groups'],
                                      fp_on_blank_gt_groups=measured['fp_on_blank_gt_groups'])
        result[method] = dict(pooled_current_checkpoint_oof=aggregate(all_groups), by_fold=per_fold)
    yolo_epochs = {str(s['fold']): s['yolo_epoch'] for s in summaries}
    equal = all(epoch == native_stage for epoch in yolo_epochs.values())
    return dict(status='complete_current_checkpoint_oof', study_identity=study_identity,
                physical_images=len(seen), folds=list(range(5)), native_epoch=native_stage,
                yolo_epochs=yolo_epochs, equal_training_budgets=equal,
                same_budget_30epoch_oof=equal and native_stage == 30,
                scope='Equal declared stage OOF' if equal else 'Mixed-budget current checkpoints; not equal30epoch OOF',
                training_executed=False, ground_truth_used_only_after_prediction_csvs=True,
                results=result, historical_91_image_pq_is_not_a_direct_comparison=True)


def package(output, root, summaries, limit=23*1024*1024):
    """Only new snapshot artifacts and newly inferred raw masks, never weights."""
    candidates = []
    for path in sorted(output.rglob('*')):
        if (path.is_file() and path.suffix not in ('.pt', '.zip', '.log', '.pending', '.tmp', '.restoring')
                and path.name != 'result.json' and 'bundles' not in path.relative_to(output).parts):
            # Per-image final records duplicate full_records.jsonl; raw YOLO is needed for later experiments.
            if 'records' in path.relative_to(output).parts:
                continue
            candidates.append((path, path.relative_to(output).as_posix()))
    for name in ('study_config.json', 'fold_plan.json', 'stage030_plan.json', 'stage030_progress.json'):
        path = root/name
        if path.is_file():
            candidates.append((path, 'original_metadata/'+name))
    for summary in summaries:
        fold = summary['fold']
        for name in ('train_split.csv', 'val_split.csv', 'split_manifest.json', 'config.json',
                     'history.json', 'yolo/config.json', 'yolo/history.json'):
            path = root/f'fold{fold}'/name
            if path.is_file():
                candidates.append((path, f'original_metadata/fold{fold}/'+name))
    folder = output/'bundles';folder.mkdir(exist_ok=True)
    archives = [];handle = None;used = 0
    index = []
    try:
        for path, relative in candidates:
            size = path.stat().st_size
            if size > limit-65536:
                raise ValueError('Unexpectedly large single snapshot artifact: '+str(path))
            if handle is None or used+size > limit-65536:
                if handle is not None:
                    handle.close()
                target = folder/f'current_epoch_results_part{len(archives)+1:02d}.zip'
                archives.append(target)
                handle = zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED)
                used = 0
            handle.write(path, relative);used += size
            index.append(dict(archive=archives[-1].name, path=relative, sha256=digest(path), bytes=size))
    finally:
        if handle is not None:
            handle.close()
    manifest = dict(files=index, archives=[path.name for path in archives])
    atomic_json(folder/'archive_manifest.json', manifest)
    for path in archives:
        with zipfile.ZipFile(path, 'a', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('archive_manifest.json', json.dumps(manifest, indent=2))
    assert all(path.stat().st_size < 24*1024*1024 for path in archives)
    return [str(path) for path in archives]


def run(root, images, annotations, output, device='cuda', native_stage=30, _provider=None):
    root, images, annotations, output = map(Path, (root, images, annotations, output))
    if output.resolve().is_relative_to(root.resolve()) and output.resolve() == root.resolve():
        raise ValueError('Evaluation output must be a separate snapshot directory')
    if not output.resolve().is_relative_to((root/'current_epoch_evaluation').resolve()):
        raise ValueError('Output must be below study/current_epoch_evaluation to preserve original artifacts')
    output.mkdir(parents=True, exist_ok=True)
    configure_cuda()
    study = read(root/'study_config.json');plan = read(root/'fold_plan.json')
    if (fingerprint(study['identity_payload']) != study['study_identity']
            or study['identity_payload']['annotations_sha256'] != digest(annotations)
            or study['identity_payload']['fold_plan_sha256'] != digest(root/'fold_plan.json')):
        raise ValueError('Official source/study identity changed')
    sources = {name: digest(Path(__file__).with_name(name)) for name in SOURCE_NAMES}
    request = dict(study_identity=study['study_identity'], native_stage=native_stage,
                   source_sha256=sources, training_executed=False)
    preserve_json(output/'request.json', request)
    execution_source = output/'execution_source'
    execution_source.mkdir(exist_ok=True)
    for name in SOURCE_NAMES:
        source = Path(__file__).with_name(name)
        saved = execution_source/name
        if saved.exists() and digest(saved) != sources[name]:
            raise ValueError('Snapshot execution source changed; choose a new output directory')
        if not saved.exists():
            pending = saved.with_name(saved.name+'.pending')
            pending.write_bytes(source.read_bytes())
            pending.replace(saved)
    folders = selected_stage_folders(root, native_stage, study['study_identity'], 5)
    if set(folders) != set(range(5)):
        raise ValueError('All five native stage exports are required')
    all_records = {};summaries = [];regimes = []
    for fold, (native_folder, expected) in sorted(folders.items()):
        print(f'Fold{fold}: auditing saved native{native_stage} predictions and selecting saved YOLO weights.', flush=True)
        rich, _, regime = metadata_fold(native_folder, study, plan, fold, native_stage,
                                        digest(annotations), digest(root/'fold_plan.json'))
        if regimes and regime != regimes[0]:
            raise ValueError('Native inference regime differs across folds')
        regimes.append(regime)
        native = read(native_folder/'stage_summary.json')
        if expected and native != expected:
            raise ValueError('Native progress and stage summary differ')
        fold_output = output/f'fold{fold}';fold_output.mkdir(exist_ok=True)
        yolo = select_yolo(root, native, fold, native_stage, fold_output)
        records, summary = export_fold(root, images, native, yolo, rich, output, native_stage,
                                      device, sources, _provider=_provider)
        all_records[fold] = records;summaries.append(summary)
    # This is the first GT decoding operation; every prediction CSV is already saved.
    report = evaluate_records(read(annotations), all_records, summaries, study['study_identity'], native_stage)
    report.update(source_sha256=sources, fold_summaries=summaries)
    atomic_json(output/'evaluation.json', report)
    archives = package(output, root, summaries)
    result = dict(status=report['status'], report=str(output/'evaluation.json'), archives=archives,
                  native_epoch=native_stage, yolo_epochs=report['yolo_epochs'],
                  equal_training_budgets=report['equal_training_budgets'], training_executed=False)
    atomic_json(output/'result.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True);parser.add_argument('--images', required=True)
    parser.add_argument('--annotations', required=True);parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda');parser.add_argument('--native-stage', type=int, default=30)
    args = parser.parse_args()
    import fcntl
    key = hashlib.sha256(str(Path(args.output).resolve()).encode()).hexdigest()[:20]
    with (Path('/tmp')/('solar_current_eval_'+key+'.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('This evaluation snapshot is already running in this runtime') from error
        run(args.root, args.images, args.annotations, args.output, args.device, args.native_stage)
