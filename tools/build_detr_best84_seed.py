"""Build prediction-only fixed-validation inputs for the T4 best84 comparison.

No annotations or GT masks are read.  The seed contains only the frozen YOLO
reference/weak predictions and the reviewed DETR70/80 bounding boxes.  This is
an audited cache for the existing91 validation images, never a test predictor.
"""
from __future__ import annotations

from collections import defaultdict
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import cv2

import submission_native_recipe as native
from submission_yolo_recipe import build_yolo_reference


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT.parent/'artifacts'
SOURCE_FILES = {
    'geometric': ARTIFACTS/'validation_results_20261027T120958Z_444da7f4/raw_views.jsonl',
    'preprocessing': ARTIFACTS/'validation_results_20261027T125130Z_2312e759/raw_preprocess_views.jsonl',
    'tiled': ARTIFACTS/'tiled_results_20261027T205426Z_eba01cfb/raw_tiled_views.jsonl',
    'native': ARTIFACTS/'best84_rich_predictions_tta/predictions.jsonl',
}
SOURCE_SHA = {
    'geometric': '86ce8a9f105216f3564eb39379f4f8e39f8533d35226db80e37d5ddc931113dc',
    'preprocessing': '85ab4d569a8355b1002409712089b93c1db5b0fa442b73234c31fce749dd7972',
    'tiled': 'ea88cc5d8bc3819043b48290431b2df17138e1422cffbb3334be3140b3db985d',
    'native': 'b5049be7391cd750cb40eda3371bb518d6422a45b35fcb95fbc7e08712a3e32c',
}
FROZEN_MODULE_SHA = {
    'submission_yolo_recipe.py': '28bfdd616ad30fe8eeb43d6d288e729936add4334acee58770f2240f4fcc36ad',
    'submission_native_recipe.py': 'cd861374baa377eca6b0cdd5af1de4b53a21927d9d8d51f11327a7c484244606',
}
DETR_ROOT = ARTIFACTS/'detr_longrun_uploaded_c27c4813ec0a'
DETR_FOLDERS = {
    '70': DETR_ROOT/'evaluations/epoch070_f2a1caef71e961ad',
    '80': DETR_ROOT/'evaluations/epoch080_fdcb1ab83ad1009f',
}
DETR_EXPORT_ID = 'c27c4813ec0a6a87747a375a042acf2de7cad440a339aa68650b71d7063ac52e'
WEIGHTS = {
    'yolo': '81c2a27d95821b5ede283169c0da8c08d360ecc4c2a60a1b9c6baf96d76bcccc',
    'native_best84': 'ae6b209f72355fd52d2cb07182e69028e97c703a1977ad1e7b38b27cf8cebe69',
    'roi30': '4d1a97c0baf3b7d820754de141d68f1f56cffb0f5fb1b06be6f1b93a9a90ad7f',
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_ready(value):
    if isinstance(value, bytes):
        return value.decode('ascii')
    if isinstance(value, dict):
        return {key: json_ready(item) for key,item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def compact(value):
    return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def rles_from_seed(values):
    return [native._rle(row) for row in values]


def native_from_seed(frame, rich, return_stages=False):
    """Apply the exact frozen native stages to their sufficient inputs.

    The old weak-proposal builder uses confidence only to select >=.6 masks;
    remaining operations only consume those masks, not raw scores/views.
    """
    if Path(rich['file_name']).name != frame['file_name']:
        raise ValueError('Native prediction and seed image differ')
    original = rles_from_seed(frame['yolo_reference'])
    weak = rles_from_seed(frame['weak'])
    reference, added = native._add_native_reference(original, rich, weak)
    refined, decisions = native._refine_boundaries(reference, rich)
    final, pruning = native._final_admission_pruning(refined, rich)
    if return_stages:
        return dict(native_reference=reference, rich_refined=refined, final=final,
                    decisions=dict(native_reference_added=added, **decisions, **pruning))
    return final


def checked_record(path, expected_file_sha):
    if digest(path) != expected_file_sha:
        raise ValueError('DETR prediction file changed: '+str(path))
    envelope = json.loads(gzip.decompress(Path(path).read_bytes()))
    payload = envelope['payload']
    # ROI/detector caches use native_instance_training.identity, which is
    # SHA256 of sorted compact JSON (including native JSON float precision).
    if hashlib.sha256(compact(payload)).hexdigest() != envelope['sha256']:
        raise ValueError('DETR record payload checksum mismatch')
    return payload


def build(output=ARTIFACTS/'detr_best84_seed'):
    started = time.monotonic()
    cv2.setNumThreads(1)
    for name,expected in FROZEN_MODULE_SHA.items():
        if digest(ROOT/'tools'/name) != expected:
            raise ValueError('Frozen recipe source changed: '+name)
    stages = defaultdict(dict)
    for family,path in SOURCE_FILES.items():
        if digest(path) != SOURCE_SHA[family]:
            raise ValueError('Original prediction input changed: '+str(path))
        for line in path.open():
            row = json.loads(line)
            if family == 'native':
                stages[Path(row['file_name']).stem][family] = row
            else:
                stages[row['image']].setdefault(family, []).append(row)
    if len(stages) != 91:
        raise ValueError('Expected all91 fixed validation photos')
    old_summary = json.loads((SOURCE_FILES['tiled'].parent/'summary.json').read_text())
    evaluation_protocol = json.loads((DETR_ROOT/'evaluation_protocol.json').read_text())['protocol']
    export_manifest = json.loads((DETR_ROOT/'export_manifest.json').read_text())
    if export_manifest['export_id'] != DETR_EXPORT_ID:
        raise ValueError('Expected the reviewed DETR70/80 export')
    reports = {epoch:json.loads((path/'report.json').read_text()) for epoch,path in DETR_FOLDERS.items()}
    original_csv = ARTIFACTS/'rich_boundary_composition_v1/tta_precision.csv'
    if digest(original_csv) != '0b651ca12822c740e7bdc06ea5f76e700c071a7ef660711b225760f378b5278b':
        raise ValueError('Expected original frozen production replay CSV')
    expected = defaultdict(list)
    with original_csv.open() as handle:
        for row in csv.DictReader(handle):
            image,_ = row['filament_id'].rsplit('_',1)
            expected[image].append(row['segmentation_rle'])
    frames=[]; counts=dict(yolo_reference=0,weak=0,boxes70=0,boxes80=0,final_masks=0)
    for index,stem in enumerate(sorted(stages),1):
        group=stages[stem]; rich=group['native']; name=rich['file_name']
        if any(len(group[key]) != count for key,count in [('geometric',4),('preprocessing',8),('tiled',18)]):
            raise ValueError('Missing full frozen YOLO view family: '+stem)
        image_sha=rich['image_sha256']
        if image_sha != old_summary['image_sha256'][stem] or image_sha != evaluation_protocol['images'][name]:
            raise ValueError('Source pixels differ between predictions: '+stem)
        yolo=build_yolo_reference(stem,group['geometric'],group['preprocessing'],group['tiled'])
        weak=native._weak_proposals(stem,group['geometric'],group['preprocessing'],group['tiled'])
        frame=dict(file_name=name,image_sha256=image_sha,yolo_reference=json_ready(yolo),
                   weak=json_ready(weak),boxes={})
        for epoch,folder in DETR_FOLDERS.items():
            relative='records/'+stem+'.json.gz'
            row=checked_record(folder/relative,reports[epoch]['record_sha256'][relative])
            if (row['image_name'],row['image_sha256']) != (name,image_sha):
                raise ValueError('DETR frame identity changed: '+stem)
            boxes=row['actual_boxes']
            if any(set(p) != {'box','score'} for p in boxes):
                raise ValueError('Unexpected detector box record')
            frame['boxes'][epoch]=boxes
            counts['boxes'+epoch]+=len(boxes)
        original=native.build_native_submission(stem,yolo,group['geometric'],group['preprocessing'],group['tiled'],rich,return_stages=True)
        replay=native_from_seed(frame,rich,return_stages=True)
        if json_ready(original) != json_ready(replay):
            raise AssertionError('Minimal seed changed a frozen native stage: '+stem)
        if [r['counts'].decode('ascii') for r in replay['final']] != expected[stem]:
            raise AssertionError('Minimal seed does not reproduce original submission: '+stem)
        if rles_from_seed(frame['yolo_reference']) != yolo or rles_from_seed(frame['weak']) != weak:
            raise AssertionError('Seed changed ordered YOLO reference/weak predictions')
        counts['yolo_reference']+=len(yolo);counts['weak']+=len(weak);counts['final_masks']+=len(replay['final'])
        frames.append(frame)
        if index%20==0 or index==91:
            print('Seed frozen replay',index,'/91',flush=True)
    result_metadata=json.loads((DETR_ROOT/'result.json').read_text())
    detr={epoch:dict(epoch=int(epoch),evaluation_identity=report['identity'],
        full_checkpoint_sha256=report['checkpoint_sha256'],raw_box_min_confidence=.05,
        detector_image_size=1280,detector_precision='bf16',bbox_head_precision='fp32',
        source_gpu='NVIDIA A100-SXM4-40GB',
        evaluation_source='reviewed frozen A100 predictions, reused identically in all T4 arms')
        for epoch,report in reports.items()}
    detr['70']['available_checkpoint_sha256']=result_metadata['best_checkpoint_sha256']
    detr['80']['available_checkpoint_sha256']=result_metadata['current_checkpoint_sha256']
    seed=dict(schema_version=1,
        scope='Fixed91 validation-only prediction cache; not a test-set inference database',
        inference_contains_GT=False,physical_images=91,metadata=evaluation_protocol['metadata'],
        source_checkpoints=dict(WEIGHTS,detr=detr),
        source_predictions_sha256=SOURCE_SHA,frozen_recipe_sha256=FROZEN_MODULE_SHA,
        source_export_id=DETR_EXPORT_ID,
        notes=['Only saved YOLO masks and DETR boxes are included; no original images, GT masks, labels or native/ROI output probabilities.',
               'The YOLO weak pool is the original ordered list of>=.6 proposals, excluding tile-edge-clipped candidates.',
               'Re-evaluate full and selected best84 inference under the same T4 precision; historicalA100 scores are not the paired control.'],
        frames=frames)
    raw=compact(seed);compressed=gzip.compress(raw,compresslevel=9,mtime=0)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/'prediction_seed.json.gz';path.write_bytes(compressed)
    report=dict(schema_version=1,seed_path=str(path),seed_gzip_sha256=digest(path),
        seed_json_sha256=hashlib.sha256(raw).hexdigest(),compressed_bytes=len(compressed),json_bytes=len(raw),
        physical_images=91,counts=counts,all_ordered_reference_and_weak_RLEs_identical=True,
        all_native_stages_and_final_ordered_RLEs_identical=True,
        original_full_pipeline_csv_sha256=digest(original_csv),
        historical_original_full_pipeline_PQ=.47259304445299505,
        original_historical_result_not_newT4_PQ=True,
        GT_or_annotations_read=False,source_prediction_sha256=SOURCE_SHA,
        code_sha256=digest(__file__),frozen_recipe_sha256=FROZEN_MODULE_SHA,
        input_image_sha_agrees_across_tiledYOLO_native_and_DETR=True,
        scope=seed['scope'],elapsed_seconds=time.monotonic()-started)
    (output/'seed_provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ARTIFACTS/'detr_best84_seed')
    args=parser.parse_args();build(args.output)
