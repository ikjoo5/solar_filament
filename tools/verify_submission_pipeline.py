"""Whole frozen portable submission replay on cached validation predictions."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time
import cv2

from experiment_mask_postprocess import overlap_stats, read_rows
from filament_eval import evaluate
from submission_driver import export_csv, finalize_frame
from verify_native_training_results import GT, ROOT, VAL


def main():
    cv2.setNumThreads(1)
    started=time.monotonic()
    files={
        'geometric':Path('/workspace/artifacts/validation_results_20261027T120958Z_444da7f4/raw_views.jsonl'),
        'preprocessing':Path('/workspace/artifacts/validation_results_20261027T125130Z_2312e759/raw_preprocess_views.jsonl'),
        'tiled':Path('/workspace/artifacts/tiled_results_20261027T205426Z_eba01cfb/raw_tiled_views.jsonl'),
        'native':Path('/workspace/artifacts/best84_rich_predictions_tta/predictions.jsonl'),
    }
    stages=defaultdict(dict)
    for name,path in files.items():
        for line in path.open():
            row=json.loads(line)
            if name=='native':stages[Path(row['file_name']).stem][name]=row
            else:stages[row['image']].setdefault(name,[]).append(row)
    expected_path=Path('/workspace/artifacts/rich_boundary_composition_v1/tta_precision.csv')
    expected=defaultdict(list)
    for row in read_rows(expected_path):expected[row.image_id].append(row.counts)
    records=[]
    assert len(stages)==91
    for index,image in enumerate(sorted(stages),1):
        payload=stages[image]
        assert all(len(payload[k])==n for k,n in [('geometric',4),('preprocessing',8),('tiled',18)])
        predictions=finalize_frame(image,payload)
        assert [r['counts'].decode('ascii') for r in predictions]==expected[image],image
        records.append(dict(file_name=image+'.jpeg',rles=predictions))
        if index%20==0:print('Frozen replay',index,'/91',flush=True)
    output=Path('/workspace/artifacts/submission_pipeline_validation_replay.csv')
    assert export_csv(output,records)==625
    rows=read_rows(output)
    groups={r['image_id'] for r in csv.DictReader(VAL.open())}
    truth=read_rows(GT,gt=True,selected=groups)
    score=evaluate(truth,rows,2048,2048);audit=overlap_stats(rows)
    assert abs(score['pq']-.47259304445299505)<1e-12
    assert not any(audit.values()) and score['tp_pairs']==708-score['fn']
    report=dict(scope='Offline91-frame cached-prediction replay of production submission modules; not newGPUtest inference',
        images=91,annotation_groups=101,gt_instances=708,all_ordered_rles_identical=True,
        prediction_rows=625,**{k:score[k] for k in ('pq','sq','rq','tp_pairs','fp','fn')},**audit,
        original_csv_sha256=hashlib.sha256(expected_path.read_bytes()).hexdigest(),
        replay_csv_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        entire_csv_identical=expected_path.read_bytes()==output.read_bytes(),
        source_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()},
        portable_modules_sha256={name:hashlib.sha256((ROOT/'tools'/name).read_bytes()).hexdigest() for name in (
            'submission_yolo_recipe.py','submission_native_recipe.py','submission_inference.py','submission_driver.py','submission_native_process.py')},
        elapsed_seconds=time.monotonic()-started)
    (ROOT/'docs/evidence/submission_pipeline_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
