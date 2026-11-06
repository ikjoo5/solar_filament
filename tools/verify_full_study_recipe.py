"""Historical91-image replay proves recipe parity, not a new5fold score."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from fivefold_full_pipeline import compose
from experiment_mask_postprocess import group_rows,read_rows
from filament_eval import Filament,evaluate
from verify_submission_native_recipe import OLD,PREP,TILES,RICH,EXPECTED,load_raw
from verify_native_training_results import GT,VAL


def main():
    geo,prep,tiles=map(load_raw,(OLD,PREP,TILES))
    rich={Path(r['file_name']).stem:r for r in map(json.loads,RICH.open())}
    expected=group_rows(read_rows(EXPECTED['final']));rows=[]
    for image,record in sorted(rich.items()):
        result=compose(image,geo[image],prep[image],tiles[image],record,variants=False)['full_selected']
        assert Counter(v['counts'] for v in result)==Counter(r.counts for r in expected.get(image,[]))
        rows.extend(Filament(f'{image}_{i}',image,image,v['counts'],0) for i,v in enumerate(result,1))
    truth=read_rows(GT,gt=True,selected={r['image_id'] for r in csv.DictReader(VAL.open())})
    measured=evaluate(truth,rows,2048,2048)
    report=dict(scope='Historicalcached91 replay only; not actualCVtraining/inference',
        baseline_mask_parity=True,images=len(rich),physical_masks=len(rows),
        **{k:measured[k] for k in ('pq','tp_pairs','fp','fn')},
        actual_fivefold_models_trained=False,weights_reused_for_actual_CV=False,
        composition_sha256=hashlib.sha256(Path(__file__).with_name('fivefold_full_pipeline.py').read_bytes()).hexdigest())
    (Path(__file__).resolve().parents[1]/'docs/evidence/fivefold_full_recipe_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
