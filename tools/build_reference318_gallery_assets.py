"""Freeze reference318 visualization inputs; no images, training or inference.

The raw candidate chosen here maximizes similarity to the public reference
only for visual diagnosis. It is not a new model-selected prediction.
"""
from pathlib import Path
from collections import defaultdict,Counter
import csv,gzip,hashlib,json,shutil
import numpy as np
from pycocotools import mask as mu
from reference_panel_audit import load_panel

REPO=Path(__file__).resolve().parents[1]
ART=REPO.parent/'artifacts'
DEST=ART/'reference318_gallery_assets_20261103'
PUBLIC=REPO.parent/'attachments/d03f8d87-a646-4f04-89dd-4ad6ab534cec/submission.csv'
OWN=ART/'best84_r1_uploaded_b7a6d1ce5267/submission_pipeline_baseline.csv'
PROTOCOL=ART/'best84_r1_uploaded_b7a6d1ce5267/protocol.json'
TRACE=ART/'gate_trace_uploaded_28a5be'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rle(value):return dict(size=[2048,2048],counts=value.encode('ascii') if isinstance(value,str) else value)
def csv_rows(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    assert sha(PUBLIC)=='0ab3932122a563ba96f4d243934d5c00680dd49b9d1165912a4f180907d2c603'
    assert sha(OWN)=='10432b73d7318f272fbb905e334381bf9b5746e7a9ca63d9402084f5732571f9'
    summary=json.loads((TRACE/'summary.json').read_text())
    protocol=json.loads(PROTOCOL.read_text())
    assert hashlib.sha256(json.dumps(protocol['protocol'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()==protocol['identity']==summary['cache_identity']
    assert sha(PROTOCOL) in summary['input_sha256'].values()
    provenance={v['stem']:v for v in protocol['protocol']['images']}
    public={r['filament_id']:r for r in csv_rows(PUBLIC)}
    own=defaultdict(list)
    for row in csv_rows(OWN):
        if row['segmentation_rle']:own[row['filament_id'].rsplit('_',1)[0]].append(row)
    targets=defaultdict(list)
    for panel in [216,102]:
        for row in load_panel(REPO/'configs'/f'reference{panel}.json',panel):
            targets[row['image']].append(dict(row,panel=panel))
    DEST.mkdir(parents=True,exist_ok=True)
    images=[];counts=Counter();all_ids=set()
    for image,rows in sorted(targets.items()):
        path=TRACE/'own_trace'/f'{image}.json.gz'
        assert sha(path)==summary['own_trace_sha256'][path.name]
        trace=json.loads(gzip.decompress(path.read_bytes()))
        assert trace['cache_identity']==protocol['identity'] and trace['image']==image
        candidates=trace['raw_yolo']
        ours=own[image]
        data=[]
        for row in sorted(rows,key=lambda v:(v['panel'],v['reference_id'])):
            refid=row['reference_id'];assert refid not in all_ids;all_ids.add(refid)
            reference=public[refid]['segmentation_rle'];target=rle(reference)
            own_iou=np.asarray(mu.iou([target],[rle(v['segmentation_rle']) for v in ours],[0]*len(ours)))[0] if ours else np.array([])
            raw_iou=np.asarray(mu.iou([target],[rle(v['mask']['counts']) for v in candidates],[0]*len(candidates)))[0] if candidates else np.array([])
            baseline_max=float(own_iou.max()) if len(own_iou) else 0.
            best_baseline=ours[int(own_iou.argmax())]['filament_id'] if baseline_max>0 else None
            best_raw=None
            if len(raw_iou) and raw_iou.max()>0:
                # Stable raw order resolves exact ties; no confidence/GT sorting.
                idx=int(raw_iou.argmax());c=candidates[idx]
                best_raw=dict(source=c['source'],view=c['view'],index=c['index'],confidence=float(c['confidence']),
                    mask_rle=c['mask']['counts'],reference_iou=float(raw_iou[idx]),
                    selection='best public-reference IoU among frozen own YOLO30 candidates; visualization only')
            matched=baseline_max>.5
            data.append(dict(reference_id=refid,panel=row['panel'],reference_rle=reference,
                baseline_max_iou=baseline_max,matched=matched,best_baseline_id=best_baseline,best_raw=best_raw))
            counts['targets']+=1;counts['matched' if matched else 'unmatched']+=1
            counts[f'panel{row["panel"]}']+=1
            counts['raw_matching']+=int(best_raw is not None and best_raw['reference_iou']>.5)
            counts['raw_absent']+=int(best_raw is None)
        src=provenance[image]
        assert Path(src['path']).name==src['name']==image+'.jpeg'
        images.append(dict(image=image,file_name=src['name'],image_sha256=src['sha256'],targets=data))
    assert len(images)==126 and counts['targets']==318 and counts['unmatched']==315 and counts['matched']==3
    # Compare independent earlier318audit, without choosing a new panel.
    audit=csv_rows(ART/'quality_coverage_analysis_20261103/reference_feasibility/reference318_static_feasibility.csv')
    expected={r['reference_id']:float(r['baseline_iou']) for r in audit}
    assert all(abs(t['baseline_max_iou']-expected[t['reference_id']])<1e-12 for i in images for t in i['targets'])
    data=dict(schema_version=1,scope='Public prediction reference visual review; not test ground truth, labels or PQ',
        model_scope='Frozen own YOLO+best84 T4 baseline; new quality model does not replace it',
        raw_scope='Best matching own YOLO30 raw candidate uses reference similarity for diagnosis only',
        source_identity=protocol['identity'],images=images)
    (DEST/'targets.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')))
    shutil.copyfile(OWN,DEST/'own_baseline.csv')
    manifest=dict(schema_version=1,files={n:sha(DEST/n) for n in ['targets.json','own_baseline.csv']},
        expected=dict(images=126,targets=318,unmatched=315,matched_controls=3,full_test_images=180,
            panel216=216,panel102=102,raw_matching=counts['raw_matching']),
        image_directory='/content/drive/MyDrive/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images',
        input_sha256={str(p):sha(p) for p in [PUBLIC,OWN,PROTOCOL,REPO/'configs/reference216.json',REPO/'configs/reference102.json',TRACE/'summary.json']},
        generated_with=sha(__file__),reference_is_ground_truth=False,new_model_inference=False)
    (DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(dict(output=str(DEST),images=len(images),counts=dict(counts),asset_bytes={p.name:p.stat().st_size for p in DEST.iterdir()}),indent=2))
if __name__=='__main__':main()
