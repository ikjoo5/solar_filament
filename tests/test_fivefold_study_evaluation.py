"""Portable native methods, common-budget coverage, and stage provenance."""
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import cv2
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from grouped_cv_training import DEFAULTS, IDENTITY_KEYS, digest
from prepare_grouped_fivefold import build_plan
import evaluate_fivefold_study as study


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def encoded(x=20, y=20, width=30):
    image = np.zeros((2048, 2048), np.uint8)
    image[y:y+width, x:x+width] = 1
    value = mu.encode(np.asfortranarray(image))
    return dict(size=[2048, 2048], counts=value['counts'].decode())


def component(rle, mean=.95):
    decoded = study.checked_rle(rle)
    return dict(rle=rle, area=int(mu.area(decoded)), bbox=[int(v) for v in mu.toBbox(decoded)],
                fg_mean=mean, stats=dict(foreground=dict(mean=mean)))


def branch(values):
    total = mu.merge([study.checked_rle(row['rle']) for row in values]) if values else dict(size=[2048,2048], counts=b'PPPP4')
    union = dict(size=[2048,2048], counts=total['counts'].decode())
    return dict(components={threshold: values for threshold in ('0.35','0.5','0.65','0.8')},
                unions=dict(foreground={threshold: union for threshold in ('0.2','0.35','0.5','0.65','0.8')}))


def signed_record(record):
    unsigned = {key:value for key,value in record.items() if key != 'record_sha256'}
    record['record_sha256'] = hashlib.sha256(json.dumps(unsigned, ensure_ascii=False, sort_keys=True,
                                                       separators=(',', ':')).encode()).hexdigest()
    return record


def fixture(root):
    data = dict(images=[], annotations=[])
    for index in range(10):
        captured = date(2011,1,1)+timedelta(days=30*index)
        name = captured.strftime('%Y%m%d')+'000000'+('Bh' if index%2 else 'Ch')+'.jpeg'
        for rater in range(2 if index == 0 else 1):
            sid = f'{rater}-{name[:-5]}'
            data['images'].append(dict(id=sid, file_name=name))
            for obj in range(index%3+1):
                data['annotations'].append(dict(image_id=sid, segmentation=encoded(20+50*obj)))
    annotations = root/'annotations.json'; save(annotations,data)
    plan = build_plan(data); plan['sources_sha256'] = dict(annotations=digest(annotations))
    plan_path = root/'plan.json'; save(plan_path,plan)
    common = dict(DEFAULTS, validate_every=100)
    source = dict(training='a'*64, runtime='b'*64, study='c'*64, rich_inference='d'*64)
    identity_payload = dict(training_settings=common, annotations_sha256=digest(annotations),
        fold_plan_sha256=digest(plan_path), training_source_sha256={key:source[key] for key in ('training','runtime')})
    identity = study.identity(identity_payload)
    save(root/'study_config.json',dict(study_identity=identity, identity_payload=identity_payload))
    save(root/'stage030_plan.json',dict(stage_epoch=30,training_epochs=100,
        expected_folds=list(range(5)),evaluation_kind='fixed_budget_exploration'))
    for fold in range(5):
        folder = root/f'fold{fold}'/'stage030'; folder.mkdir(parents=True)
        info = plan['fold_summaries'][fold]
        names = sorted(row['file_name'] for row in plan['observations'] if info['roles'][row['physical_identifier']] == 'validation')
        cfg = dict(common, fold=fold, seed=42+fold, amp_dtype='None')
        cfg['source_sha256'] = dict(annotations=digest(annotations), fold_plan=digest(plan_path),
            train_split=study.split_sha(plan,info,'train'),val_split=study.split_sha(plan,info,'validation'),
            base_runtime=source['runtime'],cv_runner=source['training'])
        payload = dict(settings={key:cfg[key] for key in IDENTITY_KEYS}, fold=fold,
            sources=cfg['source_sha256'], precision='None',final_selection='fixed_epoch')
        cfg['cv_identity_payload'] = payload; cfg['cv_identity'] = study.identity(payload)
        model_sha = str(fold+1)*64
        records = []
        for name in names:
            base = branch([component(encoded())] if fold else [])
            tta = branch([component(encoded())])
            record = dict(**base, file_name=name,study_identity=identity,stage_epoch=30,fold=fold,
                checkpoint_sha256=model_sha,image_sha256='e'*64,tta=tta,
                preprocess_views=dict(clahe=tta),inference_identity='fixture-inference',
                view_unions={view:tta['unions']['foreground']['0.5'] for view in ('base','hflip','vflip','hvflip')})
            records.append(signed_record(record))
        rich_path = folder/'rich_records.jsonl'
        rich_path.write_text(''.join(json.dumps(row)+'\n' for row in records))
        exporter = dict(stage_epoch=30,fold=fold,
            provenance=dict(study_identity=identity,model_sha256=model_sha),
            rich_predictions_sha256=digest(rich_path), records_sha256={r['file_name']:r['record_sha256'] for r in records},
            inference_identity='fixture-inference',settings=dict(profile='tta_clahe'))
        save(folder/'export_summary.json',exporter)
        summary = dict(status='complete',fold=fold,stage_epoch=30,study_identity=identity,
            training_config=cfg,training_cv_identity=cfg['cv_identity'],source_sha256=source,
            train_physical_identifiers=sorted(k for k,role in info['roles'].items() if role == 'train'),
            image_names=names,image_coverage=len(names),model_sha256=model_sha,
            rich_predictions_sha256=digest(rich_path),export_summary_sha256=digest(folder/'export_summary.json'))
        save(folder/'stage_summary.json',summary)
    return annotations, plan_path


class StudyEvaluationTests(unittest.TestCase):
    def test_progress_selects_current_variant_and_does_not_pool_old_canonical_folds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); annotations, plan = fixture(root)
            canonical = root/'fold0/stage030'
            variant = canonical/'variants/test_variant'; shutil.copytree(canonical,variant)
            current=json.loads((variant/'stage_summary.json').read_text())
            current['export']=dict(rich_predictions='/old/drive/fold0/stage030/variants/test_variant/rich_records.jsonl')
            save(variant/'stage_summary.json',current)
            old=json.loads((canonical/'stage_summary.json').read_text()); old['stage_epoch']=40
            save(canonical/'stage_summary.json',old)
            save(root/'stage030_progress.json',dict(study_identity=current['study_identity'],stage_epoch=30,
                completed_folds=[0],summaries=[current]))
            result=study.run(annotations,plan,root,30,root/'selected',root/'selected.json')
            self.assertEqual(result['completed_folds'],[0])
            self.assertFalse(result['complete_stage'])
            self.assertEqual(result['selected_inference_paths']['0'],'fold0/stage030/variants/test_variant')

    def test_actual_rich_exporter_schema_is_accepted(self):
        from fivefold_rich_inference import export_stage
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); annotations, plan_path = fixture(root)
            plan = json.loads(plan_path.read_text()); stage = root/'fold0/stage030'
            summary = json.loads((stage/'stage_summary.json').read_text())
            info = plan['fold_summaries'][0]
            images = root/'images'; images.mkdir()
            pixels = np.full((2048,2048),20,np.uint8); pixels[20:50,20:50] = 240
            csv_path = root/'val_split.csv'
            with csv_path.open('w',newline='') as stream:
                import csv
                writer=csv.writer(stream); writer.writerow(['image_id','file_name'])
                for row in plan['observations']:
                    if info['roles'][row['physical_identifier']] == 'validation':
                        cv2.imwrite(str(images/row['file_name']),pixels)
                        writer.writerows((sid,row['file_name']) for sid in row['annotation_ids'])
            class Store:
                def __init__(self,*args): pass
                def load(self,name): return pixels.copy()
            runtime=SimpleNamespace(ImageStore=Store,__file__=__file__,
                predict_foreground=lambda model,image,*args: image.astype(np.float32)/255)
            cfg=dict(summary['training_config'],images=str(images),val_split=str(csv_path),
                cache_root=str(root/'cache'),device='cpu',inference_profile='tta_clahe',stage_epoch=30)
            provenance=dict(study_identity=summary['study_identity'],fold=0,stage_epoch=30,
                model_sha256=summary['model_sha256'],source_sha256=summary['source_sha256'])
            exported=export_stage(runtime,SimpleNamespace(eval=lambda:None),cfg,stage,provenance)
            summary['rich_predictions_sha256']=digest(exported['rich_predictions'])
            summary['export_summary_sha256']=digest(exported['summary'])
            save(stage/'stage_summary.json',summary)
            study_config=json.loads((root/'study_config.json').read_text())
            records,_,_=study.metadata_fold(stage,study_config,plan,0,30,digest(annotations),digest(plan_path))
            self.assertEqual(len(records),len(summary['image_names']))

    def test_complete_common_budget_and_partial_reports_generate_before_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); annotations, plan = fixture(root)
            actual = study.truth_for
            def checked_truth(*args, **kwargs):
                for fold in range(5):
                    for method in study.METHODS:
                        self.assertTrue((root/'outputs'/f'fold{fold}'/(method+'.csv')).is_file())
                return actual(*args, **kwargs)
            with patch.object(study,'truth_for',side_effect=checked_truth):
                result = study.run(annotations,plan,root,30,root/'outputs',root/'report.json')
            self.assertTrue(result['complete_stage'])
            self.assertEqual(result['completed_physical_images'],10)
            self.assertEqual(result['results']['base']['pooled_oof']['annotation_groups'],11)
            self.assertGreater(result['deltas_from_base']['guarded_consensus']['tp_pairs'],0)
            self.assertFalse(result['adopted'])
            for fold in range(1,5):
                (root/f'fold{fold}'/'stage030/stage_summary.json').unlink()
            partial = study.run(annotations,plan,root,30,root/'partial',root/'partial.json')
            self.assertEqual(partial['status'],'partial_stage')
            self.assertIsNone(partial['results']['base']['pooled_oof'])
            self.assertGreater(partial['results']['base']['pooled_completed_folds']['fn'],0)

    def test_mixed_stage_or_altered_coverage_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); annotations,plan = fixture(root)
            path = root/'fold0/stage030/stage_summary.json'
            summary = json.loads(path.read_text()); summary['stage_epoch'] = 40; save(path,summary)
            with self.assertRaisesRegex(ValueError,'Mixed/incomplete stage'):
                study.run(annotations,plan,root,30,root/'badstage',root/'report.json')
            summary['stage_epoch'] = 30; save(path,summary)
            rich = root/'fold0/stage030/rich_records.jsonl'
            lines = rich.read_text().splitlines(); rich.write_text('\n'.join(lines[1:])+'\n')
            summary['rich_predictions_sha256'] = digest(rich); save(path,summary)
            exported_path = rich.parent/'export_summary.json'; exported = json.loads(exported_path.read_text())
            exported['rich_predictions_sha256'] = digest(rich); save(exported_path,exported)
            summary['export_summary_sha256'] = digest(exported_path); save(path,summary)
            with self.assertRaisesRegex(ValueError,'prediction coverage'):
                study.run(annotations,plan,root,30,root/'badcoverage',root/'report.json')

    def test_portable_rules_keep_baseline_and_reject_overlap(self):
        base = component(encoded())
        small = component(encoded(200,200,10),.85)
        low_mean = component(encoded(400,400,10),.79)
        record = dict(**branch([base,small,low_mean]),tta=branch([base]),preprocess_views=dict(clahe=branch([base])),
            view_unions={view:encoded() for view in ('base','hflip','vflip','hvflip')})
        result = study.methods_for(record)
        self.assertEqual(len(result['base']),1)
        self.assertEqual(len(result['small_object_recall']),2)
        self.assertEqual(result['small_object_recall'][0],result['base'][0])
        with self.assertRaisesRegex(ValueError,'Overlapping'):
            study.disjoint([result['base'][0],result['base'][0]])


if __name__ == '__main__':
    unittest.main()
