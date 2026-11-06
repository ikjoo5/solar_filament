"""OOF provenance, blank-image coverage, and count-pooled score contracts."""
from datetime import date, timedelta
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from grouped_cv_training import DEFAULTS, IDENTITY_KEYS, digest, metric
from prepare_grouped_fivefold import build_plan
import evaluate_grouped_oof as audit


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def encoded(x, y, width=30):
    mask = np.zeros((2048, 2048), np.uint8)
    mask[y:y+width, x:x+width] = 1
    value = mu.encode(np.asfortranarray(mask))
    return dict(size=[2048, 2048], counts=value['counts'].decode())


def fixture(root):
    data = dict(images=[], annotations=[])
    for index in range(10):
        day = date(2011, 1, 1)+timedelta(days=index*30)
        name = day.strftime('%Y%m%d')+'000000'+('Bh' if index % 2 else 'Ch')+'.jpeg'
        for rater in range(2 if index == 0 else 1):
            sid = f'{rater}-{name[:-5]}'
            data['images'].append(dict(id=sid, file_name=name))
            for obj in range(index % 3+1):
                data['annotations'].append(dict(image_id=sid, segmentation=encoded(20+50*obj, 20)))
    annotations = root/'annotations.json'; write_json(annotations, data)
    plan = build_plan(data)
    plan['sources_sha256'] = dict(annotations=digest(annotations))
    plan_path = root/'plan.json'; write_json(plan_path, plan)
    model_source = 'a'*64
    for fold in range(5):
        folder = root/f'fold{fold}'; folder.mkdir()
        info = plan['fold_summaries'][fold]
        names = sorted(r['file_name'] for r in plan['observations'] if info['roles'][r['physical_identifier']] == 'validation')
        cfg = dict(DEFAULTS, fold=fold, amp_dtype='None')
        cfg['source_sha256'] = dict(annotations=digest(annotations), fold_plan=digest(plan_path),
            train_split=audit.split_sha(plan, info, 'train'), val_split=audit.split_sha(plan, info, 'validation'),
            base_runtime=model_source, cv_runner=model_source)
        payload = dict(settings={key: cfg[key] for key in IDENTITY_KEYS}, fold=fold,
            sources=cfg['source_sha256'], precision='None', torch_version='test-fixture', final_selection='fixed_epoch')
        cfg['cv_identity_payload'] = payload
        cfg['cv_identity'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        write_json(folder/'config.json', cfg)
        truth, mapping = audit.truth_for(data, set(names))
        predictions, records = {}, []
        for name in names:
            image = Path(name).stem
            gt = next(truth[sid] for sid in truth if mapping[sid] == image)
            original = [dict(size=[2048, 2048], counts=r['counts'].decode()) for r in gt]
            selected = (original if fold == 0 else [] if fold == 1 else original[:1]
                        if fold == 2 else original[:-1] if fold == 3 else original+[encoded(1700, 1700)])
            predictions[image] = selected
            # The rich export retains small components, but selected CSV min_area300 excludes them.
            values = [dict(area=int(mu.area(r)), fg_mean=.9, rle=r) for r in selected]
            values.append(dict(area=100, fg_mean=.8, rle=encoded(1900, 1900, 10)))
            records.append(dict(file_name=name, image_sha256='b'*64, checkpoint_sha256='c'*64,
                cv_identity=cfg['cv_identity'], fold=fold, selected_epoch=100, component_min_area=100,
                components={'0.5': values}))
        (folder/'oof_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        with (folder/'oof_predictions.csv').open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['filament_id', 'segmentation_rle'])
            writer.writerows((f'{image}_{i}', r['counts']) for image, values in sorted(predictions.items())
                             for i, r in enumerate(values, 1))
        summary = dict(fold=fold, status='complete', selected_epoch=100, epochs_completed=100,
            fixed_final_epoch=100, oof_score_available=True, diagnostic_scores_select_checkpoint=False,
            cv_identity=cfg['cv_identity'], source_sha256=cfg['source_sha256'],
            threshold=.5, min_area=300, image_names=names, image_coverage=len(names),
            checkpoint_sha256='c'*64, train_physical_identifiers=sorted(k for k, role in info['roles'].items() if role == 'train'),
            metric=metric(truth, predictions, mapping),
            oof_csv_sha256=digest(folder/'oof_predictions.csv'), oof_records_sha256=digest(folder/'oof_records.jsonl'))
        write_json(folder/'oof_summary.json', summary)
    return annotations, plan_path


def refresh_record_checksum(folder):
    path = folder/'oof_summary.json'; summary = json.loads(path.read_text())
    summary['oof_records_sha256'] = digest(folder/'oof_records.jsonl')
    write_json(path, summary)


class OOFTests(unittest.TestCase):
    def test_complete_oof_pools_counts_instead_of_mean_fold_pq(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); annotations, plan = fixture(root)
            result = audit.evaluate_oof(annotations, plan, root)
            self.assertTrue(result['complete_oof'])
            self.assertEqual(result['completed_physical_images'], 10)
            self.assertEqual(result['pooled_oof']['annotation_groups'], 11)
            tp, fp, fn, iou = [sum(r[key] for r in result['by_fold'])
                               for key in ('tp_pairs', 'fp', 'fn', 'matched_iou_sum')]
            self.assertAlmostEqual(result['pooled_oof']['pq'], iou/(tp+.5*(fp+fn)))
            self.assertNotAlmostEqual(result['pooled_oof']['pq'], sum(r['pq'] for r in result['by_fold'])/5)
            self.assertEqual(set(result['by_site']), {'bh', 'ch'})

    def test_partial_folds_are_not_advertised_as_complete_oof(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); annotations, plan = fixture(root)
            for fold in range(1, 5):
                (root/f'fold{fold}'/'oof_summary.json').unlink()
            result = audit.evaluate_oof(annotations, plan, root)
            self.assertEqual(result['status'], 'partial_folds')
            self.assertIsNone(result['pooled_oof'])
            self.assertEqual(result['completed_folds'], [0])
            self.assertGreater(result['pooled_completed_folds']['pq'], 0)

    def test_blank_prediction_image_must_still_have_coverage_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); annotations, plan = fixture(root)
            folder = root/'fold1'
            lines = (folder/'oof_records.jsonl').read_text().splitlines()
            (folder/'oof_records.jsonl').write_text('\n'.join(lines[1:])+'\n')
            refresh_record_checksum(folder)
            with self.assertRaisesRegex(ValueError, 'including empty predictions'):
                audit.evaluate_oof(annotations, plan, root)

    def test_tampered_checkpoint_or_csv_cannot_pass_with_refreshed_file_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); annotations, plan = fixture(root)
            folder = root/'fold0'; path = folder/'oof_records.jsonl'
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[0]['checkpoint_sha256'] = 'd'*64
            path.write_text(''.join(json.dumps(r)+'\n' for r in records)); refresh_record_checksum(folder)
            with self.assertRaisesRegex(ValueError, 'checkpoint/source provenance'):
                audit.evaluate_oof(annotations, plan, root)
            records[0]['checkpoint_sha256'] = 'c'*64
            path.write_text(''.join(json.dumps(r)+'\n' for r in records)); refresh_record_checksum(folder)
            csv_path = folder/'oof_predictions.csv'
            csv_path.write_text('filament_id,segmentation_rle\n')
            summary_path = folder/'oof_summary.json'; summary = json.loads(summary_path.read_text())
            summary['oof_csv_sha256'] = digest(csv_path); write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, 'CSV masks differ'):
                audit.evaluate_oof(annotations, plan, root)

    def test_nonfinal_or_overlapping_exports_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); annotations, plan = fixture(root)
            folder = root/'fold0'; summary_path = folder/'oof_summary.json'
            summary = json.loads(summary_path.read_text()); summary['selected_epoch'] = 99
            write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, 'fixed-final-epoch'):
                audit.evaluate_oof(annotations, plan, root)
            summary['selected_epoch'] = 100; write_json(summary_path, summary)
            path = folder/'oof_records.jsonl'
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[0]['components']['0.5'].append(records[0]['components']['0.5'][0])
            path.write_text(''.join(json.dumps(r)+'\n' for r in records)); refresh_record_checksum(folder)
            with self.assertRaisesRegex(ValueError, 'Overlapping/duplicate'):
                audit.evaluate_oof(annotations, plan, root)


if __name__ == '__main__':
    unittest.main()
