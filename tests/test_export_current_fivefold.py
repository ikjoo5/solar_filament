"""Stopped checkpoint provenance, no-training resume, and honest OOF budgets."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import shutil

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_current_fivefold as current
from test_fivefold_study_evaluation import encoded, fixture, signed_record


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class StoppedSnapshotTests(unittest.TestCase):
    def test_compact_preserves_actual_epoch_and_rejects_history_or_identity_changes(self):
        model = torch.nn.Linear(2, 1)
        history = [dict(epoch=i) for i in range(1, 26)]
        payload = dict(cv_epoch=25, epoch=24, cv_identity='owned', cv_history=history,
                       ema=model, optimizer={'old': 'state'}, cv_rng={'a': 1})
        compact, epoch = current.compact_payload(payload, dict(identity='owned'), history)
        self.assertEqual((epoch, compact['cv_epoch'], compact['epoch']), (25, 25, 24))
        self.assertIs(compact['ema'], model)
        self.assertIsNone(compact['optimizer'])
        self.assertEqual(payload['optimizer'], {'old': 'state'})
        with self.assertRaisesRegex(ValueError, 'history'):
            current.compact_payload(payload, dict(identity='owned'), history[:-1])
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            current.compact_payload(payload, dict(identity='other'), history)

    def test_last_completed_yolo_snapshot_leaves_original_unchanged_and_resumes_without_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);source = root/'fold4/yolo';output = root/'snapshot'
            source.mkdir(parents=True);output.mkdir()
            sources = {k: k for k in ('annotations', 'fold_plan', 'train_split', 'val_split', 'base_runtime')}
            settings = dict(fold=4, architecture='yolo26s-seg.pt', sources=sources)
            identity = current.fingerprint(settings)
            save(source/'config.json', dict(settings=settings, identity=identity))
            history = [dict(epoch=i) for i in range(1, 26)]
            save(source/'history.json', history)
            payload = dict(cv_identity=identity, cv_epoch=25, epoch=24, cv_history=history,
                           ema=torch.nn.Linear(2, 1), optimizer={'a': torch.tensor(3)})
            torch.save(payload, source/'last_full.pt')
            originals = {p.name: p.read_bytes() for p in source.iterdir()}
            native = dict(training_config=dict(source_sha256=sources))
            result = current.select_yolo(root, native, 4, 30, output)
            self.assertEqual(result['epoch'], 25)
            compact = torch.load(result['checkpoint'], weights_only=False)
            self.assertEqual(compact['cv_epoch'], 25)
            self.assertIsNone(compact['optimizer'])
            self.assertEqual(originals, {p.name: p.read_bytes() for p in source.iterdir()})
            with patch.object(current.torch, 'load', side_effect=AssertionError('Unnecessary checkpoint reload')):
                self.assertEqual(current.select_yolo(root, native, 4, 30, output), result)
            changed = dict(payload, cv_epoch=26)
            torch.save(changed, source/'last_full.pt')
            with self.assertRaisesRegex(ValueError, 'changed since snapshot'):
                current.select_yolo(root, native, 4, 30, output)

    def test_mixed_budget_oof_is_explicit_and_pools_counts_not_fold_pq_means(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);annotations, plan_path = fixture(root)
            data = json.loads(annotations.read_text());plan = json.loads(plan_path.read_text())
            records = {};summaries = []
            for info in plan['fold_summaries']:
                fold = info['fold']
                names = [r['file_name'] for r in plan['observations']
                         if info['roles'][r['physical_identifier']] == 'validation']
                records[fold] = [dict(file_name=name, masks=dict(full_selected=[encoded()] if fold else [])) for name in names]
                summaries.append(dict(fold=fold, yolo_epoch=25 if fold == 4 else 30))
            result = current.evaluate_records(data, records, summaries, 'study', 30)
            self.assertFalse(result['equal_training_budgets'])
            self.assertFalse(result['same_budget_30epoch_oof'])
            self.assertEqual(result['yolo_epochs']['4'], 25)
            measured = result['results']['full_selected']
            total = measured['pooled_current_checkpoint_oof']
            by_fold = measured['by_fold'].values()
            self.assertEqual(total['tp_pairs'], sum(row['tp_pairs'] for row in by_fold))
            self.assertAlmostEqual(total['pq'], total['matched_iou_sum']/(total['tp_pairs']+.5*(total['fp']+total['fn'])))
            records[4][0]['file_name'] = records[0][0]['file_name']
            with self.assertRaisesRegex(ValueError, 'multiple folds'):
                current.evaluate_records(data, records, summaries, 'study', 30)

    def test_export_resumes_predictions_without_repeating_model_inference_and_detects_tampering(self):
        import cv2
        import numpy as np
        from test_fivefold_full_pipeline import EmptyProvider
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);images = root/'images';images.mkdir();output = root/'evaluation';output.mkdir()
            name = '20110101000000Ch.jpeg'
            cv2.imwrite(str(images/name), np.full((2048, 2048), 100, np.uint8))
            rich = dict(file_name=name, image_sha256=current.digest(images/name))
            native = dict(fold=4, study_identity='study', model_sha256='a'*64, rich_predictions_sha256='b'*64)
            yolo = dict(epoch=25, checkpoint='not_loaded', checkpoint_sha256='c'*64)
            sources = {'submission_inference.py': 'd'*64}
            masks = dict(full_selected=[encoded()])
            with patch.object(current, 'compose', return_value=masks):
                rows, summary = current.export_fold(root, images, native, yolo, [rich], output, 30,
                                                     'cpu', sources, _provider=EmptyProvider())
            self.assertFalse(summary['reused_completed_full_pipeline'])
            with patch.object(EmptyProvider, 'geometric', side_effect=AssertionError('Repeated inference')):
                rows2, _ = current.export_fold(root, images, native, yolo, [rich], output, 30,
                                               'cpu', sources, _provider=EmptyProvider())
            self.assertEqual(rows, rows2)
            path = output/'fold4/records'/f'{Path(name).stem}.json'
            row = json.loads(path.read_text());row['yolo_epoch'] = 30;save(path, row)
            with self.assertRaisesRegex(ValueError, 'cache changed'):
                current.export_fold(root, images, native, yolo, [rich], output, 30, 'cpu', sources,
                                    _provider=EmptyProvider())

    def test_package_contains_new_raw_caches_but_no_weights_or_native_rich_duplication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);output = root/'current_epoch_evaluation/test';output.mkdir(parents=True)
            save(output/'fold4/yolo_records/test.json.gz', dict(example='new raw cache placeholder'))
            save(output/'fold4/summary.json', dict(fold=4))
            (output/'fold4/yolo_snapshot.pt').write_bytes(b'weights excluded')
            save(output/'evaluation.json', dict(status='test'))
            (output/'colab_execution.log').write_text('mutable active log excluded')
            (output/'result.json').write_text('{}')
            (output/'unfinished.json.pending').write_text('{}')
            archives = current.package(output, root, [dict(fold=4)])
            with zipfile.ZipFile(archives[0]) as archive:
                self.assertIn('fold4/yolo_records/test.json.gz', archive.namelist())
                self.assertIn('archive_manifest.json', archive.namelist())
                self.assertFalse(any(n.endswith('.pt') for n in archive.namelist()))
                self.assertFalse(any('rich_records' in n for n in archive.namelist()))
                self.assertNotIn('colab_execution.log', archive.namelist())
                self.assertNotIn('result.json', archive.namelist())
                self.assertNotIn('unfinished.json.pending', archive.namelist())

    def test_complete_run_reuses_four_completed_folds_exports_stopped_fifth_and_never_trains(self):
        import cv2
        import numpy as np
        import fivefold_full_pipeline as full
        import grouped_cv_training
        from test_fivefold_full_pipeline import EmptyProvider
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);annotations, plan = fixture(root)
            shutil.copyfile(plan, root/'fold_plan.json')
            images = root/'images';images.mkdir()
            summaries = []
            for fold in range(5):
                stage = root/f'fold{fold}/stage030'
                summary = json.loads((stage/'stage_summary.json').read_text())
                path = stage/'rich_records.jsonl'
                records = [json.loads(line) for line in path.read_text().splitlines()]
                for record in records:
                    cv2.imwrite(str(images/record['file_name']), np.full((2048, 2048), 100, np.uint8))
                    record['image_sha256'] = current.digest(images/record['file_name'])
                    record['size'] = [2048, 2048]
                    signed_record(record)
                path.write_text(''.join(json.dumps(r)+'\n' for r in records))
                exported = json.loads((stage/'export_summary.json').read_text())
                exported.update(rich_predictions_sha256=current.digest(path),
                                records_sha256={r['file_name']: r['record_sha256'] for r in records})
                save(stage/'export_summary.json', exported)
                summary.update(rich_predictions_sha256=current.digest(path),
                               export=dict(rich_predictions=str(path)),
                               export_summary_sha256=current.digest(stage/'export_summary.json'))
                save(stage/'stage_summary.json', summary);summaries.append(summary)
                sources = {key: summary['training_config']['source_sha256'][key]
                           for key in ('annotations', 'fold_plan', 'train_split', 'val_split', 'base_runtime')}
                settings = dict(fold=fold, architecture='yolo26s-seg.pt', sources=sources)
                signature = current.fingerprint(settings)
                yolo = root/f'fold{fold}/yolo';yolo.mkdir()
                save(yolo/'config.json', dict(settings=settings, identity=signature))
                if fold != 4:
                    weight = stage/'yolo.pt';weight.write_bytes(b'fixture inference checkpoint')
                    save(yolo/'stage030.json', dict(identity=signature, fold=fold, stage_epoch=30,
                         checkpoint=str(weight), checkpoint_sha256=current.digest(weight)))
                else:
                    history = [dict(epoch=i) for i in range(1, 26)]
                    save(yolo/'history.json', history)
                    torch.save(dict(cv_identity=signature, cv_epoch=25, epoch=24, cv_history=history,
                                    ema=torch.nn.Linear(2, 1), optimizer={'a': torch.tensor(3)}), yolo/'last_full.pt')
            save(root/'stage030_progress.json', dict(study_identity=summaries[0]['study_identity'],
                 stage_epoch=30, completed_folds=list(range(5)), summaries=summaries))
            for fold in range(4):
                original = full.run_export(dict(study_root=str(root), stage_epoch=30, fold=fold,
                                                images=str(images), device='cpu'), _provider=EmptyProvider())
                save(root/f'fold{fold}/stage030/full_export_result.json', original)
            before = {str(p.relative_to(root)): current.digest(p) for p in root.rglob('*') if p.is_file()}
            output = root/'current_epoch_evaluation/stopped'
            with patch.object(grouped_cv_training, 'run', side_effect=AssertionError('Training entry was called')):
                result = current.run(root, images, annotations, output, device='cpu', _provider=EmptyProvider())
            self.assertEqual(result['yolo_epochs'], {'0': 30, '1': 30, '2': 30, '3': 30, '4': 25})
            self.assertFalse(result['equal_training_budgets'])
            self.assertFalse(result['training_executed'])
            report = json.loads((output/'evaluation.json').read_text())
            self.assertEqual(report['physical_images'], 10)
            self.assertEqual([s['reused_completed_full_pipeline'] for s in report['fold_summaries']], [True]*4+[False])
            self.assertEqual(set(report['results']), set(full.METHODS))
            self.assertTrue((output/'execution_source/export_current_fivefold.py').is_file())
            self.assertEqual(before, {name: current.digest(root/name) for name in before})
            with patch.object(EmptyProvider, 'geometric', side_effect=AssertionError('Repeated completed inference')):
                again = current.run(root, images, annotations, output, device='cpu', _provider=EmptyProvider())
            self.assertEqual(again['yolo_epochs'], result['yolo_epochs'])


if __name__ == '__main__':
    unittest.main()
