"""Fixed external-reference agreement is neither actual recall nor copied-model skill."""
from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest

import numpy as np
from pycocotools import mask as mu

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
import track_reference216 as track
from filament_eval import Filament


def instance(identifier,width=10):
    pixels=np.zeros((2048,2048),np.uint8);pixels[5:15,5:5+width]=1
    mask=mu.encode(np.asfortranarray(pixels));stem=identifier.split('_')[0]
    return Filament(identifier,stem,stem,mask['counts'].decode(),int(mu.area(mask)))


def audit_file(path,targets,overrides=None):
    rows=[dict(image=t['image'],reference_id=t['reference_id'],baseline_max_iou='0',raw_max_iou='',r1_max_iou='') for t in targets]
    if overrides:
        for index,value in overrides.items():rows[index].update(value)
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


class Reference216Tests(unittest.TestCase):
    def test_strict_half_iou_and_empty_predictions(self):
        target=instance('a_1');half=instance('a_2',width=5)
        row=track.agreement([target],[half])[0]
        self.assertEqual(row['max_iou'],.5);self.assertFalse(row['matched'])
        blank=track.agreement([target],[])[0]
        self.assertEqual(blank['max_iou'],0)

    def test_missing_is_not_zero(self):
        unknown=track.summarize_scores([None,None]);blank=track.summarize_scores([0.,0.])
        self.assertIsNone(unknown['coverage']);self.assertIsNone(unknown['matched_count'])
        self.assertEqual(blank['coverage'],0);self.assertEqual(blank['matched_count'],0)

    def test_gained_and_lost_are_identity_matched(self):
        old=[dict(reference_id='a',max_iou=.6),dict(reference_id='b',max_iou=.5)]
        new=[dict(reference_id='b',max_iou=.8),dict(reference_id='a',max_iou=.49)]
        result=track.changes(old,new)
        self.assertEqual(result['gained_reference_ids'],['b']);self.assertEqual(result['lost_reference_ids'],['a'])

    def test_fixed_panel_mutation_and_duplicate_rejected(self):
        config=json.loads(track.DEFAULT_TARGETS.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'targets.json'
            changed=json.loads(json.dumps(config));changed['targets'][0]['reference_id']='wrong_1'
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'identity changed'):track.load_targets(path)
            changed['targets'][0]=changed['targets'][1];path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'Duplicate'):track.load_targets(path)

    def test_audit_absence_image_and_duplicates_rejected(self):
        targets=track.load_targets(track.DEFAULT_TARGETS)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'audit.csv'
            audit_file(path,targets[:-1])
            with self.assertRaisesRegex(ValueError,'Missing fixed216'):track.load_audit(path,targets)
            audit_file(path,targets,{0:dict(image='wrong')})
            with self.assertRaisesRegex(ValueError,'image mismatch'):track.load_audit(path,targets)
            audit_file(path,targets+[targets[0]])
            with self.assertRaisesRegex(ValueError,'Duplicate audit'):track.load_audit(path,targets)

    def test_audit_mode_preserves_unknown_and_labels_hybrid(self):
        targets=track.load_targets(track.DEFAULT_TARGETS)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'audit.csv'
            audit_file(path,targets,{0:dict(baseline_max_iou='.9')})
            before=path.read_bytes()
            result=track.run(root/'out',origin='hybrid',audit_csv=path)
            self.assertFalse(result['model_recovery_interpretation_allowed'])
            self.assertEqual(result['audit']['baseline']['matched_count'],1)
            self.assertEqual(result['audit']['baseline']['coverage'],1/216)
            self.assertIsNone(result['audit']['raw']['coverage'])
            self.assertIsNone(result['audit']['raw_present_baseline_missing_count'])
            self.assertEqual(before,path.read_bytes())

    def test_explicit_origin_completeness_and_not_measured(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaisesRegex(ValueError,'Explicit origin'):
                track.run(root/'bad',origin='')
            with self.assertRaisesRegex(ValueError,'complete-test-inference'):
                track.run(root/'bad',origin='own_model',prediction_csv=root/'missing.csv')
            result=track.run(root/'out',origin='own_model',not_measured=True)
            self.assertEqual(result['measurement_status'],'not_measured')
            self.assertIsNone(result['prediction']['coverage'])
            self.assertIsNone(result['prediction']['matched_count'])

    def test_safe_rle_csv_rejects_duplicate_and_truncation(self):
        mask=instance('a_1')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pred.csv'
            with path.open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(['filament_id','segmentation_rle'])
                writer.writerow(['a_1',mask.counts]);writer.writerow(['a_1',mask.counts])
            with self.assertRaisesRegex(ValueError,'Duplicate prediction'):track.read_masks(path)
            path.write_text('filament_id,segmentation_rle\na_1,1\n')
            with self.assertRaises(ValueError):track.read_masks(path)


if __name__=='__main__':unittest.main()
