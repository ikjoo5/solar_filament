"""Reference-panel reports must preserve missing-data and identity semantics."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'tools'))
import reference_panel_audit as audit


class PanelAuditTests(unittest.TestCase):
    def test_strict_threshold_and_unknown(self):
        self.assertEqual(audit.summarize([.5, .50001, 0])['matched_count'], 1)
        self.assertIsNone(audit.summarize([None, None])['matched_count'])
        self.assertIsNone(audit.summarize([None, .7])['coverage'])
        self.assertEqual(audit.summarize([0, 0])['coverage'], 0)

    def test_partial_raw_does_not_claim_zero_missed(self):
        result = audit.panel_summary([
            dict(raw_max_iou='', baseline_max_iou='0', r1_max_iou=''),
            dict(raw_max_iou='.8', baseline_max_iou='.2', r1_max_iou='.5',
                 similarity_loss_transitions='a->b|a->b')])
        self.assertIsNone(result['raw_present_baseline_missing_count'])
        self.assertIsNone(result['raw_present_r1_missing_count'])
        self.assertEqual(result['similarity_loss_transition_counts'], {'a->b': 1})

    def test_bad_values_rejected(self):
        for value in ('nan', 'inf', '-0.1', '1.01'):
            with self.assertRaises(ValueError):
                audit.number(value)

    def test_frozen_panel_mutation_rejected(self):
        config = json.loads((REPO/'configs/reference102.json').read_text())
        config['targets'][0]['reference_id'] = 'changed_1'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'panel.json'; path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                audit.load_panel(path, 102)

    def test_output_and_missing_target_validation(self):
        panel216 = REPO/'configs/reference216.json'; panel102 = REPO/'configs/reference102.json'
        targets = audit.load_panel(panel216, 216)+audit.load_panel(panel102, 102)
        rows = [dict(image=t['image'], reference_id=t['reference_id'], baseline_max_iou='0',
                     raw_max_iou='', r1_max_iou='') for t in targets]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); summary = root/'summary.json'; csvpath = root/'audit.csv'
            summary.write_text(json.dumps(dict(input_sha256={'reference': audit.REFERENCE_SHA})))
            def write(selected):
                with csvpath.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(selected)
            write(rows)
            result = audit.run(csvpath, summary, panel216, panel102, root/'good')
            self.assertEqual(result['panel216']['baseline']['matched_count'], 0)
            self.assertIsNone(result['panel102']['raw']['matched_count'])
            self.assertTrue((root/'good/reference102.csv').is_file())
            write(rows[:-1])
            with self.assertRaisesRegex(ValueError, 'Missing panel target'):
                audit.run(csvpath, summary, panel216, panel102, root/'bad')
            self.assertFalse((root/'bad').exists())


if __name__ == '__main__':
    unittest.main()
