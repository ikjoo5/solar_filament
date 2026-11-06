"""Summarize the frozen216/102 external-reference panels from an existing audit.

No model inference, mask modification, ground truth or training is performed.
Missing raw-stage observations remain unavailable, rather than failed detections.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

REFERENCE_SHA = '0ab3932122a563ba96f4d243934d5c00680dd49b9d1165912a4f180907d2c603'
PANELS = {
    216: 'aada8589533eabe42f4260e891ed640d79bd0204e246536c5798dc3e7730fe3f',
    102: '5bc6e884bc9fdf5111a210911f5497b7d4593ec58bf3e7fc889818bbf7bc55e0',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_panel(path, count):
    config = json.loads(Path(path).read_text())
    rows = sorted((dict(image=r['image'], reference_id=r['reference_id'])
                   for r in config['targets']), key=lambda r: r['reference_id'])
    if config.get('count') != count or len(rows) != count or len({r['reference_id'] for r in rows}) != count:
        raise ValueError('Wrong panel size or duplicate target')
    # The102 identity also pins which historical mask was replaced.
    identity_rows = sorted(config['targets'], key=lambda r: r['reference_id']) if count == 102 else rows
    digest = hashlib.sha256(json.dumps(identity_rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != PANELS[count] or config.get('target_set_sha256') != PANELS[count]:
        raise ValueError('Frozen panel identity changed')
    if config.get('reference_csv_sha256', config.get('source_public_sha256')) != REFERENCE_SHA:
        raise ValueError('Frozen panel reference source changed')
    return rows


def number(value):
    if value in ('', None):
        return None
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Audit IoU outside [0, 1]')
    return value


def summarize(values):
    available = [v for v in values if v is not None]
    matched = sum(v > .5 for v in available)
    return dict(target_count=len(values), available_count=len(available),
                unavailable_count=len(values)-len(available),
                matched_count=matched if available else None,
                coverage=matched/len(values) if len(available) == len(values) and values else None,
                mean_max_iou=statistics.mean(available) if available else None,
                median_max_iou=statistics.median(available) if available else None)


def panel_summary(rows):
    stages = {name: summarize([number(r.get(name+'_max_iou')) for r in rows])
              for name in ('raw', 'baseline', 'r1')}
    for label in ('baseline', 'r1'):
        pairs = [(number(r.get('raw_max_iou')), number(r.get(label+'_max_iou'))) for r in rows]
        stages['raw_present_'+label+'_missing_count'] = (
            sum(a > .5 and b <= .5 for a, b in pairs)
            if all(a is not None and b is not None for a, b in pairs) else None)
    stages['raw_best_source_counts'] = dict(Counter(r['raw_best_source'] for r in rows if r.get('raw_best_source')))
    stages['similarity_loss_transition_counts'] = dict(Counter(
        token for r in rows for token in set(filter(None, r.get('similarity_loss_transitions', '').split('|')))))
    stages['similarity_gain_transition_counts'] = dict(Counter(
        token for r in rows for token in set(filter(None, r.get('similarity_gain_transitions', '').split('|')))))
    return stages


def run(audit_csv, audit_summary, panel216, panel102, output_root):
    paths = [Path(p) for p in (audit_csv, audit_summary, panel216, panel102)]
    snapshot = {str(p.resolve()): sha(p) for p in paths}
    audit = json.loads(Path(audit_summary).read_text())
    if REFERENCE_SHA not in audit.get('input_sha256', {}).values():
        raise ValueError('Audit summary does not identify the pinned public source')
    with Path(audit_csv).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not {'image', 'reference_id', 'baseline_max_iou'}.issubset(fields or []):
            raise ValueError('Not a reference_instances.csv audit')
        by_id = {}
        for row in reader:
            if row['reference_id'] in by_id:
                raise ValueError('Duplicate audit reference ID')
            by_id[row['reference_id']] = row
    selected = {}
    for count, path in ((216, panel216), (102, panel102)):
        selected[count] = []
        for target in load_panel(path, count):
            if target['reference_id'] not in by_id:
                raise ValueError('Missing panel target: '+target['reference_id'])
            row = by_id[target['reference_id']]
            if row['image'] != target['image']:
                raise ValueError('Audit target image mismatch')
            if number(row['baseline_max_iou']) is None:
                raise ValueError('Missing baseline IoU')
            if str(row.get('excluded_placeholder', '')).lower() == 'true':
                raise ValueError('Panel unexpectedly contains placeholder')
            selected[count].append(row)
    result = dict(schema_version=1, prediction_origin='own_model',
        scope='External prediction agreement only; not test PQ, recall, TP, FP or FN.',
        rule='Strict mask IoU > 0.5; fixed216 and102 reference panels, no inference modification',
        transition_scope='Reference-similarity transitions, not proof of the cause of candidate rejection',
        summary_claims_source_identity_only=True, audit_csv_authenticated_by_summary=False,
        cache_identity=audit.get('cache_identity'), input_sha256=snapshot,
        panel216=panel_summary(selected[216]), panel102=panel_summary(selected[102]))
    for path, digest in snapshot.items():
        if sha(path) != digest:
            raise ValueError('Input changed during panel audit')
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=False)
    for count, rows in selected.items():
        path = output/f'reference{count}.csv'
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    result['source_sha256'] = sha(__file__)
    result['output_sha256'] = {p.name: sha(p) for p in sorted(output.glob('*.csv'))}
    (output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audit-csv', 'audit-summary', 'panel216', 'panel102', 'output-root'):
        parser.add_argument('--'+name, required=True, type=Path)
    result = run(**vars(parser.parse_args()))
    for count in PANELS:
        panel = result[f'panel{count}']
        parts = [f"{stage} {panel[stage]['matched_count']}/{count} (확인 {panel[stage]['available_count']}개)"
                 for stage in ('raw', 'baseline', 'r1')]
        print(f'참조 {count}개: '+', '.join(parts), flush=True)
    print('공개 예측과의 일치율이며 실제 test PQ/recall이 아닙니다.', flush=True)


if __name__ == '__main__':
    main()
