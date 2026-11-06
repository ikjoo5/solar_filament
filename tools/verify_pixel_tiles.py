"""Independent native PNG, source provenance, ROI coverage and reference audit.

This tool does not score predictions or alter refinement settings. Public JPEG
comparisons establish pixel equality only; they do not estimate performance.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy.interpolate import UnivariateSpline


SIZE = 2048


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def original_limb():
    script_file = Path(__file__).with_name('colab_paste_evaluate.py')
    outer = ast.parse(script_file.read_text())
    script = next(ast.literal_eval(node.value) for node in outer.body
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id == 'SCRIPT')
    names = {'yy', 'xx', 'rr', 'disk', 'radii', 'bins'}
    nodes = []
    for node in ast.parse(script).body:
        if isinstance(node, ast.Assign):
            targets = {n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)}
            if targets and targets <= names:
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == 'limb':
            nodes.append(node)
    namespace = dict(np=np, UnivariateSpline=UnivariateSpline)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<exact-baseline-limb>', 'exec'), namespace)
    return namespace['limb']


def read_predictions(path, images):
    groups = defaultdict(list); identifiers = set()
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle):
            fid = row['filament_id']; stem = fid.split('_', 1)[0]
            assert fid not in identifiers, f'Duplicate prediction ID: {fid}'
            identifiers.add(fid)
            assert stem in images, f'Prediction outside validation set: {stem}'
            rle = dict(size=[SIZE, SIZE], counts=row['segmentation_rle'].encode('ascii'))
            assert int(mu.area(rle)) > 0, f'Empty prediction: {fid}'
            x, y, w, h = map(int, mu.toBbox(rle))
            assert 0 <= x < x+w <= SIZE and 0 <= y < y+h <= SIZE
            groups[stem].append(dict(id=fid, rle=rle, bbox=[x, y, w, h]))
    return groups


def verify(args):
    cv2.setNumThreads(1)
    root = Path(args.pixels).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['schema_version'] == 1
    assert manifest['image_size'] == [SIZE, SIZE] and manifest['tile_size'] == 128
    with Path(args.valsplit).open(newline='', encoding='utf-8-sig') as handle:
        validation = list(csv.DictReader(handle))
    filenames = {Path(row['file_name']).stem: row['file_name'] for row in validation}
    assert set(filenames) == set(manifest['images']), 'Validation and manifest image sets differ'
    source_files = [args.annotation, args.checkpoint, args.trainsplit, args.valsplit]
    sources = {}
    for value in source_files:
        path = Path(value); actual = digest(path); expected = manifest['source_sha256'][path.name]
        assert actual == expected, f'Source differs: {path}'
        sources[path.name] = dict(path=str(path), expected_sha256=expected, actual_sha256=actual, matches=True)
    assert set(sources) == set(manifest['source_sha256'])
    prior_summaries = []
    for value in args.prior_summary:
        path = Path(value)
        prior = json.loads(path.read_text())['source_sha256']
        assert prior == manifest['source_sha256'], f'Prior cache provenance differs: {path}'
        prior_summaries.append(dict(path=str(path), sha256=digest(path), all_four_sources_match=True))
    selected_by_cache = {stem: set() for stem in filenames}
    selection_caches = []
    visited_caches = set()
    for value in args.prior_summary:
        for name, threshold in [('raw_views.jsonl', .15), ('raw_preprocess_views.jsonl', .25)]:
            path = Path(value).parent/name
            if not path.is_file() or path in visited_caches:
                continue
            visited_caches.add(path); records = candidates = 0
            for line in path.open(encoding='utf-8'):
                row = json.loads(line); stem = row['image']; records += 1
                assert stem in filenames and row['coordinate_space'] == 'original'
                assert row['shape'] == [SIZE, SIZE]
                for instance in row['instances']:
                    if instance['confidence'] < threshold:
                        continue
                    rle = dict(size=[SIZE, SIZE], counts=instance['rle'].encode('ascii'))
                    x, y, w, h = map(int, mu.toBbox(rle))
                    if not w or not h:
                        continue
                    candidates += 1
                    for top in range(max(0, (y-24)//128)*128, min(SIZE, y+h+24), 128):
                        for left in range(max(0, (x-24)//128)*128, min(SIZE, x+w+24), 128):
                            selected_by_cache[stem].add((top, left))
            selection_caches.append(dict(path=str(path), sha256=digest(path),
                confidence_threshold=threshold, records=records, qualifying_candidates=candidates))
    inputs = {}
    for value in args.prediction:
        label, path = value.split('=', 1)
        assert label and label not in inputs
        groups = read_predictions(path, filenames)
        inputs[label] = dict(path=path, sha256=digest(path), groups=groups,
            rows=sum(map(len, groups.values())), roi_bands={str(band): dict(
                band=band, padding=band+3, full_coverage=0, partial_coverage=0,
                missing_roi_pixels_total=0, partial_cases=[]) for band in (3, 5, 7)})
    public = {}; limb = None
    if args.public_reference:
        public_root = Path(args.public_reference)
        references = json.loads((public_root/'source_manifest.json').read_text())
        public = {Path(row['file_name']).stem: row for row in references['images']}
        assert set(public) <= set(filenames)
        limb = original_limb()
    declared_paths = set(); geometries = Counter(); image_reports = {}; reference_reports = []
    tile_count = covered_total = 0
    for stem, info in manifest['images'].items():
        assert info['file_name'] == filenames[stem] and Path(info['file_name']).name == info['file_name']
        assert re.fullmatch('[0-9a-f]{64}', info['raw_sha256'])
        assert re.fullmatch('[0-9a-f]{64}', info['limb_sha256'])
        gray = np.zeros((SIZE, SIZE), np.uint8); covered = np.zeros((SIZE, SIZE), bool)
        for tile in info['tiles']:
            relative = tile['path']; path = (root/relative).resolve()
            assert path.is_relative_to(root) and relative not in declared_paths
            declared_paths.add(relative)
            payload = path.read_bytes()
            assert payload[:8] == b'\x89PNG\r\n\x1a\n'
            assert hashlib.sha256(payload).hexdigest() == tile['sha256'], f'PNG SHA differs: {path}'
            pixels = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
            x, y, w, h = [tile[k] for k in ('x', 'y', 'width', 'height')]
            assert all(type(v) is int for v in (x, y, w, h))
            assert w == h == 128 and x % 128 == y % 128 == 0
            assert 0 <= x < x+w <= SIZE and 0 <= y < y+h <= SIZE
            assert relative == f'tiles/{stem}/{y}_{x}.png'
            assert pixels is not None and pixels.dtype == np.uint8 and pixels.shape == (h, w)
            region = np.s_[y:y+h, x:x+w]
            assert not covered[region].any(), f'Unexpected duplicate/overlapping tiles: {path}'
            gray[region] = pixels; covered[region] = True
            geometries[f'{w}x{h}'] += 1; tile_count += 1
        if selection_caches:
            actual_positions = {(tile['y'], tile['x']) for tile in info['tiles']}
            assert actual_positions == selected_by_cache[stem], f'Tile selection differs from raw candidate caches: {stem}'
        covered_pixels = int(covered.sum()); covered_total += covered_pixels
        image_reports[stem] = dict(file_name=info['file_name'], tiles=len(info['tiles']),
            covered_pixels=covered_pixels, image_fraction_covered=covered_pixels/(SIZE*SIZE))
        for label, entry in inputs.items():
            for row in entry['groups'].get(stem, []):
                x, y, w, h = row['bbox']
                for band, counts in entry['roi_bands'].items():
                    pad = counts['padding']
                    x1, y1 = max(0, x-pad), max(0, y-pad)
                    x2, y2 = min(SIZE, x+w+pad), min(SIZE, y+h+pad)
                    roi = covered[y1:y2, x1:x2]
                    missing = int(roi.size-roi.sum())
                    counts['missing_roi_pixels_total'] += missing
                    if missing:
                        counts['partial_coverage'] += 1
                        mask = mu.decode(row['rle']).astype(bool)
                        counts['partial_cases'].append(dict(filament_id=row['id'],
                            roi=[x1, y1, x2, y2], missing_roi_pixels=missing,
                            missing_original_foreground_pixels=int((mask & ~covered).sum())))
                    else:
                        counts['full_coverage'] += 1
        if stem in public:
            reference = public[stem]; source = public_root/'images'/reference['file_name']
            raw_hash = digest(source)
            assert raw_hash == reference['sha256'], f'Public reference source changed: {source}'
            native = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
            assert native is not None and native.shape == (SIZE, SIZE) and native.dtype == np.uint8
            processed = limb(native); processed_hash = hashlib.sha256(processed.tobytes()).hexdigest()
            delta = gray[covered].astype(np.int16)-processed[covered].astype(np.int16)
            reference_reports.append(dict(file_name=info['file_name'], compared_pixels=len(delta),
                export_raw_sha256=info['raw_sha256'], public_raw_sha256=raw_hash,
                raw_hash_equal=raw_hash == info['raw_sha256'],
                export_limb_sha256=info['limb_sha256'], public_limb_sha256=processed_hash,
                limb_hash_equal=processed_hash == info['limb_sha256'],
                covered_pixels_equal=int((delta == 0).sum()),
                covered_pixels_different=int((delta != 0).sum()),
                covered_mean_absolute_difference=float(np.mean(np.abs(delta))) if len(delta) else None,
                covered_max_absolute_difference=int(np.abs(delta).max()) if len(delta) else None))
    actual_paths = {str(p.relative_to(root)) for p in root.rglob('*.png')}
    assert actual_paths == declared_paths, 'PNG files outside manifest or missing declared PNGs'
    for entry in inputs.values():
        del entry['groups']
        for counts in entry['roi_bands'].values():
            assert counts['full_coverage']+counts['partial_coverage'] == entry['rows']
    report = dict(scope='Independent pixel integrity/provenance/ROI coverage audit; no PQ scoring',
        pixels_root=str(root), manifest_sha256=digest(root/'manifest.json'),
        validation_physical_images=len(filenames), validation_annotation_groups=len(validation),
        png_files=tile_count, png_sha256_validated=tile_count, png_native_geometry_validated=tile_count,
        png_geometries=dict(geometries), extra_or_missing_png_files=0,
        declared_native_pixels=len(filenames)*SIZE*SIZE, covered_native_pixels=covered_total,
        images_without_tiles=[s for s, info in image_reports.items() if info['tiles'] == 0],
        source_hashes=sources, prior_summaries=prior_summaries,
        candidate_only_tile_selection=dict(caches=selection_caches, padding=24, tile_size=128,
            all_image_tile_sets_match=bool(selection_caches), gt_mask_used=False),
        prediction_roi_coverage=inputs,
        images=image_reports, public_reference_comparisons=dict(
            scope='Covered pixel and complete limb-hash equality only; no performance inference',
            images=len(reference_reports), compared_pixels=sum(r['compared_pixels'] for r in reference_reports),
            raw_hash_equal_images=sum(r['raw_hash_equal'] for r in reference_reports),
            limb_hash_equal_images=sum(r['limb_hash_equal'] for r in reference_reports),
            covered_equal_pixels=sum(r['covered_pixels_equal'] for r in reference_reports),
            covered_different_pixels=sum(r['covered_pixels_different'] for r in reference_reports),
            rows=reference_reports), all_integrity_checks_passed=True)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('png_files', 'validation_physical_images', 'images_without_tiles',
        'prediction_roi_coverage', 'public_reference_comparisons', 'all_integrity_checks_passed')}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pixels', 'annotation', 'checkpoint', 'trainsplit', 'valsplit', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--prediction', action='append', required=True, help='label=path.csv')
    parser.add_argument('--prior-summary', action='append', default=[])
    parser.add_argument('--public-reference')
    verify(parser.parse_args())


if __name__ == '__main__':
    main()
