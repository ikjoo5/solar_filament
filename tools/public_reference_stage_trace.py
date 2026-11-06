"""CPU replay of frozen intermediate masks for external-prediction diagnostics.

This module never runs a model or reads annotations. It checks that the recipe
source files match the inference manifest, then reproduces the cached baseline
exactly before exposing intermediate masks. A reference-match transition means
geometric agreement changed; it is not an object-identity or correctness label.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path


RECIPE_FILES = ('submission_yolo_recipe.py', 'submission_native_recipe.py')
STAGE_ORDER = ('yolo_consensus', 'yolo_preprocessing_rescue', 'yolo_smooth',
               'yolo_tiled', 'native_reference', 'native_rich_refined', 'baseline_final')


def verify_recipe_sources(expected_source_sha256):
    """Require both sibling recipe files to match the recorded inference run."""
    if not isinstance(expected_source_sha256, dict):
        raise ValueError('Intermediate replay requires protocol.sources SHA-256 values')
    actual = {}
    for name in RECIPE_FILES:
        expected = expected_source_sha256.get(name)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError('Missing recorded recipe SHA-256: '+name)
        path = Path(__file__).resolve().with_name(name)
        actual[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual[name] != expected:
            raise ValueError('Intermediate replay source differs from inference: '+name)
    return actual


def _rle_key(mask):
    if not isinstance(mask, dict) or mask.get('size') != [2048, 2048]:
        raise ValueError('Intermediate replay expects original 2048-square RLE masks')
    counts = mask.get('counts')
    if isinstance(counts, str):
        counts = counts.encode('ascii')
    if not isinstance(counts, bytes) or not counts:
        raise ValueError('Invalid intermediate compressed RLE')
    return counts


def replay_stages(stem, stages, baseline_masks, expected_source_sha256=None):
    """Return seven ordered stage -> RLE-list entries, after exact final parity.

``stages`` contains the validated raw cache data for geometric, preprocessing,
tiled and native. ``baseline_masks`` is the same-image cached baseline. The
caller owns cache envelope/RLE validation and provenance checks. Recipe source
hashes are mandatory, supplied from ``protocol['sources']``. R1 is deliberately
excluded: callers compare the already saved final cache as the next stage.
    """
    before = verify_recipe_sources(expected_source_sha256)
    required = {'geometric', 'preprocessing', 'tiled', 'native'}
    if not isinstance(stages, dict) or not required.issubset(stages):
        raise ValueError('Intermediate replay requires all four raw inference stages')
    modules = {}
    for name in RECIPE_FILES:
        module = importlib.import_module(Path(name).stem)
        if Path(module.__file__).resolve() != Path(__file__).resolve().with_name(name):
            raise ValueError('Recipe import resolved outside the audit bundle: '+name)
        modules[name] = module
    yolo = modules['submission_yolo_recipe.py'].build_yolo_reference(
        stem, stages['geometric'], stages['preprocessing'], stages['tiled'], return_stages=True)
    native = modules['submission_native_recipe.py'].build_native_submission(
        stem, yolo['tiled'], stages['geometric'], stages['preprocessing'],
        stages['tiled'], stages['native'], return_stages=True)
    result = dict(zip(STAGE_ORDER, (
        yolo['consensus'], yolo['preprocessing_rescue'], yolo['smooth'], yolo['tiled'],
        native['native_reference'], native['rich_refined'], native['final'])))
    for masks in result.values():
        for mask in masks:
            _rle_key(mask)
    if [_rle_key(mask) for mask in result['baseline_final']] != [
            _rle_key(mask) for mask in baseline_masks]:
        raise ValueError('Intermediate replay does not reproduce the cached baseline RLE sequence')
    if verify_recipe_sources(expected_source_sha256) != before:
        raise ValueError('Recipe source changed during intermediate replay')
    return result
