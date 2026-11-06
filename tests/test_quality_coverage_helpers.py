import sys
from pathlib import Path

import numpy as np
import contextlib
import unittest


@contextlib.contextmanager
def raises(exception):
    try:
        yield
    except exception:
        return
    raise AssertionError("Expected " + exception.__name__)
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from quality_coverage_sampling import select_training
from quality_append_api import append_q50


def candidates(counts):
    return [dict(target=[.6 if b < 2 else 0, 1 if b < 2 else 0], confidence=.1 if b % 2 == 0 else .8)
            for b, count in enumerate(counts) for _ in range(count)]


def sample(rows, epoch=1, **kw):
    return select_training(rows, epoch=epoch, seed=42, image="one.jpeg", role="train", **kw)


def test_deterministic_quota_cycle_and_resume():
    rows = candidates([60, 60, 60, 60])
    coverage = set()
    for epoch in range(1, 6):
        ids = sample(rows, epoch)
        assert ids == sample(rows, epoch)
        assert len(ids) == len(set(ids)) == 48
        assert [sum(b * 60 <= i < (b + 1) * 60 for i in ids) for b in range(4)] == [12] * 4
        assert coverage.isdisjoint(ids)
        coverage.update(ids)
    assert len(coverage) == 240
    assert sample(rows, 3) == sample(rows, 8)


def test_backfill_small_buckets_and_no_duplicates():
    for sizes in ([2, 3, 90, 0], [1, 2, 3, 4], [0, 0, 0, 0]):
        rows = candidates(sizes)
        for epoch in range(1, 10):
            selected = sample(rows, epoch)
            assert len(selected) == len(set(selected)) == min(48, len(rows))


def test_train_only():
    for role in ["validation", "test", None]:
        with raises(ValueError):
            select_training([], epoch=1, seed=1, image="a", role=role)


def test_invalid_targets_fail_closed():
    for row in [dict(target=[0, float("nan")], confidence=.1),
                dict(target=[0, 1], confidence=float("nan")), dict(target=[2, 1], confidence=.1),
                dict(target=[0, -1], confidence=.1), dict(confidence=.1)]:
        with raises(ValueError):
            sample([row])


def rect(x, y, w=20, h=20):
    pixels = np.zeros((2048, 2048), np.uint8)
    pixels[y:y+h, x:x+w] = 1
    return mu.encode(np.asfortranarray(pixels))


def inputs(masks, confidence=.5):
    raw = [dict(rle=m["counts"].decode(), confidence=confidence, target="must be ignored") for m in masks]
    records = [dict(view=v, instances=[]) for v in ("base", "scale", "hflip", "vflip")]
    for i, row in enumerate(raw):
        records[i % 4]["instances"].append(row)
    return raw, records


def test_two_distinct_masks_and_views_accept_only_one_overlapping_candidate():
    raw, geometry = inputs([rect(100, 100), rect(101, 100)])
    output, stats, trace = append_q50([], raw, [.8, .7], geometry)
    assert len(output) == 1 and stats["accepted"] == 1
    assert stats["reject_total_clipping"] == 1
    assert trace[0]["source"] == 0
    assert trace[0]["support"]["unique_masks"] == 2


def test_identical_rle_multiple_views_not_a_second_vote():
    raw, geometry = inputs([rect(100, 100)])
    geometry[1]["instances"] = [raw[0]]
    output, stats, _ = append_q50([], raw, [.8], geometry)
    assert output == [] and stats["reject_geometry_quorum"] == 1


def test_frozen_thresholds_and_empty_support():
    raw, geometry = inputs([rect(100, 100), rect(101, 100)], confidence=.09)
    assert append_q50([], raw, [.8, .7], geometry)[1]["reject_confidence"] == 2
    raw, geometry = inputs([rect(100, 100), rect(101, 100)])
    assert append_q50([], raw, [.49, .49], geometry)[1]["reject_quality"] == 2


def test_baseline_preserved_and_small_overlap_clipped():
    original = rect(80, 100, 21, 20)
    original["counts"] = original["counts"].decode()
    saved = dict(original)
    raw, geometry = inputs([rect(100, 100), rect(101, 100)])
    output, stats, trace = append_q50([original], raw, [.8, .7], geometry)
    assert output[0] is original and original == saved
    assert stats["accepted"] == 1 and trace[0]["clipped_fraction"] == .05
    normalized = [dict(size=m["size"], counts=m["counts"].encode() if isinstance(m["counts"], str) else m["counts"]) for m in output]
    assert int(mu.area(mu.merge(normalized))) == sum(int(mu.area(m)) for m in normalized)


def test_mismatched_score_alignment_and_geometry_rejected():
    raw, geometry = inputs([rect(100, 100)])
    with raises(ValueError):
        append_q50([], raw, [], geometry)
    bad = [dict(raw[0], confidence=.6)]
    with raises(ValueError):
        append_q50([], bad, [.8], geometry)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for key, value in list(globals().items()) if key.startswith("test_"))


if __name__ == "__main__":
    suite = load_tests(None, None, None)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
