"""Deterministic train-only candidate exposure; labels never enter inference.

Four strata split positive/negative targets and confidence below/above .2.
Each epoch advances a fixed shuffled cycle in each stratum. At cap48 the
primary quota is12 per stratum; vacant slots are filled without duplicates.
The sampler changes exposure only, not the model's target or objective.
"""
import hashlib
import math
import random


def _probability(value, field):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid " + field) from exc
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("Invalid " + field + ": expected finite probability")
    return result


def select_training(instances, *, epoch, seed, image, role, cap=48):
    """Return sorted raw indices for one training image and one 1-based epoch.

    The primary cycle covers a stratum of N candidates after ceil(N/quota)
    epochs (when quota>0); backfill may expose it sooner. Resume needs only
    the same seed, image identity, candidate order and epoch. Invalid labels
    fail closed rather than silently being assigned to a negative stratum.
    """
    if role != "train":
        raise ValueError("Coverage sampling is permitted only for role='train'")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 4:
        raise ValueError("cap must be an integer >=4")
    if not isinstance(image, str) or not image:
        raise ValueError("image must be a nonempty stable identity")
    buckets = [[] for _ in range(4)]
    for i, row in enumerate(instances):
        target = row.get("target")
        if target is None or len(target) != 2:
            raise ValueError("Candidate target must have two probabilities")
        _probability(target[0], "IoU target")
        positive = _probability(target[1], "match target") > 0
        low = _probability(row.get("confidence"), "confidence") < .2
        bucket = (0 if positive else 2) + (0 if low else 1)
        buckets[bucket].append(i)
    quotas = [cap // 4 + int(i < cap % 4) for i in range(4)]
    cursors = []
    selected = set()
    for b, indices in enumerate(buckets):
        token = (str(seed) + "\0" + image + "\0" + str(b)).encode("utf-8")
        random.Random(int.from_bytes(hashlib.sha256(token).digest(), "big")).shuffle(indices)
        start = ((epoch - 1) * quotas[b]) % len(indices) if indices else 0
        cycle = indices[start:] + indices[:start]
        cursors.append(iter(cycle))
        for _ in range(min(quotas[b], len(indices))):
            selected.add(next(cursors[b]))
    # Rotate the backfill priority across epochs; primary cycles remain fixed.
    target_count = min(cap, len(instances))
    while len(selected) < target_count:
        progressed = False
        for step in range(4):
            b = ((epoch - 1) + step) % 4
            candidate = next(cursors[b], None)
            if candidate is not None:
                selected.add(candidate)
                progressed = True
            if len(selected) == target_count:
                break
        if not progressed:
            raise RuntimeError("Coverage sampler exhausted before filling cap")
    return sorted(selected)
