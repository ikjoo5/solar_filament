"""Label-free, cardinality-preserving paired mask editor.

Requires 102 features in the exported candidate-context schema. The feature
extractor requires images; compressed RLE-only public submissions are not enough.
No annotation, filename, model family or leaderboard score is an input feature.
"""
import numpy as np
from pycocotools import mask as mu

SHAPE = (2048, 2048)
PAIR_COLUMNS = ['mask_iou', 'intersection_over_incumbent',
                'intersection_over_alternative', 'log_clipped_area_ratio',
                'alternative_clipped_fraction', 'incumbent_clipped_fraction',
                'center_distance_over_image', 'raw_mask_iou']


def feature_names(columns):
    return (['incumbent__'+c for c in columns] +
            ['alternative__'+c for c in columns] +
            ['difference__'+c for c in columns] + PAIR_COLUMNS)


def baseline_with_sources(items):
    """Exact confidence .30 frozen decoder, additionally retaining raw row IDs."""
    values = [dict(size=list(SHAPE), counts=r['rle'].encode('ascii')) for r in items]
    selected = []
    for i in sorted(range(len(items)), key=lambda j: (-float(items[j]['confidence']), j)):
        if items[i]['confidence'] < .30 or int(mu.area(values[i])) < 300:
            continue
        if selected and np.asarray(mu.iou([values[i]], [values[j] for j in selected], [0]*len(selected))).max() > .5:
            continue
        selected.append(i)
    owned = np.zeros(SHAPE, bool)
    masks, sources = [], []
    for i in selected:
        remaining = mu.decode(values[i]).astype(bool) & ~owned
        if remaining.sum() >= 300:
            owned |= remaining
            masks.append(mu.encode(np.asfortranarray(remaining.astype(np.uint8))))
            sources.append(i)
    return values, masks, sources


def build_pairs(items, features):
    """Construct fixed top-three geometry neighbors without any target access."""
    features = np.asarray(features, dtype=np.float64)
    if features.shape != (len(items), 102):
        raise ValueError('Exactly 102 context features per candidate required')
    if any(set(item) != {'rle', 'confidence'} for item in items):
        raise ValueError('Pass only RLE and confidence; labels/metadata are forbidden')
    raw, baseline, sources = baseline_with_sources(items)
    pairs, vectors = [], []
    audit = dict(raw=len(raw), incumbent=len(baseline), proposed=0,
                 rejected_clipping=0, rejected_small=0, rejected_identical=0)
    if not baseline or not raw:
        return baseline, sources, pairs, np.empty((0, 314)), audit
    area = np.asarray(mu.area(raw), dtype=float)
    incarea = np.asarray(mu.area(baseline), dtype=float)
    overlaps = np.asarray(mu.iou(raw, baseline, [0]*len(baseline)))
    source_set = set(sources)
    for k, (inc, source) in enumerate(zip(baseline, sources)):
        options = [j for j in range(len(raw)) if j not in source_set and
                   items[j]['confidence'] >= .05 and area[j] >= 300 and overlaps[j, k] >= .25]
        options.sort(key=lambda j: (-float(overlaps[j, k]), j))
        others = mu.merge([m for t, m in enumerate(baseline) if t != k]) if len(baseline) > 1 else None
        for j in options[:3]:
            audit['proposed'] += 1
            clipped_area = int(mu.area(mu.merge([raw[j], others], intersect=True))) if others else 0
            clipping = clipped_area / area[j]
            if clipping > .05:
                audit['rejected_clipping'] += 1
                continue
            if area[j]-clipped_area < 300:
                audit['rejected_small'] += 1
                continue
            alternative = raw[j]
            if clipped_area:
                bitmap = mu.decode(raw[j]).astype(bool) & ~mu.decode(others).astype(bool)
                alternative = mu.encode(np.asfortranarray(bitmap.astype(np.uint8)))
            aa = float(mu.area(alternative))
            intersection = float(mu.area(mu.merge([inc, alternative], intersect=True)))
            if intersection == aa == incarea[k]:
                audit['rejected_identical'] += 1
                continue
            box_inc, box_alt = mu.toBbox(inc), mu.toBbox(alternative)
            center_inc, center_alt = box_inc[:2]+box_inc[2:]/2, box_alt[:2]+box_alt[2:]/2
            pair_geometry = [intersection/(aa+incarea[k]-intersection), intersection/incarea[k],
                             intersection/aa, np.log(aa/incarea[k]), clipping,
                             1-incarea[k]/area[source], np.linalg.norm(center_inc-center_alt)/2048,
                             float(overlaps[j, k])]
            vector = np.r_[features[source], features[j], features[j]-features[source], pair_geometry]
            assert vector.shape == (314,)
            vectors.append(vector)
            pairs.append(dict(incumbent=k, incumbent_source=source, alternative_source=j,
                              mask=alternative, clipped_fraction=float(clipping)))
    x = np.asarray(vectors).reshape(-1, 314)
    assert not np.isinf(x).any()
    return baseline, sources, pairs, x, audit


def apply_pairs(baseline, pairs, gains, gate):
    """Apply predicted-positive, nonconflicting one-to-one swaps; do not re-NMS."""
    gains = np.asarray(gains, dtype=float)
    if gains.shape != (len(pairs),) or not np.isfinite(gains).all():
        raise ValueError('One finite predicted gain per pair is required')
    output, accepted = list(baseline), []
    used_inc, used_alt, accepted_masks = set(), set(), []
    for p in sorted(range(len(pairs)), key=lambda j: (-float(gains[j]), pairs[j]['incumbent'], pairs[j]['alternative_source'])):
        row = pairs[p]
        if gains[p] <= gate:
            continue
        if row['incumbent'] in used_inc or row['alternative_source'] in used_alt:
            continue
        if any(int(mu.area(mu.merge([row['mask'], other], intersect=True))) for other in accepted_masks):
            continue
        output[row['incumbent']] = row['mask']
        used_inc.add(row['incumbent']); used_alt.add(row['alternative_source'])
        accepted_masks.append(row['mask']); accepted.append(p)
    assert len(output) == len(baseline)
    assert not output or int(mu.area(mu.merge(output))) == sum(int(mu.area(m)) for m in output)
    assert all(output[k]['counts'] == baseline[k]['counts'] for k in range(len(output)) if k not in used_inc)
    return output, accepted


def edit_candidates(items, features, estimator, gate=.05):
    """Deployment entry point. Estimator trained on paired features, no GT needed."""
    baseline, sources, pairs, x, audit = build_pairs(items, features)
    scores = estimator.predict(x) if len(pairs) else np.empty(0)
    output, accepted = apply_pairs(baseline, pairs, scores, gate)
    return output, dict(**audit, accepted=len(accepted), gate=gate,
                        edits=[dict(incumbent=pairs[p]['incumbent'],
                                    alternative_source=pairs[p]['alternative_source'],
                                    predicted_gain=float(scores[p])) for p in accepted])
