"""Fixed minimal decoders; independent of the historical pipeline and input mode."""
import numpy as np
from scipy import ndimage
from pycocotools import mask as mu

SIZE = 2048


def _rle(value):
    if isinstance(value, (str, bytes)):
        value = dict(size=[SIZE, SIZE], counts=value)
    if value.get('size') != [SIZE, SIZE]:
        raise ValueError('Expected original 2048-square mask')
    counts = value['counts']
    if isinstance(counts, bytes):
        counts = counts.decode('ascii')
    if not isinstance(counts, str) or not counts:
        raise ValueError('Expected compressed COCO RLE')
    runs, position, total = [], 0, 0
    while position < len(counts):
        number = shift = 0
        while True:
            if position == len(counts) or shift > 65:
                raise ValueError('Truncated or oversized RLE count')
            digit = ord(counts[position])-48; position += 1
            if not 0 <= digit <= 63:
                raise ValueError('Invalid RLE character')
            number |= (digit & 31) << shift; shift += 5
            if not digit & 32:
                if digit & 16: number |= -1 << shift
                break
        if len(runs) > 2: number += runs[-2]
        total += number
        if number < 0 or total > SIZE*SIZE:
            raise ValueError('Invalid RLE coverage')
        runs.append(number)
    if total != SIZE*SIZE:
        raise ValueError('Incomplete RLE coverage')
    return dict(size=[SIZE, SIZE], counts=counts.encode('ascii'))


def _encode(local, top=0, left=0):
    yy, xx = np.nonzero(local)
    linear = np.sort((xx+left)*SIZE+yy+top)
    splits = np.flatnonzero(np.diff(linear) > 1)
    starts = np.r_[linear[0], linear[splits+1]]
    ends = np.r_[linear[splits]+1, linear[-1]+1]
    counts, previous = [], 0
    for start, end in zip(starts, ends):
        counts.extend([int(start-previous), int(end-start)]); previous = end
    counts.append(int(SIZE*SIZE-previous))
    return mu.frPyObjects(dict(size=[SIZE, SIZE], counts=counts), SIZE, SIZE)


def validate_masks(values):
    masks = [_rle(value) for value in values]
    areas = np.asarray(mu.area(masks)) if masks else np.empty(0)
    if np.any(areas == 0): raise ValueError('Empty instance mask')
    if masks and int(mu.area(mu.merge(masks))) != int(areas.sum()):
        raise ValueError('Overlapping instance masks')
    return masks


def native_masks(foreground_rle, min_area=100):
    """One base-view FG>=.5 union; eight-connected components, area>=100."""
    if type(min_area) is not int or min_area < 1:
        raise ValueError('Positive integer component area required')
    binary = mu.decode(_rle(foreground_rle))
    labels, _ = ndimage.label(binary, structure=np.ones((3, 3), np.uint8))
    areas = np.bincount(labels.ravel())
    return [_encode(labels[region] == label, region[0].start, region[1].start)
            for label, region in enumerate(ndimage.find_objects(labels), 1)
            if region is not None and areas[label] >= min_area]


def yolo_masks(record, confidence=.2):
    """Base-view instances; stable confidence priority; no area/shape filter."""
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Invalid confidence threshold')
    items = list(enumerate(record['instances']))
    if any(not np.isfinite(x['confidence']) or not 0 <= x['confidence'] <= 1
           for _, x in items): raise ValueError('Invalid prediction confidence')
    items = sorted((pair for pair in items if pair[1]['confidence'] >= confidence),
                   key=lambda pair: (-pair[1]['confidence'], pair[0]))
    occupied = np.zeros((SIZE, SIZE), bool); output = []
    for _, item in items:
        mask = mu.decode(_rle(item['rle'])).astype(bool) & ~occupied
        if mask.any():
            occupied |= mask
            output.append(_encode(mask))
    return output


def score_groups(groups, predicted):
    """Official global all-pairs IoU>.5; explicitly include blank predictions."""
    if set(predicted) != {g['image'] for g in groups.values()}:
        raise ValueError('Prediction image set differs from validation')
    predicted = {key: validate_masks(value) for key, value in predicted.items()}
    tp = fp = fn = 0; total = 0.
    for group in groups.values():
        truth = [_rle(x) for x in group['masks']]; masks = predicted[group['image']]
        matrix = (np.asarray(mu.iou(truth, masks, [0]*len(masks))) if truth and masks
                  else np.zeros((len(truth), len(masks))))
        hits = matrix > .5
        tp += int(hits.sum()); fp += int((~hits.any(0)).sum())
        fn += int((~hits.any(1)).sum()); total += float(matrix[hits].sum())
    denominator = tp+.5*(fp+fn)
    return dict(tp_pairs=tp, fp=fp, fn=fn, matched_iou_sum=total,
        pq=total/denominator if denominator else 0., rq=tp/denominator if denominator else 0.,
        sq=total/tp if tp else 0., physical_images=len(predicted), annotation_groups=len(groups),
        gt_instances=sum(len(g['masks']) for g in groups.values()),
        prediction_instances=sum(map(len, predicted.values())))
