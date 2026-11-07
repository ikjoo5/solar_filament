import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mu
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import swefil_training_data as data


def document():
    images = []
    for n, name in enumerate(('20100101000000Ch', '20100102000000Ch', '20100103000000Ch')):
        for raw in (False, True):
            images.append(dict(id=2*n+int(raw), file_name=f'images/{name}{"_raw" if raw else ""}.jpg',
                               height=2048, width=2048))
    annotations = []
    for image_id, category in ((0, 0), (1, 0), (2, 3), (3, 3)):
        annotations.append(dict(id=image_id, image_id=image_id, category_id=category, iscrowd=0,
                                segmentation=[[1, 1, 5, 1, 5, 5, 1, 5]], area=16, bbox=[1, 1, 4, 4]))
    return dict(images=images, annotations=annotations,
                categories=[dict(id=k, name=v) for k, v in data.CATEGORIES.items()])


def test_physical_selection_retains_sunspot_only_not_unknown():
    rows, counts = data.select_observations({'train': document()}, expected_counts=None)
    assert [r['observation_id'] for r in rows] == ['20100101000000Ch', '20100102000000Ch']
    assert all('_raw' not in r['file_name'] and r['spine_valid'] is False for r in rows)
    assert counts['physical_observations'] == 3
    assert counts['annotated_observations'] == 2
    assert counts['excluded_unannotated'] == 1
    assert counts['filament_positive_observations'] == 1
    assert counts['sunspot_only_observations'] == 1
    assert counts['filament_instances'] == counts['sunspot_instances'] == 1


def test_default_contract_rejects_partial_source_and_empty():
    with pytest.raises(ValueError, match='count contract'):
        data.select_observations({'train': document()})
    empty = document(); empty['annotations'] = []
    with pytest.raises(ValueError, match='No annotated'):
        data.select_observations({'train': empty}, expected_counts=None)


@pytest.mark.parametrize('mutation, message', [
    (lambda d: d['images'][0].update(file_name='../escape.jpg'), 'unsafe'),
    (lambda d: d['images'][0].update(width=1024), '2048'),
    (lambda d: d['categories'][0].update(name='SUNSPOT'), 'category'),
    (lambda d: d['annotations'][0].update(category_id=4), 'category'),
    (lambda d: d['annotations'][0].update(image_id=500), 'unknown image'),
    (lambda d: d['annotations'].pop(), 'annotation mismatch'),
])
def test_rejects_bad_sources(mutation, message):
    doc = document(); mutation(doc)
    with pytest.raises(ValueError, match=message):
        data.select_observations({'train': doc}, expected_counts=None)


def test_duplicate_partition_rejected_instead_of_oversampling():
    with pytest.raises(ValueError, match='duplicated across'):
        data.select_observations({'train': document(), 'test': document()}, expected_counts=None)


def test_multiple_polygons_remain_one_instance_and_overlap_is_union():
    ann = document()['annotations'][0]
    ann['segmentation'] += [[3, 1, 7, 1, 7, 5, 3, 5], [9, 9, 11, 9, 11, 11, 9, 11]]
    canonical = data.canonical_annotation(ann, height=16, width=16)
    encoded = dict(canonical['segmentation']); encoded['counts'] = encoded['counts'].encode()
    mask = mu.decode(encoded)
    assert mask.shape == (16, 16)
    assert canonical['area'] == int(mask.sum()) == 28
    assert canonical['bbox'] == [1.0, 1.0, 10.0, 10.0]
    assert canonical['source_polygon_count'] == 3
    assert canonical['category_id'] == 0


@pytest.mark.parametrize('polygons', [[], [[1, 2, 3]], [[1, 1, 3, 1, float('nan'), 3]],
                                     [[-1, 1, 3, 1, 3, 3]], [[1, 1, 1, 1, 1, 1]]])
def test_bad_or_empty_polygon_rejected(polygons):
    ann = document()['annotations'][0]; ann['segmentation'] = polygons
    with pytest.raises(ValueError):
        data.canonical_annotation(ann)


def test_near_far_are_diagnostic_without_removing_training_records():
    records, _ = data.select_observations({'train': document()}, expected_counts=None)
    before = copy.deepcopy(records)
    result = data.validation_proximity(['20100109000000Ch.jpeg', '20100110000001Ch.jpeg'], records)
    assert result['near'] == ['20100109000000Ch.jpeg']  # Exactly7 days counts as near.
    assert result['far'] == ['20100110000001Ch.jpeg']
    assert result['training_exclusions'] == 0
    assert records == before


def test_revision_refuses_unreviewed_source_before_io(tmp_path):
    with pytest.raises(ValueError, match='pinned SWEFil revision'):
        data.prepare_swefil(tmp_path, tmp_path/'local', revision='main')
    assert list(tmp_path.iterdir()) == []


class Response:
    def __init__(self, body, status=200, headers=None):
        self.body = body; self.status_code = status; self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def raise_for_status(self):
        assert self.status_code in (200, 206)

    def iter_content(self, size):
        yield self.body


class Session:
    def __init__(self, responses):
        self.responses = list(responses); self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def spec(body):
    return dict(size=len(body), sha256=hashlib.sha256(body).hexdigest())


def test_download_resumes_then_reuses_verified_cache(tmp_path):
    body = b'actual pinned bytes'; path = tmp_path/'image.jpg'
    path.with_suffix('.jpg.part').write_bytes(body[:7])
    session = Session([Response(body[7:], 206, {'Content-Range': f'bytes 7-{len(body)-1}/{len(body)}'})])
    assert data.download_verified('https://source/pinned', path, spec(body), session) == path
    assert path.read_bytes() == body
    assert session.calls[0][1]['headers']['Range'] == 'bytes=7-'
    data.download_verified('https://source/pinned', path, spec(body), session)
    assert len(session.calls) == 1


def test_ignored_range_restarts_without_appending(tmp_path):
    body = b'complete body'; path = tmp_path/'source.json'
    path.with_suffix('.json.part').write_bytes(body[:3])
    data.download_verified('https://source/pinned', path, spec(body), Session([Response(body)]))
    assert path.read_bytes() == body


def test_corrupt_complete_cache_preserved_and_replaced(tmp_path):
    body = b'good bytes'; path = tmp_path/'source.json'; path.write_bytes(b'bad bytes!')
    data.download_verified('https://source/pinned', path, spec(body), Session([Response(body)]))
    assert path.read_bytes() == body
    assert [p.read_bytes() for p in tmp_path.glob('*.corrupt-*')] == [b'bad bytes!']


def test_bad_transfer_never_published(tmp_path):
    path = tmp_path/'source.json'; body = b'good'
    with pytest.raises(ValueError, match='checksum'):
        data.download_verified('https://source/pinned', path, spec(body), Session([Response(b'evil')]))
    assert not path.exists()
    assert path.with_suffix('.json.part').is_file()


def test_git_blob_integrity_is_not_plain_sha1(tmp_path):
    body = b'{"value": 1}\n'; path = tmp_path/'test.json'; path.write_bytes(body)
    git_hash = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
    assert data._matches(path, dict(size=len(body), git_blob_sha1=git_hash))
    assert not data._matches(path, dict(size=len(body), git_blob_sha1=hashlib.sha1(body).hexdigest()))


def test_local_copy_verified_before_use(tmp_path):
    source = tmp_path/'drive'/'image.jpg'; source.parent.mkdir(); source.write_bytes(b'pixels')
    target = tmp_path/'local'/'image.jpg'
    data._local_copy(source, target, spec(b'pixels'))
    assert target.read_bytes() == b'pixels'
    target.write_bytes(b'corruption')
    data._local_copy(source, target, spec(b'pixels'))
    assert target.read_bytes() == b'pixels'
    source.write_bytes(b'bad'); target.unlink()
    with pytest.raises(ValueError, match='Persistent'):
        data._local_copy(source, target, spec(b'pixels'))


def test_atomic_json_has_no_partial_success_artifact(tmp_path):
    path = tmp_path/'manifest.json'
    data.atomic_json(path, {'old': 1})
    with pytest.raises(ValueError):
        data.atomic_json(path, {'new': float('nan')})
    assert json.loads(path.read_text()) == {'old': 1}
    assert list(tmp_path.glob('*.pending')) == []


def test_real_pinned_dataset_contract_is_explicit():
    assert data.EXPECTED_COUNTS['annotated_observations'] == 250
    assert data.EXPECTED_COUNTS['filament_instances'] == 1670
    assert data.EXPECTED_COUNTS['sunspot_instances'] == 402
    assert data.VARIANT == 'processed'
