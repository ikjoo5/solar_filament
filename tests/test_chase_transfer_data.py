"""Independent geometry, source integrity and supervision regression checks."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import zlib

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'tools'))
import chase_transfer_data as data


def disk(side=512, center=(251, 263), radius=228):
    image = np.zeros((side, side), np.uint8)
    cv2.circle(image, center, radius, 110, -1)
    return image


def test_frozen_manifest_has_only695_new_observations():
    path = Path(__file__).parents[1]/'configs/chase_ahas2024_2025_manifest.json'
    manifest = data.validate_manifest(json.loads(path.read_text()))
    assert len(manifest['pairs']) == 695
    assert len({p['id'] for p in manifest['pairs']}) == 695
    assert {p['date'][:4] for p in manifest['pairs']} == {'2024', '2025'}
    assert all('/train/' in p['image']['member'] for p in manifest['pairs'])
    for pair in manifest['pairs']:
        assert pair['image']['member'].replace('/images/', '/masks/') == pair['mask']['member']
    altered = json.loads(path.read_text())
    altered['pairs'][0]['image']['member'] = '../untrusted.png'
    with pytest.raises(ValueError, match='unsafe'):
        data.validate_manifest(altered)


def test_alignment_fits_geometry_and_moves_mask_with_image():
    image = disk()
    mask = np.zeros_like(image); mask[200:210, 300:320] = 255
    image[mask > 0] = 30
    geometry = data.estimate_disk(image)
    assert abs(geometry['cx']-251) < 1
    assert abs(geometry['cy']-263) < 1
    assert abs(geometry['radius']-228) < 2
    aligned, fg, valid, audit = data.align_and_screen(image, mask, target_size=256, target_radius=110)
    assert aligned.shape == fg.shape == valid.shape == (256, 256)
    assert set(np.unique(fg)) == {0., 1.}
    assert set(np.unique(valid)) <= {0., 1.}
    assert aligned[fg > 0].mean() < 50
    assert audit['exclusion_boxes'] == []
    assert audit['ignored_foreground'] == 0
    assert np.all((fg == 0) | (valid == 1))
    assert aligned.max() <= image.max()  # No contrast stretch/sharpening.


def test_branched_crossing_thin_and_small_masks_are_retained_without_metadata():
    image = disk(center=(256, 256), radius=230)
    mask = np.zeros_like(image)
    cv2.line(mask, (180, 150), (180, 280), 255, 7)
    cv2.line(mask, (120, 200), (240, 200), 255, 7)
    mask[330:340, 300] = 255
    mask[290, 350] = 255
    aligned, fg, valid, audit = data.align_and_screen(image, mask, target_size=512, target_radius=230,
        source_geometry={'cx': 256, 'cy': 256, 'radius': 230})
    np.testing.assert_array_equal(aligned, image)
    np.testing.assert_array_equal(fg, mask > 0)
    assert valid.all()
    assert audit['exclusion_boxes'] == []
    assert audit['usable_foreground'] == int((mask > 0).sum())
    assert 'No junction' in audit['ambiguity_policy']


def test_explicit_exclusion_ignores_positive_without_relabeling_background():
    image = disk(center=(256, 256), radius=230)
    mask = np.zeros_like(image); mask[200:230, 200:240] = 255
    _, fg, valid, audit = data.align_and_screen(image, mask, target_size=512, target_radius=230,
        source_geometry={'cx': 256, 'cy': 256, 'radius': 230},
        manual_boxes=[dict(x0=195, y0=195, x1=245, y1=235)])
    assert fg.sum() == 1200
    assert (fg*valid).sum() == 0
    assert audit['ignored_foreground'] == 1200
    assert audit['exclusion_boxes'][0]['reason'] == 'explicit_overlap_metadata'
    with pytest.raises(ValueError, match='outside'):
        data.align_and_screen(image, mask, manual_boxes=[dict(x0=-1, y0=1, x1=10, y1=10)])


def test_failed_disk_fit_is_explicit():
    with pytest.raises(ValueError, match='separation'):
        data.estimate_disk(np.zeros((128, 128), np.uint8))


def test_decode_accepts_native2025_size_but_keeps_binary_mask_contract():
    image = np.full((2313, 2304), 90, np.uint8)
    raw = cv2.imencode('.png', image)[1].tobytes()
    np.testing.assert_array_equal(data._decode(raw, source='2025_image.png'), image)
    with pytest.raises(ValueError, match=r'published2048.*2025_mask.png.*2313, 2304'):
        data._decode(raw, mask=True, source='2025_mask.png')
    rgb = cv2.imencode('.png', np.repeat(image[..., None], 3, axis=2))[1].tobytes()
    with pytest.raises(ValueError, match=r'grayscale uint8.*rgb.png'):
        data._decode(rgb, source='rgb.png')
    mask = np.zeros((2048, 2048), np.uint8); mask[900, 1000] = 128
    with pytest.raises(ValueError, match=r'binary mask.*soft_mask.png'):
        data._decode(cv2.imencode('.png', mask)[1].tobytes(), mask=True, source='soft_mask.png')


def test_native2025_image_uses_authors_full_frame_mask_grid_before_alignment():
    image = np.zeros((2313, 2304), np.uint8)
    cv2.circle(image, (1152, 1156), 1040, 120, -1)
    cv2.rectangle(image, (900, 1050), (1000, 1080), 30, -1)
    # The published mask already belongs to the authors' resized2048 grid.
    reference_image = cv2.resize(image, (2048, 2048), interpolation=cv2.INTER_LINEAR)
    mask = np.zeros((2048, 2048), np.uint8)
    mask[930:950, 800:870] = 255
    before = mask.copy()
    actual = data.align_and_screen(image, mask)
    reference = data.align_and_screen(reference_image, mask)
    for observed, expected in zip(actual[:3], reference[:3]):
        np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(mask, before)
    grid = actual[3]['source_grid']
    assert grid['image_shape'] == [2313, 2304]
    assert grid['mask_shape'] == [2048, 2048]
    assert grid['resized'] and not grid['mask_resized']
    assert not reference[3]['source_grid']['resized']
    assert not actual[3]['photometric_preprocessing']


def test_sampling_is_deterministic_and_spine_never_supervised(monkeypatch):
    records = [dict(id='fixture', image_path='unused', mask_path='unused',
                    image_sha256='a', mask_sha256='b')]
    dataset = data.ChaseDataset(records, crop=64, length=4, seed=73)
    image = np.full((2048, 2048), 80, np.uint8)
    foreground = np.zeros_like(image); foreground[900:920, 1000:1030] = 1
    valid = np.ones_like(image)
    monkeypatch.setattr(dataset, '_aligned', lambda _: (image, foreground, valid))
    first = dataset.sample(1, epoch=3); repeat = dataset.sample(1, epoch=3)
    for a, b in zip(first, repeat):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    assert first[0].shape == (3, 64, 64)
    assert first[1].shape == first[2].shape == (2, 64, 64)
    assert not first[1][1].any() and not first[2][1].any()
    expected = (80/255-np.array([.485, .456, .406]))/np.array([.229, .224, .225])
    np.testing.assert_allclose(first[0][:, 0, 0], expected, atol=2e-7)
    assert first[2][0].any()
    monkeypatch.setattr(dataset, '_aligned', lambda _: (image, foreground, np.zeros_like(valid)))
    with pytest.raises(ValueError, match='No valid'):
        dataset.sample(1, epoch=3)


def test_invalid_random_crop_falls_back_to_observed_valid_region(monkeypatch):
    dataset = data.ChaseDataset([dict(id='fixture')], crop=64, length=1)
    image = np.full((2048, 2048), 100, np.uint8)
    valid = np.zeros_like(image); valid[1000, 1000] = 1
    monkeypatch.setattr(dataset, '_aligned', lambda _: (image, np.zeros_like(image), valid))
    _, _, loss_valid = dataset.sample(0)
    assert loss_valid[0].sum() == 1


def test_explicit_record_selection_preserves_crop_rng_and_validates_index(monkeypatch):
    records = [dict(id='first'), dict(id='second')]
    dataset = data.ChaseDataset(records, crop=64, length=20, seed=31)
    image = np.full((2048, 2048), 80, np.uint8)
    foreground = np.zeros_like(image); foreground[1000:1020, 1000:1025] = 1
    valid = np.ones_like(image)
    selected = []
    def aligned(record):
        selected.append(record['id'])
        return image, foreground, valid
    monkeypatch.setattr(dataset, '_aligned', aligned)
    positives = 0
    for index in range(20):
        normal = dataset.sample(index, epoch=4)
        forced = dataset.sample(index, epoch=4, record_index=1)
        assert selected[-1] == 'second'
        for a, b in zip(normal, forced):
            np.testing.assert_array_equal(a.numpy(), b.numpy())
        positives += int(forced[1][0].sum() > 0)
    assert positives > 0
    dataset.sample(0, record_index=np.int64(0))
    assert selected[-1] == 'first'
    for bad in (-1, 2, True, 0.5, '0'):
        with pytest.raises(ValueError, match='record_index'):
            dataset.sample(0, record_index=bad)


class Response:
    def __init__(self, status, body, headers=None):
        self.status_code = status; self.body = body; self.headers = headers or {}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def raise_for_status(self):
        assert self.status_code in (200, 206)
    def iter_content(self, _):
        yield self.body


class Session:
    def __init__(self, response):
        self.response = response; self.headers = None
    def get(self, _, **kwargs):
        self.headers = kwargs['headers']; return self.response


def test_archive_resume_and_ignored_range_restart(tmp_path):
    content = b'preserved source bytes'*20
    archive = dict(url='https://example.invalid/read-only', size=len(content), md5=hashlib.md5(content).hexdigest())
    part = tmp_path/'ahas_21582245_data.zip.part'; part.write_bytes(content[:17])
    client = Session(Response(206, content[17:], {'Content-Range': f'bytes 17-{len(content)-1}/{len(content)}'}))
    path = data.download_archive(archive, tmp_path, session=client)
    assert client.headers['Range'] == 'bytes=17-'
    assert path.read_bytes() == content and not part.exists()
    path.unlink(); part.write_bytes(b'old partial')
    ignored = Session(Response(200, content))
    path = data.download_archive(archive, tmp_path, session=ignored)
    assert path.read_bytes() == content
    # Once complete and verified, no network is needed.
    assert data.download_archive(archive, tmp_path, session=object()) == path


def test_wrong_range_and_checksum_rejected(tmp_path):
    body = b'abcdefghij'; archive = dict(url='irrelevant', size=10, md5=hashlib.md5(body).hexdigest())
    (tmp_path/'ahas_21582245_data.zip.part').write_bytes(body[:3])
    wrong = Session(Response(206, body[3:], {'Content-Range': 'bytes 0-6/10'}))
    with pytest.raises(ValueError, match='range'):
        data.download_archive(archive, tmp_path, wrong)
    corrupt = Session(Response(200, b'xxxxxxxxxx'))
    with pytest.raises(ValueError, match='MD5'):
        data.download_archive(archive, tmp_path, corrupt)


@pytest.mark.parametrize('native2025', [False, True])
def test_prepare_resumes_sources_without_network_and_repairs_crc(tmp_path, monkeypatch, native2025):
    if native2025:
        image = np.zeros((2313, 2304), np.uint8)
        cv2.circle(image, (1152, 1156), 1040, 110, -1)
    else:
        image = disk(2048, center=(1024, 1024), radius=940)
    mask = np.zeros((2048, 2048), np.uint8); mask[800:810, 900:950] = 255
    image_bytes = cv2.imencode('.png', image)[1].tobytes()
    mask_bytes = cv2.imencode('.png', mask)[1].tobytes()
    archive = tmp_path/'fixture.zip'; pair = dict(id='RSM20240101T010101_0000_HA', date='2024-01-01')
    with zipfile.ZipFile(archive, 'w') as z:
        for kind, value in [('image', image_bytes), ('mask', mask_bytes)]:
            member = f'AHAS/train/{kind}s/'+pair['id']+'.png'
            z.writestr(member, value)
            pair[kind] = dict(member=member, size=len(value), crc32=zlib.crc32(value)&0xffffffff)
        z.writestr('../../do_not_extract.txt', b'excluded member')
    manifest = dict(schema_version=1, archive={'fixture': True}, pairs=[pair])
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(manifest))
    monkeypatch.setattr(data, 'validate_manifest', lambda value: value)
    downloads = []
    monkeypatch.setattr(data, 'download_archive', lambda *a: downloads.append(1) or archive)
    root, cache = tmp_path/'persistent', tmp_path/'local'
    first, report_path = data.prepare_chase(root, cache, path)
    assert len(downloads) == 1
    assert Path(first[0]['image_path']).read_bytes() == image_bytes
    assert Path(first[0]['mask_path']).read_bytes() == mask_bytes
    assert first[0]['image_shape'] == list(image.shape)
    assert first[0]['mask_shape'] == [2048, 2048]
    assert json.loads(report_path.read_text())['image_to_mask_resize_count'] == int(native2025)
    again, _ = data.prepare_chase(root, cache, path)
    assert again == first and len(downloads) == 1
    assert not (tmp_path.parent/'do_not_extract.txt').exists()
    Path(first[0]['mask_path']).write_bytes(b'broken source')
    repaired, _ = data.prepare_chase(root, cache, path)
    assert repaired == first and len(downloads) == 2
    assert json.loads(report_path.read_text())['records'] == first
    dataset = data.ChaseDataset(first, crop=64, length=2, cache_dir=cache/'aligned')
    report = dataset.preflight()
    assert report['observations'] == 1 and report['retained_all_source_observations']
    assert report['records'][0]['ignored_foreground'] == 0
    sample = dataset[0]
    assert sample[0].shape == (3, 64, 64)
