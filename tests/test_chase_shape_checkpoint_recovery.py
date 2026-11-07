"""Read-only discovery never substitutes another model or trusts filenames."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('recovery',ROOT/'tools/chase_shape_checkpoint_recovery.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)

@pytest.fixture
def project(tmp_path,monkeypatch):
    root=tmp_path/'project';train=root/'fixed/train';train.mkdir(parents=True)
    state=dict(payload=dict(epoch=51,best_epoch=42,best_pq=.4092245503356435,
        best=dict(path='checkpoints/best_epoch042.pt',sha256='earlier_weight')))
    (train/'state.json').write_text(json.dumps(state))
    monkeypatch.setattr(worker,'CHECKPOINT_SHA',hashlib.sha256(b'reviewed82').hexdigest())
    return root,train.parent


def test_finds_exact_reviewed_file_despite_earlier_state_and_preserves_inputs(project):
    root,training=project
    target=root/'backup/train/checkpoints/best_epoch082.pt';target.parent.mkdir(parents=True)
    target.write_bytes(b'reviewed82')
    wrong=training/'train/checkpoints/best_epoch082.pt';wrong.parent.mkdir();wrong.write_bytes(b'wrong')
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    report=root/'output/search.json'
    assert worker.find_reviewed_checkpoint(root,training,report_path=report)==target
    assert all(p.read_bytes()==content for p,content in before.items())
    info=json.loads(report.read_text());assert info['training_state']['best_epoch']==42
    assert info['status']=='found' and len(info['checked'])==2


def test_accepts_renamed_weight_only_when_metadata_points_to_exact_sha(project):
    root,training=project;other=root/'other/train';other.mkdir(parents=True)
    weight=other/'renamed_checkpoint.pth';weight.write_bytes(b'reviewed82')
    (other/'state.json').write_text(json.dumps({'payload':{'best':{'path':weight.name,'sha256':worker.CHECKPOINT_SHA}}}))
    assert worker.find_reviewed_checkpoint(root,training)==weight


def test_missing_reviewed_weights_reports_state_and_never_selects42(project):
    root,training=project
    older=training/'train/checkpoints/best_epoch042.pt';older.parent.mkdir();older.write_bytes(b'older42')
    with pytest.raises(FileNotFoundError,match='current state selects epoch 42'):
        worker.find_reviewed_checkpoint(root,training,report_path=root/'search.json')
    report=json.loads((root/'search.json').read_text())
    assert report['status']=='not_found' and report['checked']==[]
    assert str(older) in report['weight_files']


def test_pruned_image_directories_and_external_symlinks_are_not_scanned(project,tmp_path):
    root,training=project
    hidden=root/'train_images/best_epoch082.pt';hidden.parent.mkdir();hidden.write_bytes(b'reviewed82')
    outside=tmp_path/'outside.pt';outside.write_bytes(b'reviewed82')
    (root/'best_epoch082.pt').symlink_to(outside)
    with pytest.raises(FileNotFoundError):worker.find_reviewed_checkpoint(root,training)
    with pytest.raises(ValueError,match='symlink'):
        worker.find_reviewed_checkpoint(root,training,checkpoint=root/'best_epoch082.pt')


def test_explicit_path_accepts_exact_bytes_and_rejects_other_weights(project,tmp_path):
    root,training=project;weight=tmp_path/'renamed.pt';weight.write_bytes(b'reviewed82')
    assert worker.find_reviewed_checkpoint(root,training,checkpoint=weight)==weight
    weight.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='does not match'):
        worker.find_reviewed_checkpoint(root,training,checkpoint=weight)


def test_directory_limit_is_reported_as_incomplete_search(project):
    root,training=project
    with pytest.raises(RuntimeError,match='directory limit'):
        worker.find_reviewed_checkpoint(root,training,max_directories=1,report_path=root/'report.json')
    assert json.loads((root/'report.json').read_text())['status']=='search_failed'


def test_early_exact_candidate_wins_before_later_directory_limit(project):
    root,training=project;weight=root/'best_epoch082.pt';weight.write_bytes(b'reviewed82')
    assert worker.find_reviewed_checkpoint(root,training,max_directories=1)==weight
