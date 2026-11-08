"""Full-pass worker contracts: frozen coverage, separate branches and PQ masks."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from pycocotools import mask as mu
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import fullpass_both_training as worker


def config_fixture(tmp_path, monkeypatch):
    project = tmp_path/'project'
    previous = project/'fixed_split_experiments/chase_shape_best84_recipe_t4_v1'
    previous.mkdir(parents=True)
    (previous/'experiment_config.json').write_text('{}')
    old = dict(worker.original.DEFAULTS, annotations=str(tmp_path/'annotations.json'),
        images=str(tmp_path/'images'), train_split=str(tmp_path/'train.csv'), val_split=str(tmp_path/'val.csv'),
        baseline_checkpoint='/old/solar/best84.pt', initial_checkpoint=None)
    monkeypatch.setattr(worker.original, 'normalize_config', lambda value:old)
    return dict(base=str(project), chase_training_root=str(previous),
        output_root=str(project/'fixed_split_experiments/fullpass_both_t4_v1'),
        cache_root=str(tmp_path/'local_cache'))


def test_config_has_all_tiles_and_never_inherits_solar_weights(tmp_path, monkeypatch):
    cfg = worker.normalize_config(config_fixture(tmp_path, monkeypatch))
    assert cfg['samples_per_epoch'] == cfg['steps_per_epoch'] == 5928
    assert cfg['updates_per_epoch'] == 1482
    assert cfg['epochs'] == 100 and cfg['warmup_epochs'] == cfg['real_finish_epochs'] == 0
    assert cfg['data_counts'] == {'real':616,'swefil':250,'synthetic':616}
    assert 'baseline_checkpoint' not in cfg and 'initial_checkpoint' not in cfg
    assert cfg['checkpoint_interval_updates'] == 100


@pytest.mark.parametrize('change', [dict(steps_per_epoch=400),dict(initial_checkpoint='best84.pt'),
    dict(yolo_checkpoint='old.pt'),dict(warmup_epochs=3),dict(epochs=60)])
def test_config_rejects_changed_coverage_and_old_initialization(tmp_path, monkeypatch, change):
    with pytest.raises(ValueError):
        worker.normalize_config(dict(config_fixture(tmp_path, monkeypatch), **change))


def test_config_preserves_previous_swefil_run(tmp_path, monkeypatch):
    cfg = config_fixture(tmp_path, monkeypatch)
    cfg['output_root'] = str(Path(cfg['base'])/'fixed_split_experiments/swefil_all_mixed_native_t4_v1')
    with pytest.raises(ValueError, match='preserve'):
        worker.normalize_config(cfg)


def test_mask_resolution_filters_final_area_without_overlapping():
    masks = np.zeros((3,8,8), np.uint8)
    masks[0,1:5,1:5] = 1
    masks[1,3:7,3:7] = 1
    masks[2,0:1,0:1] = 1
    output = worker.disjoint_yolo_masks(masks, [.8,.9,.99], .5, 5)
    assert len(output) == 2
    assert int(mu.area(mu.merge(output))) == sum(int(mu.area(x)) for x in output)
    assert int(mu.area(output[0])) == 16 and int(mu.area(output[1])) == 12
    assert len(worker.disjoint_yolo_masks(masks, [.8,.9,.99], .85, 5)) == 1


@pytest.mark.parametrize('value', ['', 'yolo,yolo','native','yolo,test'])
def test_models_reject_unknown_or_duplicate_names(value):
    with pytest.raises(ValueError): worker.parse_models(value)


def test_branches_run_sequentially_with_separate_store_and_no_solar_weights(tmp_path, monkeypatch):
    import chase_shape_cache
    import swefil_training_data
    import fullpass_training_data
    import fullpass_native_training
    import fullpass_yolo_training
    import fullpass_checkpoint_store
    import verified_drive_checkpoints
    cfg = config_fixture(tmp_path, monkeypatch)
    previous = Path(cfg['chase_training_root'])
    (previous/'prepared_summary.json').write_text(json.dumps({'identity':'synthesis-v1'}))
    (tmp_path/'annotations.json').write_text('{}')
    monkeypatch.setattr(chase_shape_cache,'restore',lambda root:{'status':'restored'})
    monkeypatch.setattr(worker.base,'prepare_split',lambda data:(set(),['val.jpeg'],{}))
    monkeypatch.setattr(swefil_training_data,'prepare_swefil',lambda *a,**k:
        {'records':[{}]*250, 'manifest_sha256':'external-source-sha'})
    monkeypatch.setattr(worker.original,'source_identity',lambda *a:({'inputs':'frozen'},{}))
    monkeypatch.setattr(worker.original,'configure_cuda',lambda:None)
    monkeypatch.setattr(worker.torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(worker.torch.cuda,'empty_cache',lambda:None)
    class Dataset:
        domains = {key:{'files':list(range(n))} for key,n in worker.COUNTS.items()}
        def __len__(self): return 5928
        def coverage_manifest(self): return {'tiles_per_epoch':5928}
    monkeypatch.setattr(fullpass_training_data,'prepare_dataset',lambda *a:(Dataset(),object(),['val.jpeg']))
    monkeypatch.setattr(fullpass_training_data,'export_yolo_dataset',lambda *a:(tmp_path/'dataset.yaml',{}))
    monkeypatch.setattr(verified_drive_checkpoints,'make_drive_service',lambda:object())
    folders = []
    def resolve(service, parts, identity, **kwargs):
        folders.append((parts,identity)); return parts[-1]
    monkeypatch.setattr(verified_drive_checkpoints,'resolve_experiment_folder',resolve)
    monkeypatch.setattr(fullpass_checkpoint_store,'BoundedDriveCheckpointStore',
        lambda service,folder,identity,*a,**k:SimpleNamespace(folder=folder))
    calls = []
    def run_native(config,dataset,validator,output,identity,store,**kwargs):
        calls.append(('unet',output,config,store.folder))
        return {'epochs_completed':3,'status':'paused'}
    def run_yolo(config,yaml,metadata,output,identity,store,**kwargs):
        calls.append(('yolo',output,config,store.folder))
        return {'epochs_completed':3,'status':'paused'}
    monkeypatch.setattr(fullpass_native_training,'run_native',run_native)
    monkeypatch.setattr(fullpass_yolo_training,'run_yolo',run_yolo)
    result = worker.run(cfg,run_until=3,models=['unet','yolo'])
    assert [row[0] for row in calls] == ['unet','yolo']
    assert [row[3] for row in calls] == ['unet','yolo']
    assert calls[0][1] != calls[1][1]
    assert all('baseline_checkpoint' not in row[2] for row in calls)
    assert result['epochs_completed'] == {'unet':3,'yolo':3}
    assert result['status'] == 'paused' and Path(result['archive']).is_file()
    assert len({identity for _,identity in folders}) == 2
