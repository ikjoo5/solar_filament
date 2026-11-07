"""Offline checks for immutable TTA launch snapshots and verified result routing."""
import copy
import json
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import swefil_tta_test_colab as launcher


@pytest.fixture
def layout(tmp_path):
    repository = tmp_path/'repository'
    for name in launcher.SOURCE_FILES:
        path = repository/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('# Offline source fixture.\n')
    training = tmp_path/'training'
    (training/'train').mkdir(parents=True)
    for name in ('identity.json','config.json'):
        (training/'train'/name).write_text('{}')
    images = tmp_path/'test_images'
    images.mkdir()
    return types.SimpleNamespace(repository=repository,training=training,images=images,output=tmp_path/'output')


def install(layout, **kwargs):
    return launcher.install(layout.repository,layout.training,layout.images,layout.output,**kwargs)


def complete_result(output, **changes):
    csv = output/launcher.CSV_NAME
    csv.write_text('filament_id,segmentation_rle\n')
    archive = output/launcher.ARCHIVE_NAME
    archive.write_bytes(b'review fixture')
    result = dict(status='complete',images_completed=180,images_total=180,
                  checkpoint_epoch=100,checkpoint_sha256=launcher.CHECKPOINT_SHA,
                  recipe=copy.deepcopy(launcher.RECIPE),output_root=str(output),overlap_pixels=0,
                  validation_reference={'pq':launcher.VALIDATION_PQ},
                  csv=str(csv),csv_sha256=launcher.shared.sha(csv),
                  archive=str(archive),archive_sha256=launcher.shared.sha(archive))
    result.update(changes)
    (output/'latest_result.json').write_text(json.dumps(result))
    return result


def test_snapshot_resumes_despite_later_checkout_changes(layout):
    expected = install(layout)
    for path in (layout.repository/'tools').iterdir():
        path.write_text('raise RuntimeError("updated code must not execute")\n')
    assert install(layout) == expected
    _,code,_ = expected
    assert set(json.loads((code/'source_manifest.json').read_text())) == set(launcher.SOURCE_FILES)
    assert (code/'tools/swefil_tta_test_inference.py').read_text() == '# Offline source fixture.\n'


@pytest.mark.parametrize('name',['tools/swefil_test_colab.py','tools/swefil_tta_test_inference.py',
                                 'tools/swefil_inference_ablation.py'])
def test_imported_dependency_and_worker_changes_are_detected(layout,name):
    _,code,_ = install(layout)
    (code/name).write_text('# altered\n')
    with pytest.raises(ValueError,match='source checksum'):
        install(layout)


def test_changed_image_or_training_path_cannot_mix_results(layout):
    install(layout)
    with pytest.raises(ValueError,match='inputs/settings changed'):
        launcher.restore(layout.output,test_images=layout.images.parent/'other_images')
    with pytest.raises(ValueError,match='inputs/settings changed'):
        launcher.restore(layout.output,training_root=layout.training.parent/'other_training')


def test_checkpoint_locator_may_move_without_changing_model_identity(layout):
    config,_,_ = install(layout,checkpoint=layout.repository/'old.pt')
    restored,_,_ = install(layout,checkpoint=layout.repository/'new.pt')
    assert restored == config and restored['checkpoint_sha256']==launcher.CHECKPOINT_SHA


def test_interrupted_snapshot_commit_can_resume(layout,monkeypatch):
    actual = launcher.shared.atomic_write
    def interrupted(path,payload):
        if Path(path).name == launcher.STATE_NAME:
            raise OSError('interrupted')
        actual(path,payload)
    monkeypatch.setattr(launcher.shared,'atomic_write',interrupted)
    with pytest.raises(OSError,match='interrupted'):
        install(layout)
    assert not (layout.output/launcher.STATE_NAME).exists()
    monkeypatch.setattr(launcher.shared,'atomic_write',actual)
    config,code,_ = install(layout)
    assert config['recipe']['tta'] and code.is_dir()


@pytest.mark.parametrize('marker',['test_bootstrap_state.json','ablation_bootstrap_state.json',
                                   'predictions','submission_swefil_epoch100_native.csv'])
def test_old_baseline_or_ablation_output_is_preserved(layout,marker):
    layout.output.mkdir()
    path=layout.output/marker
    path.write_text('old result')
    with pytest.raises(ValueError,match='preserve'):
        install(layout)
    assert path.read_text()=='old result'
    assert not (layout.output/launcher.STATE_NAME).exists()


def test_output_must_be_separate_from_data_and_training(layout):
    for output in (layout.training,layout.training/'sub',layout.training.parent,layout.images/'sub'):
        with pytest.raises(ValueError,match='separate'):
            launcher.settings(layout.training,layout.images,output)


def test_snapshot_source_symlink_is_rejected(layout):
    _,code,_ = install(layout)
    target=code/'tools/swefil_test_colab.py'
    target.unlink();target.symlink_to(layout.repository/'tools/swefil_test_colab.py')
    with pytest.raises(ValueError,match='Unsafe saved source'):
        install(layout)


@pytest.mark.parametrize('change',[
    {'status':'running'}, {'images_completed':179}, {'images_total':179},
    {'checkpoint_epoch':82}, {'checkpoint_sha256':'0'*64}, {'overlap_pixels':1},
    {'recipe':dict(launcher.RECIPE,tta=False)},
    {'recipe':dict(launcher.RECIPE,threshold=.5)},
    {'validation_reference':{'pq':.41086632237070175}},
])
def test_wrong_or_incomplete_result_is_not_downloadable(layout,change):
    layout.output.mkdir();complete_result(layout.output,**change)
    with pytest.raises((RuntimeError,ValueError)):
        launcher.checked_result(layout.output)


@pytest.mark.parametrize('key',['csv','archive'])
def test_artifact_bytes_bound_to_completed_result(layout,key):
    layout.output.mkdir();result=complete_result(layout.output)
    assert launcher.checked_result(layout.output)==result
    Path(result[key]).write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        launcher.checked_result(layout.output)


def test_run_invokes_frozen_tta_worker_without_changing_training(layout,monkeypatch):
    before={p.name:p.read_bytes() for p in (layout.training/'train').iterdir()}
    calls=[]
    monkeypatch.setattr(launcher.shared,'ensure_dependencies',lambda output:None)
    def download(training,checkpoint,file_id):
        calls.append(('download',training,checkpoint,file_id))
        return Path('/tmp/pinned_epoch100.pt')
    monkeypatch.setattr(launcher.shared,'download_checkpoint',download)
    def monitored(command,log,cwd,env):
        calls.append(('worker',command,log,cwd,env));complete_result(layout.output)
    monkeypatch.setattr(launcher.shared,'monitored',monitored)
    result=launcher.run(layout.training,layout.images,layout.output,repository=layout.repository,
                        checkpoint_file_id='reviewed-fixture')
    command=next(x[1] for x in calls if x[0]=='worker')
    assert Path(command[2]).name=='swefil_tta_test_inference.py'
    assert command[command.index('--test-images')+1]==str(layout.images)
    assert '--checkpoint' in command and '--run-until' not in command
    assert next(x[3] for x in calls if x[0]=='download')=='reviewed-fixture'
    assert before=={p.name:p.read_bytes() for p in (layout.training/'train').iterdir()}
    assert result['recipe']==launcher.RECIPE


def test_recipe_and_bundle_match_actual_worker():
    import swefil_tta_test_inference as worker
    assert launcher.RECIPE==worker.RECIPE
    assert launcher.CSV_NAME==worker.CSV_NAME
    assert launcher.ARCHIVE_NAME==worker.ARCHIVE_NAME
    assert all('tools/'+name in launcher.SOURCE_FILES for name in worker.SOURCE_FILES)


def test_notebook_is_one_plain_cell_with_distinct_tta_output():
    notebook=json.loads((ROOT/'notebooks/SolarFilament_SWEFil_TTA_Test_T4.ipynb').read_text())
    cells=[c for c in notebook['cells'] if c['cell_type']=='code']
    assert len(cells)==1 and not cells[0]['outputs']
    source=''.join(cells[0]['source']);compile(source,'colab','exec')
    assert 'base64' not in source and 'PAYLOAD' not in source
    assert launcher.STATE_NAME in source and 'submissions/'+launcher.SCOPE in source
    assert launcher.CSV_NAME in source and launcher.ARCHIVE_NAME in source
    assert 'for name, expected in manifest.items()' in source
