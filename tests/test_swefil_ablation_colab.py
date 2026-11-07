"""Offline checks for validation-only launch routing and durable source recovery."""
import json
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import swefil_ablation_colab as launcher


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
    return types.SimpleNamespace(repository=repository,training=training,output=tmp_path/'output')


def install(layout, **kwargs):
    return launcher.install(layout.repository,layout.training,layout.output,**kwargs)


def complete_result(output, **changes):
    archive = output/'review.zip'
    archive.write_bytes(b'review fixture')
    result = dict(status='complete',validation_images=91,annotation_groups=101,gt_instances=708,
                  checkpoint_epoch=100,checkpoint_sha256=launcher.CHECKPOINT_SHA,
                  output_root=str(output),recipes=list(launcher.DEFAULT_RECIPES),
                  archive=str(archive),archive_sha256=launcher.shared.sha(archive))
    result.update(changes)
    (output/'latest_result.json').write_text(json.dumps(result))
    return result


def test_snapshot_resumes_without_repository(layout):
    config,code,state = install(layout)
    # A later checkout update must not replace code already used for cached maps.
    for path in (layout.repository/'tools').iterdir():
        path.write_text('raise RuntimeError("new checkout must not execute")\n')
    restored = install(layout)
    assert restored == (config,code,state)
    assert set(json.loads((code/'source_manifest.json').read_text())) == set(launcher.SOURCE_FILES)
    assert (code/'tools/swefil_inference_ablation.py').read_text() == '# Offline source fixture.\n'


@pytest.mark.parametrize('name', ['tools/swefil_test_colab.py','tools/swefil_inference_ablation.py'])
def test_dependency_and_worker_changes_are_detected(layout, name):
    _,code,_ = install(layout)
    (code/name).write_text('# changed\n')
    with pytest.raises(ValueError,match='source checksum'):
        install(layout)


def test_changed_validation_inputs_and_recipes_cannot_reuse_output(layout):
    install(layout,images=layout.repository/'images')
    with pytest.raises(ValueError,match='inputs/recipes changed'):
        install(layout,images=layout.repository/'other_images')
    with pytest.raises(ValueError,match='inputs/recipes changed'):
        install(layout,recipes=['baseline'])
    with pytest.raises(ValueError,match='inputs/recipes changed'):
        launcher.restore(layout.output,layout.repository/'other_training')


def test_local_checkpoint_locator_can_change_without_changing_model_identity(layout):
    config,_,_ = install(layout,checkpoint=layout.repository/'original.pt')
    restored,_,_ = install(layout,checkpoint=layout.repository/'relocated.pt')
    assert config == restored
    assert restored['checkpoint_sha256'] == launcher.CHECKPOINT_SHA


def test_partial_install_retries_without_losing_source(layout, monkeypatch):
    actual = launcher.shared.atomic_write
    def interrupted(path, payload):
        if Path(path).name == launcher.STATE_NAME:
            raise OSError('simulated interruption')
        actual(path,payload)
    monkeypatch.setattr(launcher.shared,'atomic_write',interrupted)
    with pytest.raises(OSError,match='simulated interruption'):
        install(layout)
    assert not (layout.output/launcher.STATE_NAME).exists()
    monkeypatch.setattr(launcher.shared,'atomic_write',actual)
    config,code,_ = install(layout)
    assert config['inference_only'] and code.is_dir()


def test_source_link_escape_rejected(layout):
    _,code,_ = install(layout)
    path = code/'tools/swefil_test_colab.py'
    path.unlink()
    path.symlink_to(layout.repository/'tools/swefil_test_colab.py')
    with pytest.raises(ValueError,match='Unsafe saved source'):
        install(layout)


@pytest.mark.parametrize('output_kind', ['training','inside_training','ancestor'])
def test_output_must_not_replace_training_inputs(layout, output_kind):
    output = {'training':layout.training,'inside_training':layout.training/'inference',
              'ancestor':layout.training.parent}[output_kind]
    with pytest.raises(ValueError,match='separate'):
        launcher.settings(layout.training,output)


def test_unsupported_recipe_cannot_start_an_experiment(layout):
    with pytest.raises(ValueError,match='Recipes must'):
        install(layout,recipes='test_public_reference')
    assert not layout.output.exists()
    assert launcher.normalize_recipes('raw,tta4,raw') == ['baseline','tta4','raw']


@pytest.mark.parametrize('changes', [
    {'status':'baseline_mismatch'}, {'validation_images':90}, {'annotation_groups':91},
    {'gt_instances':707}, {'checkpoint_epoch':82}, {'checkpoint_sha256':'0'*64},
    {'recipes':['baseline']},
])
def test_incomplete_or_wrong_result_is_not_downloadable(layout, changes):
    layout.output.mkdir()
    complete_result(layout.output,**changes)
    with pytest.raises((RuntimeError,ValueError)):
        launcher.checked_result(layout.output)


def test_archive_is_bound_to_completed_result(layout):
    layout.output.mkdir()
    expected = complete_result(layout.output)
    assert launcher.checked_result(layout.output) == expected
    Path(expected['archive']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        launcher.checked_result(layout.output)


def test_run_dispatches_only_validation_and_preserves_training(layout, monkeypatch):
    before = {p.name:p.read_bytes() for p in (layout.training/'train').iterdir()}
    calls = []
    monkeypatch.setattr(launcher.shared,'ensure_dependencies',lambda output:calls.append(('deps',output)))
    def checkpoint(training,local,file_id):
        calls.append(('checkpoint',training,local,file_id))
        return Path('/tmp/reviewed_epoch100.pt')
    monkeypatch.setattr(launcher.shared,'download_checkpoint',checkpoint)
    def monitor(command,log,cwd,env):
        calls.append(('worker',command,log,cwd,env))
        complete_result(layout.output)
    monkeypatch.setattr(launcher.shared,'monitored',monitor)
    result = launcher.run(layout.training,layout.output,repository=layout.repository,
                          checkpoint_file_id='reviewed-fixture-locator')
    command = next(row[1] for row in calls if row[0]=='worker')
    assert Path(command[2]).name == 'swefil_inference_ablation.py'
    assert command[command.index('--recipes')+1] == 'baseline,tta4,dense,raw'
    assert '--checkpoint' in command and '--training-root' in command
    assert '--test-images' not in command and '--run-until' not in command
    assert result['validation_images'] == 91
    assert before == {p.name:p.read_bytes() for p in (layout.training/'train').iterdir()}
    assert next(row for row in calls if row[0]=='checkpoint')[3] == 'reviewed-fixture-locator'


def test_notebook_has_one_plain_source_cell_and_no_payload():
    notebook = json.loads((ROOT/'notebooks/SolarFilament_SWEFil_Ablation_T4.ipynb').read_text())
    cells = [cell for cell in notebook['cells'] if cell['cell_type']=='code']
    assert len(cells) == 1
    source = ''.join(cells[0]['source'])
    compile(source,'colab','exec')
    assert 'base64' not in source and 'PAYLOAD' not in source
    assert 'ablation_bootstrap_state.json' in source
    assert 'files.download(str(archive))' in source
