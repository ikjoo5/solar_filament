"""Run portable CPU contract tests using generated fixtures, without model weights.

From a checkout: ``python tools/check_core.py``.
Additional arguments are forwarded to pytest, e.g. ``--maxfail=1``.
The historical full test suite also contains private-data integration checks;
this explicit suite is the public, data-free contract check.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TESTS = (
    'tests/test_filament_eval.py',
    'tests/test_convert_annotations.py',
    'tests/test_checkpoint_utils.py',
    'tests/test_submission_driver.py',
    'tests/test_submission_yolo_recipe.py',
    'tests/test_submission_native_recipe.py',
    'tests/test_submission_inference.py',
    'tests/test_chase_independent_decode.py',
    'tests/test_chase_transfer_data.py',
    'tests/test_targeted_synthetic_data.py',
    'tests/test_github_colab_notebook.py',
    'tests/test_chase_shape_synthesis.py',
    'tests/test_chase_shape_training.py',
    'tests/test_chase_shape_colab.py',
)


def main():
    required = [ROOT / name for name in TESTS]
    required.append(ROOT / 'configs/chase_ahas2024_2025_manifest.json')
    required.append(ROOT / 'notebooks/SolarFilament_GitHub_Colab.ipynb')
    required.append(ROOT / 'notebooks/SolarFilament_CHASE_Shape_T4.ipynb')
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit('Incomplete code checkout: ' + ', '.join(missing))
    return subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', *TESTS, *sys.argv[1:]],
        cwd=ROOT,
    ).returncode


if __name__ == '__main__':
    raise SystemExit(main())
