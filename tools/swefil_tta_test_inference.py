"""Test inference for the measured epoch100 four-view TTA validation recipe.

Weights stay fixed. Average inverse-aligned foreground probabilities, then
apply threshold .65 and area300 once. No reference RLEs or test labels are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch

import swefil_test_inference as shared
from swefil_inference_ablation import checked_probability, transform

VERSION = 'swefil_epoch100_tta4_test_v1'
CHECKPOINT_SHA = shared.CHECKPOINT_SHA
EXPECTED_IMAGES = shared.EXPECTED_IMAGES
CSV_NAME = 'submission_swefil_epoch100_tta4.csv'
ARCHIVE_NAME = 'swefil_epoch100_tta4_test_results.zip'
SOURCE_FILES = ('swefil_tta_test_inference.py', 'swefil_test_inference.py',
                'swefil_inference_ablation.py', 'solar_a100_train_standalone.py',
                'chase_best84_training.py', 'chase_independent_decode.py')
RECIPE = dict(shared.RECIPE, tta=True, views=['base', 'h', 'v', 'hv'], channel=0,
              averaging='probability_mean_after_inverse_flip')
PROFILE = dict(recipe=RECIPE, csv_name=CSV_NAME, archive_name=ARCHIVE_NAME,
               source_files=SOURCE_FILES)
REVIEWED_PQ = 0.4207475911810596
VALIDATION_REFERENCE = dict(
    recipe='tta4', threshold=.65, min_area=300, pq=REVIEWED_PQ,
    sq=0.6784217048379826, rq=0.6201859229747676,
    tp_pairs=467, fp=331, fn=241, physical_images=91,
    annotation_groups=101, gt_instances=708,
    prediction_csv_sha256='5b2a6d69f0601d8eb2ea7c4efedb122df842f3f38d394200a5b3f3fe85614fcb',
    ablation_identity='314d7af3f37bfa01ffcd7bd652904305abe9a91e129d10da727baf01f2015516',
    scope='Measured on the existing development split; not a test or leaderboard score.')


def tta_probability(raw, predictor):
    """Match the audited ablation's order and float32 averaging exactly."""
    corrected = shared.base.limb(raw)
    total = None
    for view in RECIPE['views']:
        foreground = checked_probability(predictor(transform(corrected, view)))
        restored = transform(foreground, view)
        if total is None:
            # Preserve the producer's memory if it returns a reusable buffer.
            total = restored.copy()
        else:
            total += restored
    total /= len(RECIPE['views'])
    return checked_probability(total)


def run(training_root, test_images, output, checkpoint):
    metadata = shared.training_metadata(training_root, checkpoint)
    training_root, output = Path(training_root).resolve(), Path(output).resolve()
    images = Path(test_images or metadata['config']['test_images']).resolve()
    if output == training_root or output in training_root.parents or training_root in output.parents:
        raise ValueError('TTA submission output must be separate from training')
    if Path(metadata['checkpoint']).resolve().is_relative_to(output):
        raise ValueError('TTA output must not contain the verified checkpoint')
    sources = shared.inventory(images)
    if not torch.cuda.is_available():
        raise RuntimeError('Select a Colab T4 GPU runtime for test inference')
    shared.configure_cuda()
    cv2.setNumThreads(2)
    runtime = shared.runtime_identity()
    protocol = dict(version=VERSION, training_identity=shared.TRAINING_IDENTITY,
        checkpoint_sha256=CHECKPOINT_SHA, recipe=RECIPE,
        export_profile=dict(csv_name=CSV_NAME, archive_name=ARCHIVE_NAME, source_files=list(SOURCE_FILES)),
        source_images=sources, test_names_sha256=shared.TEST_NAMES_SHA,
        source_code={name: shared.digest(Path(__file__).with_name(name)) for name in SOURCE_FILES},
        runtime=runtime, validation_reference=VALIDATION_REFERENCE,
        checkpoint_native_validation=metadata['validation'],
        training_runtime=metadata['training_runtime'])
    print('Inference only: fixed epoch100, limb correction, native1024 overlap.25, '
          'four inverse-aligned probability views, threshold.65, minimum area300.', flush=True)
    print('Measured development PQ: 0.4207476. Test PQ requires Kaggle evaluation.', flush=True)
    model = None
    device = torch.device('cuda')

    def predict_view(image):
        nonlocal model
        if model is None:
            payload = shared.load_reviewed_checkpoint(metadata['checkpoint'])
            model = shared.base.build_model(RECIPE['encoder'], pretrained=False)
            model.load_state_dict(payload['model_state_dict'], strict=True)
            del payload
            model = model.to(device).eval()
        return shared.base.predict_foreground(model, image, device, tile=1024,
                                               overlap=.25, batch=1, amp_dtype=torch.float16)

    return shared._run_images(images, output, protocol,
        lambda raw: tta_probability(raw, predict_view),
        provenance=metadata.get('checkpoint_selection'), profile=PROFILE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training-root', 'output', 'checkpoint'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--test-images', type=Path)
    parser.add_argument('--result', type=Path)
    args = parser.parse_args()
    result = run(args.training_root, args.test_images, args.output, args.checkpoint)
    if args.result:
        shared.atomic_json(args.result, result)
    print(json.dumps({key: value for key, value in result.items() if key != 'per_image'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
