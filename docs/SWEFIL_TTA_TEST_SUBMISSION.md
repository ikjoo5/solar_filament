# Reviewed epoch100 TTA test submission

[Open the single-cell T4 notebook](https://colab.research.google.com/github/ikjoo5/solar_filament/blob/main/notebooks/SolarFilament_SWEFil_TTA_Test_T4.ipynb)

This generates a new submission from the same epoch100 ConvNeXt-Tiny U-Net with **four-view flip TTA**. Its measured validation PQ is **0.420748**, compared with **0.410866** for the original native-only recipe. FP decreased from 350 to 331 and FN from 247 to 241. The original native-only test submission received a user-reported public LB of **0.34**. The new TTA leaderboard score is unknown.

## Fixed inference recipe

- Exact reviewed epoch100 weights; no fine-tuning or optimizer updates.
- Original 2048 × 2048 test images and the same limb correction and ImageNet input normalization.
- Tile size 1024, overlap .25, Hann blending, FP16 inference with batch size one.
- Base, horizontal flip, vertical flip and both flips. Inverse-align foreground probabilities and average them before decoding.
- Foreground threshold .65, eight-connected components, minimum area 300 pixels.
- No YOLO, public prediction masks, extra cleanup, or test-driven parameter selection.

The selected recipe comes from the [completed validation comparisons](SWEFIL_INFERENCE_ABLATIONS.md). The checkpoint's original native validation score remains recorded separately from the TTA recipe's validation score. The model weights are identical.

## Run and resume

1. Open the notebook, select a **T4 GPU**, and run its single code cell.
2. Mount and authenticate the same Google account that holds the completed training folder.
3. The launcher restores its saved source snapshot and downloads the exact verified checkpoint when required.
4. If the session stops, run the same cell again. Completed image predictions are reused; an unfinished image is repeated.
5. After all **180 test images** are complete, download `submission_swefil_epoch100_tta4.csv` and the review ZIP. Submit the CSV to Kaggle manually, then share the score and review ZIP.

Default folders:

```text
Training:
/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/swefil_all_mixed_native_t4_v1

Test images:
/content/drive/MyDrive/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images

New output:
/content/drive/MyDrive/filament-segmentation-2026/submissions/swefil_epoch100_tta4_t4_v1
```

The baseline submission stays in its original folder. The new output contains completed-image caches, `inference.log`, the CSV and review archive. The final export checks the official filename inventory, empty-image coverage, mask validity and non-overlap. It does not submit to Kaggle automatically.

The review archive enables comparison with the previous native predictions and tracking of the existing 216-addition/102-replacement reference panels. Those panels are public predictions, not test ground truth; agreement with them is a diagnostic rather than PQ.

The selected TTA method has been measured on validation. This new test launcher is checked with CPU tests; test inference and public LB must be obtained in Colab and Kaggle respectively.
