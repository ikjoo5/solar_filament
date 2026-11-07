# SWEFil mixed epoch100 test submission

[Open the one-cell T4 notebook](https://colab.research.google.com/github/ikjoo5/solar_filament/blob/main/notebooks/SolarFilament_SWEFil_Test_T4.ipynb).

This runs the completed GONG/SWEFil/CHASE model on the 180 competition test images, without additional training. Its selected development PQ is **0.410866**, below the historical best84 native result of 0.429419. The purpose of this submission is to measure actual leaderboard performance; development PQ is not a predicted leaderboard score.

## Run and submit

1. Open the notebook and select a **T4 GPU** runtime.
2. Run its single code cell. Mount Drive and authenticate with the account holding the completed SWEFil experiment.
3. Wait for all 180 images and the CSV checks to finish. The notebook downloads `submission_swefil_epoch100_native.csv` and a small review ZIP.
4. Submit the **CSV** to Kaggle. The review ZIP contains predictions and provenance for later analysis, not model weights.

## Fixed inference recipe

| Setting | Value |
|---|---|
| Checkpoint | Selected best at epoch100 |
| Model | ConvNeXt-Tiny encoder + U-Net decoder |
| Input | Original grayscale2048 image, existing limb correction and normalization |
| Inference | 1024 tiles, 25% overlap, FP16, batch1 |
| Mask decoding | Foreground threshold0.65, eight-connected components, minimum area300 |
| Additional models/views | No YOLO, no TTA, no public prediction blending |
| Export | Non-overlapping instance masks in native2048 coordinates, compressed COCO RLE CSV |

The weight SHA-256 is `2b22ed327e2f64902bb7b9b3984586d1ba893f52ada74f557622cce4dac27e40`. The file size is 128,514,783 bytes. The worker also verifies the checkpoint's training identity, epoch, architecture, and selected decoding settings before inference.

## Paths and resume

All defaults are under `/content/drive/MyDrive/filament-segmentation-2026`:

- Training metadata: `fixed_split_experiments/swefil_all_mixed_native_t4_v1/train/`.
- Best weight: a Drive API file in the experiment's **`verified_checkpoints/`** folder. The launcher reads its file ID from the matching best record; it does not assume a mounted `train/checkpoints/best_epoch100.pt` exists.
- Test inputs: `MAGFiLO_1.0_Kaggle_2026/test/test_images/`.
- Output: `submissions/swefil_epoch100_native_t4_v1/`.
- Submission: `submission_swefil_epoch100_native.csv` in the output folder.
- Live log: `inference.log` in the output folder.

Run the same notebook cell after an interruption. A source snapshot and per-image prediction cache are stored in the submission folder. Verified completed images are reused; an unfinished image is repeated. Cache identity includes the model, recipe, source image hashes, code and runtime. A changed runtime or recipe cannot silently mix predictions; preserve the old output and use a new output directory for a deliberate change.

The weight download is read-only and uses a temporary local cache. Downloaded bytes must match the reviewed hash. On a fresh runtime, the weight and some dependencies may need downloading again. No image synthesis, optimizer restoration, or training is invoked. This inference run does not require the training launcher's 50 GB storage reserve.

If mounted metadata points to another checkpoint, set `CHECKPOINT_FILE_ID` to the reviewed best file's Drive ID, or provide its exact local copy as `CHECKPOINT`. The launcher still requires the same weight hash. It never silently selects an earlier or alternative model.

## Validation and limits

The worker checks the exact official 180-image filename inventory and native image dimensions. Empty predictions are recorded as completed images and produce no dummy instance row. Before publishing the CSV, it verifies identifiers, RLE masks, and non-overlap. The model and test ground truth are not embedded in this repository.

CPU fixtures exercise checkpoint selection/download integrity, the threshold and area rules, interrupted resume, cache mismatch rejection, and CSV validation. Actual GPU inference and authenticated Drive access occur in the user's Colab session. A complete submission file can only be produced there with the stored weights and test images.
