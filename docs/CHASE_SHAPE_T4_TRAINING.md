# GONG + CHASE-shape training on T4

This is the current implemented experiment following the user's November 6 decisions. It trains **one native U-Net with a ConvNeXt-Tiny encoder**, starting from a generic ImageNet encoder and a new decoder. It does not initialize training from best84 or the CHASE-only checkpoint. YOLO training, architecture changes, and an additional control-training run are outside this execution.

[Open the one-code-cell Colab notebook](https://colab.research.google.com/github/ikjoo5/solar_filament/blob/main/notebooks/SolarFilament_CHASE_Shape_T4.ipynb). Select a T4 GPU and run the cell. Source code is saved to Drive on first execution; rerunning the same cell restores that saved version after a runtime reset.

## Confirmed best84 reference

The original run completed **100 epochs**, selecting **epoch 84** by development validation PQ. The earlier 60-epoch setting belongs to the separate CHASE-only continuation experiment. Verified source/config/history hashes and the original recipe are in [best84_training_reference.json](../configs/best84_training_reference.json).

| Setting | Original best84 | This experiment |
|---|---|---|
| Model | SMP U-Net, ConvNeXt-Tiny, foreground + spine heads | Same |
| Initialization | ImageNet encoder + new decoder | Same; no solar checkpoint warm start |
| Training schedule | 100 epochs | 100 epochs |
| Input crop | 1024 × 1024 | Same |
| Batch / accumulation | 2 / 2 | **1 / 4** for T4 |
| Crop presentations / optimizer updates per epoch | 400 / 100 | Same budget; AMP may skip an update on overflow, which is recorded |
| Initial learning rate | 0.0001 | Same |
| Optimizer | AdamW, weight decay 0.01, betas 0.9/0.999, epsilon 1e-8 | Same |
| Scheduler | Cosine, T_max=100, minimum LR=0.000005 | Same |
| Gradient clipping | Norm 1.0 | Same |
| Synthetic sampling probability | 20% during mixed phase | **50%**, explicitly requested |
| Synthetic source | Historical GONG-derived v2 | CHASE shapes on the supplied clean background |
| Real-only epochs | 1–3 and 95–100 | Same |
| Mixed epochs | 4–94 | Same |
| Preprocessing | Limb-only correction once, grayscale repeated to RGB, ImageNet normalization | Same |
| Augmentation | Gamma 0.9–1.1, brightness multiplier 0.85–1.15, D4 | Same |
| Loss | Foreground BCE + Dice + 0.1 balanced genuine-spine BCE | Same; synthetic samples have no spine loss |
| Precision / workers / inference batch | A100 BF16 / 2 / 4 | **T4 FP16 / 0 / 1** |
| Seed | 42 | Same |

An epoch is **400 image-first random crops**, as in best84, not a full pass over every GONG or synthetic image. Within a chosen image, the sampler selects a small positive crop with probability 0.5, another positive crop with probability 0.25, and a uniform crop with probability 0.25. During mixed epochs, approximately 200 crops per epoch come from each domain; exact counts are recorded. The first three real-only epochs are a data schedule, not learning-rate warmup.

Batch-size, precision, runtime, and hardware changes prevent an exact numerical reproduction of best84. Changing both synthetic source and sampling ratio also means this is a practical candidate comparison, not an experiment isolating donor geometry alone.

## How synthesis works

1. Verify the frozen GONG train/validation split and annotations: 616 training observations and 91 validation observations. Derive intensity statistics only from training observations. Validation remains evaluation-only.
2. Read the existing CHASE AHAS 2024–2025 dataset. Verify its 695 image/mask pairs and reuse valid downloads. Match published image/mask grids and align the solar disk using the existing CHASE loader.
3. Extract connected-component shapes from CHASE binary masks. These are geometry donors, not confirmed physical instance IDs. Respect explicit exclusions and unobserved support; branching alone is not a reason to remove a shape. The renderer's placement/size/visibility checks can still reject an unsuitable insertion and record the reason.
4. Load `clean_solar.jpg` read-only, convert it to grayscale if necessary, and align a working copy to the native 2048 grid. The user's provenance statement is: **an external website image observed to contain no filaments; URL not provided**. This is a user-declared empty background, not independently verified ground truth. The manifest records that limitation. Exact duplicate checks against available held-out files do not establish broader provenance.
5. Generate **616 synthetic variants**, each containing **three nonoverlapping inserted shapes**. Sample a CHASE observation before sampling its component. Apply rotations/reflections and modest scale changes; retain a margin between inserted objects.
6. Estimate attenuation, edge contrast, and spatial variation jointly from real GONG train objects and their local foreground-excluded surroundings. Select a measured tuple using target brightness, solar radius, and object area. Render dark absorption inside the transformed shape using the background's existing texture. This is a controlled image augmentation, not a physical conversion of CHASE observations into GONG observations.
7. Store exact instance RLEs for the inserted shapes. Check quantized visibility, overlap, clipping, and preservation of every raw aligned-background pixel outside inserted masks. There are no genuine spine annotations for these synthetic frames, so their spine loss is disabled, following the original best84 synthetic policy.

Original GONG files and the clean-background file are never rewritten. Background alignment and subsequent shared limb preprocessing act on copies. The pixel-preservation assertion concerns the aligned raw composite before limb correction.

With one supplied clean image, this is **one unique background and 616 synthetic variants**, not 616 independent solar observations. The manifest reports the number of unique backgrounds and CHASE donors actually used. More shapes and a higher sampling ratio may help or hurt; the current code does not establish an improvement in PQ.

The generator produces the original clean-source panel and up to 96 composite previews under `synthesis/preview/`. Inspect these for artificial contrast, incomplete geometry, and unintended background features. Automated checks cannot certify that a website image contains no unlabeled filament. Run with `PHASE = "prepare"` to generate these without starting training, then change back to `"all"` when ready; this is optional, not an approval step.

## Paths and persistent state

The notebook's default project root is:

```text
/content/drive/MyDrive/filament-segmentation-2026
```

It reads:

- `MAGFiLO_1.0_Kaggle_2026/train/train_images/`
- `MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json`
- `clean_solar.jpg`
- `external_CHASE/ahas_2024_2025/` (existing original downloads are reused)
- Original `train_split.csv` and `val_split.csv` in `/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes/`
- Original best84 at `/content/drive/MyDrive/SolarFilament_NativeUNet_A100/20261028T030808Z_bedf3e21/best.pt`, for **frozen baseline evaluation only**.

Outputs go into a new folder:

```text
fixed_split_experiments/chase_shape_best84_recipe_t4_v1/
  code/                    saved code + dependency manifests
  synthesis/               provenance, annotations, calibration, previews
  train/                   history, metrics, state, best/last checkpoints
  training.log             child-process output
  latest_result.json       completion status and result ZIP path
  chase_shape_native_results.zip
```

Full synthetic PNGs and processed image caches stay on local Colab storage at `/content/chase_shape_best84_recipe_t4_cache`, reducing Drive use. They can be regenerated from persistent recipes after a reset. Completed original CHASE downloads remain on Drive. Resuming may verify data and rebuild local caches; that does not imply downloading all source images again.

To preserve the generated PNGs before deleting a runtime, stop the training cell while keeping the runtime connected, then copy the [synthetic-cache backup cell](../notebooks/SolarFilament_CHASE_Cache_Backup.ipynb) into that **same training notebook/runtime** and execute it. A separate runtime cannot access the generated local PNGs. It saves all generated PNGs in `synthetic_images_cache.zip` inside the experiment's Drive folder. Wait for the backup-completed message before disconnecting/deleting the runtime. Existing model checkpoints and data remain intact.

On a new runtime, the updated main training notebook automatically restores that archive before launching the saved training code. Restoration verifies the saved synthesis identity and each PNG's hash; it cannot silently substitute a different synthetic dataset. Valid local images are reused. Source integrity checks and recreation of processed limb/target caches can still take time, so this is not an instant-resume guarantee. The archive contains raw synthetic PNGs only, not all temporary caches or model weights. The first backup prints its actual byte count and requires that much additional Drive space; PNG entries are stored without redundant compression.

## Resume behavior

- Run the **same cell** again with the same output folder. It validates the saved code/configuration and restores the last committed completed epoch, including model, optimizer, scheduler, AMP scaler, and random-number state.
- An interrupted partial epoch is repeated. A completed checkpoint is committed through a small state pointer only after its files have been written; an incomplete newer file does not replace that pointer.
- Keep `RUN_UNTIL_EPOCH = 100` for the full run. A smaller value pauses at that completed epoch while preserving the 100-epoch cosine schedule. Increasing it later resumes instead of restarting.
- A finished 100-epoch run is not trained for another 100 epochs when rerun.
- Code/data/runtime mismatches stop with an explicit error rather than silently changing the experiment. Deliberate protocol changes need a new output directory.
- The cell starts a separate child process, writes full logs to Drive, relays bounded progress tails through the notebook kernel, and terminates its own child group when interrupted. Early notebook versions inherited child stdout directly, which could leave only the Drive-mounted message visible even while training progressed. Use `training.log` to check an already-running older cell; there is no need to interrupt active learning solely to refresh its display. Updating this notebook wrapper does not alter the saved training recipe or checkpoint identity. Colab or Drive failures can still interrupt execution; this is checkpoint recovery, not a guarantee of uninterrupted service.
- Colab's installed torch/torchvision CUDA combination is preserved. Small supporting dependencies are installed separately. A GPU probe checks the T4 FP16 depthwise-convolution path before training.

## Evaluation and result review

The original best84 checkpoint is evaluated once under this runtime and the same limb-only validation grid; its weights never initialize the new training run. New-model validation runs at epoch 1, every two epochs, and at epoch 100. It uses thresholds 0.35/0.5/0.65 and areas 100/200/300, retaining the best development PQ and corresponding predictions. This is the original selection procedure, not the full YOLO + native submission pipeline.

The fixed split has 91 physical images, 101 annotator groups, and 708 reference instances. Local PQ uses strict IoU > 0.5 matching and accumulation across annotation groups. Compare the same-runtime native results, PQ/SQ/RQ, TP/FP/FN, curves, and domain exposure counts before adding an ensemble or changing decoding. Repeated development selection can overfit; an improvement must later be checked with the intended 5-fold/submission procedure.

The small result ZIP contains metrics, histories, source code, and a preview subset. Weights stay on Drive and are not embedded in the ZIP. Upload the ZIP for analysis when training completes or after a deliberate pause.

## Implementation and verification scope

- [Synthesis](../tools/chase_shape_synthesis.py): train-only appearance calibration, clean-background shape rendering, canonical RLE, provenance, and deterministic cache recovery.
- [Training](../tools/chase_shape_training.py): original-recipe sampler/loss, fresh initialization, validation, and checkpoint transaction.
- [Colab bootstrap](../tools/chase_shape_colab.py): dependency setup, persistent code snapshot, child process, and resume.

Automated CPU tests cover rendering and source preservation, target/loss behavior, deterministic resume, and bootstrap integrity. They do not replace an actual T4 run on the user's complete data. No new training PQ or leaderboard gain is claimed before that execution.
