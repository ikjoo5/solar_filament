# Frozen epoch100 inference experiments

The current mixed-data ConvNeXt-Tiny U-Net received a **user-reported public LB of 0.34**. Its saved native validation result is **PQ 0.410866**, with TP 461, FP 350 and FN 247. The original YOLO + best84 pipeline has validation PQ 0.472593 and a user-reported public LB of 0.39. Validation and leaderboard scores are separate measurements.

## Completed CPU experiments — 7 November 2026

The saved predictions were evaluated against the original **91 physical images, 101 annotator–image groups and 708 annotated instances**. Predictions are evaluated for every annotation group, including images with no predictions. Matching uses the supplied evaluator's strict mask IoU > 0.5 and global PQ aggregation. Exported masks were checked for overlaps and rescored independently.

The experiments covered:

- **81 morphology/area settings:** three foreground thresholds; area pruning; dilation, erosion, opening, closing and hole filling.
- **40 topology settings:** high-confidence core selection, bounded expansion and marker watershed using the three saved threshold masks.
- **24 model-combination settings:** branch controls, novel-instance additions, and union/intersection replacement of matching masks.
- **12 confirmation gates:** retain an existing YOLO instance only when the new native foreground supports a specified fraction of its area, optionally preserving large instances.
- **One fixed transfer:** apply the selected confirmation rule to the old complete YOLO + best84 pipeline, without a new parameter search.

Each family's settings were fixed before scoring that family. The confirmation family and transfer were added after reviewing the earlier comparisons. Controls appear in more than one family; these are not 158 independent studies.

| Configuration | Validation PQ | TP | FP | FN |
|---|---:|---:|---:|---:|
| Current native baseline, threshold .65 / area 300 | 0.410866 | 461 | 350 | 247 |
| Best morphology setting: minimum area 400 | 0.411225 | 442 | 302 | 266 |
| Best non-baseline topology setting: bounded expansion | 0.409252 | 458 | 350 | 250 |
| Historical processed YOLO branch alone | 0.464939 | 481 | 213 | 227 |
| YOLO first + novel current-native instances | 0.460504 | 493 | 248 | 215 |
| YOLO confirmed by current-native foreground | 0.468970 | 476 | 192 | 232 |
| Original full YOLO + best84 pipeline | 0.472593 | 488 | 204 | 220 |
| Original full pipeline + the fixed confirmation gate | 0.472140 | 482 | 195 | 226 |

The historical YOLO branch already contains multi-view, tiled inference and postprocessing developed on this validation set. It is not a single-view YOLO control. Its annotation, split and validation image hashes match the current experiment.

### What the experiments support

- **Simple mask cleanup is insufficient.** The best morphology improvement is only +0.000358 PQ, with 19 additional misses. No morphology setting reduced both FP and FN.
- **Lowering the threshold mostly changes existing boundaries.** Saved foreground unions at thresholds .35 and .65 have aggregate pixel IoU 0.95247. All 726 low-threshold components with area at least 300 already contain a higher-threshold core. Components smaller than 100 were not saved, so this result does not characterize all raw probability pixels.
- **Adding native predictions is not automatically useful.** The best addition rule recovers 12 instances compared with the processed YOLO branch but adds 35 FP. PQ falls by 0.004435.
- **The new model has a possible confirmation role.** Keeping a YOLO mask when at least 15% of its area intersects the native threshold-.35 foreground removes 21 FP while losing five TP. All twelve tested confirmation settings improve the YOLO branch's PQ on this development split.
- **The confirmation rule does not improve the original full pipeline.** Its fixed transfer removes nine FP but loses six TP, slightly reducing PQ. It is not adopted as a replacement for the original pipeline.

The selected YOLO confirmation rule improves four of five chronological diagnostic blocks. Its conditional image-bootstrap PQ-difference interval includes zero; this is not strong evidence of generalization. The blocks are diagnostics, not newly trained cross-validation folds. All images already participated in checkpoint or method selection. No test labels, public reference RLEs, or leaderboard probes were used for these CPU comparisons.

## Completed T4 inference comparisons

The uploaded `swefil_inference_ablation_results.zip` contains all 36 configurations. Independent rescoring reproduced every result to floating-point precision. Its ground truth and baseline prediction CSV are byte-identical to the previous trusted exports. Recorded checkpoint, validation source and source-code hashes agree with the reviewed run at commit `d1f5bfa`. The ZIP does not contain the original images or weights, so their bytes were not rehashed locally.

All four inference families reached their highest PQ at the same threshold .65 and minimum area 300:

| Inference family | PQ | SQ | RQ | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.410866 | 0.676904 | 0.606978 | 461 | 350 | 247 |
| **Four-view TTA** | **0.420748** | **0.678422** | **0.620186** | **467** | **331** | **241** |
| Uniform overlapping tile placement | 0.411463 | 0.676865 | 0.607895 | 462 | 350 | 246 |
| Remove limb correction | 0.410442 | 0.678122 | 0.605263 | 460 | 352 | 248 |

TTA increases PQ by **0.009881**, reducing FP by 19 and FN by six. Precision changes from 56.84% to 58.52%; recall changes from 65.11% to 65.96%. It retains 448 previous matches, gains 19 and loses 13. Among the 19 gains, fourteen already had baseline mask IoU between .25 and .5, and only four had no baseline overlap. Most improvement therefore comes from making existing partial predictions valid matches, rather than locating entirely unseen regions.

Both timestamp-proximity subsets improve: near17 PQ .372678 → .399064, far74 .416925 → .424184. TTA improves four of five chronological diagnostic blocks. The conditional paired image-bootstrap interval for its PQ change is approximately [−.00196, +.02108], so this reused development set does not establish a leaderboard improvement.

At TTA threshold .50 / area300, PQ is **0.420568**, TP472/FP342/FN236. It finds five more instances than threshold .65 while adding eleven FP, with an almost equal PQ. Threshold .65 remains the predefined maximum-PQ submission choice; this small difference is not strong evidence that .65 will generalize better.

Applying the earlier frozen 15%-support gate with the new TTA masks gives PQ .468351 on the processed YOLO branch and .472127 on the old complete pipeline. This does not improve the prior confirmation result or the original complete pipeline, so it is not added to the next submission.

The original best84 native-only historical PQ is .429419, and the original complete pipeline's PQ is .472593. The TTA improvement measures an inference change to the current model; it does not establish that the mixed-data training outperformed those models. The reported inference-loop time was 1300.6 seconds on T4; the ZIP does not establish how many caches were already present.

The **native TTA-only test submission** completed with the reviewed epoch100 weights and all other selected settings retained. The user reported the same displayed public LB of **0.34** as the native-only submission; the unrounded scores were not supplied. The development PQ improvement therefore has no demonstrated public leaderboard gain. The original own-model YOLO + best84 pipeline remains the stronger submitted baseline at 0.39. [TTA submission record and reproduction instructions](SWEFIL_TTA_TEST_SUBMISSION.md).

The TTA export contains 1,479 valid non-overlapping masks and complete records for all 180 test images. Fixed public-reference agreement changes from 28 to 18 for the 216 panel and stays at 29 for the 102 panel (57 → 47 combined). Six of the ten lost agreements in the 216 panel have no intersecting final TTA mask; their formerly agreeing native masks were small, 349–489 pixels. Reference agreement is not test correctness. A bounded six-setting validation follow-up at minimum areas 200/400 did not establish a useful recovery rule: area 200 at threshold .65 gives PQ .412751 with FP 381/FN 231, while area 400 gives PQ .421063 with FP 290/FN 258. Neither is adopted over the existing operating point.

## Reproduce the validation experiment

[Open the single-cell T4 notebook](https://colab.research.google.com/github/ikjoo5/solar_filament/blob/main/notebooks/SolarFilament_SWEFil_Ablation_T4.ipynb)

The notebook evaluates four predefined families on the same validation images:

| Family | Change |
|---|---|
| `baseline` | Reproduce the submitted native recipe: limb correction, tile 1024, overlap .25, single view |
| `tta4` | Average inverse-aligned base, horizontal, vertical and combined-flip probabilities |
| `dense` | Change overlap to .5 for more uniform tile placement |
| `raw` | Remove limb correction; retain the same input normalization and single-view decoding |

Each family uses thresholds .35/.50/.65 and minimum areas 100/300/500, giving **36 validation configurations**. This is inference only: the reviewed epoch100 checkpoint remains unchanged, and no optimizer or training loop runs. At image size 2048 and tile size 1024, overlap .25 and .5 both produce nine tiles; the difference is tile placement, not tile count. The complete suite requires six view passes per image because TTA reuses the baseline pass.

The initial baseline must reproduce validation PQ 0.410866 within 0.003 before alternative inference runs. A larger discrepancy produces a diagnostic result and stops the comparisons. The `raw` input is an ablation, not a recommended preprocessing change before measurement.

### Execution and resume

1. Select a Colab T4 GPU and run the notebook's single code cell.
2. Mount and authenticate the same Google account used for the completed training run.
3. The exact reviewed epoch100 checkpoint is located through the training metadata and verified by SHA-256 and file size.
4. Run the same cell after an interruption. Completed, verified probability maps are reused; an unfinished view is repeated.
5. Upload the resulting `swefil_inference_ablation_results.zip` for comparison. It includes all 36 prediction CSVs, ground truth, per-group scores and provenance. It does not include weights, source images or probability arrays.

Default training folder:

```text
/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/swefil_all_mixed_native_t4_v1
```

Default experiment folder:

```text
/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/swefil_epoch100_inference_ablation_t4_v1
```

The experiment saves its code and settings on Drive. Probability maps are stored losslessly in compressed form and can occupy several GB; the six float32 view maps for all 91 images would occupy about 8.5 GiB before compression. The full log is `inference.log` in the experiment folder. No synthetic data are regenerated and no test CSV is produced by this notebook.

Both the CPU comparisons and the uploaded T4 validation results above are measured. Unit and CPU integration checks cover checkpoint and source verification, flip restoration, scoring, interruption recovery, complete export, and cached resume. The T4 runs were performed by the user in Colab; the cloud workspace independently verified their exported predictions and validation metrics. The user-reported displayed public LB is 0.34 for both native and TTA submissions. Private test labels and unrounded leaderboard scores are unavailable.
