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

## Next experiment: change inference while keeping weights fixed

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

The CPU experiment results above are measured. The new T4 families remain **unmeasured until this notebook runs with the user's images and weights**. Unit and CPU integration checks cover checkpoint and source verification, flip restoration, scoring, interruption recovery, complete export, and cached resume. Live Colab authentication and GPU execution have not been tested from the cloud workspace.
