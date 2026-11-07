# CHASE-shape epoch-82 test submission

This entry produces a **native U-Net-only** submission from the selected GONG + CHASE-shape checkpoint. The checkpoint was selected on fixed validation before looking at any new test score. It is a separate submission from the established YOLO + best84 ensemble and from CSVs mixing external public predictions.

[Open the one-code-cell T4 notebook](https://colab.research.google.com/github/ikjoo5/solar_filament/blob/main/notebooks/SolarFilament_CHASE_Test_T4.ipynb). Select a GPU runtime and run the cell. Original test images and the completed training folder must be available on the user's Drive; the repository does not contain either the images or the checkpoint.

## Frozen model and decoding

| Item | Value |
|---|---|
| Architecture | SMP U-Net with ConvNeXt-Tiny encoder, foreground/spine outputs |
| Checkpoint | `train/checkpoints/best_epoch082.pt` |
| Checkpoint SHA-256 | `8ea1198734dd434c42889aee2c79928afd8eb8ce4491e826c13d07e41e03d330` |
| Selected development PQ | 0.4313392463 |
| Inputs | Original 2048 × 2048 test images |
| Preprocessing | The same limb correction and ImageNet normalization used in validation |
| Inference | Native 1024 tiles, 25% overlap, batch 1, FP16 |
| View count | One base view; no TTA |
| Foreground threshold | **0.35**, inclusive |
| Components | Eight-connected components |
| Minimum instance area | **300 pixels**, inclusive, in original coordinates |
| Fusion/refinement | No YOLO, public-prediction input, or additional boundary refinement |

Overlapping tile probabilities are blended before thresholding, matching the validation implementation. The foreground channel is decoded; the auxiliary spine channel is not used to split masks. Every exported mask has original 2048-square coordinates.

The final epoch-100 checkpoint had lower development PQ (0.420170) and is not selected. With the selected epoch-82 checkpoint, validation TP/FP/FN were 472/295/236. These local metrics are not the public test score.

## Expected Drive paths

```text
/content/drive/MyDrive/filament-segmentation-2026/
  MAGFiLO_1.0_Kaggle_2026/test/test_images/
  fixed_split_experiments/chase_shape_best84_recipe_t4_v1/
    train/config.json
    train/identity.json
    train/state.json
    train/best_epoch.json
    train/checkpoints/best_epoch082.pt
  submissions/chase_shape_epoch082_native_t4_recovery_v2/
```

The worker verifies the selected model's identity/checksum and required inference-helper source hashes. It checks that test filenames match the previously supplied complete 180-image inventory using a digest of the sorted names; no private filename list is embedded in the notebook. Image hashes bind the prediction cache to actual input bytes.

## Recovering the reviewed checkpoint from an older training state

The uploaded completed-run archive records epoch100 with best epoch82/PQ0.431339. A later read of the user's Drive returned epoch51 with best epoch42/PQ0.409225. That latter score exactly matches epoch42 in the completed history. The shared training identity describes the training protocol; it does not prove that two folders or mounted views contain the same checkpoint state. The available evidence does not establish whether the difference came from an old mounted view, another copy, or a changed training folder.

The updated notebook refreshes the Drive mount and searches the configured project directory for the reviewed epoch82 checkpoint. Search checks the actual file SHA-256, rather than selecting a filename or a newer modification time. A verified match is passed explicitly to the inference worker. The worker checks the checkpoint's training identity, epoch, model configuration and validation selection, while leaving the current `train/state.json` and other training files unchanged.

The recovery output has a new directory so the previous pinned launcher is preserved. If no matching checkpoint is found, the cell reports its search scope and observed training state and stops before GPU inference. It does not substitute epoch42 or begin more training. The uploaded review ZIP contains metrics and code but no model weights, so it cannot recreate a missing checkpoint. A matching original or backup weight file is required; the checkpoint path or search scope can be supplied in the notebook.

`SEARCH_ROOT` and `CHECKPOINT` can be changed on retry without replacing the saved source snapshot. The selected file must still match the same reviewed SHA. Current training-state and checkpoint-location observations are saved separately in `checkpoint_selection.json`: a later view of epoch100 instead of epoch51 does not invalidate completed predictions from that same verified model.

### If the project-folder search returns `not_found`

The mounted-folder search does not inspect Drive trash. Use the separate [Drive recovery Colab notebook](https://colab.research.google.com/github/ikjoo5/solar_filament/blob/main/notebooks/SolarFilament_CHASE_Drive_Recovery.ipynb). It authenticates with Google, lists accessible active and trashed checkpoint candidates without a parent-folder constraint, and downloads candidates only to verify the reviewed SHA. Its filename-prefix search includes the original epoch82 filename but cannot establish the absence of arbitrarily renamed or permanently deleted copies. API permission failures and search limits are reported separately from a completed search without a match.

A verified copy is saved under `/content/solar_chase_epoch082_recovered/` on the Colab runtime disk. This avoids consuming Drive space for another large weight file: the user's screenshot showed 99.97GB of100GB used, approximately30MB remaining. That makes checkpoint upload/synchronization failure a possibility, but does not establish the cause of the earlier state discrepancy. The notebook then passes the verified local copy to the existing saved inference launcher. It performs no Drive API update, delete, move, or untrash operation. If a trashed candidate cannot be downloaded, its link is included for manual inspection/restoration. A missing match produces `drive_recovery_report.json` and stops normally without attempting another model.

The recovered local weight and local search report disappear when the Colab runtime is deleted. Rerunning the cell can fetch the same accessible source again, while verified completed predictions remain in the Drive inference folder. This local copy is not a permanent weight backup; preserve the original Drive file. Small prediction caches, manifests and CSV/ZIP outputs still require some Drive space.

The training code retains only the current best and last checkpoint generations after each epoch. An older resumed state or a concurrent writer could consequently remove later files in the same checkpoint folder; the available logs do not establish that this happened here. Do not edit `state.json` to claim epoch100 or start training simply to bypass the inference guard.

If the reviewed weights cannot be recovered, continuing training is a separate next step. It requires both current `last_epoch051.pt` and `best_epoch042.pt` to exist and match their recorded hashes. The existing training runner can restore model/optimizer/scheduler/scaler/RNG/history and continue at epoch52 through100, keeping the original schedule. Preserve the available weights first. Such a continuation is a new result: identical epoch82 weights or PQ0.431339 are not guaranteed, and its selected checkpoint must be evaluated and registered before a submission.

No synthesis preparation, training, train-image reading, or test-label evaluation is performed. Synthetic PNG backups do not have to be restored for this task. The model is constructed without downloading ImageNet initialization and then loaded strictly from the selected checkpoint.

## Resume and outputs

- First execution stores the inference code/configuration in the new submission folder on Drive. Later execution can use that saved snapshot after the temporary Git checkout disappears.
- Each completed image has a small validated prediction cache. Interrupted images are rerun; completed compatible images are reused.
- Resume checks model, image, code, recipe, and runtime identity instead of silently mixing incompatible predictions.
- Blank predictions are recorded as completed images with no instance rows; a dummy or empty mask is not written to the submission CSV.
- Only after all 180 images are complete does the worker publish the submission CSV and review ZIP. Each CSV row contains `filament_id,segmentation_rle`, and masks are checked for exact RLE coverage, valid coordinates, nonempty area, uniqueness, and absence of overlaps.
- The notebook verifies the completed result's CSV/ZIP checksums before downloading them. A partial run does not download a submission-ready file.
- Full output is saved to `inference.log` and progress is relayed to the Colab cell. Interrupting the cell stops its own subprocess group.
- Result caches and new CSVs are separate from the training folder; the model checkpoint and completed training state remain unchanged.

Submit the generated CSV to Kaggle to measure the new model's public score. The worker does not calculate test PQ without ground truth or automatically send a submission to Kaggle. Preserve the CSV checksum and its returned score as one experiment record.

The files saved in the submission folder are `submission_chase_shape_best82_native.csv` and `chase_shape_best82_test_results.zip`. The ZIP contains predictions and provenance for review; weights and test images are not included.

## Comparison limits

The established public LB 0.39 used the full YOLO + best84 ensemble. This new CSV uses the new native model and its basic validation decoder, so their LB difference will reflect the complete submitted configurations, not isolate the effect of CHASE synthesis. No score improvement is assumed before submission.

CPU fixture checks cover serialization, cancellation/resume, source binding, and invalid or overlapping masks. Actual 180-image GPU prediction requires the user's original Drive images and epoch-82 checkpoint; those assets are absent from this cloud workspace.
