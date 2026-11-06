# Continuing YOLO and best84 training on CHASE only

Historical experiment specification. Statements about work not yet executed describe the status when this specification was written. See [current pipeline](CURRENT_PIPELINE.md), [independent comparison results](CHASE_INDEPENDENT_RESULTS.md), and [next experiment design](CHASE_GONG_NEXT_EXPERIMENT.md) for the subsequent results and current plan.

User-confirmed conditions: continue training both models from the existing public-LB-0.39 pipeline on CHASE alone, using a T4. Do not use GONG replay or a final GONG calibration stage. Exclude SWEFil. Apply only disk alignment to CHASE, and do not train chirality classification. Exclude overlapping cases only when the data explicitly identify them; do not delete them automatically based on their appearance.

## Components being trained

- YOLO26s-seg: continue from the existing checkpoint. Convert connected regions in the CHASE binary masks into proxy object polygons. This does not recover true object IDs. Report representation differences caused by hole filling, disconnected regions, and the model's internal polygon resampling. Do not exclude shapes merely because they are small or branched.
- best84: continue training the model containing the existing ConvNeXt-Tiny encoder and U-Net decoder. CHASE supervises foreground only. Exclude the unavailable spine labels from the loss rather than treating them as negative labels. The shared encoder/decoder is updated, so the spine output is not completely frozen.
- The boundary refiner in the 0.39 pipeline is a rule combining the two models' masks and support across views, rather than a separate learned model. Preserve those rules and thresholds. The new best84 outputs also participate in boundary refinement, so training can affect the postprocessed results.
- Preserve the original YOLO and best84 files and save new weights in a separate experiment directory.

## Data and inputs

- CHASE AHAS, 695 images from 2024–2025: align disk center and size only. Do not use CLAHE, additional limb correction, gamma adjustment, sharpening, blur, super-resolution, or photometric augmentation.
- PNG format correction: the verified 2025 source images are 2304×2313, while the published masks are 2048×2048. Following the author's [pinned generation code](https://github.com/irisaltHu/MORDEN/blob/e8186480a37395272ba414a0143b2b1232294ed1/segmentation/inference.py#L102), bilinearly resize the entire source image to the mask grid, then apply the same disk alignment to the image and mask. Do not modify the mask during this preliminary image resize; retain nearest-neighbor interpolation for masks during disk alignment. Preserve the original stored PNG files and their CRC/SHA values.
- GONG train, 616 images: not used in this continuation training. Keep the original train/validation split files for data identification and leakage checks.
- GONG validation, 91 images, 101 annotator groups, and 708 GT objects: evaluation only. Retain the existing pipeline's limb correction during evaluation.
- The training candidate pool contains 695 distinct CHASE observations. Additional samples or crops do not increase the count of unique images.
- Of the 815 public AHAS images, the 120 observations shared with MHAS are outside this experiment's scope. All 695 remaining images are eligible unless explicit overlap exclusions are provided. Report file corruption or alignment failure as errors with the observation ID; do not silently omit them.
- Do not infer overlap automatically from CHASE binary masks. If explicit exclusion information becomes available, use native loss-ignore regions or exclude the entire YOLO observation; do not turn removed filament pixels into background labels.
- Preserve fixed numerical input normalization and grayscale-channel replication. Geometric interpolation of images and nearest-neighbor transformation of masks must use the same coordinates.

## Default continuation-training budget

- Train each model for 60 epochs, sequentially, for 120 model-epochs in total. Start from the existing deployed weights.
- One epoch uses each of the 695 CHASE images exactly once. Set `epoch_sampling='full_pass'` and remove the 400-image limit.
- Use CHASE only throughout all 60 epochs. Each model receives 41,700 inputs and zero GONG training samples. Each of the selected 695 images is used 60 times.
- Shuffle the input order every epoch. For the native model, exposure to an image does not mean that its crop covered every pixel or filament in that image.
- T4 defaults: batch size 1. U-Net uses gradient accumulation 4; YOLO uses accumulation 1, for which completed-epoch resume has been verified. The continuation-training learning rate is 2e-5.
- As requested by the user, this experiment measures CHASE continuation training without GONG replay. Measure changes in GONG performance on the fixed validation set.
- This comparison is between the original weights and the CHASE continuation result. It does not run a matched-budget GONG-only continuation control, so it cannot establish superiority over other continuation-training strategies.

## Full-image coverage per epoch

- Reproducibly shuffle all 695 images each epoch and use each one once. No image is duplicated or omitted within an epoch.
- Shuffle again for the next epoch after completing the previous one. Restore the same input order when resuming from a completed epoch.
- Each image's cumulative exposure count equals the number of completed epochs. Do not automatically inherit checkpoints from the former 400-image schedule.
- YOLO uses the selected image's entire 2048×2048 frame. Native uses a 1024×1024 crop from that image; its position may change between epochs.
- If valid positive pixels exist, a native crop has a 75% probability of being selected to contain one of them; otherwise, select a random position. Random crops may also contain filaments, so this does not mean a 75% positive / 25% background split.

## Actual hyperparameters

| Setting | YOLO26s-seg | best84 ConvNeXt-Tiny U-Net |
|---|---|---|
| Initial weights | Existing deployed YOLO | Existing epoch 84 |
| Continuation training | 60 epochs × 695 full frames | 60 epochs × 695 crops |
| Input size | 2048×2048 | 1024×1024 |
| Batch / accumulation | 1 / 1 | 1 / 4 |
| Optimizer | AdamW | AdamW |
| Initial learning rate | 0.00002 | 0.00002 |
| Learning-rate schedule | 60-epoch cosine, minimum ratio 0.1 | 60-epoch cosine, minimum ratio 0.1 |
| Weight decay | 0.01, using the library's parameter-group policy | 0.01 |
| AMP | Enabled | Enabled |
| Warmup | None | None |
| Random seed / workers | 42 / 0 | 42 / 0 |
| Augmentation | Mosaic, MixUp, rotation, flips, brightness transforms, etc. disabled | Crop sampling without CHASE augmentation |
| Early stopping | No early stopping within 60 epochs | No early stopping within 60 epochs |
| Loss | Library's YOLO26 segmentation loss | Foreground BCE + Dice; no spine supervision |

The learning rate, input size, sampling volume, accumulation, preprocessing, and augmentation limits were chosen for this continuation experiment. They are not all competition-optimized hyperparameters. Unspecified YOLO loss details and optimizer settings use the defaults of the pinned Ultralytics 8.4.182 release. Native AdamW also uses PyTorch's default betas and eps. The initial learning rate of 2e-5 is intended to update existing weights gradually, but it has not yet been verified as optimal for CHASE-only training.

Accumulating 695 native crops in groups of 4 results in 174 attempted optimizer updates per epoch. The final 3 crops are included using their mean loss rather than discarded. YOLO attempts 695 updates per epoch. AMP overflow may skip updates; record the actual counts.

## Evaluation

1. Original YOLO + original best84.
2. New YOLO + original best84.
3. Original YOLO + new best84.
4. New YOLO + new best84.

All combinations use the same frozen finalizer and official IoU > 0.5 all-pairs PQ calculation. Do not judge full-pipeline improvement from YOLO mAP or U-Net Dice alone. Share native predictions and YOLO view caches to reduce redundant inference.

- Use YOLO weights from the fixed completed epoch; do not describe them as mAP-selected.
- Every 5 epochs, record native full-pipeline PQ in combination with the original YOLO and retain the best checkpoint as well. Compare the final four combinations using both models' weights from the same completed epoch.
- Historical validation PQ 0.472593 and public LB 0.39 are different measurements. Compare the baseline and new combinations generated in the same environment.
- The public test references 216+102 are not training labels in this experiment. This code evaluates continuation training on validation data; it does not automatically measure new test reference coverage or a new LB score.

## Execution, resume, and verification scope

- A single-code-cell Colab package is provided. Keep code, configuration, weights, and logs on Drive; keep downloaded ZIP files and preprocessing caches under local `/content`.
- The corrected-PNG experiment directory is `fixed_split_experiments/chase_only_both_alignment_t4_60ep_fullpass_pngfix_v1`. The earlier failed run was still preparing data, so training had not begun. Preserve the original CHASE storage directory and `/content/chase_only_both_alignment_60ep_fullpass_cache` to reuse downloaded ZIPs and partially extracted files. Resume the corrected version with the same cell.
- Restore optimizer, scheduler, AMP, RNG, and YOLO EMA states from the completed epoch. Some batches from an interrupted epoch may run again. Reuse completed validation/inference images and stages as well.
- Check identifiers for original weights, labels, splits, images, and code to avoid mixing incompatible experiment states.
- CPU tests and real-sample audits had been completed when this specification was written, but the full 695-image download and actual T4 training had not run in this cloud environment. Actual continuation-training PQ and LB results were not yet available at that time.
