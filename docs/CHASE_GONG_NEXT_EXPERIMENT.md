# CHASE shape synthesis and GONG image adaptation: next experiment

Status: **historical experiment design**. A clean-background CHASE-shape renderer and fresh native-training entry are now implemented in [GONG + CHASE-shape T4 training](CHASE_SHAPE_T4_TRAINING.md), following the user's later choices. That guide supersedes this document's proposed continuation pilot and full-pass defaults for the current execution. Whole-image CHASE-to-GONG conversion and the broader A–D comparisons below remain proposals. No new training result is claimed. SWEFFI remains deferred until the final stage, as requested by the user.

The earlier [synthesis protocol](CHASE_SHAPE_SYNTHESIS_PROTOCOL.md) records proposed backgrounds, shape sampling, rendering, labels, and matched controls. Use the current execution guide above for active defaults.

## Initial recommendation

With the code and data currently available, **a small pilot using CHASE mask shapes for local synthesis on GONG images is the more practical starting point for implementation and causal analysis.** It preserves GONG backgrounds and existing labels while allowing control over the contrast, size, position, and labels of added objects. Much of the tested copy-paste tooling can be reused.

Adapting complete CHASE images to GONG is also a useful comparison. It preserves real image structure and context, but the current 8-bit PNG files cannot exactly reproduce GONG's spectral response, atmospheric blur, or calibration process. Prioritizing synthesis concerns initial implementation cost and experimental control; it does not mean that a PQ improvement or final superiority has been demonstrated.

Another advantage of local synthesis is that **only selected positive CHASE shapes need to be added**. This reduces dependence on retraining with every CHASE pixel outside an AHAS mask treated as certain background. This has not been established as the sole cause of the current regression, but it allows us to separate the risks from missing external automatic labels and changing observation backgrounds. Existing GONG backgrounds can also contain unlabeled objects, so insertion-site checks and preservation of known ground truth remain necessary.

| Criterion | CHASE shapes → local GONG synthesis | Real CHASE → GONG image adaptation |
|---|---|---|
| Information transferred | Masks, centerlines, length, curvature, and branching | Real filament structure, intensity, and surrounding context |
| GONG background | Uses actual GONG training images directly | Must approximate the distribution of background brightness, texture, and noise |
| New implementation needed | A renderer that gives shapes GONG-like contrast | Image-statistics estimation and a constrained observation-characteristics converter |
| Reusable code | Split checks, insertion, overlap prevention, label preservation, logging, and resume | CHASE downloads and integrity checks, image/mask alignment, and Dataset |
| Main failure risks | Edge artifacts, unnatural texture, and excessive tiny fragments | Spectral and labeling differences remain even if the images look similar |
| Label meaning | Masks of explicitly inserted synthetic objects | Existing automatic AHAS binary labels |
| Initial purpose | Test the added value of external shapes under target observation conditions | Test the added value of real external images and the effect of image conversion |

## What the current results establish

The fixed GONG validation set contains 91 observations, 101 annotation groups, and 708 ground-truth objects. Models were evaluated independently with a single base view and the same minimal decoder. These values are neither the results of the original complete ensemble pipeline nor public leaderboard scores.

| Model | Raw-input PQ | Limb-input PQ |
|---|---:|---:|
| Original YOLO | 0.399286 | 0.396637 |
| CHASE 60-epoch YOLO | 0.077358 | 0.091471 |
| Original native U-Net | 0.400068 | 0.406802 |
| CHASE 60-epoch native U-Net | 0.072098 | 0.066450 |

- The current procedure of continuing training on CHASE alone for 60 epochs did not preserve GONG performance. This does not establish that CHASE itself is unusable.
- For the new native model with raw input, lowering the foreground threshold from 0.5 to 0.05 changed PQ only from 0.072098 to 0.084986. Dilation with a fixed radius of 3 increased PQ to 0.120052, but FP also increased from 518 to 559.
- For the new YOLO model with raw input and confidence 0.2, 317 of 579 false positives were small fragments with at least 90% of their predicted area inside a single GT object, while covering no more than half of that object. Another 181 false positives had no overlap with any GT.
- Lower confidence thresholds or switching limb correction on/off therefore do not resolve the problem on their own. Both the observation distribution and the supervision supplied by the external dataset need investigation.

## What synthesis transfers

A center coordinate `(x, y)` alone cannot represent a filament's shape. Use the **mask/contour and, when useful, the centerline** as shape material. The absolute coordinates on the original solar disk are not essential.

1. Align the original CHASE image and binary mask using the existing alignment code. Preserve the original image, mask, and alignment history.
2. Select donor shapes. AHAS connected components are not expert-assigned physical object IDs. Do not automatically group unrelated fragments as one original filament. Selecting shapes for synthesis is a separate operation from deleting labels from the original external data.
3. Estimate relative contrast, width, edge transitions, and blur ranges from real filaments and their surroundings in GONG **training** images. Avoid filling all CHASE masks with uniform black or applying the same contrast to every shape.
4. Create GONG-like dark intensity variations inside the shape and insert it into an actual GONG training background. Check for abrupt intensity changes at edges, rectangular patch artifacts, and labeled areas with no visible inserted signal. Record reuse of the relative absorption pattern from the CHASE image and shape-only synthesis as separate variants.
5. Preserve existing GONG objects and labels. Use the union of foreground masks from all training annotation groups, plus a margin, as an exclusion region for insertion. Added objects must also not overlap one another. This prevents conflicts with known annotations; it does not guarantee the absence of unlabeled objects.
6. Record the added mask as a synthetic instance. Apply coordinate changes consistently to the image and mask, and reject outputs that retain an object label when the object is no longer visible.

If a CHASE mask contains only a dark central fragment, copying it unchanged does not supply the broader object definition used by the competition. Derive width and area rules from the GONG training distribution and apply them consistently to both the image and mask. **Do not dilate, split, or merge donors to fit validation ground truth.**

The first renderer does not require a model that generates an entire solar image. Local contrast/texture synthesis with conservative quality checks is sufficient for a pilot. The result must not be presented as a physical observation equivalent to an actual GONG image.

## Comparing image adaptation

- After disk alignment, make limited adjustments to radial background brightness and local contrast using GONG training statistics. Do not indiscriminately match a single histogram that includes the black region outside the disk.
- Measure blur after matching spatial resolution and the disk's pixel scale. Do not assume every CHASE image is sharper than GONG and blur them all uniformly.
- GONG-like noise and compression ranges can be used, but separately check label visibility when transformations make small filaments difficult to see.
- A single PNG contains no wavelength-resolved information. CHASE spectral cubes and instrument response functions could enable an approximation of intensity through the GONG filter band, but conversion of the current PNG files is not equivalent to that physical reconstruction.
- Simultaneous CHASE/GONG observation pairs would provide more direct evidence for validating conversion and registering labels. Even registered labels should be treated as weak supervision because of time offsets, spectral response, viewing direction, and alignment errors.

## Sequential experiments with matched budgets

For the initial continuation-training comparisons A–D, start every branch from the original GONG-trained weights. Using the CHASE-only 60-epoch weights as a shared starting point would conflate the existing regression with the effect of the new data-use strategy.

| Experiment | Training data | Question |
|---|---|---|
| A | GONG train + GONG augmentation with the same budget | What is the effect of additional updates alone? |
| B | GONG train + GONG synthesis using CHASE shapes | Does placing external shapes into target backgrounds and contrast help? |
| C | GONG train + aligned real CHASE | What is the auxiliary-training benefit of real external images? |
| D | GONG train + real CHASE converted to GONG characteristics | What is the isolated effect of image adaptation relative to C? |

- **Implement and compare A/B first** to assess synthetic quality and added value. Compare real-image adaptation with the C/D pair. There is no need to launch four long training runs at once.
- Design the data layer for both YOLO and the native model. Measure independent-model PQ separately from final ensemble-pipeline PQ to determine which model benefits.
- Record initial weights, GONG exposure counts, total optimizer updates, seeds, and evaluation/checkpoint-selection rules. Do not attribute extra updates caused by adding more samples to the external data itself.
- Define an epoch as a complete pass through the specified training list. A fixed-size subset is not a full epoch. Fill the control's matching update budget with GONG augmentation.
- Do not claim full convergence from the first inspection interval. Extend only promising conditions under the same schedule, based on training curves and GONG PQ/FP/FN. A decrease in loss on automatic CHASE labels alone is not sufficient for adoption.
- Keep external-label weighting, selection, and uncertainty handling identical between C and D. Do not change the loss or batch-normalization policy while measuring the effect of image conversion.
- AHAS has automatic binary labels, so evaluate it as weaker auxiliary supervision than GONG's manual instance labels. External binary foreground loss and YOLO proxy-instance training are different problems. Audit YOLO's connected-component definition and bias toward small fragments separately.
- Compare training methods and hyperparameters on the existing fixed split. Subsequent generalization checks on the prescribed 5-fold setup and actual submissions are separate stages. Do not report the best value selected on repeatedly reused validation data as independent performance.

### Added proposal: continuation pilot versus a fresh training run

This subsection is a new clarification added to the English experiment plan in response to the user's question; it is not part of the original Korean translation. It proposes comparisons only and does not report a new training run.

There are two distinct questions: whether CHASE-derived shapes add useful information, and which initialization/training schedule produces the best final model. A short continuation pilot answers the first question economically; a fresh-training comparison can then address the second.

1. **Continuation pilot:** start both A and B from the same original GONG-trained checkpoint. Continue A with GONG-only augmentation and B with GONG plus CHASE-shape synthesis. Keep optimizer-update budgets, evaluation rules, and relevant GONG exposure counts matched and recorded. Repeated GONG images are not new independent data; their role is to retain supervision for the target observation domain while the weights are updated with new examples. The A control measures improvement from continued training itself, so that improvement is not automatically attributed to CHASE. This retention benefit must be evaluated rather than assumed. Do not start this pilot from the degraded CHASE-only 60-epoch checkpoint.
2. **Fresh-training comparison after evidence:** if synthesis quality and the continuation results justify a larger experiment, train matched GONG-only and GONG-plus-synthetic arms anew from the same generic pretrained initialization with no competition-image training history. This removes the historical GONG-only fine-tuning stage from both arms. Keep the original fixed train/validation split, synthetic provenance rules, update budget, and checkpoint-selection procedure comparable.
3. **Random initialization is a separate experiment:** starting a new task-specific run from generic pretrained weights is not the same as initializing every parameter randomly. A fully random initialization changes the amount of prior visual information and may require a different convergence budget. It is not automatically the fairer comparison or the better model, and fresh training is not guaranteed to outperform continuation.

In all cases, use only training observations and training-derived statistics to generate synthetic data. Keep validation and test inputs out of generation and training. If a later 5-fold study retrains these arms, each fold must use its own permitted training observations and initialization; a checkpoint that has already seen that fold's validation images cannot provide an unbiased OOF result.

## Leakage prevention and reproducibility records

- Use training data only to derive GONG backgrounds, textures, contrast statistics, and donor rules. Do not use validation/test images, labels, or public prediction RLEs as synthesis material or target labels.
- Track originals by observation filename/date, duplicate images, and annotator-specific variants. Keep synthesized or transformed derivatives in the same training role as their source observation.
- If a CHASE quality-evaluation split is created, split by original observation/time group. Do not call it a new independent holdout for weights already trained on all CHASE images.
- For each synthetic sample, record the GONG background SHA, CHASE image/mask SHA, source ID, seed, transformation matrix, insertion position, contrast parameters, mask areas before/after insertion, insertion-failure reason, and full configuration/code version.
- Keep large images, labels, training weights, and predictions locally or on Drive. Store code, designs, example configurations, and generation specifications on GitHub.

## Existing implementation and missing components

| File/API | Current behavior | Use in this design |
|---|---|---|
| `tools/chase_transfer_data.py`: `prepare_chase`, `align_and_screen`, `ChaseDataset.aligned` | Integrity checks for CHASE695 files, matching the published mask grid, disk alignment | Reuse image/mask loading in shared coordinates |
| `tools/targeted_synthetic_data.py`: `split_roles`, `annotation_mask` | Checks the fixed GONG split/annotation groups and creates masks | Build training-only background and shape statistics |
| `tools/targeted_synthetic_data.py`: `make_donor_patch` | Extracts measured dark log residuals from existing GONG donors | Reference for synthesis principles; this is not an implemented CHASE shape-only renderer |
| `tools/targeted_synthetic_data.py`: `paste_residual`, `_cached_record` | Local synthesis, exclusion-region checks, resumable image/label/provenance storage | Candidates for reuse as a shared synthesis/cache layer |
| `tools/targeted_synthetic_data.py`: `mask_to_yolo` | Converts masks to YOLO polygons and checks raster IoU | Check synthetic-object output format; inspect subsequent training-library resampling separately |
| `tools/chase_independent_plain_worker.py` | Independent original/CHASE-model × raw/limb inference and validation caches | Reference for an evaluation adapter with new weights/input contracts |

Do not assume the existing targeted generator's GONG donor ID list or historical limits such as a minimum of 300 px and maximum of 5,000 px are appropriate for CHASE. `prepare_dataset` has a fixed input contract for the existing v2/v3 data; new CHASE data must not be connected by merely renaming files. **A new input manifest, shape renderer, mixed dataset, matched-budget launcher, and corresponding validation are needed.** Preserve existing evaluation/submission code and develop this work in a new experiment namespace.

## Evidence

- [MORDEN paper](https://arxiv.org/abs/2607.24525), Sections 3.1.1 and 4.2: AHAS uses automatic binary labels produced by a model, DenseCRF, and hole filling, followed by human screening. The paper explains the need for separate, trustworthy validation data.
- [Public AHAS data](https://zenodo.org/records/21582245): source of the 695 observations from 2024–2025 used for this continuation training. Do not confuse the 120 overlapping MHAS observations with these additional 695 observations.
- [CHASE mission description](https://doi.org/10.1007/s11433-022-1893-3), [GONG H-alpha data description](https://nso.edu/data/nisp-data/h-alpha/).
- The repository's [CHASE/GONG comparison](CHASE_VS_GONG_20261104.md) is a historical record of the user's CHASE-only settings at that time. This document defines the current GONG+CHASE experiment design.
