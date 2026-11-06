# Independent CHASE evaluation and postprocessing diagnostics

[Korean original](CHASE_INDEPENDENT_RESULTS.ko.md)

The original and CHASE60 weights were compared with raw and limb-corrected inputs on the fixed GONG validation set: 91 images, 101 annotation groups, and 708 objects. YOLO and native were evaluated separately, using a single base view and minimal decoding. These are development-validation measurements and must be distinguished from public LB scores and the PQ of the original full ensemble.

The uploaded ZIP's 758 files and 728 frames were verified. Masks were reconstructed using a known local decoder, and CSV ordering and aggregate scores were checked. PQ/SQ/RQ/TP/FP/FN and the IoU sum were reproduced within 1e-12. The identities of the weights, original images, code, runtime, and validation split were also checked.

| Model | Input | PQ | TP | FP | FN |
|---|---|---:|---:|---:|---:|
| Original YOLO | Raw |0.399286|439|335|269|
| Original YOLO | Limb-corrected |0.396637|440|350|268|
| CHASE YOLO | Raw |0.077358|85|579|623|
| CHASE YOLO | Limb-corrected |0.091471|99|550|609|
| Original native | Raw |0.400068|468|408|240|
| Original native | Limb-corrected |0.406802|485|414|223|
| CHASE native | Raw |0.072098|77|518|631|
| CHASE native | Limb-corrected |0.066450|71|526|637|

## Completed additional CPU experiments using the caches

- Native foreground thresholds of 0.05/0.1/0.2/0.35/0.5/0.65/0.8 were compared using the same 8-connected-component and area-100 rules. With raw inputs, CHASE native produced PQ 0.072098 at 0.5 and 0.084986 at 0.05. Pixel recall increased from 23.25% to 26.01%.
- A prespecified dilation with radius 3 increased raw-input CHASE native PQ to 0.120052, but also increased FP from 518 to 559. The same dilation reduced the original native model's raw-input PQ from 0.400068 to 0.242995. This suggests that insufficient mask extent is one issue; it does not mean that this rule was adopted for deployment.
- In a comparison of YOLO confidence thresholds 0.05/0.1/0.2/0.35/0.5/0.7, the highest observed CHASE PQ was 0.089523 for raw inputs and 0.101528 for limb-corrected inputs. These values were selected on a repeatedly used development set and are not independent validation results.
- Among the saved single-base-view candidates, a candidate with IoU > 0.5 existed for 585/708 ground-truth objects with the original raw-input YOLO model, versus 99/708 with the raw-input CHASE model. These counts include the effects of confidence 0.05 and NMS 0.7 applied before saving. They must not be generalized to candidate recall for the full 30-view pipeline.
- Of the 579 raw-input CHASE YOLO false positives, 317 had at least 90% of their predicted pixels inside one ground-truth instance but covered no more than half of that instance. A small fragment in the correct location can therefore produce both FP and FN under instance-IoU evaluation.

## Interpretation and next direction

Simply removing limb correction, changing thresholds, or applying a small dilation did not recover the original performance. This does not establish that CHASE is unusable; it indicates that the CHASE-only 60-epoch adaptation strategy needs redesign. The individual causal contributions of image distributions, automatic-label extent and object granularity, and loss of the original GONG representation have not yet been isolated.

Given the currently available implementation and data, the recommended first comparison is local synthesis that places shapes from positive CHASE masks into GONG training backgrounds with GONG contrast. Mixing real CHASE data and adjusting its observation characteristics remain planned follow-up controls. See the [rationale, unimplemented components, and matched-budget experiment design](CHASE_GONG_NEXT_EXPERIMENT.md). SWEFFI is reserved for the final stage.

Raw predictions, ground truth, splits, images, and checkpoints are not included in the repository. Reproduction code is in `tools/chase_independent_plain_worker.py`, `tools/chase_independent_decode.py`, and `tools/chase_postprocess_ablation.py`. This document records completed experiments and makes no claim of improvements from new training or new image transformations.
