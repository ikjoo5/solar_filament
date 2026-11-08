# Frozen best84 postprocessing experiments — 8 November 2026

The reference for this study is the original **YOLO26s-seg + best84** ensemble:
development PQ **0.472593**, with **488 TP, 204 FP and 220 FN**. Its user-reported
public leaderboard score is **0.39**. This is separate from the public-prediction
hybrids and the newer mixed-data models.

None of the 30 configurations improved this reference. Existing model
weights, production postprocessing and the ongoing full-pass training were left
unchanged.

Nine configurations produced the baseline result and 21 reduced PQ.

## Evaluation protocol

- Reused the frozen validation cohort: **91 physical images, 101 annotator–image
  groups and 708 annotated instances**. The baseline contains 625 physical
  predicted instances; counts in the metric include evaluation against each
  annotator group.
- Replayed cached geometric, photometric and tiled YOLO predictions and the
  original best84 base/TTA predictions. Original RLEs reproduce the historical
  baseline exactly. Nearby T4 replays with PQ 0.472274 are not substituted for it.
- Froze each family's rules and prediction files before scoring that family's
  alternatives. The native-only residual controls and native-veto follow-up
  were added after earlier results; the entire study was not prespecified at
  the outset.
- Computed global PQ using strict mask IoU > 0.5, with every validation group
  included. Independently decoded and rescored exported CSVs and checked mask
  overlaps.
- Checked physical-image paired bootstrap differences and chronological
  diagnostic blocks. These are repeatedly used development images, not new
  training folds or an independent test set. Conditional intervals do not
  account for historical checkpoint/method selection.
- No public-reference RLEs, test labels, new GPU inference or leaderboard
  submissions were used. Original best84 caches contain threshold masks and
  component statistics, not arbitrary continuous probability maps.
- The fixed public-reference panels of 216 additions and 102 replacements are
  not rescored here: no new test predictions were generated. Their correspondence
  remains a separate diagnostic for a future candidate that passes validation.

## Results

| Family | Configurations | Best result within family | PQ | FP | FN |
|---|---:|---|---:|---:|---:|
| Frozen reference | — | Original pipeline | **0.472593** | **204** | **220** |
| Residual instance recovery | 12 | Stronger YOLO support makes no changes | 0.472593 | 204 | 220 |
| Cross-family boundary replacement | 8 | Agreement from all three other families | 0.471950 | 205 | 221 |
| Transverse intensity-width agreement | 4 | Strict width rule | 0.442054 | 337 | 202 |
| Release occupancy of already-pruned masks | 2 | Both rules make no changes | 0.472593 | 204 | 220 |
| Restore masks rejected by final native confirmation | 4 | Strong YOLO confirmation makes no changes | 0.472593 | 204 | 220 |

These are configuration comparisons, not 30 independent studies. Several
configurations produce identical predictions. None is adopted.

### Residual instance recovery

Some native components overlap an existing instance and are rejected by the
whole-component novelty gate. This experiment subtracted occupied pixels and
tested whether a remaining connected region represented another filament.
Native high-threshold cores, agreement between views and independent YOLO
support were used as confirmation.

- Four of eight YOLO-supported settings left the output unchanged; four added
  the same single FP, giving PQ **0.472256**.
- Four follow-up native-only settings added 13–29 physical masks and recovered
  **zero GT matches**. PQ fell to **0.462679–0.467914**.
- Strong agreement on leftover pixels did not establish a complete additional
  instance. In the threshold-0.35/area-300 native-only diagnostic, the largest
  added-mask GT IoU was only **0.4068**.

### Cross-family boundary replacement

Candidates were compared against other source families, excluding their own
family from the supporting score. A proposal could replace one incumbent mask
only; ambiguous multi-instance matches were rejected and other instances'
pixels remained reserved.

The strongest rule replaced 37 masks. It gained eight GT matches but lost nine.
It shrank 32 of the changed masks, with a median replacement area of 70% of the
original. Agreement often favored a compact core rather than the full annotated
object. All eight rules reduced overall PQ. The conditional paired-bootstrap
interval for the best rule's PQ difference was approximately **[−0.00769,
+0.00679]**, which provides no evidence of an improvement.

### Transverse intensity-width agreement

Rather than using average darkness, this test sampled intensity across a
candidate's local centerline. It compared the width and edges of the dark
intensity trough with the proposed mask, while accounting for local illumination
and noise. Pixels absent from the saved tile cache were excluded, not treated
as black background.

- Usable profiles existed for 1,094 candidate masks. Other candidates lacked
  mask/profile pixels or adequate geometry.
- Adding all eligible candidates gave PQ **0.422152**, FP 448 and FN 184.
- The strict rule reduced these to FP 337 and FN 202, but relative to the
  original pipeline it still added **133 FP** while recovering **18 TP**.
- It reduced PQ in four of five chronological diagnostic blocks. Its conditional
  paired-bootstrap PQ-difference interval was approximately **[−0.04263,
  −0.01786]**.

This measured width descriptor does not reliably separate the difficult
remaining candidates. It is not adopted or inverted after observing labels.

### Occupancy release

The original pipeline tests native admission before its final pruning step.
The experiment checked whether regions occupied by subsequently removed masks
were blocking otherwise acceptable new instances.

Of 500 native components passing the non-occupancy gates, 488 were rejected by
occupancy. The pipeline subsequently removed 17 masks, but **none of those 488
candidates intersected a removed region**. Releasing these regions therefore
changed no predictions under either the three-view or four-view rule. This
ordering did not explain the misses in this cohort.

### Restore masks rejected by final native confirmation

After the candidate-availability audit below, four follow-up rules tested only
the 17 masks removed by final native confirmation. All incumbent masks stayed
unchanged. Restored masks had to retain at least 95% of their area after clipping
against occupied pixels and remain at least 300 pixels.

Restoring all 17 recovered two GT instances but added 16 FP, giving **PQ
0.468704, TP 490 / FP 220 / FN 218**. Three stricter rules requiring two or three
supporting geometric views, or geometric plus photometric support, restored
nothing. Directly reversing these final deletions therefore does not recover
most of the missed instances available in the raw pool. Earlier candidate
selection and changes to instance boundaries remain the larger unresolved issue.

## Candidate availability diagnostic

The exact current baseline was also audited against the complete exported
candidate pool, using ground truth only for this diagnostic:

- **666/708 GT instances (94.07%)** have at least one candidate with IoU > 0.5.
- Of the baseline's **220 FN**, **178 (80.91%)** already have such a candidate;
  42 do not.
- Those 178 comprise **112 with a matching YOLO mask only**, **65 with both
  YOLO and native matching masks**, and **one with a native match only**.
- "YOLO only" means no exported native component at the examined thresholds
  reaches IoU > 0.5. It does not imply absent native foreground support.
- Independent per-GT RLE recalculation reproduced all 708 rows. One-to-one
  candidate matching within each annotator group preserves these coverage
  counts, but does not enforce a jointly nonoverlapping output across all
  selected candidates.

These are candidate-availability counts, not achieved recall, guaranteed
recovery or achievable PQ. They support investigating candidate acceptance and
instance completeness while continuing to improve the detector for the 42
uncovered misses. They do not support indiscriminately adding the raw pool.

## Where the valid raw candidates disappear

An additional diagnostic replayed seven unchanged production stages. This is
not another parameter search; every stage's RLEs were frozen before evaluation.

| Selected stage | PQ | TP | FP | FN |
|---|---:|---:|---:|---:|
| Geometric consensus |0.457152|467|199|241|
| Photometric rescue |0.459118|479|215|229|
| Boundary smoothing |0.460759|479|215|229|
| Tiled-mask refinement |0.464939|481|213|227|
| Native-reference additions |0.465562|483|215|225|
| Native boundary refinement |0.467844|485|213|223|
| Final admission and pruning |**0.472593**|**488**|**204**|**220**|

Of the **178 final misses with a valid raw candidate**, **174 never reach IoU >
0.5 at any of these selected stages**. Only four are correct at an earlier
selected stage and later lost permanently: two during smoothing and two in the
final stage. Among the 112 YOLO-only raw-covered misses, 110 never match at a
selected stage and two are lost in final pruning.

Thus, broad removal of downstream refinements is not supported: the complete
pipeline improves aggregate PQ over every listed intermediate stage. The larger
opportunity lies in getting the correct raw proposal, with its complete shape,
through selection and fusion. This diagnostic does not distinguish rejection
of a raw proposal from a damaging shape change during selection/fusion, and does
not provide a deployable rule for identifying the 174 instances without labels.

## Reproducibility artifacts

The workspace retains each family's generating script, frozen protocol, input
hashes, prediction CSVs, full metric records and independent audits:

```text
/workspace/artifacts/best84_probability_20261108/
/workspace/artifacts/best84_candidate_20261108/
/workspace/artifacts/best84_width_20261108/
/workspace/artifacts/best84_occupancy_20261108/
/workspace/artifacts/best84_occupancy_20261108/stage_loss_audit/
/workspace/artifacts/best84_screen_20261108/independent_report.json
```

The combined independent scorer is
`best84_screen_20261108/independent_score.py`. No new test submission is warranted
by these results. The public leaderboard reference remains **0.39**.
