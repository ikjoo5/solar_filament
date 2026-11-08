# Fresh YOLO and U-Net training with complete image coverage

This experiment trains two new solar models using every available training image
in every epoch. It replaces the previous400-crop epoch and probabilistic domain
mixture. It does not continue the old solar YOLO, best84, or SWEFil epoch100
weights.

[Open the one-cell Colab notebook](https://colab.research.google.com/github/ikjoo5/solar_filament/blob/main/notebooks/SolarFilament_FullPass_Both_T4.ipynb).

## Data coverage

| Training source | Physical images |1024×1024 tiles per epoch |
|---|---:|---:|
| Official GONG training split |616|2,464|
| Annotated SWEFil observations |250|1,000|
| Existing CHASE-shape synthetic images |616|2,464|
| **Total, for each model** |**1,482**|**5,928**|

- Each2048×2048 image contributes its four nonoverlapping quadrants, exactly once
  per epoch. Only their order is shuffled. No image is selected with replacement.
- All three sources appear in every epoch, including the beginning and end of
  training. Their resulting image shares are approximately41.57%,16.87%,41.57%;
  these are consequences of the pool sizes, not sampling probabilities.
- Blank tiles and the two sunspot-only SWEFil observations remain in training.
  SWEFil sunspots are background for the filament task.
- The91 official validation images remain excluded from official training.
  All250 annotated SWEFil observations are included as requested; external
  temporal proximity remains reported, so validation is a development estimate.
- Several GONG images have separate observer annotations. One observer is used
  consistently for all four tiles of that image during an epoch. The observer
  rotates across epochs; conflicting labels are not combined into duplicate
  instances.
- Existing CHASE synthetic PNGs are restored from their verified Drive archive.
  They are not regenerated. Verified SWEFil downloads are reused. Local
  preprocessed/tile caches may be rebuilt after a Colab runtime reset.

The source images stay intact. Four-tile training exposes every source pixel,
but objects crossing a tile boundary are seen as clipped fragments. Full-field
validation measures the effect of this choice.

## Models and optimization

| Setting |YOLO segmentation |Native U-Net |
|---|---|---|
| Architecture |YOLO26s-seg |ConvNeXt-Tiny encoder + U-Net decoder |
| Fresh initialization |Official COCO weights, new one-class solar task |ImageNet encoder, new decoder |
| Existing solar weights |None |None |
| Schedule |100 epochs |100 epochs |
| Training input |1024×1024 native tiles |1024×1024 native tiles |
| Physical batch / effective batch |1 /4 |1 /4 |
| Optimizer |AdamW, learning rate1e-4, weight decay0.01 |Same |
| Learning-rate schedule |Cosine across100 epochs, minimum5% |Same |
| Precision / data workers |FP16 /0 |FP16 /0 |
| Update attempts per complete epoch |1,482 |1,482 |

"Fresh" means a new solar training run using general-purpose pretrained weights,
not fully random weights. The two models run sequentially on oneT4, by default
YOLO followed by U-Net. Completing both means100+100 model epochs. Each epoch
contains14.82 times as many training tiles as the earlier400-crop native run;
wall time should be estimated from the first completed epoch on the actual GPU.

Limb correction is applied to each complete image before cutting the tiles.
The U-Net additionally uses its existing ImageNet normalization and intensity/
rotation/reflection augmentations. YOLO disables mosaic, mixing, spatial
scaling, translation, and random cropping so they cannot defeat full coverage.
Its instance polygons and any raster representation differences are recorded
in the tile manifest. The native loss remains foreground BCE+Dice with a0.1
auxiliary spine term only where genuine GONG spine labels exist; external and
synthetic data do not acquire fabricated spine supervision.

Each model is selected separately using official-style PQ on the fixed91-image
development set at epoch1, every second epoch, and epoch100. Native validation
uses overlapping1024 tiles and the existing threshold/area grid. YOLO validation
uses full2048 images at native scale, confidence thresholds0.1/0.2/0.3/0.4 and
minimum areas100/200/300, with confidence-ordered nonoverlapping masks. This run
does not yet measure or select an ensemble, and it makes no public-LB improvement
claim.

## Running and resuming

Run the notebook's single cell in aT4 runtime. Defaults:

```python
MODELS = ["yolo", "unet"]
RUN_UNTIL_EPOCH = 100
PHASE = "all"
```

The output is stored in:

```text
MyDrive/filament-segmentation-2026/fixed_split_experiments/fullpass_both_t4_v1/
  code/                       # immutable source snapshot
  experiment_config.json
  coverage_manifest.json
  prepared_summary.json
  training.log
  yolo/                       # history, PQ grids, validation CSVs, checkpoint links
  unet/                       # history, PQ grids, validation CSVs, checkpoint links
  verified_checkpoints/
    yolo/                     # Drive API verified model/optimizer checkpoints
    unet/
  fullpass_both_results.zip    # small review bundle, without images or weights
```

Run the **same cell** after a reset. It restores the saved code and verifies its
checksums, queries server-confirmed checkpoint records, and skips already
completed training. `RUN_UNTIL_EPOCH` is an execution cap; changing it does not
shorten or restart the fixed100-epoch learning-rate schedule. For example, a cap
of5 runs each selected model through epoch5; returning it to100 continues both.
`MODELS=["unet"]` or `["yolo"]` can resume one branch independently.

- U-Net saves within-epoch progress every100 optimizer updates, normally400
  tiles. Resume starts at the next saved tile with optimizer, scheduler, scaler
  and RNG state restored. Work after the last confirmed save may repeat.
- YOLO resumes from the last **completed epoch**; an interrupted epoch repeats.
- Checkpoints are uploaded through the Drive API and verified before committing
  progress. Mounted Drive files alone are not treated as proof of durability.
- To bound storage, the new run retains checkpoint references from its latest
  two completed epochs and latest two native progress commits, including the
  selected best weights. Older committed weights belonging only to this new
  run are removed after a newer verified commit. Existing experiments and source
  data are not touched. Interrupted, uncommitted uploads are not swept.
- One active Colab runtime should write to this experiment at a time.

After training or an execution cap completes, the notebook offers the review ZIP.
It contains per-model histories, validation results and checkpoint links. The
weights remain in Drive; large images and checkpoints are not embedded in the
notebook or uploaded to GitHub.
