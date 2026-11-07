# GONG + all labeled SWEFil + CHASE-shape training

Status: the uploaded run completed 100 epochs and selected **epoch100, native-only development PQ 0.410866**. All nine exported validation CSVs were independently rescored and passed mask-overlap checks. This is below the historical best84 native result of 0.429419 and the previous CHASE-shape result of 0.431339. On November 7, 2026, the user explicitly requested testing the entire labeled SWEFil pool without temporal or duplicate-based admission filtering. This replaces the earlier proposal to exclude nearby observations and the historical decision to defer SWEFil.

The next step is **test inference with these existing weights**, without further training: [one-cell T4 test notebook](https://colab.research.google.com/github/ikjoo5/solar_filament/blob/main/notebooks/SolarFilament_SWEFil_Test_T4.ipynb) · [submission guide](SWEFIL_TEST_SUBMISSION.md). The selected recipe is threshold0.65 / minimum area300; no leaderboard score is available yet.

## Run in Colab

[Open the single-cell T4 notebook](https://colab.research.google.com/github/ikjoo5/solar_filament/blob/main/notebooks/SolarFilament_SWEFil_Mixed_T4.ipynb).

1. Select a T4 GPU runtime and run the code cell with `PHASE="all"` and `RUN_UNTIL_EPOCH=100`.
2. Mount the existing Drive and authorize the Drive API with the same account. SWEFil downloads automatically; no separate ZIP upload is needed.
3. Keep the existing CHASE run at `filament-segmentation-2026/fixed_split_experiments/chase_shape_best84_recipe_t4_v1`, including its synthesis metadata and `synthetic_images_cache.zip`. The new run restores these assets without regenerating the synthesis or loading the old weights.
4. After interruption or a runtime reset, run the same cell. It restores the saved source version and resumes after the last verified complete epoch; a partially trained epoch is repeated.

New output directory:

```text
/content/drive/MyDrive/filament-segmentation-2026/fixed_split_experiments/swefil_all_mixed_native_t4_v1
```

`training.log` records live progress. `train/` contains human-readable metrics, history and state mirrors. Model weights and authoritative state generations are stored through the Drive API in **`verified_checkpoints/`**, with file IDs, SHA-256 hashes and best/last links recorded in `latest_result.json`. The review archive is `swefil_mixed_native_results.zip`; it does not contain model weights. Only one runtime should train this experiment folder at a time.

The source snapshot keeps resume independent of later GitHub edits. Dependency downloads may still be needed after a Colab runtime reset. Use `PHASE="prepare"` to download and verify data without training. A smaller `RUN_UNTIL_EPOCH` pauses at that completed epoch without shortening the 100-epoch cosine schedule.

## Data and purpose

[SWEFil](https://huggingface.co/datasets/antonio-reche/SWEFil) supplies manually annotated GONG H-alpha observations. It can add real filament appearances and labeled sunspot regions without first translating CHASE images into the GONG observation domain. This follows the organizer's external-data permission already reported by the user. Improvement remains an experimental question.

Dataset credit: Antonio Reche and Consuelo Cid, *Solar filament detection, classification, and tracking with deep learning*, SPAICE2024, pp. 69–74.

The pinned revision is `bc818c69a79266ad0fa91a539359485af57967cf` (last modified October 22, 2025), licensed CC-BY-4.0. The actual public COCO annotations contain:

- **250 labeled physical observations:** 248 contain filaments; two contain sunspots only.
- **1,670 filament instances:** 1,129 QRF, 285 IRF and 256 ARF; all three types map to filament foreground.
- **402 sunspot annotations:** these remain background for filament segmentation, except where a filament-positive annotation takes precedence.
- Another 27 observations have no annotations at all. They are omitted because missing labels do not establish empty background.
- Raw and processed records represent the same observations. Use one processed 2048 × 2048 JPEG per observation, rather than counting both representations as independent samples. The 250 JPEGs total approximately 75.9 MiB.

Both public SWEFil annotation splits contribute to this training pool. SWEFil's file named `test.json` is a publicly labeled external split, not the competition test set. It consequently cannot serve as a held-out SWEFil benchmark in this experiment.

The dataset README's larger object totals count raw and processed representations twice; its stated 227 unique observations also differs from the 277 in the actual metadata. The announced extended edition is not assumed available.

## What is included and what validation means

**All 250 labeled SWEFil observations are admitted. No date-proximity or competition-overlap filter removes an observation from this run.** This is the user's requested broad-data development experiment.

The original competition split stays at 616 training and 91 validation images. Competition validation labels are used for evaluation and checkpoint selection, not gradient updates; competition test images remain prediction-only. Existing CHASE shapes and the external clean background remain the synthesis sources.

There is no exact normalized observation-name match between SWEFil and the 887 competition image names. That does not establish independence: solar structures persist, and differently timed observations may show the same filament. Six SWEFil observations share dates with held-out competition observations (one validation, five test). Pixel-level duplication has not been ruled out.

Report validation in three ways:

- **Full 91-image validation:** preserves the previous development-score comparison.
- **17 validation images within seven days of a labeled SWEFil observation.**
- **74 validation images farther than seven days from all labeled SWEFil observations.**

These groups are reporting diagnostics only. They never filter the training pool. The seven-day boundary does not guarantee independence, and repeatedly tuned validation scores are development evidence. Compare each diagnostic group against the same images from the baseline, rather than interpreting difficulty differences between groups as gains. A subsequent competition submission is needed to measure the new model's actual LB result; the public leaderboard itself is also a development signal.

## Training recipe

Start a **new native model with an ImageNet encoder and a new decoder**. Neither the original solar best84 weights nor the surviving epoch42/51 weights initialize this run. Existing CHASE synthesis assets are reused, not old trained weights.

| Setting | Value |
|---|---|
| Model | ConvNeXt-Tiny U-Net, foreground and spine outputs |
| Training | One run, 100 epochs |
| Device / precision | Colab T4 / FP16 |
| Crop / batch / gradient accumulation | 1024 × 1024 / 1 / 4 |
| Epoch budget | 400 random image-first crop presentations; 100 attempted optimizer updates |
| Optimizer | AdamW, LR 1e-4, weight decay 0.01 |
| Schedule | Cosine across 100 epochs; minimum LR 5e-6 |
| Input processing / augmentation | Existing best84 recipe: limb correction, ImageNet normalization, gamma/brightness/D4 |
| Epochs 1–3 and 95–100 | Competition GONG training images only |
| Epochs 4–94 | **GONG 30% / SWEFil 60% / CHASE-shape 10%** |
| Validation | Original fixed split; PQ, SQ, RQ, TP, FP, FN and recall |
| Selection grid | Foreground thresholds 0.35/0.50/0.65 × minimum areas 100/200/300 |

The user requested this **30/60/10 mixture**, increasing exposure to SWEFil's real filament examples. Each mixed epoch averages 120 original GONG crops, 240 SWEFil crops and 40 CHASE-shape crops. Across 91 mixed epochs, SWEFil contributes approximately **21,840 crop presentations**, or **87 per observation** on average. Random crops do not guarantee full-image coverage each epoch; all 250 observations are eligible. The first three and final six GONG-only epochs retain the existing training schedule.

This changes both external-data exposure and the previous GONG/CHASE balance: CHASE sampling drops from 50% in the completed shape experiment to 10% during mixed epochs. These are declared experiment choices, not measured optimum values.
The model, optimization and crop budget match the best84 comparison recipe; the data mixture changes. This is a practical broad-data experiment, not a matched causal estimate of SWEFil alone. There is no additional compulsory 100-epoch control run or five-fold training.

## Label integration

- Merge QRF/IRF/ARF into filament foreground, retaining instance identities for diagnostics.
- Union all polygon parts within one annotation. In this release, 581 filament instances have multiple polygons; their parts must not become extra object labels.
- Filament-positive pixels take precedence where filament and sunspot annotations overlap.
- SWEFil has no genuine MAGFiLO spine labels. Disable its spine loss rather than treating a missing spine label as an all-negative target.
- Keep the existing two-output architecture. No additional sunspot head, context block or YOLO training is introduced here.
- Preserve original competition GONG pixels and existing CHASE clean-background synthesis. Downloaded SWEFil pixels are cached with checksums; processing occurs in a separate cache.

## Checkpoint durability

The prior completed-run ZIP recorded epoch100 metrics and a selected best82, but the reviewed best82 weights were not found in the completed Drive searches. The available state instead points to epoch51/best42. Drive was almost full at the time, and older checkpoint files appeared in trash; these observations do not establish the exact cause. The user has expanded total storage to 200 GB.

The new run must:

1. Stage and hash a complete checkpoint on Colab's local disk.
2. Upload an immutable generation through the Drive API and verify the remote file before committing a durable epoch state.
3. Stop on publication failure. A computed epoch is not reported as safely saved until remote verification and state commitment succeed.
4. Verify the selected best separately from the resumable last checkpoint.
5. Keep checkpoint generations. Do not automatically delete `.pt` files, empty trash, or alter the previous experiment folder.
6. Check API write access and available quota before GPU training. At the previous checkpoint sizes, retaining 100 full checkpoints and up to 51 best checkpoints needs approximately **43 GB before overhead**. The fresh run requires 50 GB free; the reserve decreases with the remaining epoch budget on resume. Total account capacity alone is not sufficient.
7. Publish verified best/last file IDs and hashes at completion. Keep the small metrics ZIP distinct from the model weights.

Runtime interruption can still lose work since the last remotely committed epoch. Resume must use the newest verified remote state and verify downloaded model, optimizer, scheduler, scaler and RNG state; a local JSON file alone is not proof of remote checkpoint durability.

## Reference measurements

- Original best84 native model in the same T4 evaluation: **PQ 0.429419**.
- Completed GONG + CHASE-shape run, selected epoch82: **PQ 0.431339**; reviewed weights currently unavailable.
- Surviving epoch42: recorded **PQ 0.409225**.
- Original production YOLO + best84 ensemble: **public LB 0.39**, development PQ 0.472593.

These are distinct evaluation configurations. The new run evaluates a native model alone; neither its eventual validation PQ nor external-data availability guarantees a leaderboard gain.

## Checks completed before release

- Downloaded and verified all 250 pinned processed JPEGs and decoded each at native 2048 × 2048.
- Rasterized all 2,072 labeled objects, keeping multiple polygons within each instance.
- Repeated complete preparation with network access disabled: zero network requests, identical manifest, approximately 2.3 seconds for the already populated local cache. A new Colab runtime still needs to restore its local cache from Drive.
- CPU tests cover domain sampling, missing-spine supervision, identical interrupted/uninterrupted optimizer and RNG state, failed remote publication, cached source loading and cancellation. Fake Drive tests exercise API contracts.
- Authenticated Drive publication and full GPU training cannot be exercised in this cloud session; the Colab worker checks API access and quota before building the training model.
