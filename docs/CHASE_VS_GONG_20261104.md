# Differences between CHASE and the competition's GONG inputs

Written: 2026-11-04. This is a historical experiment record; see the [current pipeline](CURRENT_PIPELINE.md), [independent results](CHASE_INDEPENDENT_RESULTS.md), and [next experiment design](CHASE_GONG_NEXT_EXPERIMENT.md) for subsequent findings and plans. As requested by the user, SWEFil was excluded from this experiment and CHASE was considered first. The plan follows the organizer's external-data permission reported by the user. At the time of writing, no PQ improvement from the additional data had been measured.

## User-confirmed scope

- Learn filament shape, contrast, and boundary features from CHASE. Right/left chirality classification is not a training objective.
- Under the user's latest instruction at the time, exclude overlap cases only when the data explicitly identify them. Otherwise, retain them for training. Do not use automatic exclusion rules based on crossings, branching, or connected-component size.
- Public binary masks do not contain original object IDs, so actual overlap cannot be identified completely automatically. Handle explicit exclusion regions with loss-ignore or exclusion of the entire YOLO observation; do not relabel filament pixels as background.
- Use CHASE alone for this continuation training, without GONG replay or a final GONG calibration stage. Reconsider additional GONG training at a future final-model stage. Retain the requirement that final output masks have no overlapping pixels.
- Continue training both YOLO and best84 from the existing 0.39 pipeline, then evaluate the four original/new model combinations. Follow the [execution design](CHASE_BOTH_TRAINING.md).

## Confirmed CHASE input preprocessing

- Apply **disk alignment only**, following the user's latest choice at the time. The previously proposed comparison with mild brightness flattening is also excluded from this baseline experiment.
- Estimate disk center and radius from the distributed PNGs and align their centers and sizes to the GONG reference coordinates. Use translation and uniform scaling, without nonlinear shape distortion from unequal horizontal/vertical scaling.
- Apply the same coordinate transformation to image and mask. Use appropriate interpolation/antialiasing for downsampling images and nearest-neighbor transformation for masks. Preserve original files and masks.
- Do not apply additional limb-darkening correction, CLAHE, percentile stretching, gamma/contrast changes, sharpening, blur, black-hat processing, or learned super-resolution to CHASE. Brightness and blur augmentation are also disabled in this condition.
- Preserve the model's required uint8-to-float scaling, replication of grayscale into 3 channels, and the existing encoder's fixed mean/std normalization. These are numerical input representations, rather than image corrections that change local contrast or boundary shape.
- Alignment itself involves interpolation, so inspect preservation of thin masks and the disk edge. Do not force images into an arbitrary transformation when disk estimation fails.
- These choices apply to CHASE continuation-training inputs. Keep the original preprocessing for GONG validation to separate the effect of additional CHASE training from changes to GONG preprocessing. Do not use GONG training images in this continuation run.

## Observation methods

| Property | Competition GONG | CHASE |
|---|---|---|
| Location | Multiple ground-based observatories | Space-based satellite |
| Atmospheric effects | Seeing, clouds, and differences between stations | No terrestrial seeing or cloud effects; other instrument noise and optical limits remain |
| H-alpha observation | Filter images centered at 6562.8 Å; station-dependent bandwidth about 0.5–0.6 Å | HIS measures spectra around H-alpha, with a wavelength sampling interval of 0.024 Å |
| Nominal spatial resolution | Approximately 2 arcsec according to NSO | Approximately 1.2 arcsec according to the research team |
| Temporal acquisition | Full-disk images, approximately 1 minute per station | Slit raster scanning acquires the full disk in about 46 seconds to 1 minute |
| Files examined for training | MAGFiLO 2048×2048 JPEGs | Verified AHAS/MHAS samples were 2048×2048, 8-bit grayscale PNGs |

- Wavelength sampling interval and filter bandwidth are different physical quantities. Dividing 0.5 by 0.024 does not provide a spatial-resolution or image-quality multiplier.
- CHASE's nominal spatial resolution and absence of terrestrial seeing are advantages, but the public PNG samples do not contain more pixels than GONG. Effective resolution also depends on observing mode, binning, export conversion, and the optical PSF.
- H-alpha intensity at a narrow wavelength differs from intensity integrated through a broader filter. Even the same filament can differ in darkness, width, and visibility because of motion-induced Doppler shifts and filter bandwidth.
- Do not confuse CHASE continuum/Fe I or other channels with H-alpha. This review examined the H-alpha images in the public segmentation dataset.
- CHASE images acquired at different times or in different observations are not sharpened ground-truth counterparts to the same GONG images. They alone cannot form supervised GONG-to-CHASE super-resolution pairs.

## Findings from actual files

The ZIP central directory and selected members were fetched using HTTP Range requests and checked against their CRCs. The full 1.2 GB dataset was not downloaded.

| Dataset | Verified image | Image | Mask |
|---|---|---|---|
| AHAS | `RSM20240101T061747_0000_HA.png` | 2048×2048, uint8, grayscale | 2048×2048, pixel values 0/255 |
| MHAS | `01.png` | 2048×2048, uint8, grayscale | 2048×2048, pixel values 0/255 |

- The verified masks are **semantic binary masks of filament/background**. They have no per-filament instance IDs.
- MHAS contains 120 images whose initial k-means labels were manually corrected. The paper describes adding dark filaments above active regions, connecting fragments within the same filament channel, and correcting faint boundaries.
- AHAS contains 815 images: 695 observations from 2024–2025 plus relabeled versions of the 120 MHAS images. Its labels were produced by a model, DenseCRF, and hole filling, then screened by people. This does not mean every pixel was independently checked by hand.
- The 120 MHAS images and the 120 included in AHAS are the same observations. Do not count them as 935 independent images. Split and sample by original observation.
- Verifying the 2048 dimensions of two samples is not validation of every file in the dataset. Recheck individual file dimensions, formats, and integrity during full preparation.
- Actual comparison image: `/workspace/artifacts/chase_comparison_20261104/chase_samples_preview.png`. It preserves the supplied PNG brightness and shows original crops beside the public masks. The two inspected samples had disk radii of approximately 940 pixels, differing from the GONG-specific 900 pixels. Estimate and align disk geometry separately for training.

## Connecting the data to our models

1. Estimate each CHASE solar disk's center and radius and align it to the GONG reference. Under the confirmed conditions above, apply no additional contrast correction. Do not crop it to the GONG-specific radius of 900 before alignment.
2. Initially use it to train the U-Net encoder and foreground segmentation. Avoid extracting binary-mask connected components and declaring them to be competition instance ground truth. A single filament can be disconnected, and multiple objects can touch.
3. For separate native heads such as spine, omit the loss on external samples when labels with the same definition are unavailable, or establish a separately validated label-generation rule. Do not indiscriminately feed the same label to both existing heads.
4. Do not assume CHASE automatic labels and competition manual labels have equal quality. Start by evaluating the user-selected CHASE-only continuation. Additional GONG fine-tuning or a mixed-data control remains a future option and is not part of this run.
5. Start new-model comparisons on the existing fixed validation set. The current comparison measures additional CHASE training against the existing weights. It does not run a matched-budget GONG-only control and therefore cannot establish superiority over that approach. Validation is not input for external-mask training or label correction.
6. Evaluate actual PQ, TP, FP, FN, and detection of small/low-contrast objects. An improvement in CHASE semantic IoU/Dice is not a substitute for competition instance PQ. Keep the public references 216+102 as a separate candidate-coverage diagnostic.

At this stage, the work had examined public sample images/labels and observation descriptions; it had not performed GPU continuation training or PQ calculation.

## Sources and storage

- [CHASE mission overview, 2022](https://doi.org/10.1007/s11433-022-1893-3)
- [CHASE research team first results, 2023](https://doi.org/10.5194/egusphere-egu23-3035)
- [NSO GONG H-alpha data description](https://nso.edu/data/nisp-data/h-alpha/)
- [MORDEN paper, 2026](https://arxiv.org/abs/2607.24525)
- [Public AHAS/MORDEN data](https://zenodo.org/records/21582245)
- [Public MHAS data](https://zenodo.org/records/17230605)

File samples, SHA256 values, CRC-checking code, and retrieved source material are stored under `/workspace/artifacts/chase_comparison_20261104/`. The earlier broad candidate-data survey is under `/workspace/artifacts/external_data_research_20261104/`.
