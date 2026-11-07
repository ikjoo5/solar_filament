# README image sources and reproduction

This directory contains the approved competition overview artwork, a credited NASA photograph, real Hα observations and annotations, archived model predictions, and a rendering illustration of data synthesis. The sections below identify each source and how it is used. Full training images, prediction exports and weights stay outside Git.

## Competition overview

`competition_overview.png` is original artwork created with OpenAI image generation and approved by the repository owner for the README. It presents the competition task as observation → filament regions → separate instance masks. The approved PNG is included unchanged.

- SHA-256: `91257ae0a75a069ff444f8ab53d2577652d324e2818c2e4001a26af18ba038b5`.

## Color introduction: NASA solar filament eruption

`solar_eruption_nasa.jpg` is the unmodified 1920 × 1080 image from [NASA SVS: August 31, 2012 Magnificent CME](https://svs.gsfc.nasa.gov/11095/).

- Credit: **NASA/Goddard Space Flight Center**.
- [Original JPEG](https://svs.gsfc.nasa.gov/vis/a010000/a011000/a011095/304-171_Overlay_Blend_Crop.jpg).
- The image combines SDO extreme-ultraviolet observations at 304 Å and 171 Å. It introduces the physical subject; it is not a GONG Hα competition input or a model result.
- Use follows [NASA's images and media guidance](https://www.nasa.gov/nasa-brand-center/images-and-media/); the README credits the source and does not imply NASA endorsement. NASA media terms are separate from the SWEFil and CHASE CC BY licenses below.
- SHA-256: `e67cce379ebd4f2a4afc180e3bd9f5c3a57b26585b65411dfb615010ddf0b5ae`.

## Shape-synthesis illustration

`synthesis_demo.jpg` shows the repository's shape-rendering method as four panels: background, donor shapes, rendered composite, and the corresponding instance labels. It uses public CHASE mask geometry and a procedural grayscale background, rather than redistributing the private clean-background image or a training sample. Display colors distinguish shapes and labels; solar intensity panels remain grayscale.

The figure is generated with `tools/build_readme_synthesis_demo.py`, which calls the existing `render_at` function in `tools/chase_shape_synthesis.py`. The demo's appearance parameters are illustrative, not values estimated from the private GONG training images. No model inference or training is performed. See [`synthesis_demo_provenance.json`](synthesis_demo_provenance.json) for the public source, hashes, seed, rendering parameters, and output checks.

The actual CHASE-shape training method measures appearance from GONG training observations and places CHASE shapes on copies of the supplied clean background; see the [training recipe](../../CHASE_SHAPE_T4_TRAINING.md). The README figure explains the rendering operation with independently reproducible inputs.

- Source: [CHASE AHAS dataset, Zenodo record 21582245](https://zenodo.org/records/21582245), DOI `10.5281/zenodo.21582245`, by **Jiarui Hu, Qi Hao, Zheng Zhi, P. F. Chen, Chuan Li and Yifan Meng**, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Geometry donor: `RSM20240101T061747_0000_HA`, connected components 38, 18 and 12 from the automatic binary mask. The figure translates these shapes onto a new background and adds display colors; component boundaries are preserved.
- Seed: `20261107`; attenuation settings: `center_tau=0.25`, `edge_ratio=0.3`, `variation=0.15`, `width=5`.
- The script downloads only one 13,917-byte mask through HTTP byte ranges and checks it against the repository manifest and a pinned SHA-256. It needs NumPy, OpenCV, Matplotlib, pycocotools, SciPy and PyTorch, as used by the existing synthesis renderer.

From the repository root:

```bash
python tools/build_readme_synthesis_demo.py --cache /path/to/demo-cache
```

The script checks that instance masks are disjoint, donor geometry is preserved, labeled pixels are darkened, and pixels outside the inserted shapes are unchanged.

## Observation and annotation credits

`filament_intro.jpg` and `preprocessing.jpg` use observation **20100604185114Th** from [SWEFil](https://huggingface.co/datasets/antonio-reche/SWEFil), by **Antonio Reche and Consuelo Cid**, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Related publication: *Solar filament detection, classification, and tracking with deep learning*, SPAICE 2024, pp. 69–74.

- Dataset revision: `bc818c69a79266ad0fa91a539359485af57967cf`.
- [Processed source image](https://huggingface.co/datasets/antonio-reche/SWEFil/resolve/bc818c69a79266ad0fa91a539359485af57967cf/images/20100604185114Th.jpg).
- [Raw source image](https://huggingface.co/datasets/antonio-reche/SWEFil/resolve/bc818c69a79266ad0fa91a539359485af57967cf/images/20100604185114Th_raw.jpg).
- Introductory outline: dataset annotation ID `1`; the image is cropped and a colored outline/transparent overlay is added. This is a **human reference annotation**.
- Preprocessing panels: the raw observation, the repository's `limb` correction, then `preprocess_variant(..., 'clahe16')`. Every panel uses the same grayscale display range, 0–255. The contrast view illustrates the YOLO branch.

Solar observations are from **GONG / NSO Integrated Synoptic Program**. NSO is operated by AURA under a cooperative agreement with the National Science Foundation; GONG also receives NOAA support. See [GONG](https://gong.nso.edu/) for observatory information.

## Actual model-prediction example

`model_predictions.jpg` uses observation **20120330063134Lh** from the project's GONG/MAGFiLO competition validation data. It displays the same crop, `(x0, y0, x1, y1) = (384, 384, 768, 768)`, in four panels:

1. Limb-corrected pixels reconstructed from nine verified 128 × 128 image tiles.
2. Archived YOLO candidates after its view/tile fusion.
3. Archived best84 U-Net candidates, decoded at foreground threshold 0.5 and minimum component area 300 pixels.
4. Archived final masks from the adopted YOLO + best84 pipeline.

The native prediction record and pixel tiles are tied to the **same original image SHA-256**. The final CSV is the frozen validation export used by the adopted pipeline. This is an illustrative crop, not a before/after benchmark or a ground-truth comparison. All masks retain their saved coordinates and shapes; display colors are only overlays. No full competition images, RLE exports, or weights are included here.

[`provenance.json`](provenance.json) records source checksums, crop coordinates, visible instance identifiers, preprocessing-code checksum, and output-image checksums. The SWEFil license above applies to its source material; it is not a blanket license for the separate competition data.

## Reproduce the figures

Use [`tools/build_readme_figures.py`](../../../tools/build_readme_figures.py) from the repository root with Python 3.11+ and NumPy, OpenCV, Matplotlib, pycocotools, SciPy, and PyTorch installed. The builder imports the preprocessing functions from `tools/submission_inference.py`; it performs no model inference or training and needs no GPU.

Create an external JSON file with these input keys and paths to the corresponding local archives:

| Key | Expected input |
|---|---|
| `swefil_manifest` | Prepared SWEFil `training_manifest.json`, containing records, annotations, image checksums, and revision |
| `swefil_images` | Directory containing `20100604185114Th.jpg` |
| `swefil_raw` | Path to `20100604185114Th_raw.jpg` |
| `pixel_tiles` | Export directory containing `manifest.json` and `tiles/20120330063134Lh/` |
| `native_predictions` | best84 `predictions.jsonl`, with its `manifest.json` in the same directory |
| `yolo_csv` | Archived `c0.35_iou0.3_majority_guard0.85.csv` candidate export |
| `final_csv` | Archived `tta_precision.csv` final validation export |

```bash
python tools/build_readme_figures.py --inputs /path/to/readme_inputs.json
```

The script checks image-tile, native-record, and final-prediction provenance before rendering. The input archives must be available separately; cloning the repository alone is not enough to regenerate the prediction figure.
