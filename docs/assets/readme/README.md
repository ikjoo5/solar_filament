# README image sources and reproduction

These figures explain the project using real observations, a human annotation, and archived model predictions. They are original figure layouts, not copied illustrations or newly generated model results. Only the small rendered JPEGs are included in Git.

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
