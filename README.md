# Solar Filament Segmentation 2026

![Solar Filament Segmentation Challenge 2026: observe the Sun, trace filament regions, and separate each filament with its own mask.](docs/assets/readme/competition_overview.png)

## What are we looking for?

**Solar filaments** are long strands of relatively cool, dense plasma held above the Sun by magnetic fields. Against the bright solar disk, they look like dark ribbons in **Hα**, a wavelength that reveals hydrogen in the solar atmosphere. Identifying their shapes helps researchers study these changing solar structures. [Learn more at NSO](https://nso.edu/for-public/sun-science/prominence-filaments/).

The [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026) asks a computer to turn a telescope image into **a separate outline for every filament**. This project builds that image-to-mask pipeline.

![From a full-disk Hα observation, to a close-up of a dark filament, to its outline highlighted in teal.](docs/assets/readme/filament_intro.jpg)

*Read left to right: find the structure on the Sun → look closer → trace its shape. The teal region is a human reference annotation from the public SWEFil dataset.*

| What goes in | What comes out | What makes it difficult |
|---|---|---|
| A 2048 × 2048 GONG Hα image | A pixel mask for each individual filament | Faint branches, uneven brightness, and nearby dark features |

**A mask marks which pixels belong to an object.** Different mask colors identify different filaments; the colors are labels, not the Sun's observed colors. A good prediction follows the full shape while keeping neighboring filaments separate. The challenge's **Panoptic Quality (PQ)** score measures both shape agreement and object recognition.

![Solar observations and magnified filament regions progressing from image preparation to foreground masks and separately colored filament instances.](docs/assets/readme/detection_process.png)

After brightness correction, YOLO and U-Net analyze the image in parallel. Their predictions are combined to trace each filament and keep neighboring objects separate.

## Model pipeline at a glance

The submitted baseline combines **YOLO26s-seg** with a **ConvNeXt-Tiny encoder and U-Net decoder**. YOLO proposes object masks; U-Net estimates filament probability at each pixel. The two networks process images independently, bringing complementary evidence to instance reconstruction. `best84` identifies the selected U-Net checkpoint.

```mermaid
flowchart LR
    A["GONG image<br/>2048 × 2048"] --> B["Limb correction<br/>Views / tiles"]
    B --> Y["YOLO26s-seg"]
    B --> N["ConvNeXt-Tiny<br/>U-Net · best84"]
    Y --> M["Mask fusion<br/>Refine / filter"]
    N --> M
    M --> R["Non-overlap masks<br/>RLE CSV"]
```

The encoder extracts features at multiple spatial resolutions. The decoder combines these features through skip connections to recover fine spatial detail. Its foreground probability map supplies additional candidates and boundary evidence to the ensemble.

Brightness correction reduces the Sun's natural darkening toward the edge. Overlapping crops preserve small details, and predictions from several orientations are brought back to the original image coordinates before fusion.

![The same observation crop shown as input, with YOLO masks, with U-Net masks, and with the combined colored filament instances.](docs/assets/readme/model_predictions.jpg)

*An actual archived prediction example: input → the two models' candidates → combined instances. Each output pixel belongs to at most one filament.*

## Using the surrounding context

A faint dark patch is easier to interpret when we can also see the longer structure around it. The **context-block method** adds this wider view inside the U-Net: parallel dilated convolutions examine several scales, and pooling summarizes the current crop. A learned gate controls how much this information changes the deepest features; skip connections carry fine detail to the decoder.

```mermaid
flowchart LR
    I["Image crop"] --> E["ConvNeXt-Tiny encoder"]
    E --> F["Deepest features"]
    F --> C["Wider context<br/>Multi-scale convolutions<br/>+ crop summary"]
    F --> G["Learned residual gate"]
    C --> G
    G --> D["U-Net decoder"]
    E -->|"Fine-detail skips"| D
    D --> M["Pixel predictions"]
    classDef base fill:#e7f1ff,stroke:#397ac0,color:#143555
    classDef context fill:#fff0d6,stroke:#b56916,color:#33220c
    classDef output fill:#ddf6ef,stroke:#23816c,color:#123e34
    class I,E,F,D base
    class C,G context
    class M output
```

[Explore the model architecture →](docs/MODEL_ARCHITECTURE.md)

## Making more training examples

To give the model more shapes to learn from, we also explore **shape-based data synthesis**. Filament shapes from CHASE masks are rotated, reflected, and resized, then placed on a clean solar background. Darkening and edge characteristics measured from GONG training observations guide their appearance, preserving the background texture. The inserted shapes provide the matching instance masks automatically.

![Four steps of shape-based synthesis: background, donor shapes, rendered dark filaments, and colored instance labels.](docs/assets/readme/synthesis_demo.jpg)

*A rendering illustration of the synthesis method, using public CHASE shapes and a procedural background. Colors in the last panel show the automatically generated labels. [Image sources and reproduction](docs/assets/readme/README.md).*

The CHASE-shape experiment creates three non-overlapping insertions per image. Real observations and synthetic examples are mixed during training; validation and test images are kept out of the synthesis ingredients.

[Synthesis method and training recipe →](docs/CHASE_SHAPE_T4_TRAINING.md)

## Evaluation

**Panoptic Quality (PQ)** evaluates both the accuracy of filament shapes and the detection of individual filaments. Higher scores indicate better performance.

$$
\mathrm{PQ} =
\frac{
\displaystyle\sum_{(p,g)\in\mathrm{TP}}\mathrm{IoU}(p,g)
}{
|\mathrm{TP}|+\frac{1}{2}|\mathrm{FP}|+\frac{1}{2}|\mathrm{FN}|
}
$$

**IoU** measures the overlapping area divided by the union of two masks. A prediction matches a reference mask when **IoU > 0.5**.

- **TP:** correctly matched filaments
- **FP:** false detections
- **FN:** missed filaments

| Method | Validation PQ ↑ |
|---|---:|
| ConvNeXt-Tiny U-Net | 0.4294 |
| U-Net + context block | 0.4332 |
| U-Net + CHASE-shape synthesis | 0.4313 |
| U-Net with GONG / SWEFil / CHASE-shape training | 0.4109 |
| Mixed-data U-Net + 4-view TTA | 0.4207 |
| **YOLO26s-seg + U-Net ensemble** | **0.4726** |

*Validation scores are measured on the fixed 91-image development set.*

## Explore the project

| I want to… | Start here |
|---|---|
| Follow preprocessing, mask fusion, and submission generation | [Pipeline walkthrough](docs/CURRENT_PIPELINE.md) |
| Understand U-Net and the context method | [Model architecture](docs/MODEL_ARCHITECTURE.md) |
| Explore training with GONG, SWEFil, and synthetic examples | [Mixed-data training](docs/SWEFIL_RETRAINING_PLAN.md) |
| Check image credits and reproduce the figures | [Figure sources](docs/assets/readme/README.md) |

### References

- [U-Net](https://arxiv.org/abs/1505.04597) · [ConvNeXt](https://arxiv.org/abs/2201.03545) · [Panoptic Quality](https://arxiv.org/abs/1801.00868)
- Solar observations: **GONG / NSO**. Public introductory image and annotation: **SWEFil, Antonio Reche and Consuelo Cid, CC BY 4.0**. Full credits are in the [figure guide](docs/assets/readme/README.md).
