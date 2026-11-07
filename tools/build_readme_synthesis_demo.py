"""Illustrate the existing CHASE-shape renderer with clearly identified demo inputs.

The donor is one public AHAS mask, fetched by byte ranges from its pinned ZIP.
The background and appearance parameters are procedural examples, not the
private background or GONG-calibrated statistics used for model training.
No model inference, training, or competition-data access is performed.

From the repository root:
    python tools/build_readme_synthesis_demo.py --cache /path/to/demo-cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request
import zlib

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

from chase_shape_synthesis import render_at


ROOT = Path(__file__).resolve().parents[1]
DONOR_ID = "RSM20240101T061747_0000_HA"
DONOR_SHA256 = "cc22091a6f40b230a546b236d8366095a4644377ebdd920bd8544f77aa4335e7"
SEED = 20261107
COLORS = ("#087f8c", "#b44b79", "#7761ba")
INK = "#172b3b"
PROFILE = dict(center_tau=.25, edge_ratio=.3, variation=.15, width=5)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive_range(url, first, last):
    """Fetch only the requested range; never fall back to a whole-archive GET."""
    request = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 206:
            raise ValueError("Source must support byte ranges; full ZIP download refused")
        if response.headers.get("Content-Range", "").split("/")[0] != f"bytes {first}-{last}":
            raise ValueError("Source returned a different archive range")
        payload = response.read(last - first + 2)
    if len(payload) != last - first + 1:
        raise ValueError("Truncated or oversized archive range")
    return payload


def donor_mask(cache):
    """Extract one manifest-verified public mask without downloading the 1.2 GB ZIP."""
    manifest_path = ROOT / "configs/chase_ahas2024_2025_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = next(row for row in manifest["pairs"] if row["id"] == DONOR_ID)
    target, archive = record["mask"], manifest["archive"]
    path = cache / Path(target["member"]).name
    if not path.exists():
        tail = archive_range(archive["url"], archive["size"] - 65536, archive["size"] - 1)
        position = tail.rfind(b"PK\x05\x06")
        if position < 0:
            raise ValueError("Missing ZIP end-of-directory record")
        end = struct.unpack_from("<4s4H2LH", tail, position)
        if end[1:3] != (0, 0) or end[3] != end[4]:
            raise ValueError("Multi-disk ZIP is unsupported")
        directory = archive_range(archive["url"], end[6], end[6] + end[5] - 1)
        cursor = 0
        while cursor < len(directory):
            entry = struct.unpack_from("<4s6H3L5H2L", directory, cursor)
            if entry[0] != b"PK\x01\x02":
                raise ValueError("Invalid ZIP central directory")
            name = directory[cursor + 46:cursor + 46 + entry[10]].decode("utf-8")
            if name == target["member"]:
                break
            cursor += 46 + entry[10] + entry[11] + entry[12]
        else:
            raise ValueError("Pinned donor is absent from the ZIP directory")
        if entry[3] & 1 or entry[7] != target["crc32"] or entry[9] != target["size"]:
            raise ValueError("ZIP donor differs from the repository manifest")
        offset, compressed_size = entry[16], entry[8]
        local = struct.unpack("<4s5H3L2H", archive_range(archive["url"], offset, offset + 29))
        if local[0] != b"PK\x03\x04" or local[3] != entry[4]:
            raise ValueError("Invalid or inconsistent local ZIP header")
        first = offset + 30 + local[9] + local[10]
        compressed = archive_range(archive["url"], first, first + compressed_size - 1)
        if entry[4] == 8:
            pixels = zlib.decompress(compressed, -15)
        elif entry[4] == 0:
            pixels = compressed
        else:
            raise ValueError("Unsupported ZIP compression")
        if len(pixels) != target["size"] or zlib.crc32(pixels) != target["crc32"]:
            raise ValueError("Extracted donor failed manifest size/CRC verification")
        if hashlib.sha256(pixels).hexdigest() != DONOR_SHA256:
            raise ValueError("Extracted donor failed the pinned SHA-256 check")
        path.write_bytes(pixels)
    if digest(path) != DONOR_SHA256 or path.stat().st_size != target["size"]:
        raise ValueError("Cached donor changed")
    source = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if source is None or source.shape != (2048, 2048) or set(np.unique(source)) != {0, 255}:
        raise ValueError("Expected the native 2048-pixel binary AHAS mask")
    return source, dict(
        observation_id=DONOR_ID,
        url=archive["url"],
        archive_record="https://zenodo.org/records/21582245",
        doi="10.5281/zenodo.21582245",
        license="CC BY 4.0",
        credit="Jiarui Hu, Qi Hao, Zheng Zhi, P. F. Chen, Chuan Li and Yifan Meng",
        member=target["member"],
        sha256=DONOR_SHA256,
        bytes=target["size"],
        crc32=target["crc32"],
        manifest="configs/chase_ahas2024_2025_manifest.json",
        manifest_sha256=digest(manifest_path),
        retrieval="Four HTTP byte ranges; member CRC, length and pinned SHA-256 checked. Whole-archive MD5 not recomputed.",
        annotation_scope="AHAS automatic binary mask; connected components supply geometry, not verified physical instance IDs.",
    )


def procedural_background():
    """A grayscale texture patch created from seeded noise and a smooth gradient."""
    rng = np.random.default_rng(SEED)
    yy, xx = np.indices((512, 512))
    fine = cv2.GaussianBlur(rng.normal(size=(512, 512)).astype(np.float32), (0, 0), 1.0)
    broad = cv2.GaussianBlur(rng.normal(size=(512, 512)).astype(np.float32), (0, 0), 12.)
    fine /= fine.std()
    broad /= broad.std()
    illumination = 183 - 10 * ((xx - 120) / 650) ** 2 - 8 * ((yy - 150) / 650) ** 2
    return np.rint(illumination + 2.4 * fine + 1.8 * broad).clip(0, 255).astype(np.uint8)


def build(cache, output):
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    donor, source = donor_mask(cache)
    _, components, stats, _ = cv2.connectedComponentsWithStats((donor > 0).astype(np.uint8), 8)
    background = procedural_background()
    composite = background.copy()
    labels = np.zeros_like(background)
    geometry = dict(cx=256., cy=256., radius=400.)
    # Editorial selection of three visible, differently oriented donor shapes.
    placements = [(38, 60, 85), (18, 100, 220), (12, 340, 280)]
    records, masks = [], []
    for instance, (component, x, y) in enumerate(placements, 1):
        left, top, width, height, _ = stats[component]
        shape = (components[top:top + height, left:left + width] == component).astype(np.uint8)
        forbidden = cv2.dilate((labels > 0).astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
        composite, placed, quality = render_at(
            composite, shape, forbidden, x, y, geometry, PROFILE, np.random.default_rng(SEED + instance)
        )
        if np.any(placed & (labels > 0)) or not np.array_equal(placed[y:y + height, x:x + width], shape):
            raise AssertionError("Renderer changed donor geometry or overlapping instances")
        if composite.dtype != np.uint8 or composite.shape != background.shape:
            raise AssertionError("Unexpected composite shape or storage type")
        labels[placed > 0] = instance
        masks.append(placed.astype(bool))
        records.append(dict(instance=instance, source_component=int(component),
                            source_bbox_xywh=[int(left), int(top), int(width), int(height)],
                            placement_xy=[x, y], transform="translation only; native mask pixel size preserved",
                            render_seed=SEED + instance, renderer_quality=quality,
                            mask_sha256=hashlib.sha256(placed.tobytes()).hexdigest()))
    if not np.array_equal(composite[labels == 0], background[labels == 0]):
        raise AssertionError("Pixels outside inserted shapes changed")
    if not np.all(composite[labels > 0] < background[labels > 0]):
        raise AssertionError("Some labeled pixels are not visibly attenuated")
    if sum(int(mask.sum()) for mask in masks) != int((labels > 0).sum()):
        raise AssertionError("Instance masks overlap")

    fig = plt.figure(figsize=(14.4, 5.7), facecolor="white")
    fig.text(.025, .945, "More filament shapes, paired with exact labels", fontsize=23,
             fontweight="bold", color=INK, va="top")
    fig.text(.025, .865, "Illustrative renderer demo  /  procedural background + public CHASE geometry",
             fontsize=12.5, color="#586875", va="top")
    lefts = [.025, .270, .515, .760]
    titles = ["01  Background", "02  Donor shapes", "03  Composite", "04  Instance labels"]
    captions = ["Seeded grayscale texture", "Three CHASE mask components",
                "Darken only inside each shape", "Each color is one inserted object"]
    donor_rgb = np.full((512, 512, 3), to_rgb("#eef5f7"))
    label_rgb = np.repeat(composite[:, :, None], 3, axis=2).astype(float) / 255
    for instance, color in enumerate(COLORS, 1):
        mask = labels == instance
        donor_rgb[mask] = to_rgb(color)
        label_rgb[mask] = .30 * label_rgb[mask] + .70 * np.array(to_rgb(color))
    for i, (left, title, caption) in enumerate(zip(lefts, titles, captions)):
        fig.text(left, .775, title, fontsize=15, fontweight="bold", color=INK)
        ax = fig.add_axes([left, .16, .215, .57])
        if i in (0, 2):
            ax.imshow(background if i == 0 else composite, cmap="gray", vmin=0, vmax=255,
                      interpolation="nearest")
        else:
            ax.imshow(donor_rgb if i == 1 else label_rgb, interpolation="nearest")
        if i == 3:
            for mask, color in zip(masks, COLORS):
                ax.contour(mask.astype(float), levels=[.5], colors=[color], linewidths=.9)
        ax.set_axis_off()
        fig.text(left, .13, caption, fontsize=10.8, color="#586875")
        if i < 3:
            fig.text(left + .227, .465, "→", fontsize=20, color="#8495a2", va="center", ha="center")
    fig.text(.025, .045, "Shape → intensity attenuation → image + mask", fontsize=11.5,
             fontweight="bold", color=COLORS[0])
    fig.text(.975, .045, "CHASE / Hu et al. · CC BY 4.0", fontsize=10.5,
             color="#586875", ha="right")
    image_path = output / "synthesis_demo.jpg"
    fig.savefig(image_path, dpi=150, facecolor="white", pil_kwargs={"quality":92, "optimize":True})
    plt.close(fig)
    provenance = dict(
        title="Illustrative CHASE-shape renderer demo",
        scope="Method illustration only. No model training/inference; this is not a sample from the private training synthesis archive.",
        source=source,
        background=dict(kind="procedural grayscale texture, not an observation", seed=SEED, shape=[512, 512],
                        implementation="procedural_background in tools/build_readme_synthesis_demo.py",
                        raw_pixel_sha256=hashlib.sha256(background.tobytes()).hexdigest()),
        appearance=dict(kind="fixed illustrative values, not measured GONG training statistics", **PROFILE),
        renderer=dict(function="chase_shape_synthesis.render_at", source_sha256=digest(ROOT / "tools/chase_shape_synthesis.py"),
                      geometry=geometry, records=records),
        checks=dict(outside_mask_pixels_unchanged=True, instance_masks_disjoint=True,
                    native_donor_geometry_preserved=True, every_labeled_pixel_darkened=True),
        reproduction="python tools/build_readme_synthesis_demo.py --cache /path/to/demo-cache",
        versions=dict(numpy=np.__version__, opencv=cv2.__version__, matplotlib=matplotlib.__version__),
        output=dict(path="docs/assets/readme/synthesis_demo.jpg", sha256=digest(image_path), bytes=image_path.stat().st_size),
    )
    (output / "synthesis_demo_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(dict(output=provenance["output"], checks=provenance["checks"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="External cache directory for the one public donor PNG")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets/readme")
    arguments = parser.parse_args()
    build(arguments.cache, arguments.output)
