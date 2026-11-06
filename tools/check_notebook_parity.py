#!/usr/bin/env python3
"""Compare the RLE evaluator with reviewed functions from the uploaded notebook.

Optional verification dependencies: pandas and CPU torch (tested with 2.9.1).
This only accepts the exact reviewed notebook hash and extracts function
definitions from cells 3/4. It does not execute installation/training/UI cells.
The fixtures are artificial; passing them is not a competition score.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util
import torch

import filament_eval


NOTEBOOK_SHA256 = "d4956c0e92cf999dc4417da40f3b61f6e5705e7d596c021ad22cf272507101c8"
FUNCTIONS = {
    "get_overlap_matrices", "fp_count_hit", "fn_count_hit", "rles_to_layers",
    "process_entry", "get_overlap_df", "get_pq_score",
}


def load_reviewed_functions(path: Path) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != NOTEBOOK_SHA256:
        raise ValueError("Notebook differs from reviewed upload; inspect it before updating the hash")
    notebook = json.loads(raw)
    source = "\n".join("".join(notebook["cells"][i]["source"]) for i in (3, 4))
    tree = ast.parse(source)
    if any(not isinstance(node, ast.FunctionDef) for node in tree.body):
        raise ValueError("Unexpected top-level notebook statement")
    if {node.name for node in tree.body} != FUNCTIONS:
        raise ValueError("Unexpected notebook function set")
    # The hash pins these definitions to manually inspected, non-I/O functions.
    namespace = {"pd": pd, "np": np, "torch": torch, "mask_util": mask_util}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def rectangle(y1: int, y2: int, x1: int, x2: int) -> str:
    image = np.zeros((2048, 2048), dtype=np.uint8, order="F")
    image[y1:y2, x1:x2] = 1
    return mask_util.encode(image)["counts"].decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    official = load_reviewed_functions(args.notebook)
    a = rectangle(100, 110, 100, 110)
    half = rectangle(100, 105, 100, 110)
    close = rectangle(100, 111, 100, 110)
    b = rectangle(200, 210, 300, 310)
    c = rectangle(400, 410, 500, 510)
    cases = [
        ("perfect", [("a-img_1", a)], [("img_1", a)]),
        ("strict_half", [("a-img_1", a)], [("img_1", half)]),
        ("fractional_iou", [("a-img_1", a)], [("img_1", close)]),
        ("fp_fn", [("a-img_1", a), ("a-img_2", b)], [("img_1", a), ("img_2", c)]),
        ("no_predictions", [("a-img_1", a)], []),
        ("global_not_macro", [("a-one_1", a), ("a-many_1", a),
                               ("a-many_2", b), ("a-many_3", c)], [("one_1", a)]),
        ("multi_annotator", [("a-img_1", a), ("b-img_1", close),
                             ("a-other_1", b)], [("img_1", a)]),
        ("all_pairs_overlap", [("a-img_1", a), ("a-img_2", close)],
                              [("img_1", a), ("img_2", close)]),
    ]
    results = []
    with tempfile.TemporaryDirectory() as folder:
        for name, gt, pred in cases:
            paths = []
            for filename, rows in (("gt.csv", gt), ("pred.csv", pred)):
                path = Path(folder) / filename
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["filament_id", "segmentation_rle"])
                    writer.writerows(rows)
                paths.append(path)
            gt_df, pred_df = (pd.read_csv(path, keep_default_na=False) for path in paths)
            overlap = official["get_overlap_df"](gt_df, pred_df)
            reference = official["get_pq_score"](overlap)
            fast = filament_eval.evaluate(
                filament_eval.read_csv(paths[0], ground_truth=True),
                filament_eval.read_csv(paths[1]), 2048, 2048,
            )
            error = abs(reference - fast["pq"])
            if error > 1e-6:
                raise AssertionError(f"{name}: notebook={reference}, RLE={fast['pq']}")
            results.append({"case": name, "notebook_pq": reference,
                            "rle_pq": fast["pq"], "absolute_error": error})
    print(json.dumps({"notebook_sha256": NOTEBOOK_SHA256, "mask_shape": [2048, 2048],
                      "torch_version": torch.__version__, "passed_cases": len(results),
                      "absolute_tolerance": 1e-6, "results": results,
                      "scope": "synthetic contract fixtures; not data/model readiness or measured competition PQ"},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
