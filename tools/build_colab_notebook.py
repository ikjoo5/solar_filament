#!/usr/bin/env python3
"""Build a standalone Colab notebook from the reviewed local evaluator sources."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = "solar_filament.tools.build_colab_notebook.v1"
MODULES = ("filament_eval.py", "convert_annotations.py", "checkpoint_utils.py",
           "evaluate_yolo_baseline.py")


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": textwrap.dedent(text).strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    text = textwrap.dedent(text).strip() + "\n"
    ast.parse(text)
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def build_notebook() -> dict:
    sources = {name: (ROOT / "tools" / name).read_text(encoding="utf-8") for name in MODULES}
    for name, source in sources.items():
        compile(source, name, "exec")
    payload = json.dumps(sources, ensure_ascii=False)
    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
    cells = [
        markdown("""
        # 기존 YOLO26 모델을 Google Drive 데이터로 평가

        **이미지 전체를 채팅에 업로드하거나 다른 저장소로 옮기지 않습니다.**
        Drive의 원본 JPEG, 정답 JSON, `best.pt`, `train_split.csv`를 그대로 읽습니다.
        새 학습·Kaggle 제출 없이 제공 self-evaluation notebook 의미의 **전역 누적 PQ**를 계산합니다.

        1. Colab의 **런타임 → 런타임 유형 변경 → T4 또는 A100 GPU**를 선택합니다.
        2. 아래 셀을 순서대로 실행하고 Drive 연결을 허용합니다.
        3. 경로 설정을 확인하고 평가 셀을 실행합니다.
        4. 마지막에 내려받는 작은 결과 ZIP만 채팅에 첨부하면 됩니다.

        원래 `val_split.csv`가 있으면 자동 사용합니다. 없으면 `train_split.csv`의 나머지를
        **METADATA_COMPLEMENT_DIAGNOSTIC**으로 평가합니다. 이것이 당시 validation과 같다고
        입증된 것은 아니며, 리더보드 0.37 재현이나 독립 최종 점수라고 보고하면 안 됩니다.

        이 노트북에는 실행 결과가 저장되어 있지 않습니다. 도구 테스트/CPU 확인과 별개로
        실제 Colab GPU·전체 Drive 데이터에서의 실행은 사용자의 런타임에서 확인해야 합니다.
        """),
        markdown("""
        ## 1. 실행 환경

        기존 Colab CUDA용 PyTorch를 유지합니다. CPU용 PyTorch wheel을 설치하지 않습니다.
        이 노트북은 Python 3.11 이상과 checkpoint 검사 API가 있는 PyTorch를 요구합니다.
        """),
        code('''
        import json
        import os
        from pathlib import Path
        import subprocess
        import sys
        import tempfile

        if sys.version_info < (3, 11):
            raise RuntimeError("Python 3.11 이상인 Colab 런타임이 필요합니다.")
        os.environ.setdefault("YOLO_CONFIG_DIR", "/content/solar_filament_runtime/ultralytics")
        os.environ.setdefault("MPLCONFIGDIR", "/content/solar_filament_runtime/matplotlib")
        os.environ.setdefault("XDG_CACHE_HOME", "/content/solar_filament_runtime/cache")
        check_torch = (
            "import json, torch; "
            "assert hasattr(torch.serialization, 'get_unsafe_globals_in_checkpoint'), "
            "'PyTorch checkpoint 검사 API가 없는 런타임입니다'; "
            "assert torch.cuda.is_available(), 'Colab 런타임에서 T4/A100 GPU를 선택해 주세요'; "
            "print(json.dumps({'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(0)}))"
        )
        before = json.loads(subprocess.check_output([sys.executable, "-c", check_torch], text=True))
        requirements = [
            "numpy==2.3.5", "scipy==1.17.0", "pycocotools==2.0.11",
            "ultralytics==8.4.182", "opencv-python==5.0.0.93",
        ]
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        *requirements], check=True)
        after = json.loads(subprocess.check_output([sys.executable, "-c", check_torch], text=True))
        if before["torch"] != after["torch"]:
            raise RuntimeError("설치 중 PyTorch 버전이 변경되었습니다. 런타임 의존성을 확인해 주세요.")
        print("준비된 환경:", after)
        '''),
        markdown("""
        ## 2. Drive 연결

        모델과 원본 데이터는 Drive에 유지합니다. 셀을 실행하면 Colab의 Drive 인증 화면이 열릴 수 있습니다.
        """),
        code('''
        from google.colab import drive

        if not Path("/content/drive/MyDrive").is_dir():
            drive.mount("/content/drive")
        print("Drive 연결:", Path("/content/drive/MyDrive").is_dir())
        '''),
        markdown("""
        ## 3. 경로 설정

        확인된 `train_images` 경로를 기본값으로 넣었습니다. `best.pt`와 split을 다른 폴더에
        보관했다면 해당 경로만 바꿉니다. 정답은 `.json` 또는 JSON 하나가 든 `.zip`을 지정할 수 있습니다.

        `VAL_SPLIT=None`이면 기존 run 폴더의 `val_split.csv`를 먼저 찾습니다.
        없을 때는 아래의 명시적 `USE_TRAIN_COMPLEMENT=True` 설정으로 나머지 집합을 진단합니다.
        전체 평가가 기본이며, 빠른 파일/추론 점검만 하려면 `MAX_IMAGES=2`로 바꿉니다.
        이 경우 결과는 전체 validation이 아니라 smoke subset으로 기록됩니다.
        """),
        code('''
        DATA_ROOT = Path("/content/drive/MyDrive/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
        TRAIN_IMAGES = DATA_ROOT / "train" / "train_images"
        ANNOTATIONS = DATA_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

        BASELINE_RUN = Path("/content/drive/MyDrive/SolarFilament_LimbOnly_YOLO26_mixed_FullRes")
        CHECKPOINT = BASELINE_RUN / "best.pt"
        TRAIN_SPLIT = BASELINE_RUN / "train_split.csv"
        VAL_SPLIT = None  # 원본 validation CSV가 다른 위치면 Path("...")로 지정
        USE_TRAIN_COMPLEMENT = True

        DEVICE = "0"
        CONFIDENCES = [0.15, 0.25, 0.35, 0.45, 0.55]
        POSTPROCESS = "both"  # union_cc와 preserve를 진단 비교; preserve overlap은 별도 확인
        MAX_IMAGES = None     # None: 선택된 validation 전체, 2: 두 장 smoke test
        RESULTS_ROOT = BASELINE_RUN / "official_pq_validation"
        '''),
        markdown("""
        ## 4. 내장된 평가 코드 준비

        아래 소스는 검토한 도구의 사본입니다. Git clone이나 외부 코드 다운로드가 필요 없습니다.
        새 임시 폴더에 쓰며 사용자 원본 이미지·정답·checkpoint를 수정하지 않습니다.
        """),
        code("""
        EMBEDDED_SOURCES = json.loads(%s)
        TOOLS_WORKDIR = Path(tempfile.mkdtemp(prefix="solar_filament_eval_"))
        for name, source in EMBEDDED_SOURCES.items():
            compile(source, name, "exec")
            (TOOLS_WORKDIR / name).write_text(source, encoding="utf-8")
        print("평가 코드 위치:", TOOLS_WORKDIR)
        """ % repr(payload)),
        markdown("""
        ## 5. 입력 확인과 평가 실행

        원본 2048×2048 JPEG의 존재와 split 일치 여부를 먼저 확인합니다. 같은 영상의
        여러 annotation이 train/validation 양쪽에 나뉘면 중단합니다. GPU 추론은 실제 영상당
        한 번만 실행하고 confidence와 후처리 후보를 비교합니다. 기본 limb 보정은 업로드
        베이스라인과 같은 `(1024,1024), radius=900` 설정입니다.

        기존 결과를 덮어쓰지 않고 timestamp와 무작위 suffix가 붙은 새 Drive 폴더를 만듭니다.
        """),
        code('''
        from datetime import datetime, timezone
        import uuid
        import zipfile

        annotation_input = Path(ANNOTATIONS)
        if not annotation_input.is_file() and annotation_input.suffix.lower() == ".json":
            candidates = [annotation_input.with_suffix(".zip"), Path(str(annotation_input) + ".zip")]
            matches = [p for p in candidates if p.is_file()]
            if len(matches) == 1:
                annotation_input = matches[0]
        for label, path in (("checkpoint", CHECKPOINT), ("train_split", TRAIN_SPLIT),
                            ("annotations", annotation_input)):
            if not Path(path).is_file():
                raise FileNotFoundError(f"{label} 파일이 없습니다: {path}")
        if not Path(TRAIN_IMAGES).is_dir():
            raise FileNotFoundError(f"train_images 폴더가 없습니다: {TRAIN_IMAGES}")

        resolved_annotations = annotation_input
        if annotation_input.suffix.lower() == ".zip":
            maximum_json_bytes = 128 * 1024 * 1024
            with zipfile.ZipFile(annotation_input) as archive:
                members = [m for m in archive.infolist()
                           if not m.is_dir() and m.filename.lower().endswith(".json")]
                if len(members) != 1 or members[0].file_size > maximum_json_bytes:
                    raise ValueError("ZIP에는 128MiB 이하의 annotation JSON 하나만 있어야 합니다.")
                with archive.open(members[0]) as handle:
                    raw_json = handle.read(maximum_json_bytes + 1)
                if len(raw_json) > maximum_json_bytes:
                    raise ValueError("annotation JSON이 크기 제한을 넘었습니다.")
            json.loads(raw_json)
            resolved_annotations = TOOLS_WORKDIR / "input_annotations.json"
            resolved_annotations.write_bytes(raw_json)
        elif annotation_input.suffix.lower() != ".json":
            raise ValueError("ANNOTATIONS에는 JSON 또는 JSON 하나가 든 ZIP을 지정해 주세요.")

        resolved_val_split = Path(VAL_SPLIT) if VAL_SPLIT is not None else None
        if resolved_val_split is not None and not resolved_val_split.is_file():
            raise FileNotFoundError(f"지정한 validation CSV가 없습니다: {resolved_val_split}")
        automatic_val_split = Path(BASELINE_RUN) / "val_split.csv"
        if resolved_val_split is None and automatic_val_split.is_file():
            resolved_val_split = automatic_val_split

        if resolved_val_split is not None:
            selection_args = ["--val-split", str(resolved_val_split)]
            print("USER_SUPPLIED_VALIDATION_SPLIT:", resolved_val_split)
        elif USE_TRAIN_COMPLEMENT:
            selection_args = ["--use-train-complement"]
            print("METADATA_COMPLEMENT_DIAGNOSTIC: 당시 validation 일치/리더보드 재현을 보장하지 않습니다.")
        else:
            raise ValueError("VAL_SPLIT을 지정하거나 USE_TRAIN_COMPLEMENT=True로 명시해 주세요.")

        run_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        OUTPUT_DIR = Path(RESULTS_ROOT) / run_label
        command = [sys.executable, str(TOOLS_WORKDIR / "evaluate_yolo_baseline.py"),
                   "--annotations", str(resolved_annotations), "--train-split", str(TRAIN_SPLIT),
                   *selection_args, "--images", str(TRAIN_IMAGES), "--checkpoint", str(CHECKPOINT),
                   "--output-dir", str(OUTPUT_DIR), "--device", str(DEVICE),
                   "--postprocess", POSTPROCESS, "--preprocessing", "limb_only",
                   "--confidences", *[str(value) for value in CONFIDENCES]]
        if MAX_IMAGES is not None:
            command += ["--max-images", str(MAX_IMAGES)]
        print("결과 폴더:", OUTPUT_DIR)
        subprocess.run(command, check=True)
        '''),
        markdown("""
        ## 6. 결과 확인과 작은 ZIP 다운로드

        `summary.json`의 전역 PQ, TP/FP/FN, split 출처를 확인합니다. 이 셋에서 confidence와
        후처리를 골랐다면 development tuning 결과이며 독립 최종 점수로 해석하지 않습니다.
        내려받는 ZIP에는 JSON/CSV 결과만 담고 원본 이미지·checkpoint는 넣지 않습니다.
        """),
        code('''
        summary_path = OUTPUT_DIR / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            raise RuntimeError("평가 완료 summary가 아닙니다.")
        print("평가 범위:", summary["run_scope"])
        print("분할 출처:", summary["selection"]["provenance"])
        print("실행 영상 수:", summary["selection"]["executed_physical_images"])
        print("개발 집합에서 선택된 설정:", json.dumps(summary["best_development_setting"], ensure_ascii=False))
        for row in summary["comparisons"]:
            print(f"{row['postprocess']:10} conf={row['confidence']:.2f} "
                  f"PQ={row['pq']:.6f} SQ={row['sq']:.6f} RQ={row['rq']:.6f} "
                  f"TP={row['tp_pairs']} FP={row['fp']} FN={row['fn']}")

        RESULTS_ZIP = OUTPUT_DIR.parent / (OUTPUT_DIR.name + "_results.zip")
        with zipfile.ZipFile(RESULTS_ZIP, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(OUTPUT_DIR.rglob("*")):
                if path.is_file() and path.suffix.lower() in {".json", ".csv"}:
                    archive.write(path, arcname=path.relative_to(OUTPUT_DIR))
        print("채팅에 첨부할 결과 ZIP:", RESULTS_ZIP, f"({RESULTS_ZIP.stat().st_size:,} bytes)")
        from google.colab import files
        files.download(str(RESULTS_ZIP))
        '''),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"solar-filament-{index:02d}"
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU",
        "colab": {"name": "Evaluate_Existing_YOLO_Colab.ipynb", "provenance": []},
        "solar_filament": {"generated_by": GENERATOR, "embedded_source_sha256": hashes,
                           "execution_status": "unexecuted; requires user's Drive data and GPU runtime"}},
        "nbformat": 4, "nbformat_minor": 5}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "notebooks" / "Evaluate_Existing_YOLO_Colab.ipynb")
    args = parser.parse_args()
    notebook = build_notebook()
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("metadata", {}).get("solar_filament", {}).get("generated_by") != GENERATOR:
            raise ValueError("Refusing to overwrite a notebook not owned by this generator")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"notebook": str(args.output), "bytes": args.output.stat().st_size,
                      "code_cells_syntax_checked": sum(c["cell_type"] == "code" for c in notebook["cells"]),
                      "embedded_modules": list(MODULES), "executed": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
