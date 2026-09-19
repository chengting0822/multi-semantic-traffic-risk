#!/usr/bin/env python3
"""Run the four bundled prepared demo cases through the formal scene pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_risk.scene.pipeline import SceneRiskPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the four portfolio examples through the frozen scene model and hybrid fusion."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "examples/demo")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/demo_scene_risk.csv")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a PyTorch device string")
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Also predict three boundary windows without formal evaluation labels.",
    )
    return parser.parse_args()


def friendly_risk(value: int) -> str:
    """Map the formal internal 0/1/2/3 scale to the paper's three alerts."""

    if value <= 0:
        return "無風險"
    if value >= 3:
        return "高風險"
    return "中風險"


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    manifest = json.loads((data_dir / "cases.json").read_text(encoding="utf-8"))

    result = SceneRiskPipeline(device=args.device).predict(
        dataset=data_dir / "scene_features.npz",
        scene_windows=pd.read_csv(data_dir / "scene_windows.csv", low_memory=False),
        scene_tokens=pd.read_csv(data_dir / "scene_tokens.csv", low_memory=False),
        interaction_features=pd.read_csv(data_dir / "interaction_features.csv", low_memory=False),
        labeled_only=not args.all_rows,
    )
    result["video_id"] = result["video_id"].astype(str)
    result["risk_label"] = result["risk_level"].astype(int).map(friendly_risk)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    expected = pd.read_csv(data_dir / "expected_scene_risk.csv")
    expected["video_id"] = expected["video_id"].astype(str)
    keys = ["row_idx", "video_id", "scene_window_idx"]
    compared = result[keys + ["risk_level"]].merge(expected, on=keys, validate="one_to_one")
    mismatches = int((compared["risk_level"] != compared["expected_risk_level"]).sum())

    print(f"output={args.output.resolve()}")
    print(f"verified_rows={len(compared)} mismatches={mismatches}")
    for case in manifest["cases"]:
        part = result.loc[result["video_id"] == str(case["video_id"])]
        peak = int(part["risk_level"].max())
        print(
            f"case{case['case']:02d} video={case['video_id']} "
            f"windows={len(part)} peak={peak} ({friendly_risk(peak)}) "
            f"event={case['title_zh']}"
        )
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
