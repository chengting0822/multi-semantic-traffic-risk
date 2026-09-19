from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from .paths import CONFIG_DIR, MODEL_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="traffic-risk", description="Multi-semantic traffic-risk toolkit")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("doctor", help="Check runtime dependencies and bundled assets")

    detect = commands.add_parser("detect", help="Detect and track vehicles in a video")
    detect.add_argument("args", nargs=argparse.REMAINDER)

    single = commands.add_parser("predict-single", help="Run the frozen single-vehicle GRU on a prepared feature CSV")
    single.add_argument("features", type=Path)
    single.add_argument("output", type=Path)
    single.add_argument("--device", default="auto")

    policy = commands.add_parser("apply-single-policies", help="Apply the frozen semantic policy chain")
    policy.add_argument("windows", type=Path)
    policy.add_argument("output", type=Path)
    for name in (
        "source-features",
        "trajectory-features",
        "lane-features",
        "lane-family-features",
        "redlight-zone-features",
        "tail-lane-features",
        "wrongway-tail-features",
        "double-yellow-features",
    ):
        policy.add_argument(f"--{name}", type=Path, required=True)

    scene = commands.add_parser("predict-scene-model", help="Run the frozen scene neural model on a prepared NPZ")
    scene.add_argument("dataset", type=Path)
    scene.add_argument("output", type=Path)
    scene.add_argument("--device", default="auto")
    scene.add_argument("--batch-size", type=int, default=256)
    scene.add_argument("--all-rows", action="store_true", help="Also predict rows without evaluation labels")

    scene_pipeline = commands.add_parser(
        "predict-scene",
        help="Run the frozen scene model and selected hybrid fusion",
    )
    scene_pipeline.add_argument("dataset", type=Path)
    scene_pipeline.add_argument("scene_windows", type=Path)
    scene_pipeline.add_argument("scene_tokens", type=Path)
    scene_pipeline.add_argument("interaction_features", type=Path)
    scene_pipeline.add_argument("output", type=Path)
    scene_pipeline.add_argument("--device", default="auto")
    scene_pipeline.add_argument("--batch-size", type=int, default=256)
    scene_pipeline.add_argument("--labeled-only", action="store_true")
    return parser


def doctor() -> int:
    dependencies = ["numpy", "pandas", "torch", "cv2", "ultralytics"]
    report = {
        "python": sys.version.split()[0],
        "dependencies": {name: importlib.util.find_spec(name) is not None for name in dependencies},
        "assets": {
            path.name: path.exists()
            for path in (
                MODEL_DIR / "single_vehicle_gru.pt",
                MODEL_DIR / "single_vehicle_normalization.json",
                MODEL_DIR / "scene_risk.pt",
                MODEL_DIR / "traffic_light_classifier.pt",
                CONFIG_DIR / "scene_hybrid.json",
            )
        },
    }
    if report["dependencies"]["torch"]:
        import torch

        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device_count"] = int(torch.cuda.device_count())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(report["dependencies"].values()) and all(report["assets"].values()) else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return doctor()
    if args.command == "detect":
        from .detection.cli import main as detection_main

        forwarded = args.args[1:] if args.args[:1] == ["--"] else args.args
        return detection_main(forwarded)
    if args.command == "predict-single":
        from .single_vehicle.inference import SingleVehicleRiskModel

        frame = pd.read_csv(args.features, low_memory=False)
        result = SingleVehicleRiskModel(device=args.device).predict(frame)
    elif args.command == "apply-single-policies":
        from .single_vehicle.pipeline import PolicyInputs, apply_policy_chain

        frame = pd.read_csv(args.windows, low_memory=False)
        inputs = PolicyInputs.from_csv(
            source_features=args.source_features,
            trajectory_features=args.trajectory_features,
            lane_features=args.lane_features,
            lane_family_features=args.lane_family_features,
            redlight_zone_features=args.redlight_zone_features,
            tail_lane_features=args.tail_lane_features,
            wrongway_tail_features=args.wrongway_tail_features,
            double_yellow_features=args.double_yellow_features,
        )
        result = apply_policy_chain(frame, inputs)
    elif args.command == "predict-scene-model":
        from .scene.inference import SceneRiskModel

        result = SceneRiskModel(device=args.device).predict_npz(
            args.dataset,
            batch_size=args.batch_size,
            labeled_only=not args.all_rows,
        )
    elif args.command == "predict-scene":
        from .scene.pipeline import SceneRiskPipeline

        result = SceneRiskPipeline(device=args.device).predict(
            dataset=args.dataset,
            scene_windows=pd.read_csv(args.scene_windows, low_memory=False),
            scene_tokens=pd.read_csv(args.scene_tokens, low_memory=False),
            interaction_features=pd.read_csv(args.interaction_features, low_memory=False),
            batch_size=args.batch_size,
            labeled_only=args.labeled_only,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"output={args.output} rows={len(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
