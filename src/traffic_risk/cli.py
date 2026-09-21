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

    prepare = commands.add_parser(
        "prepare-tracks",
        help="Stream a detector CSV into timestamped rows and per-track windows",
    )
    prepare.add_argument("detector_csv", type=Path)
    prepare.add_argument("video", type=Path)
    prepare.add_argument("output_dir", type=Path)
    prepare.add_argument("--video-id", help="Logical ID; defaults to the video filename stem")
    prepare.add_argument("--fps", type=float, help="Override FPS instead of reading video metadata")
    prepare.add_argument("--chunksize", type=int, default=250_000)

    overspeed = commands.add_parser(
        "build-overspeed",
        help="Build accepted IPM-based overspeed semantics from timestamped tracks",
    )
    overspeed.add_argument("timestamp_csv", type=Path)
    overspeed.add_argument("reference_windows_csv", type=Path)
    overspeed.add_argument("output_dir", type=Path)
    overspeed.add_argument("--ipm", type=Path, default=CONFIG_DIR / "ipm.json")

    redlight = commands.add_parser(
        "build-redlight",
        help="Build accepted stop-line and traffic-light violation semantics",
    )
    redlight.add_argument("timestamp_csv", type=Path)
    redlight.add_argument("reference_windows_csv", type=Path)
    redlight.add_argument("output_dir", type=Path)
    redlight.add_argument("--stop-lines", type=Path, default=CONFIG_DIR / "stop_lines.json")

    trajectory = commands.add_parser(
        "build-trajectory",
        help="Build accepted fixed20/GMM/v8 trajectory semantics from timestamped tracks",
    )
    trajectory.add_argument("timestamp_csv", type=Path)
    trajectory.add_argument("output_dir", type=Path)
    trajectory.add_argument("--lane-map", type=Path, default=CONFIG_DIR / "lane_map.json")
    trajectory.add_argument("--double-yellow", type=Path, default=CONFIG_DIR / "double_yellow.json")
    trajectory.add_argument("--model", type=Path, default=MODEL_DIR / "trajectory_gmm_b5.json")
    trajectory.add_argument("--chunksize", type=int, default=250_000)

    fusion = commands.add_parser(
        "build-fusion-v0",
        help="Align trajectory, overspeed, and red-light modules into the accepted 15D semantic table",
    )
    fusion.add_argument("trajectory_sidecar", type=Path)
    fusion.add_argument("overspeed_features", type=Path)
    fusion.add_argument("redlight_features", type=Path)
    fusion.add_argument("output_dir", type=Path)
    fusion.add_argument("--overspeed-gru", action="store_true", help="Input already contains GRU-ready overspeed columns")
    fusion.add_argument("--redlight-gru", action="store_true", help="Input already contains GRU-ready red-light columns")

    semantics = commands.add_parser(
        "build-semantics-v0",
        help="Run all three car-only upstream branches and their audited 15D fusion intermediate",
    )
    semantics.add_argument("timestamp_csv", type=Path)
    semantics.add_argument("output_dir", type=Path)
    semantics.add_argument("--lane-map", type=Path, default=CONFIG_DIR / "lane_map.json")
    semantics.add_argument("--double-yellow", type=Path, default=CONFIG_DIR / "double_yellow.json")
    semantics.add_argument("--trajectory-model", type=Path, default=MODEL_DIR / "trajectory_gmm_b5.json")
    semantics.add_argument("--ipm", type=Path, default=CONFIG_DIR / "ipm.json")
    semantics.add_argument("--stop-lines", type=Path, default=CONFIG_DIR / "stop_lines.json")
    semantics.add_argument("--chunksize", type=int, default=250_000)

    semantics_v8 = commands.add_parser(
        "build-semantics-v8",
        help="Build formal car-only v8 branches, S3.3 prefix, and C4O model inputs",
    )
    semantics_v8.add_argument("timestamp_csv", type=Path)
    semantics_v8.add_argument("output_dir", type=Path)
    semantics_v8.add_argument("--lane-map", type=Path, default=CONFIG_DIR / "lane_map.json")
    semantics_v8.add_argument("--double-yellow", type=Path, default=CONFIG_DIR / "double_yellow.json")
    semantics_v8.add_argument("--trajectory-model", type=Path, default=MODEL_DIR / "trajectory_gmm_b5.json")
    semantics_v8.add_argument("--ipm", type=Path, default=CONFIG_DIR / "ipm.json")
    semantics_v8.add_argument("--stop-lines", type=Path, default=CONFIG_DIR / "stop_lines.json")
    semantics_v8.add_argument("--chunksize", type=int, default=250_000)

    video_run = commands.add_parser("run-video", help="Run a video through detection, C4O, and final traffic risk")
    video_run.add_argument("video", type=Path)
    video_run.add_argument("output_dir", type=Path)
    video_run.add_argument("--yolo-model", type=Path, help="YOLO weights; required unless --tracks-csv is given")
    video_run.add_argument("--tracks-csv", type=Path, help="Use an existing detector CSV instead of rerunning YOLO")
    video_run.add_argument("--video-id", help="Logical video ID; defaults to the filename stem")
    video_run.add_argument("--camera-config", type=Path, default=CONFIG_DIR / "camera_roi.json")
    video_run.add_argument("--tracker-config", type=Path, default=CONFIG_DIR / "bytetrack.yaml")
    video_run.add_argument("--traffic-light-model", type=Path, default=MODEL_DIR / "traffic_light_classifier.pt")
    video_run.add_argument("--tl-roi", default="1079,426;1119,426;1119,443;1079,443", help="Traffic-light ROI polygon for this camera")
    video_run.add_argument("--classes", default="2,3,5,7,9", help="YOLO classes to track before car-majority filtering")
    video_run.add_argument("--lane-map", type=Path, default=CONFIG_DIR / "lane_map.json")
    video_run.add_argument("--double-yellow", type=Path, default=CONFIG_DIR / "double_yellow.json")
    video_run.add_argument("--stop-lines", type=Path, default=CONFIG_DIR / "stop_lines.json")
    video_run.add_argument("--ipm", type=Path, default=CONFIG_DIR / "ipm.json")
    video_run.add_argument("--annotations", type=Path, help="Optional temporal labels; omitted for unlabeled inference")
    video_run.add_argument("--detector-device", type=int, default=0)
    video_run.add_argument("--device", default="auto", help="PyTorch inference device")
    video_run.add_argument("--fps", type=float, help="Override video metadata FPS")
    video_run.add_argument("--chunksize", type=int, default=250_000)

    single = commands.add_parser("predict-single", help="Run the frozen single-vehicle GRU on a prepared feature CSV")
    single.add_argument("features", type=Path)
    single.add_argument("output", type=Path)
    single.add_argument("--device", default="auto")

    full_single = commands.add_parser(
        "run-single",
        help="Build current-run semantic features and run the complete single-vehicle pipeline",
    )
    full_single.add_argument("c4o_features", type=Path, help="Upstream/C4O window feature CSV")
    full_single.add_argument("trajectory_features", type=Path, help="Trajectory semantic sidecar CSV")
    full_single.add_argument("timestamp_dir", type=Path, help="Directory containing <video_id>.csv tracking files")
    full_single.add_argument("output_dir", type=Path)
    full_single.add_argument("--lane-map", type=Path, default=CONFIG_DIR / "lane_map.json")
    full_single.add_argument("--stop-lines", type=Path, default=CONFIG_DIR / "stop_lines.json")
    full_single.add_argument("--device", default="auto")
    full_single.add_argument("--overwrite-cache", action="store_true")

    full_risk = commands.add_parser(
        "run-risk",
        help="Run semantic rebuilding, single-vehicle risk, and scene risk",
    )
    full_risk.add_argument("c4o_features", type=Path)
    full_risk.add_argument("trajectory_features", type=Path)
    full_risk.add_argument("timestamp_dir", type=Path)
    full_risk.add_argument("output_dir", type=Path)
    full_risk.add_argument("--annotations", type=Path, default=Path("annotations"))
    full_risk.add_argument("--lane-map", type=Path, default=CONFIG_DIR / "lane_map.json")
    full_risk.add_argument("--stop-lines", type=Path, default=CONFIG_DIR / "stop_lines.json")
    full_risk.add_argument("--ipm", type=Path, default=CONFIG_DIR / "ipm.json")
    full_risk.add_argument("--device", default="auto")
    full_risk.add_argument("--overwrite-cache", action="store_true")

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

    annotate = commands.add_parser("annotate", help="Open the temporal risk annotation GUI")
    annotate.add_argument(
        "video_id",
        help="Logical video ID; bundled demo IDs provide default paths, other IDs require --video",
    )
    annotate.add_argument("--video", type=Path)
    annotate.add_argument("--tracks", type=Path)
    annotate.add_argument("--output", type=Path, dest="annotation")
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
                MODEL_DIR / "trajectory_gmm_b5.json",
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
    if args.command == "prepare-tracks":
        from .upstream.tracking import prepare_tracking_csv

        prepared = prepare_tracking_csv(
            detector_csv=args.detector_csv,
            video_path=args.video,
            output_dir=args.output_dir,
            video_id=args.video_id,
            fps=args.fps,
            chunksize=args.chunksize,
        )
        print(json.dumps({
            "video_id": prepared.video_id,
            "fps": prepared.fps,
            "timestamp_csv": str(prepared.timestamp_csv),
            "reference_windows_csv": str(prepared.reference_windows_csv),
            "detector_rows": prepared.detector_rows,
            "track_count": prepared.track_count,
            "reference_window_count": prepared.reference_window_count,
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "build-overspeed":
        from .upstream.overspeed.runner import build_overspeed_features, write_overspeed_features

        features = build_overspeed_features(
            timestamp_csv=args.timestamp_csv,
            reference_windows_csv=args.reference_windows_csv,
            ipm_json=args.ipm,
        )
        windows_path, sidecar_path = write_overspeed_features(features, args.output_dir)
        print(json.dumps({
            "windows": str(windows_path),
            "sidecar": str(sidecar_path),
            "rows": len(features.windows),
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "build-redlight":
        from .upstream.redlight.runner import build_redlight_features, write_redlight_features

        features = build_redlight_features(
            timestamp_csv=args.timestamp_csv,
            reference_windows_csv=args.reference_windows_csv,
            stop_lines_json=args.stop_lines,
        )
        windows_path, sidecar_path = write_redlight_features(features, args.output_dir)
        print(json.dumps({
            "windows": str(windows_path),
            "sidecar": str(sidecar_path),
            "rows": len(features.windows),
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "build-trajectory":
        from .upstream.trajectory.runner import build_trajectory_features, write_trajectory_features

        features = build_trajectory_features(
            timestamp_csv=args.timestamp_csv,
            lane_map_json=args.lane_map,
            double_yellow_json=args.double_yellow,
            model_json=args.model,
            chunksize=args.chunksize,
        )
        paths = write_trajectory_features(features, args.output_dir)
        print(json.dumps({"rows": len(features.windows), **{key: str(value) for key, value in paths.items()}}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "build-fusion-v0":
        from .upstream.fusion import build_fusion_features, write_fusion_features

        features = build_fusion_features(
            pd.read_csv(args.trajectory_sidecar, low_memory=False),
            pd.read_csv(args.overspeed_features, low_memory=False),
            pd.read_csv(args.redlight_features, low_memory=False),
            overspeed_is_gru=args.overspeed_gru,
            redlight_is_gru=args.redlight_gru,
        )
        paths = write_fusion_features(features, args.output_dir)
        print(json.dumps({"rows": len(features.windows), **{key: str(value) for key, value in paths.items()}}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "build-semantics-v0":
        from .upstream.runner import build_semantics_v0

        manifest = build_semantics_v0(
            timestamp_csv=args.timestamp_csv,
            output_dir=args.output_dir,
            lane_map=args.lane_map,
            double_yellow=args.double_yellow,
            trajectory_model=args.trajectory_model,
            ipm=args.ipm,
            stop_lines=args.stop_lines,
            chunksize=args.chunksize,
        )
        print(json.dumps({"car_window_count": manifest["car_window_count"], "fusion": manifest["fusion"]["fusion"]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "build-semantics-v8":
        from .upstream.runner import build_semantics_v8

        manifest = build_semantics_v8(
            timestamp_csv=args.timestamp_csv, output_dir=args.output_dir,
            lane_map=args.lane_map, double_yellow=args.double_yellow,
            trajectory_model=args.trajectory_model, ipm=args.ipm,
            stop_lines=args.stop_lines, chunksize=args.chunksize,
        )
        print(json.dumps({"car_window_count": manifest["car_window_count"], "c4o_window_count": manifest["c4o_window_count"], "c4o_features": manifest["v8"]["c4o_features"]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-video":
        from .video_runner import run_video_pipeline

        path = run_video_pipeline(
            video=args.video, output_dir=args.output_dir, yolo_model=args.yolo_model,
            tracks_csv=args.tracks_csv, video_id=args.video_id,
            camera_config=args.camera_config, tracker_config=args.tracker_config,
            traffic_light_model=args.traffic_light_model, lane_map=args.lane_map,
            tl_roi=args.tl_roi,
            classes=args.classes,
            double_yellow=args.double_yellow, stop_lines=args.stop_lines,
            ipm=args.ipm, annotation_dir=args.annotations,
            detector_device=args.detector_device, device=args.device,
            fps=args.fps, chunksize=args.chunksize,
        )
        print(f"output={path}")
        return 0
    if args.command == "predict-single":
        from .single_vehicle.inference import SingleVehicleRiskModel

        frame = pd.read_csv(args.features, low_memory=False)
        result = SingleVehicleRiskModel(device=args.device).predict(frame)
    elif args.command == "run-single":
        from .single_vehicle.runner import run_single_vehicle_pipeline

        run = run_single_vehicle_pipeline(
            c4o_features=args.c4o_features,
            trajectory_features=args.trajectory_features,
            timestamp_dir=args.timestamp_dir,
            lane_map=args.lane_map,
            stop_lines=args.stop_lines,
            output_dir=args.output_dir,
            device=args.device,
            overwrite_cache=args.overwrite_cache,
        )
        print(
            f"output={args.output_dir / 'single_vehicle_risk.csv'} "
            f"rows={len(run.final_windows)} cache={json.dumps(run.frame_cache_summary, sort_keys=True)}"
        )
        return 0
    elif args.command == "run-risk":
        from .pipeline import run_risk_pipeline

        output = run_risk_pipeline(
            c4o_features=args.c4o_features,
            trajectory_features=args.trajectory_features,
            timestamp_dir=args.timestamp_dir,
            output_dir=args.output_dir,
            annotation_dir=args.annotations,
            lane_map=args.lane_map,
            stop_lines=args.stop_lines,
            ipm=args.ipm,
            device=args.device,
            overwrite_cache=args.overwrite_cache,
        )
        print(f"output={output}")
        return 0
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
    elif args.command == "annotate":
        from .annotation.gui import launch_annotation_gui

        launch_annotation_gui(
            args.video_id,
            video=args.video,
            tracks=args.tracks,
            annotation=args.annotation,
        )
        return 0
    else:  # pragma: no cover
        raise AssertionError(args.command)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"output={args.output} rows={len(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
