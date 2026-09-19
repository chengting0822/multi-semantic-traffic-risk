from __future__ import annotations

import argparse
from pathlib import Path

from traffic_risk.paths import CONFIG_DIR, MODEL_DIR

from .tracker import run_tracking_to_csv_roi_crop
from .utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect and track vehicles in one video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/tracks.csv"))
    parser.add_argument("--yolo-model", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--camera-config", type=Path, default=CONFIG_DIR / "camera_roi.json")
    parser.add_argument("--tracker-config", type=Path, default=CONFIG_DIR / "bytetrack.yaml")
    parser.add_argument("--traffic-light-model", type=Path, default=MODEL_DIR / "traffic_light_classifier.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.18)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--cpu-threads", type=int, default=20)
    parser.add_argument("--save-video", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for label, path in {
        "video": args.video,
        "YOLO model": args.yolo_model,
        "camera config": args.camera_config,
        "tracker config": args.tracker_config,
        "traffic-light model": args.traffic_light_model,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    setup_logging("INFO")
    run_tracking_to_csv_roi_crop(
        scene_config_path=str(args.camera_config),
        video_path=str(args.video),
        out_csv=str(args.output),
        model_path=str(args.yolo_model),
        tracker_cfg=str(args.tracker_config),
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        half=True,
        classes=(2, 3, 5, 7, 9),
        use_polygon_mask=True,
        filter_ground_in_roi=False,
        keep_missing_frames=30,
        drop_track_when_leave_roi=True,
        log_every=500,
        show=False,
        show_scale=0.6,
        save_video=str(args.save_video) if args.save_video else None,
        prefetch_frames=256,
        cpu_threads=args.cpu_threads,
        tune_gpu_backend=True,
        max_infer_fps=0.0,
        ramp_frames=0,
        ramp_max_sleep_ms=0.0,
        gpu_guard=False,
        gpu_util_high=90.0,
        gpu_power_high_ratio=0.85,
        gpu_guard_sleep_ms=8.0,
        csv_flush_every=1,
        log_run_params=True,
        params_log_path=str(args.output.with_suffix(".run.jsonl")),
        run_tag="portfolio_pipeline",
        args_snapshot=vars(args),
        run_quality_check=False,
        quality_json_path="",
        tl_cnn_model=str(args.traffic_light_model),
        tl_roi="1079,426;1119,426;1119,443;1079,443",
        tl_img_size=0,
        tl_min_prob=0.55,
        tl_unknown_label="unknown",
        tl_every_n_frames=1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
