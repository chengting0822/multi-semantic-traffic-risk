from __future__ import annotations

import argparse
from pathlib import Path

from traffic_risk.paths import CONFIG_DIR, MODEL_DIR

from .tracker import run_tracking_to_csv_roi_crop
from .utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and track vehicles using the paper-time YOLO/ByteTrack preset."
    )
    parser.add_argument("video", type=Path, nargs="?", help="Input video (positional form)")
    parser.add_argument("--video", dest="video_option", type=Path, help="Input video (legacy form)")
    parser.add_argument("--output", "--out-csv", dest="output", type=Path, default=Path("outputs/tracks.csv"))
    parser.add_argument("--yolo-model", "--model", dest="yolo_model", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--camera-config", "--scene-config", dest="camera_config", type=Path, default=CONFIG_DIR / "camera_roi.json"
    )
    parser.add_argument(
        "--tracker-config", "--tracker", dest="tracker_config", type=Path, default=CONFIG_DIR / "bytetrack.yaml"
    )
    parser.add_argument(
        "--traffic-light-model", "--tl-cnn-model", dest="traffic_light_model",
        type=Path, default=MODEL_DIR / "traffic_light_classifier.pt",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.18)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--classes", default="2,3,5,7,9")
    parser.add_argument("--polygon-mask", action=argparse.BooleanOptionalAction, default=True)
    roi = parser.add_mutually_exclusive_group()
    roi.add_argument("--keep-outside-roi", dest="keep_outside_roi", action="store_true", default=True)
    roi.add_argument("--filter-ground-in-roi", dest="keep_outside_roi", action="store_false")
    parser.add_argument("--keep-missing-frames", type=int, default=30)
    parser.add_argument("--keep-tracks-after-leave-roi", action="store_true")
    parser.add_argument("--prefetch-frames", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=20)
    parser.add_argument("--gpu-tune", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-infer-fps", type=float, default=0.0)
    parser.add_argument("--ramp-frames", type=int, default=0)
    parser.add_argument("--ramp-max-sleep-ms", type=float, default=0.0)
    parser.add_argument("--gpu-guard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gpu-util-high", type=float, default=90.0)
    parser.add_argument("--gpu-power-high-ratio", type=float, default=0.85)
    parser.add_argument("--gpu-guard-sleep-ms", type=float, default=8.0)
    parser.add_argument("--csv-flush-every", type=int, default=50000)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--show-scale", type=float, default=0.6)
    parser.add_argument("--save-video", type=Path)
    parser.add_argument("--params-log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--params-log-path", type=Path)
    parser.add_argument("--run-tag", default="portfolio_paper_preset")
    parser.add_argument("--quality-check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quality-json-path", type=Path)
    parser.add_argument("--tl-roi", default="1079,426;1119,426;1119,443;1079,443")
    parser.add_argument("--tl-img-size", type=int, default=0)
    parser.add_argument("--tl-min-prob", type=float, default=0.55)
    parser.add_argument("--tl-unknown-label", default="unknown")
    parser.add_argument("--tl-every-n-frames", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    video = args.video_option or args.video
    if video is None:
        parser.error("an input video is required (positional VIDEO or --video VIDEO)")
    for label, path in {
        "video": video,
        "YOLO model": args.yolo_model,
        "camera config": args.camera_config,
        "tracker config": args.tracker_config,
        "traffic-light model": args.traffic_light_model,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    try:
        classes = tuple(int(value.strip()) for value in args.classes.split(",") if value.strip())
    except ValueError as exc:
        parser.error(f"--classes must be comma-separated integer class IDs: {exc}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    params_log_path = args.params_log_path or args.output.with_suffix(".run.jsonl")
    quality_json_path = args.quality_json_path or args.output.with_suffix(".quality.json")
    setup_logging(args.log_level)
    run_tracking_to_csv_roi_crop(
        scene_config_path=str(args.camera_config),
        video_path=str(video),
        out_csv=str(args.output),
        model_path=str(args.yolo_model),
        tracker_cfg=str(args.tracker_config),
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        half=args.half,
        classes=classes or None,
        use_polygon_mask=args.polygon_mask,
        filter_ground_in_roi=not args.keep_outside_roi,
        keep_missing_frames=args.keep_missing_frames,
        drop_track_when_leave_roi=not args.keep_tracks_after_leave_roi,
        log_every=args.log_every,
        show=args.show,
        show_scale=args.show_scale,
        save_video=str(args.save_video) if args.save_video else None,
        prefetch_frames=args.prefetch_frames,
        cpu_threads=args.cpu_threads,
        tune_gpu_backend=args.gpu_tune,
        max_infer_fps=args.max_infer_fps,
        ramp_frames=args.ramp_frames,
        ramp_max_sleep_ms=args.ramp_max_sleep_ms,
        gpu_guard=args.gpu_guard,
        gpu_util_high=args.gpu_util_high,
        gpu_power_high_ratio=args.gpu_power_high_ratio,
        gpu_guard_sleep_ms=args.gpu_guard_sleep_ms,
        csv_flush_every=args.csv_flush_every,
        log_run_params=args.params_log,
        params_log_path=str(params_log_path),
        run_tag=args.run_tag,
        args_snapshot={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        run_quality_check=args.quality_check,
        quality_json_path=str(quality_json_path),
        tl_cnn_model=str(args.traffic_light_model),
        tl_roi=args.tl_roi,
        tl_img_size=args.tl_img_size,
        tl_min_prob=args.tl_min_prob,
        tl_unknown_label=args.tl_unknown_label,
        tl_every_n_frames=args.tl_every_n_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
