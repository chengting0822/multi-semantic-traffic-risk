from pathlib import Path

from traffic_risk.detection.cli import build_parser


def test_paper_detection_defaults() -> None:
    args = build_parser().parse_args(["input.mp4", "--yolo-model", "yolo26x.pt"])

    assert args.video == Path("input.mp4")
    assert args.yolo_model == Path("yolo26x.pt")
    assert args.output == Path("outputs/tracks.csv")
    assert args.device == 0
    assert args.imgsz == 1280
    assert args.conf == 0.18
    assert args.iou == 0.55
    assert args.half is True
    assert args.classes == "2,3,5,7,9"
    assert args.keep_outside_roi is True
    assert args.prefetch_frames == 256
    assert args.cpu_threads == 20
    assert args.max_infer_fps == 0
    assert args.ramp_frames == 0
    assert args.ramp_max_sleep_ms == 0
    assert args.gpu_guard is False
    assert args.quality_check is False
    assert args.tl_min_prob == 0.55


def test_legacy_yolotest_argument_names() -> None:
    args = build_parser().parse_args(
        [
            "--scene-config",
            "scene_config.json",
            "--video",
            "video.mp4",
            "--out-csv",
            "tracks.csv",
            "--model",
            "yolo26x.pt",
            "--tracker",
            "trackers/bytetrack.yaml",
            "--device",
            "0",
            "--half",
            "--classes",
            "2,3,5,7,9",
            "--keep-outside-roi",
            "--no-gpu-guard",
            "--tl-cnn-model",
            "traffic_light.pt",
            "--tl-roi",
            "1079,426;1119,426;1119,443;1079,443",
            "--no-quality-check",
        ]
    )

    assert args.video_option == Path("video.mp4")
    assert args.output == Path("tracks.csv")
    assert args.yolo_model == Path("yolo26x.pt")
    assert args.camera_config == Path("scene_config.json")
    assert args.tracker_config == Path("trackers/bytetrack.yaml")
    assert args.traffic_light_model == Path("traffic_light.pt")
    assert args.gpu_guard is False
    assert args.quality_check is False
