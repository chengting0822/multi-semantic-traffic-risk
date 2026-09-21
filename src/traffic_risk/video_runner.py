"""One-video orchestration with camera-specific geometry and arbitrary video IDs."""

from __future__ import annotations

from pathlib import Path

from traffic_risk.paths import CONFIG_DIR, MODEL_DIR


def run_video_pipeline(
    *,
    video: Path,
    output_dir: Path,
    yolo_model: Path | None = None,
    tracks_csv: Path | None = None,
    video_id: str | None = None,
    camera_config: Path = CONFIG_DIR / "camera_roi.json",
    tracker_config: Path = CONFIG_DIR / "bytetrack.yaml",
    traffic_light_model: Path = MODEL_DIR / "traffic_light_classifier.pt",
    tl_roi: str = "1079,426;1119,426;1119,443;1079,443",
    classes: str = "2,3,5,7,9",
    lane_map: Path = CONFIG_DIR / "lane_map.json",
    double_yellow: Path = CONFIG_DIR / "double_yellow.json",
    stop_lines: Path = CONFIG_DIR / "stop_lines.json",
    ipm: Path = CONFIG_DIR / "ipm.json",
    annotation_dir: Path | None = None,
    detector_device: int = 0,
    device: str = "auto",
    fps: float | None = None,
    chunksize: int = 250_000,
) -> Path:
    """Run detector → timestamp tracks → frozen v8/C4O → GRU/scene risk.

    `tracks_csv` skips only YOLO, which is useful for reproducible testing and
    systems without a YOLO weight.  All later stages remain the same.
    """
    from traffic_risk.pipeline import run_risk_pipeline
    from traffic_risk.upstream.runner import build_semantics_v8
    from traffic_risk.upstream.tracking import prepare_tracking_csv

    video = Path(video)
    output_dir = Path(output_dir)
    if not video.is_file():
        raise FileNotFoundError(video)
    for name, path in {
        "lane map": lane_map, "double-yellow geometry": double_yellow,
        "stop lines": stop_lines, "IPM": ipm,
    }.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"{name}: {path}")
    if tracks_csv is None:
        if yolo_model is None:
            raise ValueError("--yolo-model is required unless --tracks-csv is supplied")
        from traffic_risk.detection.cli import main as detect_main

        tracks_csv = output_dir / "detection" / "tracks.csv"
        detect_main([
            str(video), "--output", str(tracks_csv), "--yolo-model", str(yolo_model),
            "--device", str(detector_device), "--camera-config", str(camera_config),
            "--tracker-config", str(tracker_config),
            "--traffic-light-model", str(traffic_light_model),
            "--tl-roi", tl_roi,
            "--classes", classes,
        ])
    else:
        tracks_csv = Path(tracks_csv)
        if not tracks_csv.is_file():
            raise FileNotFoundError(tracks_csv)
    prepared = prepare_tracking_csv(
        detector_csv=tracks_csv, video_path=video,
        output_dir=output_dir / "upstream", video_id=video_id,
        fps=fps, chunksize=chunksize,
    )
    semantics = build_semantics_v8(
        timestamp_csv=prepared.timestamp_csv, output_dir=output_dir / "semantics",
        lane_map=lane_map, double_yellow=double_yellow,
        ipm=ipm, stop_lines=stop_lines, chunksize=chunksize,
    )
    if semantics["c4o_window_count"] == 0:
        raise ValueError("No car track has at least two C4O windows; no risk prediction can be made")
    return run_risk_pipeline(
        c4o_features=Path(semantics["v8"]["c4o_features"]),
        trajectory_features=Path(semantics["trajectory"]["windows"]),
        timestamp_dir=prepared.timestamp_csv.parent,
        output_dir=output_dir / "risk",
        annotation_dir=Path(annotation_dir) if annotation_dir is not None else output_dir / "annotations",
        lane_map=lane_map, stop_lines=stop_lines, ipm=ipm,
        device=device,
    )


__all__ = ["run_video_pipeline"]
