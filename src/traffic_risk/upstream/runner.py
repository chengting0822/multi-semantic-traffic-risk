"""Build the accepted v0 three-branch semantics from any timestamp CSV."""

from __future__ import annotations

import json
from pathlib import Path

from traffic_risk.paths import CONFIG_DIR, MODEL_DIR

from .fusion import build_fusion_features, write_fusion_features
from .overspeed.runner import build_overspeed_features, write_overspeed_features
from .redlight.runner import build_redlight_features, write_redlight_features
from .trajectory.contracts import KEY_COLUMNS
from .trajectory.runner import build_trajectory_features, write_trajectory_features


def build_semantics_v0(
    *,
    timestamp_csv: Path,
    output_dir: Path,
    lane_map: Path = CONFIG_DIR / "lane_map.json",
    double_yellow: Path = CONFIG_DIR / "double_yellow.json",
    trajectory_model: Path = MODEL_DIR / "trajectory_gmm_b5.json",
    ipm: Path = CONFIG_DIR / "ipm.json",
    stop_lines: Path = CONFIG_DIR / "stop_lines.json",
    chunksize: int = 250_000,
) -> dict[str, object]:
    """Run three formal upstream branches; stop before C4O/GRU inference.

    The trajectory branch defines the car-only window set.  This avoids
    silently synthesizing a missing trajectory signal for non-car objects.
    """

    timestamp_csv = Path(timestamp_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = build_trajectory_features(
        timestamp_csv=timestamp_csv,
        lane_map_json=lane_map,
        double_yellow_json=double_yellow,
        model_json=trajectory_model,
        chunksize=chunksize,
    )
    trajectory_paths = write_trajectory_features(trajectory, output_dir / "trajectory")
    reference_csv = output_dir / "car_reference_windows.csv"
    trajectory.fixed20[[*KEY_COLUMNS, "start_sec", "end_sec"]].to_csv(reference_csv, index=False)
    overspeed = build_overspeed_features(
        timestamp_csv=timestamp_csv,
        reference_windows_csv=reference_csv,
        ipm_json=ipm,
    )
    overspeed_paths = write_overspeed_features(overspeed, output_dir / "overspeed")
    redlight = build_redlight_features(
        timestamp_csv=timestamp_csv,
        reference_windows_csv=reference_csv,
        stop_lines_json=stop_lines,
    )
    redlight_paths = write_redlight_features(redlight, output_dir / "redlight")
    fusion = build_fusion_features(trajectory.sidecar, overspeed.windows, redlight.windows)
    fusion_paths = write_fusion_features(fusion, output_dir / "fusion")
    manifest = {
        "schema_version": "traffic_risk.upstream_semantics_v0/v1",
        "stage": "three_branch_v0_not_c4o_or_final_risk",
        "timestamp_csv": str(timestamp_csv),
        "car_window_count": len(trajectory.fixed20),
        "trajectory": {key: str(value) for key, value in trajectory_paths.items()},
        "overspeed": {"windows": str(overspeed_paths[0]), "sidecar": str(overspeed_paths[1])},
        "redlight": {"windows": str(redlight_paths[0]), "sidecar": str(redlight_paths[1])},
        "fusion": {key: str(value) for key, value in fusion_paths.items()},
        "car_reference_windows": str(reference_csv),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


__all__ = ["build_semantics_v0"]
