"""Build the accepted v0 three-branch semantics from any timestamp CSV."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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


def build_semantics_v8(
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
    """Build the complete frozen three-branch v8 and C4O inference input."""

    from .canonical import build_c4o_from_v8
    from .fusion_v8 import build_fusion_v8
    from .overspeed.v8 import build_overspeed_v8_frames_from_inputs
    from .redlight.v8 import build_redlight_v8
    from .s33 import build_s33_prefix
    from .v8_common import join

    output_dir = Path(output_dir)
    prior = build_semantics_v0(
        timestamp_csv=timestamp_csv, output_dir=output_dir,
        lane_map=lane_map, double_yellow=double_yellow,
        trajectory_model=trajectory_model, ipm=ipm,
        stop_lines=stop_lines, chunksize=chunksize,
    )
    def read(path: str) -> pd.DataFrame:
        return pd.read_csv(path, low_memory=False)

    trajectory_v8 = read(prior["trajectory"]["windows"])
    trajectory_sidecar = read(prior["trajectory"]["sidecar"])
    overspeed_v0 = read(prior["overspeed"]["windows"])
    overspeed_sidecar = read(prior["overspeed"]["sidecar"])
    redlight_v0 = read(prior["redlight"]["windows"])
    redlight_sidecar = read(prior["redlight"]["sidecar"])
    fusion_v0 = read(prior["fusion"]["fusion"])
    fusion_sidecar = read(prior["fusion"]["sidecar"])

    overspeed_v8, speed_evidence, speed_report = build_overspeed_v8_frames_from_inputs(
        base_frame=trajectory_v8, module_frame=overspeed_v0,
        sidecar_frame=overspeed_sidecar, ipm_json=Path(ipm),
    )
    redlight_v8 = build_redlight_v8(redlight_v0, redlight_sidecar)
    fusion_base = join(fusion_v0, fusion_sidecar, label="fusion_sidecar")
    fusion_v8 = build_fusion_v8(fusion_base, trajectory_v8, overspeed_v8, redlight_v8)
    prefix = build_s33_prefix(fusion_v8, trajectory_sidecar)
    overspeed_full = join(overspeed_v8, speed_evidence, label="overspeed_evidence")
    canonical = build_c4o_from_v8(
        fusion_v8=fusion_v8, trajectory_v8=trajectory_v8,
        overspeed_v8=overspeed_full, redlight_v8=redlight_v8,
        s33_prefix=prefix,
    )
    final_dir = output_dir / "v8"
    final_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "trajectory_v8": trajectory_v8,
        "overspeed_v8": overspeed_v8,
        "overspeed_v8_evidence": speed_evidence,
        "redlight_v8": redlight_v8,
        "fusion_v8": fusion_v8,
        "s33_prefix": prefix,
        "c4o_features": canonical,
    }
    paths: dict[str, str] = {}
    for name, frame in tables.items():
        path = final_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = str(path)
    manifest = {
        **prior,
        "schema_version": "traffic_risk.upstream_semantics_v8/v1",
        "stage": "formal_c4o_ready_for_run_risk",
        "v8": paths,
        "overspeed_v8_report": {
            "physical_speed_source_ratio": speed_report["physical_speed_source_ratio"],
            "raw_recompute_rejected_count": speed_report["raw_recompute_rejected_count"],
        },
        "c4o_window_count": len(canonical),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


__all__ = ["build_semantics_v0", "build_semantics_v8"]
