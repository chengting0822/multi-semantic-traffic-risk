"""End-to-end trajectory branch from timestamped tracks to accepted v8 semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .contracts import KEY_COLUMNS
from .materializer import materialize_fixed20
from .scoring import build_causal_semantics, score_fixed20
from .v8 import build_oscillation_sidecar_from_samples, build_trajectory_v8_frames_from_merged


@dataclass(frozen=True)
class TrajectoryFeatures:
    windows: pd.DataFrame
    sidecar: pd.DataFrame
    fixed20: pd.DataFrame
    samples: pd.DataFrame
    causal_prefix: pd.DataFrame


def build_trajectory_features(
    *,
    timestamp_csv: Path,
    lane_map_json: Path,
    double_yellow_json: Path,
    model_json: Path,
    chunksize: int = 250_000,
) -> TrajectoryFeatures:
    fixed20, samples = materialize_fixed20(
        timestamp_csv=timestamp_csv,
        lane_map_json=lane_map_json,
        double_yellow_json=double_yellow_json,
        chunksize=chunksize,
    )
    scored = score_fixed20(fixed20, model_json)
    prefix, base_semantics = build_causal_semantics(scored)
    merged = fixed20.merge(base_semantics, on=KEY_COLUMNS + ["start_sec", "end_sec"], how="left", validate="one_to_one")
    oscillation = build_oscillation_sidecar_from_samples(window_frame=merged, sample_frame=samples)
    merged = merged.merge(oscillation, on=KEY_COLUMNS, how="left", validate="one_to_one")
    windows, sidecar = build_trajectory_v8_frames_from_merged(
        merged_frame=merged,
        geometry_source_path=str(double_yellow_json),
        geometry_source_present=True,
    )
    # The downstream semantic builder consumes the v8 sidecar.  Preserve base
    # trajectory evidence there as well so fusion can be built from one table.
    base_columns = [
        *KEY_COLUMNS, "trajectory_anomaly_score", "trajectory_anomaly_level",
        "trajectory_anomaly_duration_sec_proxy", "trajectory_anomaly_reliability",
        "trajectory_anomaly_signal_valid",
    ]
    sidecar = sidecar.merge(base_semantics[base_columns], on=KEY_COLUMNS, how="left", validate="one_to_one")
    return TrajectoryFeatures(windows, sidecar, fixed20, samples, prefix)


def write_trajectory_features(features: TrajectoryFeatures, output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "windows": output_dir / "trajectory_features.csv",
        "sidecar": output_dir / "trajectory_sidecar.csv",
        "fixed20": output_dir / "fixed20_features.csv",
        "samples": output_dir / "fixed20_samples.csv",
        "causal_prefix": output_dir / "trajectory_causal_prefix.csv",
    }
    features.windows.to_csv(paths["windows"], index=False)
    features.sidecar.to_csv(paths["sidecar"], index=False)
    features.fixed20.to_csv(paths["fixed20"], index=False)
    features.samples.to_csv(paths["samples"], index=False)
    features.causal_prefix.to_csv(paths["causal_prefix"], index=False)
    return paths


__all__ = ["TrajectoryFeatures", "build_trajectory_features", "write_trajectory_features"]
