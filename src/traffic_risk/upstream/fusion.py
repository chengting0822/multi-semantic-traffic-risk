"""Formal v0 semantic adapters and three-branch fusion.

The v0 table is an inspectable intermediate, not the 16-column C4O model
input.  It reproduces the accepted GRU adapters and fusion joiner without
labels, historical output files, or a fixed list of demo video IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from traffic_risk.identifiers import canonical_id

KEYS = ["case_key", "source_type", "source_id", "video_id", "track_id", "ts_window_idx"]
TIMES = ["start_sec", "end_sec"]
TRAJECTORY_COLUMNS = [
    "trajectory_anomaly_score", "trajectory_anomaly_level",
    "trajectory_anomaly_duration_sec_proxy", "trajectory_anomaly_reliability",
    "trajectory_anomaly_signal_valid",
]
OVERSPEED_COLUMNS = [
    "overspeed_score", "overspeed_level", "overspeed_duration_sec_proxy", "overspeed_reliability",
]
REDLIGHT_COLUMNS = [
    "redlight_confirmed", "redlight_score", "redlight_duration_sec_proxy", "redlight_signal_valid",
]
GLOBAL_COLUMNS = ["semantic_active_module_count", "semantic_history_ema", "semantic_recent_peak_level"]
MAIN_COLUMNS = [
    *KEYS, *TIMES, *TRAJECTORY_COLUMNS[:4], *OVERSPEED_COLUMNS, *REDLIGHT_COLUMNS,
    *GLOBAL_COLUMNS,
]
SIDECAR_COLUMNS = [
    *KEYS, *TIMES, "trajectory_anomaly_signal_valid", "trajectory_active",
    "overspeed_active", "redlight_active", "redlight_proxy_level",
    "semantic_current_score", "semantic_current_level",
]


@dataclass(frozen=True)
class FusionFeatures:
    windows: pd.DataFrame
    sidecar: pd.DataFrame
    overspeed_gru: pd.DataFrame
    redlight_gru: pd.DataFrame


def _normalize(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    missing = [column for column in [*KEYS, *TIMES] if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing join columns: {missing}")
    out = frame.copy()
    for column in ["source_id", "video_id", "track_id"]:
        out[column] = out[column].map(canonical_id)
    for column in ["case_key", "source_type"]:
        out[column] = out[column].astype(str).str.strip()
    out["ts_window_idx"] = pd.to_numeric(out["ts_window_idx"], errors="raise").astype(int)
    for column in TIMES:
        out[column] = pd.to_numeric(out[column], errors="raise").astype(float)
    if out.duplicated(KEYS).any():
        raise ValueError(f"{label} contains duplicate semantic window keys")
    return out


def _number(row: pd.Series, name: str, default: float = 0.0) -> float:
    value = pd.to_numeric(pd.Series([row.get(name, default)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) and np.isfinite(value) else default


def _true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y", "on"}


def _stride(frame: pd.DataFrame) -> float:
    diffs: list[float] = []
    for _, group in frame.sort_values(["case_key", "ts_window_idx", "start_sec"]).groupby("case_key", sort=False):
        delta = np.diff(group["start_sec"].to_numpy(dtype=float))
        diffs.extend(delta[np.isfinite(delta) & (delta > 1e-8)].tolist())
    return float(np.median(diffs)) if diffs else 0.3333


def adapt_overspeed(module: pd.DataFrame) -> pd.DataFrame:
    """Exact causal v0 overspeed adapter used by the research fusion joiner."""

    frame = _normalize(module, label="overspeed")
    required = ["speed_kmh", "overspeed_level", "overspeed_reliability", "timestamp_delta_valid", "ipm_source_valid", "track_motion_valid"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"overspeed is missing module columns: {missing}")
    stride = _stride(frame)
    rows: list[dict[str, object]] = []
    ordered = frame.sort_values(["case_key", "ts_window_idx", "end_sec"], kind="mergesort")
    for _, group in ordered.groupby("case_key", sort=False):
        consecutive = 0
        for _, row in group.iterrows():
            level = int(_number(row, "overspeed_level"))
            valid = level > 0 and all(_true(row[name]) for name in ["timestamp_delta_valid", "ipm_source_valid", "track_motion_valid"])
            consecutive = consecutive + 1 if valid else 0
            rows.append({
                **{column: row[column] for column in [*KEYS, *TIMES]},
                "overspeed_score": float(np.clip((_number(row, "speed_kmh") - 70.0) / 40.0, 0.0, 1.0)),
                "overspeed_level": level,
                "overspeed_duration_sec_proxy": float(np.clip(consecutive * stride, 0.0, 6.0)),
                "overspeed_reliability": float(np.clip(_number(row, "overspeed_reliability"), 0.0, 1.0)),
            })
    return pd.DataFrame(rows, columns=[*KEYS, *TIMES, *OVERSPEED_COLUMNS])


def adapt_redlight(module: pd.DataFrame) -> pd.DataFrame:
    """Exact causal v0 red-light adapter used by the research fusion joiner."""

    frame = _normalize(module, label="redlight")
    required = ["redlight_confirmed", "redlight_score", "redlight_signal_valid"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"redlight is missing module columns: {missing}")
    stride = _stride(frame)
    rows: list[dict[str, object]] = []
    ordered = frame.sort_values(["case_key", "ts_window_idx", "end_sec"], kind="mergesort")
    for _, group in ordered.groupby("case_key", sort=False):
        consecutive = 0
        for _, row in group.iterrows():
            confirmed = int(_number(row, "redlight_confirmed"))
            signal_valid = int(_number(row, "redlight_signal_valid"))
            consecutive = consecutive + 1 if confirmed == 1 and signal_valid == 1 else 0
            rows.append({
                **{column: row[column] for column in [*KEYS, *TIMES]},
                "redlight_confirmed": confirmed,
                "redlight_score": float(np.clip(_number(row, "redlight_score"), 0.0, 1.0)),
                "redlight_duration_sec_proxy": min(consecutive * stride, 6.0),
                "redlight_signal_valid": signal_valid,
            })
    return pd.DataFrame(rows, columns=[*KEYS, *TIMES, *REDLIGHT_COLUMNS])


def _join(base: pd.DataFrame, module: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    missing_columns = [column for column in columns if column not in module]
    if missing_columns:
        raise ValueError(f"{label} is missing semantic columns: {missing_columns}")
    joined = base.merge(module[[*KEYS, *TIMES, *columns]], on=KEYS, how="left", suffixes=("", f"_{label}"), validate="one_to_one", indicator=True)
    absent = joined["_merge"].ne("both")
    if absent.any():
        examples = joined.loc[absent, KEYS].head(3).to_dict("records")
        raise ValueError(f"{label} lacks {int(absent.sum())} trajectory windows; examples={examples}")
    for column in TIMES:
        other = f"{column}_{label}"
        mismatch = (joined[column] - joined[other]).abs().gt(1e-3)
        if mismatch.any():
            raise ValueError(f"{label} has {int(mismatch.sum())} mismatched {column} values")
    return joined.drop(columns=["_merge", *(f"{column}_{label}" for column in TIMES)])


def build_fusion_features(
    trajectory: pd.DataFrame,
    overspeed: pd.DataFrame,
    redlight: pd.DataFrame,
    *,
    overspeed_is_gru: bool = False,
    redlight_is_gru: bool = False,
) -> FusionFeatures:
    """Align all three branches and reproduce the accepted 15D v0 fusion."""

    trajectory = _normalize(trajectory, label="trajectory")
    missing_trajectory = [column for column in TRAJECTORY_COLUMNS if column not in trajectory]
    if missing_trajectory:
        raise ValueError(f"trajectory is missing semantic columns: {missing_trajectory}")
    overspeed_gru = _normalize(overspeed, label="overspeed") if overspeed_is_gru else adapt_overspeed(overspeed)
    redlight_gru = _normalize(redlight, label="redlight") if redlight_is_gru else adapt_redlight(redlight)
    frame = _join(trajectory[[*KEYS, *TIMES, *TRAJECTORY_COLUMNS]], overspeed_gru, OVERSPEED_COLUMNS, "overspeed")
    frame = _join(frame, redlight_gru, REDLIGHT_COLUMNS, "redlight")

    frame["trajectory_active"] = (pd.to_numeric(frame["trajectory_anomaly_level"]) > 0).astype(int)
    frame["overspeed_active"] = (pd.to_numeric(frame["overspeed_level"]) > 0).astype(int)
    frame["redlight_active"] = ((pd.to_numeric(frame["redlight_confirmed"]) == 1) | (pd.to_numeric(frame["redlight_score"]) >= 0.5)).astype(int)
    frame["semantic_active_module_count"] = frame[["trajectory_active", "overspeed_active", "redlight_active"]].sum(axis=1).astype(int)
    frame["semantic_current_score"] = frame[["trajectory_anomaly_score", "overspeed_score", "redlight_score"]].astype(float).max(axis=1).clip(0.0, 1.0)
    frame["redlight_proxy_level"] = np.select([frame["redlight_confirmed"].eq(1), frame["redlight_score"].ge(0.5)], [2, 1], default=0).astype(int)
    frame["semantic_current_level"] = frame[["trajectory_anomaly_level", "overspeed_level", "redlight_proxy_level"]].astype(int).max(axis=1)
    frame["semantic_history_ema"] = 0.0
    frame["semantic_recent_peak_level"] = 0
    for _, group in frame.groupby(["case_key", "video_id", "track_id"], sort=False):
        ordered = group.sort_values(["ts_window_idx", "start_sec"], kind="mergesort")
        previous_ema = previous_score = 0.0
        levels: list[int] = []
        for index, row in ordered.iterrows():
            current_ema = 0.2 * previous_score + 0.8 * previous_ema
            frame.at[index, "semantic_history_ema"] = float(np.clip(current_ema, 0.0, 1.0))
            previous_ema, previous_score = current_ema, float(row["semantic_current_score"])
            levels.append(int(row["semantic_current_level"]))
            frame.at[index, "semantic_recent_peak_level"] = max(levels[-6:])
    frame = frame.sort_values(["source_id", "track_id", "ts_window_idx", "start_sec", "end_sec"], kind="mergesort")
    return FusionFeatures(
        windows=frame[MAIN_COLUMNS].reset_index(drop=True),
        sidecar=frame[SIDECAR_COLUMNS].reset_index(drop=True),
        overspeed_gru=overspeed_gru.reset_index(drop=True),
        redlight_gru=redlight_gru.reset_index(drop=True),
    )


def write_fusion_features(features: FusionFeatures, output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "fusion": output_dir / "semantic_fusion_v0.csv",
        "sidecar": output_dir / "semantic_fusion_v0_sidecar.csv",
        "overspeed_gru": output_dir / "overspeed_gru_v0.csv",
        "redlight_gru": output_dir / "redlight_gru_v0.csv",
    }
    features.windows.to_csv(paths["fusion"], index=False)
    features.sidecar.to_csv(paths["sidecar"], index=False)
    features.overspeed_gru.to_csv(paths["overspeed_gru"], index=False)
    features.redlight_gru.to_csv(paths["redlight_gru"], index=False)
    return paths


__all__ = ["FusionFeatures", "adapt_overspeed", "adapt_redlight", "build_fusion_features", "write_fusion_features"]
