"""Small, path-independent helpers shared by frozen v8 policy builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from traffic_risk.identifiers import normalize_window_keys

KEYS = ["case_key", "source_type", "source_id", "video_id", "track_id", "ts_window_idx"]
TIMES = ["start_sec", "end_sec"]


def number(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default).astype(float)


def clip01(series: pd.Series) -> pd.Series:
    return series.astype(float).clip(0.0, 1.0)


def clip_nonnegative(series: pd.Series) -> pd.Series:
    return series.astype(float).clip(lower=0.0)


def safe_divide(numerator: pd.Series, denominator: pd.Series | float, default: float = 0.0) -> pd.Series:
    if not isinstance(denominator, pd.Series):
        denominator = pd.Series(float(denominator), index=numerator.index, dtype=float)
    result = pd.Series(default, index=numerator.index, dtype=float)
    valid = denominator.astype(float).abs() > 1e-9
    result.loc[valid] = numerator.loc[valid].astype(float) / denominator.loc[valid].astype(float)
    return result.replace([np.inf, -np.inf], default).fillna(default)


def build_window_key_set(frame: pd.DataFrame) -> set[tuple]:
    return set(map(tuple, normalize_window_keys(frame)[KEYS].itertuples(index=False, name=None)))


def burst_count_by_track(frame: pd.DataFrame, active: pd.Series) -> pd.Series:
    ordered = frame.sort_values(["video_id", "track_id", "ts_window_idx", "start_sec", "end_sec"], kind="mergesort")
    result = pd.Series(0.0, index=frame.index, dtype=float)
    for _, indices in ordered.groupby(["video_id", "track_id"], sort=False).groups.items():
        idx = pd.Index(indices)
        values = active.loc[idx].to_numpy(dtype=float)
        previous = np.concatenate([[0.0], values[:-1]])
        result.loc[idx] = np.cumsum((values >= 1.0) & (previous < 1.0)).astype(float)
    return result


def sigmoid(series: pd.Series, center: float, scale: float) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-(series.astype(float) - center) / max(scale, 1e-6)))


def maximum(*series: pd.Series) -> pd.Series:
    return pd.concat(series, axis=1).max(axis=1)


def join(base: pd.DataFrame, extra: pd.DataFrame, *, label: str) -> pd.DataFrame:
    base = normalize_window_keys(base)
    extra = normalize_window_keys(extra)
    for name, frame in (("base", base), (label, extra)):
        missing = [col for col in KEYS + TIMES if col not in frame]
        if missing:
            raise ValueError(f"{name} missing v8 join fields: {missing}")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{name} has duplicate v8 windows")
    payload = [col for col in extra if col not in KEYS + TIMES and col not in base]
    result = base.merge(extra[KEYS + TIMES + payload], on=KEYS, how="left", suffixes=("", f"_{label}"), validate="one_to_one", indicator=True)
    if not result["_merge"].eq("both").all():
        raise ValueError(f"{label} does not cover base windows")
    for time in TIMES:
        if (number(result, time) - number(result, f"{time}_{label}")).abs().gt(1e-3).any():
            raise ValueError(f"{label} has mismatched {time}")
    return result.drop(columns=["_merge", *(f"{time}_{label}" for time in TIMES)])


def cumulative_duration(frame: pd.DataFrame, active: pd.Series) -> pd.Series:
    ordered = frame.sort_values(["video_id", "track_id", "ts_window_idx", "start_sec", "end_sec"], kind="mergesort")
    duration = (number(frame, "end_sec") - number(frame, "start_sec")).clip(lower=0.0)
    result = pd.Series(0.0, index=frame.index, dtype=float)
    for _, indices in ordered.groupby(["video_id", "track_id"], sort=False).groups.items():
        idx = pd.Index(indices)
        result.loc[idx] = np.cumsum(duration.loc[idx].to_numpy() * active.loc[idx].astype(float).to_numpy())
    return result


numeric_series = number
cumulative_duration_by_track = cumulative_duration
WINDOW_JOIN_KEYS = KEYS


__all__ = ["KEYS", "TIMES", "number", "clip01", "sigmoid", "maximum", "join", "cumulative_duration"]
