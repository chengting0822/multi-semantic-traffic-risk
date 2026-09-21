"""Accepted per-frame traffic-light evidence formulas."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


KNOWN_TL_STATES = {"red", "yellow", "green"}
TRAFFIC_LIGHT_CLS_ID = 9


def clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))


def normalize_tl_state(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"r", "red", "紅", "紅燈"}:
        return "red"
    if text in {"y", "yellow", "amber", "黃", "黃燈"}:
        return "yellow"
    if text in {"g", "green", "綠", "綠燈"}:
        return "green"
    return "unknown"


def build_frame_traffic_light_table(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "frame" not in raw_df.columns or "tl_state" not in raw_df.columns:
        return pd.DataFrame(columns=["frame", "tl_state", "tl_prob_red", "tl_prob_green"]), {
            "traffic_light_rows": 0,
            "traffic_light_frame_count": 0,
            "traffic_light_source": "missing_frame_or_tl_state",
        }

    df = raw_df.copy()
    df["tl_state"] = df["tl_state"].map(normalize_tl_state)
    if "tl_prob_red" not in df.columns:
        df["tl_prob_red"] = np.nan
    if "tl_prob_green" not in df.columns:
        df["tl_prob_green"] = np.nan
    df["tl_prob_red"] = pd.to_numeric(df["tl_prob_red"], errors="coerce")
    df["tl_prob_green"] = pd.to_numeric(df["tl_prob_green"], errors="coerce")
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df = df[df["frame"].notna()].copy()

    traffic_mask = pd.Series(False, index=df.index)
    if "cls" in df.columns:
        traffic_mask |= pd.to_numeric(df["cls"], errors="coerce").fillna(-1).astype(int).eq(TRAFFIC_LIGHT_CLS_ID)
    if "track_id" in df.columns:
        traffic_mask |= pd.to_numeric(df["track_id"], errors="coerce").fillna(0).astype(float).lt(0)
    tl_df = df.loc[traffic_mask].copy()
    source = "traffic_light_detector_rows"
    if tl_df.empty:
        tl_df = df[df["tl_state"].isin(KNOWN_TL_STATES)].copy()
        source = "all_rows_tl_state_fallback"

    if tl_df.empty:
        return pd.DataFrame(columns=["frame", "tl_state", "tl_prob_red", "tl_prob_green"]), {
            "traffic_light_rows": 0,
            "traffic_light_frame_count": 0,
            "traffic_light_source": source,
        }

    rows: list[dict[str, Any]] = []
    for frame_value, group in tl_df.groupby("frame", sort=True):
        prob_red = pd.to_numeric(group["tl_prob_red"], errors="coerce")
        if prob_red.notna().any():
            selected = group.loc[prob_red.idxmax()]
        else:
            red_rows = group[group["tl_state"] == "red"]
            selected = red_rows.iloc[0] if not red_rows.empty else group.iloc[0]
        rows.append(
            {
                "frame": int(frame_value),
                "tl_state": normalize_tl_state(selected.get("tl_state", "unknown")),
                "tl_prob_red": clip01(float(selected.get("tl_prob_red", 0.0) or 0.0)),
                "tl_prob_green": clip01(float(selected.get("tl_prob_green", 0.0) or 0.0)),
            }
        )
    frame_table = pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)
    metadata = {
        "traffic_light_rows": int(tl_df.shape[0]),
        "traffic_light_frame_count": int(frame_table.shape[0]),
        "traffic_light_source": source,
        "tl_state_distribution": frame_table["tl_state"].value_counts().to_dict(),
    }
    return frame_table, metadata


def summarize_traffic_light_for_frames(frames: pd.Series | np.ndarray, frame_table: pd.DataFrame) -> dict[str, Any]:
    if frame_table.empty:
        return {
            "tl_state_window": "unknown",
            "tl_state_sequence": "",
            "tl_red_ratio": 0.0,
            "tl_prob_red_mean": 0.0,
            "tl_prob_red_max": 0.0,
            "tl_valid_state_count": 0,
            "tl_known_state_count": 0,
            "tl_frame_count": 0,
            "red_signal_score": 0.0,
            "traffic_light_signal_valid": 0,
        }
    frame_values = pd.to_numeric(pd.Series(frames), errors="coerce").dropna().astype(int)
    if frame_values.empty:
        return {
            "tl_state_window": "unknown",
            "tl_state_sequence": "",
            "tl_red_ratio": 0.0,
            "tl_prob_red_mean": 0.0,
            "tl_prob_red_max": 0.0,
            "tl_valid_state_count": 0,
            "tl_known_state_count": 0,
            "tl_frame_count": 0,
            "red_signal_score": 0.0,
            "traffic_light_signal_valid": 0,
        }
    lookup_df = frame_table.set_index("frame")
    matched = lookup_df.reindex(frame_values.to_numpy())
    states = matched["tl_state"].fillna("unknown").map(normalize_tl_state).tolist()
    probs = pd.to_numeric(matched["tl_prob_red"], errors="coerce").to_numpy(dtype=np.float64)
    known_states = [state for state in states if state in KNOWN_TL_STATES]
    red_count = sum(1 for state in states if state == "red")
    valid_count = len(known_states)
    frame_count = len(states)
    state_counter = Counter(known_states)
    tl_state_window = state_counter.most_common(1)[0][0] if state_counter else "unknown"
    finite_probs = probs[np.isfinite(probs)]
    tl_prob_red_mean = float(np.mean(finite_probs)) if len(finite_probs) else 0.0
    tl_prob_red_max = float(np.max(finite_probs)) if len(finite_probs) else 0.0
    tl_red_ratio = float(red_count / max(frame_count, 1))
    red_signal_score = clip01(max(tl_red_ratio, tl_prob_red_mean, tl_prob_red_max if tl_red_ratio > 0.0 else 0.0))
    return {
        "tl_state_window": tl_state_window,
        "tl_state_sequence": ",".join(states[:60]),
        "tl_red_ratio": tl_red_ratio,
        "tl_prob_red_mean": tl_prob_red_mean,
        "tl_prob_red_max": tl_prob_red_max,
        "tl_valid_state_count": valid_count,
        "tl_known_state_count": valid_count,
        "tl_frame_count": frame_count,
        "red_signal_score": red_signal_score,
        "traffic_light_signal_valid": int(valid_count > 0),
    }


def frame_is_red(frame_value: int, frame_table: pd.DataFrame) -> bool:
    if frame_table.empty:
        return False
    matched = frame_table.loc[frame_table["frame"] == int(frame_value)]
    if matched.empty:
        return False
    return normalize_tl_state(matched.iloc[0].get("tl_state", "unknown")) == "red"


def compute_redlight_score(
    *,
    red_signal_score: float,
    stopline_cross_score: float,
    forward_motion_score: float,
    signal_valid_score: float,
) -> float:
    return clip01(
        (0.35 * clip01(red_signal_score))
        + (0.30 * clip01(stopline_cross_score))
        + (0.25 * clip01(forward_motion_score))
        + (0.10 * clip01(signal_valid_score))
    )
