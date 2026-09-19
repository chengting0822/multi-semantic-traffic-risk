#!/usr/bin/env python3
"""Clean v38 trajectory-only risk3 severity cap.

This module distills the accepted v38 inference step:

    pred_v37e -> pred_v37e_v38_policy_cap

The rule is deliberately narrow.  If the current window is risk3 only because
of trajectory-only context, and there is no composite risk3 support and no
stable >=110 km/h speed support, the current window is capped to risk2.  The
track-level monotonic final risk is then rebuilt from the capped current risk.

It assumes the v38 semantic support columns already exist in the input table.
It does not rebuild C4O tensors, retrain GRU, touch labels, or generate v38
features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


GROUP_COLUMNS = ["split", "case_key", "video_id", "track_id"]
TIME_COLUMN = "ts_window_idx"


@dataclass(frozen=True)
class CleanV38PolicyCapColumns:
    capped_current: str = "v92_v38_policy_cap_capped_current"
    current: str = "v92_v38_policy_cap_current"
    cap_current_window: str = "v92_v38_cap_current_window"
    reason: str = "v92_v38_policy_cap_reason"


COLS = CleanV38PolicyCapColumns()


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["split", "case_key"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
    for col in ["video_id", "track_id", TIME_COLUMN]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(-1).astype(int)
    return out

def apply_clean_v38_policy_cap(
    frame: pd.DataFrame,
    *,
    base_current_col: str = "pred_v37e",
    out_current_col: str | None = None,
) -> pd.DataFrame:
    """Apply the accepted v38 cap and rebuild monotonic track risk."""

    if base_current_col not in frame.columns:
        raise KeyError(f"base current column not found: {base_current_col}")

    out_current_col = out_current_col or COLS.current
    out = normalize_keys(frame)
    base = num(out, base_current_col).astype(int).clip(0, 2)

    cap_current = (
        base.eq(2)
        & num(out, "v38_trajectory_only_context").ge(0.5)
        & num(out, "v38_policy_risk3_support").lt(0.5)
        & num(out, "v38_stable_ge110_support").lt(0.5)
    )

    capped_current = base.mask(cap_current, 1).astype(int)
    out[COLS.capped_current] = capped_current
    out[COLS.cap_current_window] = cap_current.astype(int)
    out[COLS.reason] = np.where(cap_current, "trajectory_only_risk3_cap_to_risk2", "")

    sort_cols = GROUP_COLUMNS + [TIME_COLUMN]
    sorted_out = out.sort_values(sort_cols, kind="mergesort").copy()
    sorted_out[out_current_col] = (
        sorted_out[COLS.capped_current]
        .astype(int)
        .groupby([sorted_out[c] for c in GROUP_COLUMNS], sort=False)
        .cummax()
        .astype(int)
    )
    out[out_current_col] = sorted_out.sort_index()[out_current_col].astype(int)

    if "pred_v37e_v38_policy_cap" not in out.columns:
        out["pred_v37e_v38_policy_cap"] = out[out_current_col]
    if "v38_cap_current_window" not in out.columns:
        out["v38_cap_current_window"] = out[COLS.cap_current_window]
    return out
