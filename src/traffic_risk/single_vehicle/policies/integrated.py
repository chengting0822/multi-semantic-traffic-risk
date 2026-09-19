#!/usr/bin/env python3
"""Clean v60 integrated upstream sidecar policy.

This module extracts the accepted v60 integrated branch used by the historical
v90 chain:

1. trajectory weaving pre-confirm sidecar can support at least risk2;
2. overspeed extreme pressure event can support risk3;
3. short/duplicate bbox guard blocks the extreme speed branch;
4. guarded ge110 onset and head-swing diagnostic branches are intentionally
   not consumed here.  Stable ge110 is handled by the later clean v61 module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleanV60IntegratedColumns:
    current: str = "v92_v60_integrated_current"
    reason: str = "v92_v60_integrated_reason"
    trajectory_weaving_risk2: str = "v92_v60_trajectory_weaving_risk2_support"
    extreme_pressure_risk3: str = "v92_v60_extreme_pressure_risk3_support"
    extreme_pressure_guarded_out: str = "v92_v60_extreme_pressure_guarded_out"


COLS = CleanV60IntegratedColumns()


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def apply_clean_v60_integrated_policy(
    frame: pd.DataFrame,
    *,
    base_current_col: str = "pred_v58c_rebuilt_latest_labels",
    out_current_col: str | None = None,
    out_reason_col: str | None = None,
) -> pd.DataFrame:
    """Apply the accepted v60 integrated sidecar policy.

    Values use SchemaC:
    0 = risk0+1, 1 = risk2, 2 = risk3.
    """

    if base_current_col not in frame.columns:
        raise KeyError(f"base current column not found: {base_current_col}")

    out = frame.copy()
    out_current_col = out_current_col or COLS.current
    out_reason_col = out_reason_col or COLS.reason

    current = num(out, base_current_col).astype(int).clip(0, 2)

    trajectory_weaving_risk2 = num(out, "trajectory_weaving_preconfirm_risk2_support_v60").gt(0)
    extreme_pressure_event = num(out, "overspeed_extreme_pressure_speed_event_v60").gt(0)
    short_duplicate_guard = num(out, "overspeed_short_duplicate_bbox_guard_v60").gt(0)
    extreme_pressure_risk3 = extreme_pressure_event & ~short_duplicate_guard

    updated = current.copy()
    updated.loc[trajectory_weaving_risk2] = np.maximum(updated.loc[trajectory_weaving_risk2], 1)
    updated.loc[extreme_pressure_risk3] = np.maximum(updated.loc[extreme_pressure_risk3], 2)

    reason = np.full(len(out), "", dtype=object)
    reason = np.where(trajectory_weaving_risk2, "v60_trajectory_weaving_preconfirm_risk2", reason)
    reason = np.where(extreme_pressure_risk3, "v60_extreme_pressure_speed_risk3", reason)

    out[COLS.trajectory_weaving_risk2] = trajectory_weaving_risk2.astype(int)
    out[COLS.extreme_pressure_risk3] = extreme_pressure_risk3.astype(int)
    out[COLS.extreme_pressure_guarded_out] = (extreme_pressure_event & short_duplicate_guard).astype(int)
    out[out_reason_col] = reason
    out[out_current_col] = updated.astype(int)
    return out
