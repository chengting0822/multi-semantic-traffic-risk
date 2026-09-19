#!/usr/bin/env python3
"""Clean v58c redlight/trajectory bridge policy.

This module extracts the accepted v58c prediction bridge:

1. redlight pre-confirm can support at least risk2;
2. redlight + strong trajectory/probability support can support risk3;
3. redlight + second consecutive weaving memory can support risk3.

It intentionally does not compute labels.  v54/v90 label policies are
evaluation-only and must not be mixed into prediction logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


GROUP = ["split", "case_key", "video_id", "track_id"]


@dataclass(frozen=True)
class CleanV58cRedlightBridgeColumns:
    current: str = "v92_v58c_rebuilt_latest_labels_current"
    reason: str = "v92_v58c_redlight_bridge_reason"
    redlight_preconfirm_r2: str = "v92_v58_redlight_preconfirm_r2"
    redlight_traj_prob_r3: str = "v92_v58_redlight_traj_prob_r3"
    redlight_weaving_second_r3: str = "v92_v58_redlight_weaving_second_r3"


COLS = CleanV58cRedlightBridgeColumns()


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def flag(frame: pd.DataFrame, col: str) -> pd.Series:
    return num(frame, col).gt(0)


def previous_flag(frame: pd.DataFrame, col: str) -> pd.Series:
    values = flag(frame, col)
    present_group = [c for c in GROUP if c in frame.columns]
    tmp = frame[present_group + ["ts_window_idx"]].copy()
    tmp["_value"] = values.astype(int).to_numpy()
    tmp = tmp.sort_values(present_group + ["ts_window_idx"], kind="mergesort")
    tmp["_prev"] = tmp.groupby(present_group, sort=False)["_value"].shift(1).fillna(0).astype(bool)
    out = pd.Series(False, index=frame.index)
    out.loc[tmp.index] = tmp["_prev"].to_numpy()
    return out


def apply_clean_v58c_redlight_bridge_policy(
    frame: pd.DataFrame,
    *,
    base_current_col: str = "pred_v52d_pure_overspeed_onset_guard",
    out_current_col: str | None = None,
    out_reason_col: str | None = None,
) -> pd.DataFrame:
    """Apply the accepted v58c redlight/trajectory bridge.

    Values use SchemaC:
    0 = risk0+1, 1 = risk2, 2 = risk3.
    """

    if base_current_col not in frame.columns:
        raise KeyError(f"base current column not found: {base_current_col}")

    out = frame.copy()
    out_current_col = out_current_col or COLS.current
    out_reason_col = out_reason_col or COLS.reason

    current = num(out, base_current_col).astype(int).clip(0, 2)
    red = num(out, "redlight_event_score")
    prob2 = num(out, "prob_class2")
    raw = num(out, "computed_speed_kmh_p95")

    redlight_preconfirm_r2 = (
        current.lt(1)
        & red.ge(0.95)
        & flag(out, "v38_policy_risk2_support")
        & raw.lt(90.0)
    )

    redlight_traj_prob_r3 = (
        current.lt(2)
        & red.ge(0.60)
        & flag(out, "v38_policy_risk3_support")
        & num(out, "v38_trajectory_risk2_support").ge(0.90)
        & prob2.ge(0.45)
    )

    weaving_prev = previous_flag(out, "v28_true_weaving_memory")
    redlight_weaving_second_r3 = (
        current.lt(2)
        & red.ge(0.60)
        & flag(out, "v38_policy_risk3_support")
        & flag(out, "v28_true_weaving_memory")
        & weaving_prev
        & prob2.ge(0.80)
    )

    updated = current.copy()
    updated.loc[redlight_preconfirm_r2] = np.maximum(updated.loc[redlight_preconfirm_r2], 1)
    updated.loc[redlight_traj_prob_r3] = np.maximum(updated.loc[redlight_traj_prob_r3], 2)
    updated.loc[redlight_weaving_second_r3] = np.maximum(updated.loc[redlight_weaving_second_r3], 2)

    reason = np.full(len(out), "", dtype=object)
    reason = np.where(redlight_preconfirm_r2, "v58_redlight_preconfirm_r2", reason)
    reason = np.where(redlight_traj_prob_r3, "v58_redlight_traj_prob_r3", reason)
    reason = np.where(redlight_weaving_second_r3, "v58_redlight_weaving_second_r3", reason)

    out[COLS.redlight_preconfirm_r2] = redlight_preconfirm_r2.astype(int)
    out[COLS.redlight_traj_prob_r3] = redlight_traj_prob_r3.astype(int)
    out[COLS.redlight_weaving_second_r3] = redlight_weaving_second_r3.astype(int)
    out[out_reason_col] = reason
    out[out_current_col] = updated.astype(int)
    return out
