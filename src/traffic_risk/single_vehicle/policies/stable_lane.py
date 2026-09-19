#!/usr/bin/env python3
"""Clean v47 stable lane-backtrack weaving policy.

This module distills the accepted v47 rule:

If a vehicle is already risk2, red-light active, and v28 reports an ambiguous
or occlusion-like lane sequence that nevertheless contains a stable A-B-A lane
backtrack with usable lane mapping and stable bbox size, promote to risk3.

It is a trajectory/redlight composite rule.  It does not compute v28 sidecar
features; those remain upstream inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleanV47StableLaneBacktrackColumns:
    current: str = "v92_v47_stable_lane_backtrack_weaving_current"
    reason: str = "v92_v47_stable_lane_backtrack_reason"
    ambiguous_lane_backtrack: str = "v92_v47_ambiguous_lane_backtrack"
    bbox_stable_for_backtrack: str = "v92_v47_bbox_stable_for_backtrack"
    lane_usable_for_backtrack: str = "v92_v47_lane_usable_for_backtrack"
    stable_lane_backtrack_weaving: str = "v92_v47_stable_lane_backtrack_weaving"
    promote: str = "v92_v47_promote_redlight_stable_backtrack_weaving"
    changed: str = "v92_v47_changed"


COLS = CleanV47StableLaneBacktrackColumns()


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def text(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series("", index=frame.index, dtype=str)
    return frame[col].fillna("").astype(str)


def apply_clean_v47_stable_lane_backtrack_policy(
    frame: pd.DataFrame,
    *,
    base_current_col: str = "pred_v46_redlight85_causal_consecutive",
    out_current_col: str | None = None,
) -> pd.DataFrame:
    """Apply v47 stable lane-backtrack weaving risk3 onset.

    Values use SchemaC:
    0 = risk0+1, 1 = risk2, 2 = risk3.
    """

    if base_current_col not in frame.columns:
        raise KeyError(f"base current column not found: {base_current_col}")

    out = frame.copy()
    out_current_col = out_current_col or COLS.current

    base = num(out, base_current_col).astype(int).clip(0, 2)
    redlight = num(out, "redlight_event_score").ge(0.55)
    backtrack = num(out, "v28_lane_id_backtrack_6w").ge(0.5) | num(
        out, "v28_straight_family_backtrack_6w"
    ).ge(0.5)
    repeated = num(out, "v28_lane_change_count_6w").ge(2)
    bbox_stable = num(out, "v28_bbox_area_cv_raw").lt(0.26) & num(out, "v28_bbox_area_log_jump_max").lt(0.16)
    lane_usable = num(out, "v28_lane_valid_ratio", 1.0).ge(0.85)
    not_already_true_weaving = num(out, "v28_true_weaving_current").lt(0.5)
    ambiguous_backtrack = text(out, "v28_transition_type_6w").eq("ambiguous_or_occlusion") & backtrack & repeated
    stable_lane_backtrack = ambiguous_backtrack & bbox_stable & lane_usable & not_already_true_weaving

    promote = base.eq(1) & redlight & stable_lane_backtrack

    current = base.copy()
    current.loc[promote] = np.maximum(current.loc[promote], 2)

    out[COLS.ambiguous_lane_backtrack] = ambiguous_backtrack.astype(int)
    out[COLS.bbox_stable_for_backtrack] = bbox_stable.astype(int)
    out[COLS.lane_usable_for_backtrack] = lane_usable.astype(int)
    out[COLS.stable_lane_backtrack_weaving] = stable_lane_backtrack.astype(int)
    out[COLS.promote] = promote.astype(int)
    out[COLS.changed] = current.ne(base).astype(int)
    out[COLS.reason] = np.where(promote, "v47_redlight_stable_lane_backtrack_promote_risk3", "")
    out[out_current_col] = current.astype(int)
    return out
