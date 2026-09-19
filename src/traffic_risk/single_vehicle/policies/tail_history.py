#!/usr/bin/env python3
"""Clean v49/v50c tail-lane history policy.

This module distills the accepted v49 and v50c rules:

* v49 consumes the upstream v48 tail-lane backtrack sidecar.  If a vehicle is
  already risk2, red-light active, and the current-window tail-lane backtrack
  onset is present, promote to risk3.
* v50c repairs the specific case where v28 lane smoothing has not yet captured
  a lane change, but the completed v48 tail-lane history already forms a
  same-direction A-B-A backtrack.  This is used only for red-light + trajectory
  composite risk3.

Label rebuild / primary-overlap refresh is intentionally not part of this
module; labels are evaluation-only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


GROUP = ["split", "case_key", "video_id", "track_id"]


@dataclass(frozen=True)
class CleanV50CTailHistoryColumns:
    v49_current: str = "v92_v49_tail_lane_backtrack_rule_stack_current"
    v49_reason: str = "v92_v49_tail_lane_backtrack_reason"
    v49_promote: str = "v92_v49_rule_redlight_tail_backtrack_promote"
    v49_changed: str = "v92_v49_changed"

    current: str = "v92_v50c_tail_history_missing_smoothing_current"
    reason: str = "v92_v50c_tail_history_reason"
    changed: str = "v92_v50c_changed"

    lane_sequence: str = "v92_v50_tail_history_lane_sequence_6w"
    bucket_sequence: str = "v92_v50_tail_history_bucket_sequence_6w"
    direction_family_sequence: str = "v92_v50_tail_history_direction_family_sequence_6w"
    backtrack: str = "v92_v50_tail_history_backtrack_6w"
    same_direction_backtrack: str = "v92_v50_tail_history_same_direction_backtrack_6w"
    direction_family_count: str = "v92_v50_tail_history_direction_family_count_6w"
    change_count: str = "v92_v50_tail_history_change_count_6w"
    stable_tail_count: str = "v92_v50_tail_history_stable_tail_count_6w"
    v28_smoothing_missed_lane_change: str = "v92_v50c_v28_smoothing_missed_lane_change"
    promote: str = "v92_v50c_rule_redlight_tail_history_backtrack_promote"


COLS = CleanV50CTailHistoryColumns()


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def compress(seq: list[str]) -> list[str]:
    out: list[str] = []
    for value in seq:
        if not value:
            continue
        if not out or out[-1] != value:
            out.append(value)
    return out


def lane_bucket(lane: str) -> str:
    """Map turn-lane aliases to adjacent straight-lane buckets."""

    lane = str(lane or "")
    if lane == "downbound_left_turn":
        return "downbound_1"
    if lane == "upbound_right_turn":
        return "upbound_4"
    return lane


def lane_direction_family(lane: str) -> str:
    lane = str(lane or "")
    if lane.startswith("downbound"):
        return "downbound"
    if lane.startswith("upbound"):
        return "upbound"
    if lane.startswith("new_lane"):
        return "new_lane"
    return ""


def has_backtrack(seq: list[str]) -> bool:
    comp = compress(seq)
    if len(comp) < 3:
        return False
    return any(comp[i] == comp[i + 2] and comp[i] != comp[i + 1] for i in range(len(comp) - 2))


def add_tail_history_features(frame: pd.DataFrame, *, history_windows: int = 6) -> pd.DataFrame:
    """Recompute the accepted v50 tail-lane history features causally."""

    out = frame.copy()
    out[COLS.lane_sequence] = ""
    out[COLS.bucket_sequence] = ""
    out[COLS.direction_family_sequence] = ""
    out[COLS.backtrack] = 0.0
    out[COLS.same_direction_backtrack] = 0.0
    out[COLS.direction_family_count] = 0.0
    out[COLS.change_count] = 0.0
    out[COLS.stable_tail_count] = 0.0

    stable_tail = (num(out, "v48_tail_lane_ratio") >= 0.50) & (num(out, "v48_tail_frame_count") >= 8)
    out["_v92_v50_tail_lane_for_history"] = out["v48_tail_lane"].fillna("").astype(str).where(stable_tail, "")
    out["_v92_v50_tail_bucket_for_history"] = out["_v92_v50_tail_lane_for_history"].map(lane_bucket)
    out["_v92_v50_tail_family_for_history"] = out["_v92_v50_tail_lane_for_history"].map(lane_direction_family)

    for _, idx in out.groupby(GROUP, sort=False).groups.items():
        part = out.loc[list(idx)].sort_values("ts_window_idx", kind="mergesort")
        ordered = part.index.tolist()
        lanes = part["_v92_v50_tail_lane_for_history"].astype(str).tolist()
        buckets = part["_v92_v50_tail_bucket_for_history"].astype(str).tolist()
        families = part["_v92_v50_tail_family_for_history"].astype(str).tolist()
        stable = [1 if x else 0 for x in lanes]

        lane_seq_texts: list[str] = []
        bucket_seq_texts: list[str] = []
        family_seq_texts: list[str] = []
        backtracks: list[int] = []
        same_direction_backtracks: list[int] = []
        family_counts: list[int] = []
        changes: list[int] = []
        stable_counts: list[int] = []
        for pos in range(len(part)):
            left = max(0, pos - history_windows + 1)
            lane_seq = compress(lanes[left : pos + 1])
            bucket_seq = compress(buckets[left : pos + 1])
            family_seq = compress(families[left : pos + 1])
            non_empty_families = [f for f in families[left : pos + 1] if f]
            family_set = set(non_empty_families)

            lane_seq_texts.append(">".join(lane_seq))
            bucket_seq_texts.append(">".join(bucket_seq))
            family_seq_texts.append(">".join(family_seq))
            backtracks.append(int(has_backtrack(bucket_seq)))
            same_direction_backtracks.append(int(has_backtrack(bucket_seq) and len(family_set) == 1))
            family_counts.append(len(family_set))
            changes.append(max(len(bucket_seq) - 1, 0))
            stable_counts.append(int(sum(stable[left : pos + 1])))

        out.loc[ordered, COLS.lane_sequence] = lane_seq_texts
        out.loc[ordered, COLS.bucket_sequence] = bucket_seq_texts
        out.loc[ordered, COLS.direction_family_sequence] = family_seq_texts
        out.loc[ordered, COLS.backtrack] = backtracks
        out.loc[ordered, COLS.same_direction_backtrack] = same_direction_backtracks
        out.loc[ordered, COLS.direction_family_count] = family_counts
        out.loc[ordered, COLS.change_count] = changes
        out.loc[ordered, COLS.stable_tail_count] = stable_counts

    return out.drop(
        columns=[
            "_v92_v50_tail_lane_for_history",
            "_v92_v50_tail_bucket_for_history",
            "_v92_v50_tail_family_for_history",
        ]
    )


def apply_clean_v50c_tail_history_policy(
    frame: pd.DataFrame,
    *,
    base_current_col: str = "pred_v47_stable_lane_backtrack_weaving",
    out_v49_current_col: str | None = None,
    out_current_col: str | None = None,
) -> pd.DataFrame:
    """Apply the accepted v49 and v50c tail-history policies.

    Values use SchemaC:
    0 = risk0+1, 1 = risk2, 2 = risk3.
    """

    if base_current_col not in frame.columns:
        raise KeyError(f"base current column not found: {base_current_col}")

    out = add_tail_history_features(frame)
    out_v49_current_col = out_v49_current_col or COLS.v49_current
    out_current_col = out_current_col or COLS.current

    base = num(out, base_current_col).astype(int).clip(0, 2)

    v49_promote = (
        base.eq(1)
        & num(out, "redlight_event_score").ge(0.55)
        & num(out, "v48_tail_lane_backtrack_weaving_onset").ge(0.5)
        & num(out, "v47_stable_lane_backtrack_weaving").lt(0.5)
    )
    v49_current = base.copy()
    v49_current.loc[v49_promote] = np.maximum(v49_current.loc[v49_promote], 2)

    out[COLS.v49_promote] = v49_promote.astype(int)
    out[COLS.v49_changed] = v49_current.ne(base).astype(int)
    out[COLS.v49_reason] = np.where(v49_promote, "v49_redlight_tail_backtrack_promote_risk3", "")
    out[out_v49_current_col] = v49_current.astype(int)

    v28_smoothing_missed_lane_change = num(out, "v28_lane_change_count_6w").le(0.0)
    v50c_promote = (
        v49_current.eq(1)
        & num(out, "redlight_event_score").ge(0.55)
        & num(out, COLS.same_direction_backtrack).ge(0.5)
        & num(out, COLS.stable_tail_count).ge(3)
        & num(out, COLS.change_count).ge(2)
        & num(out, "v48_tail_lane_backtrack_weaving_onset").lt(0.5)
        & v28_smoothing_missed_lane_change
    )
    current = v49_current.copy()
    current.loc[v50c_promote] = np.maximum(current.loc[v50c_promote], 2)

    out[COLS.v28_smoothing_missed_lane_change] = v28_smoothing_missed_lane_change.astype(int)
    out[COLS.promote] = v50c_promote.astype(int)
    out[COLS.changed] = current.ne(v49_current).astype(int)
    out[COLS.reason] = np.where(v50c_promote, "v50c_redlight_tail_history_backtrack_promote_risk3", "")
    out[out_current_col] = current.astype(int)
    return out
