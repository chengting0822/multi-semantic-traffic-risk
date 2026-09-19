#!/usr/bin/env python3
"""Clean v51 early strong-reverse onset policy.

This module distills the accepted v51 rule:

If the current prediction is normal, v41 already reports strict wrongway
pre-confirmation, and the tail motion is strongly opposite to lane direction
with enough samples/motion, promote only to risk2.

It intentionally does not create risk3 support.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleanV51EarlyReverseColumns:
    current: str = "v92_v51_early_strong_reverse_onset_current"
    reason: str = "v92_v51_early_reverse_reason"
    strict_wrongway: str = "v92_v51_strict_wrongway"
    strong_reverse_motion: str = "v92_v51_strong_reverse_motion"
    enough_motion: str = "v92_v51_enough_motion"
    enough_samples: str = "v92_v51_enough_samples"
    no_existing_traj_support: str = "v92_v51_no_existing_traj_support"
    promote_risk2: str = "v92_v51_early_strong_reverse_promote_risk2"


COLS = CleanV51EarlyReverseColumns()


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default).astype(float)


def apply_clean_v51_early_reverse_policy(
    frame: pd.DataFrame,
    *,
    base_current_col: str = "pred_v50c_tail_history_missing_smoothing",
    out_current_col: str | None = None,
    out_reason_col: str | None = None,
) -> pd.DataFrame:
    """Apply v51 early strong-reverse risk2 onset.

    Values use SchemaC:
    0 = risk0+1, 1 = risk2, 2 = risk3.
    """

    if base_current_col not in frame.columns:
        raise KeyError(f"base current column not found: {base_current_col}")

    out = frame.copy()
    out_current_col = out_current_col or COLS.current
    out_reason_col = out_reason_col or COLS.reason

    base = num(out, base_current_col).astype(int).clip(0, 2)
    strict_wrongway = num(out, "v41_family_wrongway_preconfirm_strict").ge(1.0)
    strong_reverse_motion = num(out, "v41_tail_family_motion_cos_top").le(-0.90)
    enough_motion = num(out, "v41_tail_motion_mag_px").ge(50.0)
    enough_samples = num(out, "v41_tail_raw_count").ge(15.0)
    no_existing_traj_support = num(out, "v38_trajectory_risk2_support").lt(0.5)

    promote = (
        base.eq(0)
        & strict_wrongway
        & strong_reverse_motion
        & enough_motion
        & enough_samples
        & no_existing_traj_support
    )

    updated = base.copy()
    updated.loc[promote] = np.maximum(updated.loc[promote], 1)

    out[COLS.strict_wrongway] = strict_wrongway.astype(int)
    out[COLS.strong_reverse_motion] = strong_reverse_motion.astype(int)
    out[COLS.enough_motion] = enough_motion.astype(int)
    out[COLS.enough_samples] = enough_samples.astype(int)
    out[COLS.no_existing_traj_support] = no_existing_traj_support.astype(int)
    out[COLS.promote_risk2] = promote.astype(int)
    out[out_reason_col] = np.where(promote, "v51_early_strong_reverse_promote_risk2", "")
    out[out_current_col] = updated.astype(int)
    return out
