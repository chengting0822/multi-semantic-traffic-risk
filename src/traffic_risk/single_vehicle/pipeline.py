"""Production single-vehicle risk policy chain.

The public API hides the experiment-version names used during research. The
frozen policy functions still run in the exact order used for paper evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from traffic_risk.identifiers import normalize_window_keys

from .policies.base_semantic import COLS as BASE_COLS, apply_clean_base_semantic_policy
from .policies.early_reverse import COLS as EARLY_REVERSE_COLS, apply_clean_v51_early_reverse_policy
from .policies.feature_generation import add_clean_v38_feature_generation, overwrite_historical_v38_aliases
from .policies.final_postprocess import COLS as FINAL_COLS, apply_clean_final_postprocess, num
from .policies.high_speed_onset import COLS as HIGH_SPEED_COLS, apply_clean_v61_ge110_onset_policy
from .policies.integrated import COLS as INTEGRATED_COLS, apply_clean_v60_integrated_policy
from .policies.lane_assignment import overwrite_historical_v28_lane_aliases
from .policies.lane_family import COLS as LANE_FAMILY_COLS, apply_clean_v41_lane_family_policy
from .policies.lane_family_features import overwrite_historical_v41_sidecar_aliases
from .policies.onset import COLS as ONSET_COLS, apply_clean_v41f_lite_onset_policy
from .policies.overspeed_redlight import COLS as SPEED_REDLIGHT_COLS, apply_clean_v44_v45_overspeed_redlight_policy
from .policies.redlight_bridge import COLS as REDLIGHT_BRIDGE_COLS, apply_clean_v58c_redlight_bridge_policy
from .policies.redlight_combo import COLS as REDLIGHT_COMBO_COLS, apply_clean_redlight_combo_maturity_policy
from .policies.redlight_context import COLS as REDLIGHT_CONTEXT_COLS, apply_clean_v46_redlight85_context_policy
from .policies.redlight_speed import COLS as REDLIGHT_SPEED_COLS, apply_clean_v52d_redlight_speed_policy
from .policies.redlight_zone_features import overwrite_historical_redlight_zone_movement_aliases
from .policies.redlight_zone_preconfirm import COLS as REDLIGHT_ZONE_COLS, apply_clean_v42_redlight_zone_preconfirm_policy
from .policies.root_cause_repair import COLS as REPAIR_COLS, apply_clean_v62_root_cause_repair_policy
from .policies.severity_cap import COLS as SEVERITY_COLS, apply_clean_v38_policy_cap
from .policies.stable_lane import COLS as STABLE_LANE_COLS, apply_clean_v47_stable_lane_backtrack_policy
from .policies.tail_history import COLS as TAIL_HISTORY_COLS, apply_clean_v50c_tail_history_policy
from .policies.tail_lane_features import overwrite_historical_v48_tail_lane_aliases
from .policies.trajectory_semantics import overwrite_historical_v37_trajectory_aliases
from .policies.wrongway_tail import apply_clean_v87_v88_wrongway_tail_onset_policy


WINDOW_KEYS = ["case_key", "video_id", "track_id", "ts_window_idx"]
TRACK_KEYS = ["split", "case_key", "video_id", "track_id"]
FINAL_CURRENT_COLUMN = FINAL_COLS.current
FINAL_TRACK_COLUMN = "risk_level"


@dataclass(frozen=True)
class PolicyInputs:
    """Prepared semantic evidence used by the frozen policy chain."""

    source_features: pd.DataFrame
    trajectory_features: pd.DataFrame
    lane_features: pd.DataFrame
    lane_family_features: pd.DataFrame
    redlight_zone_features: pd.DataFrame
    tail_lane_features: pd.DataFrame
    wrongway_tail_features: pd.DataFrame
    double_yellow_features: pd.DataFrame

    @classmethod
    def from_csv(
        cls,
        *,
        source_features: Path,
        trajectory_features: Path,
        lane_features: Path,
        lane_family_features: Path,
        redlight_zone_features: Path,
        tail_lane_features: Path,
        wrongway_tail_features: Path,
        double_yellow_features: Path,
    ) -> "PolicyInputs":
        paths = locals().copy()
        paths.pop("cls")
        return cls(**{name: pd.read_csv(path, low_memory=False) for name, path in paths.items()})


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    return normalize_window_keys(frame)


def merge_sidecar(frame: pd.DataFrame, sidecar: pd.DataFrame) -> pd.DataFrame:
    base = normalize_keys(frame)
    extra = normalize_keys(sidecar)
    missing = [column for column in WINDOW_KEYS if column not in extra.columns]
    if missing:
        raise ValueError(f"sidecar is missing join keys: {missing}")
    extra = extra.drop_duplicates(WINDOW_KEYS, keep="last")
    payload_columns = [column for column in extra.columns if column not in WINDOW_KEYS]
    base = base.drop(columns=[column for column in payload_columns if column in base.columns], errors="ignore")
    return base.merge(extra[WINDOW_KEYS + payload_columns], on=WINDOW_KEYS, how="left", validate="many_to_one")


def apply_policy_chain(windows: pd.DataFrame, inputs: PolicyInputs) -> pd.DataFrame:
    """Apply the frozen semantic policy chain and final track cumulative max."""

    if "pred_v37e" not in windows.columns:
        raise ValueError("windows must contain the learned-risk column 'pred_v37e'")

    trajectory = overwrite_historical_v37_trajectory_aliases(normalize_keys(inputs.trajectory_features))
    out = add_clean_v38_feature_generation(
        normalize_keys(windows),
        source_frame=normalize_keys(inputs.source_features),
        v37_sidecar=trajectory,
    )
    out = overwrite_historical_v38_aliases(out)
    out = apply_clean_v38_policy_cap(out, base_current_col="pred_v37e")

    out = merge_sidecar(out, inputs.lane_features)
    out = overwrite_historical_v28_lane_aliases(out)
    out = merge_sidecar(out, inputs.lane_family_features)
    out = overwrite_historical_v41_sidecar_aliases(out)
    out = apply_clean_v41_lane_family_policy(out, base_current_col=SEVERITY_COLS.current)
    out = apply_clean_v41f_lite_onset_policy(out, base_current_col=LANE_FAMILY_COLS.current)

    out = merge_sidecar(out, inputs.redlight_zone_features)
    out = overwrite_historical_redlight_zone_movement_aliases(out)
    out = apply_clean_v42_redlight_zone_preconfirm_policy(out, base_current_col=ONSET_COLS.current)
    out = apply_clean_v44_v45_overspeed_redlight_policy(out, base_current_col=REDLIGHT_ZONE_COLS.current)
    out = apply_clean_v46_redlight85_context_policy(out, base_current_col=SPEED_REDLIGHT_COLS.current)
    out = apply_clean_v47_stable_lane_backtrack_policy(out, base_current_col=REDLIGHT_CONTEXT_COLS.current)

    out = merge_sidecar(out, inputs.tail_lane_features)
    out = overwrite_historical_v48_tail_lane_aliases(out)
    out = apply_clean_v50c_tail_history_policy(out, base_current_col=STABLE_LANE_COLS.current)
    out = apply_clean_v51_early_reverse_policy(out, base_current_col=TAIL_HISTORY_COLS.current)
    out = apply_clean_v52d_redlight_speed_policy(out, base_current_col=EARLY_REVERSE_COLS.current)
    out = apply_clean_v58c_redlight_bridge_policy(out, base_current_col=REDLIGHT_SPEED_COLS.current)
    out = apply_clean_v60_integrated_policy(out, base_current_col=REDLIGHT_BRIDGE_COLS.current)
    out = apply_clean_v61_ge110_onset_policy(out, base_current_col=INTEGRATED_COLS.current)
    out = apply_clean_v62_root_cause_repair_policy(out, base_current_col=HIGH_SPEED_COLS.current)
    out = apply_clean_redlight_combo_maturity_policy(out, base_current_col=REPAIR_COLS.current)

    out = merge_sidecar(out, inputs.wrongway_tail_features)
    out = apply_clean_v87_v88_wrongway_tail_onset_policy(
        out,
        base_current_col=REDLIGHT_COMBO_COLS.current,
        frame_cache_dir=Path("."),
        lane_map=Path("."),
    )
    out = apply_clean_base_semantic_policy(out, base_current_col=REDLIGHT_COMBO_COLS.current)
    out = merge_sidecar(out, inputs.double_yellow_features)
    out = apply_clean_final_postprocess(out, base_current_col=BASE_COLS.current)

    out = normalize_keys(out).sort_values(TRACK_KEYS + ["ts_window_idx"], kind="mergesort")
    out[FINAL_TRACK_COLUMN] = (
        num(out, FINAL_CURRENT_COLUMN)
        .astype(int)
        .groupby([out[column] for column in TRACK_KEYS], sort=False)
        .cummax()
        .astype(int)
    )
    return out


__all__ = ["FINAL_CURRENT_COLUMN", "FINAL_TRACK_COLUMN", "PolicyInputs", "apply_policy_chain"]
