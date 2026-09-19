from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from traffic_risk.paths import MODEL_DIR

from .model import ModelConfigV2, build_model


TRACK_KEYS = ["case_key", "video_id", "track_id"]


def _formal_track_postprocess(frame: pd.DataFrame, raw: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Frozen postprocess used by the selected GRU checkpoint."""

    high_probability = probabilities[:, 1] + probabilities[:, 2]
    critical_probability = probabilities[:, 2]

    # Suppress an isolated positive prediction at the start of a sufficiently
    # long track when all later windows remain confidently normal.
    head_count = min(2, len(raw))
    positive = np.flatnonzero(raw > 0)
    peak = float(high_probability[:head_count].max()) if head_count else 0.0
    tail_high = float(high_probability[head_count:].max()) if head_count < len(raw) else 0.0
    tail_critical = float(critical_probability[head_count:].max()) if head_count < len(raw) else 0.0
    front_spike = bool(
        len(raw) >= 8
        and positive.size > 0
        and int(positive.max()) < 2
        and int(high_probability.argmax()) < 2
        and peak >= 0.55
        and tail_high <= 0.35
        and tail_critical <= 0.12
        and tail_high <= peak * 0.65
    )
    if front_spike:
        return np.zeros_like(raw)

    def values(column: str, default: float = 0.0) -> np.ndarray:
        if column not in frame:
            return np.full(len(frame), default, dtype=np.float64)
        return pd.to_numeric(frame[column], errors="coerce").fillna(default).to_numpy(np.float64)

    s33_prefix = values("s33_prefix_state_confirmed")
    overspeed = values("overspeed_event_score")
    redlight = values("redlight_event_score")
    spike_mask = overspeed > 0.30
    spike_count = int(spike_mask.sum())
    last_spike = int(np.flatnonzero(spike_mask).max()) if spike_count else -1
    overspeed_spike = bool(
        np.all(s33_prefix < 1e-4)
        and np.all(redlight < 1e-4)
        and float(overspeed.max(initial=0.0)) <= 0.70
        and 1 <= spike_count <= 4
        and (last_spike + 1 >= len(overspeed) or np.all(overspeed[last_spike + 1 :] <= 0.45))
    )
    # The formal evaluator only applies this suppressor to a track whose raw
    # GRU prediction is already positive. A fully normal raw track may still
    # be promoted later by the conservative semantic rescue.
    if (raw > 0).any() and overspeed_spike:
        return np.zeros_like(raw)

    pred = raw.copy()
    pred[(pred >= 2) & (critical_probability < 0.50)] = 1

    midband = values("r32_overspeed_midband_strength")
    duration = values("r32_overspeed_ge2_duration_strength")
    reliable = values("r32_overspeed_reliable_signal")
    rescue_signal = values("r32_overspeed_rescue_signal")
    active_family = values("active_family_count")
    risk3_policy = values("r46_risk3_policy")
    risk2_policy = values("r46_risk2_policy")
    computed_speed = values("computed_speed_kmh_p95")
    smoothed_speed = values("speed_smoothed_kmh_p95")

    rescue = (
        (midband >= 0.55)
        & (duration >= 0.80)
        & (reliable >= 0.85)
        & (rescue_signal >= 0.40)
        & (high_probability >= 0.50)
    )
    pred[rescue] = np.maximum(pred[rescue], 1)

    conservative_rescue = (
        (pred < 1)
        & (risk2_policy >= 0.5)
        & (risk3_policy < 0.5)
        & (overspeed >= 0.50)
        & (midband >= 0.50)
        & (duration >= 0.80)
        & (reliable >= 0.95)
        & (rescue_signal >= 0.40)
        & (high_probability >= 0.15)
        & (active_family + 1e-6 >= 2.0)
    )
    pred[conservative_rescue] = 1
    pred[(pred >= 2) & (risk3_policy < 0.5) & (risk2_policy >= 0.5) & (critical_probability < 0.5)] = 1

    speed_ratio = np.divide(
        computed_speed,
        smoothed_speed,
        out=np.full_like(computed_speed, np.inf),
        where=smoothed_speed != 0.0,
    )
    inconsistent_speed = (
        (smoothed_speed < 90.0)
        & ((computed_speed - smoothed_speed) >= 25.0)
        & (speed_ratio >= 1.30)
    )
    direct_bad_rescue = (raw < 1) & rescue & inconsistent_speed
    allowed_rescue = rescue & ~inconsistent_speed
    if direct_bad_rescue.any() and not (raw > 0).any() and not allowed_rescue.any():
        return np.zeros_like(raw)
    return np.maximum.accumulate(pred)


class SingleVehicleRiskModel:
    """Causal-GRU inference using the frozen paper checkpoint.

    ``pred_gru_raw`` is the neural-network argmax and ``pred_v37e`` includes
    the checkpoint's frozen track-level postprocess. The complete paper result
    additionally uses the semantic policy chain in ``pipeline``.
    """

    def __init__(
        self,
        checkpoint: Path = MODEL_DIR / "single_vehicle_gru.pt",
        normalization: Path = MODEL_DIR / "single_vehicle_normalization.json",
        device: str = "auto",
    ) -> None:
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model_cfg = ModelConfigV2(**payload["candidate"]["model_config"])
        self.model = build_model(model_cfg)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device).eval()
        norm = json.loads(normalization.read_text(encoding="utf-8"))
        self.feature_columns = list(norm["feature_columns"])
        self.mean = np.asarray([norm["mean"][name] for name in self.feature_columns], dtype=np.float32)
        self.std = np.asarray([norm["std"][name] for name in self.feature_columns], dtype=np.float32)
        self.std = np.where(self.std > 1e-8, self.std, 1.0)

    @torch.inference_mode()
    def predict(self, windows: pd.DataFrame) -> pd.DataFrame:
        missing = [name for name in [*TRACK_KEYS, "ts_window_idx", *self.feature_columns] if name not in windows.columns]
        if missing:
            raise ValueError(f"single-vehicle feature table is missing columns: {missing}")
        out = windows.copy()
        out["_original_order"] = np.arange(len(out), dtype=np.int64)
        out = out.sort_values([*TRACK_KEYS, "ts_window_idx"], kind="mergesort").reset_index(drop=True)
        prob = np.zeros((len(out), 3), dtype=np.float32)
        raw_prediction = np.zeros(len(out), dtype=np.int64)
        formal_prediction = np.zeros(len(out), dtype=np.int64)
        for _, index in out.groupby(TRACK_KEYS, sort=False).groups.items():
            rows = list(index)
            values = out.loc[rows, self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
            values = (values - self.mean) / self.std
            logits = self.model(torch.from_numpy(values).unsqueeze(0).to(self.device))[0]
            track_probability = torch.softmax(logits, dim=-1).cpu().numpy()
            track_raw = track_probability.argmax(axis=1).astype(np.int64)
            prob[rows] = track_probability
            raw_prediction[rows] = track_raw
            formal_prediction[rows] = _formal_track_postprocess(out.loc[rows], track_raw, track_probability)
        out["prob_class0"] = prob[:, 0]
        out["prob_class1"] = prob[:, 1]
        out["prob_class2"] = prob[:, 2]
        out["high_risk_prob"] = prob[:, 1] + prob[:, 2]
        out["critical_prob"] = prob[:, 2]
        out["pred_gru_raw"] = raw_prediction
        out["pred_v37e"] = formal_prediction
        return out.sort_values("_original_order").drop(columns="_original_order").reset_index(drop=True)
