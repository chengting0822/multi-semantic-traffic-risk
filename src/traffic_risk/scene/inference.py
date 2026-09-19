from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from traffic_risk.paths import MODEL_DIR

from .model import SceneTokenPoolConfig, SceneTokenPoolV1


class SceneRiskModel:
    """SceneTokenPool inference using the frozen paper checkpoint."""

    def __init__(self, checkpoint: Path = MODEL_DIR / "scene_risk.pt", device: str = "auto") -> None:
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = SceneTokenPoolV1(SceneTokenPoolConfig(**payload["model_config"]))
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device).eval()
        self.norm = {key: np.asarray(value, dtype=np.float32) for key, value in payload["normalization"].items()}
        checkpoint_config = payload.get("config", {})
        self.binary_threshold = float(checkpoint_config.get("training", {}).get("binary_threshold", 0.5))
        postprocess = checkpoint_config.get("postprocess", {})
        self.support_gate_class1 = bool(postprocess.get("support_gate_class1_no_track_support", True))
        self.binary_negative_downgrade = bool(postprocess.get("binary_negative_downgrade_class1", True))

    @torch.inference_mode()
    def predict_npz(self, dataset: Path, batch_size: int = 256, labeled_only: bool = True) -> pd.DataFrame:
        data = np.load(dataset, allow_pickle=True)
        split = data["split"].astype(str)
        if labeled_only:
            label_available = np.asarray(data["label_available"], dtype=bool)
            selected = np.concatenate([
                np.where((split == name) & label_available)[0]
                for name in ("train", "val", "test")
            ])
        else:
            selected = np.arange(len(split), dtype=np.int64)
        tokens = np.asarray(data["token_features"][selected], dtype=np.float32)
        token_mask = np.asarray(data["token_mask"][selected], dtype=np.uint8)
        global_raw = np.asarray(data["global_features"][selected], dtype=np.float32)
        global_features = global_raw.copy()
        tokens = (tokens - self.norm["token_mean"]) / np.where(self.norm["token_std"] > 1e-8, self.norm["token_std"], 1.0)
        tokens[token_mask == 0] = 0.0
        global_features = (global_features - self.norm["global_mean"]) / np.where(self.norm["global_std"] > 1e-8, self.norm["global_std"], 1.0)
        global_columns = [str(value) for value in data["global_feature_columns"].tolist()]
        support_index = global_columns.index("track_support_count") if "track_support_count" in global_columns else None
        rows: list[dict[str, float | int | str]] = []
        for start in range(0, len(tokens), batch_size):
            end = min(start + batch_size, len(tokens))
            outputs = self.model(
                torch.from_numpy(tokens[start:end]).to(self.device),
                torch.from_numpy(token_mask[start:end]).to(self.device),
                torch.from_numpy(global_features[start:end]).to(self.device),
            )
            four = torch.softmax(outputs["logits_4cls"], dim=-1).cpu().numpy()
            binary = torch.softmax(outputs["logits_binary"], dim=-1).cpu().numpy()
            severity = torch.softmax(outputs["logits_severity"], dim=-1).cpu().numpy()
            for offset in range(end - start):
                local_index = start + offset
                row_index = int(selected[local_index])
                pred_raw = int(np.argmax(four[offset]))
                pred_binary = int(binary[offset, 1] >= self.binary_threshold)
                pred_four = pred_raw
                support_count = float(global_raw[local_index, support_index]) if support_index is not None else 0.0
                if self.support_gate_class1 and pred_four == 1 and support_count <= 0:
                    pred_four = 0
                if self.binary_negative_downgrade and pred_four == 1 and pred_binary == 0:
                    pred_four = 0
                rows.append({
                    "row_idx": row_index,
                    "split": str(data["split"][row_index]),
                    "video_id": str(data["video_id"][row_index]),
                    "scene_window_idx": int(data["scene_window_idx"][row_index]),
                    "start_sec": float(data["start_sec"][row_index]),
                    "end_sec": float(data["end_sec"][row_index]),
                    "y_4cls": int(data["y_4cls"][row_index]),
                    "y_binary": int(data["y_binary"][row_index]),
                    "y_severity": int(data["y_severity"][row_index]),
                    "pred_4cls_raw": pred_raw,
                    "pred_4cls": pred_four,
                    "pred_binary": pred_binary,
                    "pred_severity": int(np.argmax(severity[offset])),
                    **{f"prob4_{i}": float(four[offset, i]) for i in range(four.shape[1])},
                    "prob_binary_positive": float(binary[offset, 1]),
                    **{f"prob_severity_{i}": float(severity[offset, i]) for i in range(severity.shape[1])},
                    "track_support_count": support_count,
                    "token_count": int(token_mask[local_index].sum()),
                })
        return pd.DataFrame(rows)
