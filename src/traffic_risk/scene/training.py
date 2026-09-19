#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from .model import SceneTokenPoolConfig, SceneTokenPoolV1


class SceneDataset(Dataset):
    def __init__(self, payload: dict[str, Any], indices: np.ndarray, norm: dict[str, np.ndarray]) -> None:
        self.indices = indices.astype(np.int64)
        token = payload["token_features"][self.indices].astype(np.float32, copy=True)
        mask = payload["token_mask"][self.indices].astype(np.uint8, copy=False)
        glob = payload["global_features"][self.indices].astype(np.float32, copy=True)
        token = (token - norm["token_mean"]) / norm["token_std"]
        token[mask == 0] = 0.0
        glob = (glob - norm["global_mean"]) / norm["global_std"]
        self.token = torch.from_numpy(token)
        self.mask = torch.from_numpy(mask)
        self.glob = torch.from_numpy(glob)
        self.y4 = torch.from_numpy(payload["y_4cls"][self.indices].astype(np.int64, copy=False))
        self.ybin = torch.from_numpy(payload["y_binary"][self.indices].astype(np.int64, copy=False))
        self.ysev = torch.from_numpy(payload["y_severity"][self.indices].astype(np.int64, copy=False))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        return self.token[item], self.mask[item], self.glob[item], self.y4[item], self.ybin[item], self.ysev[item], torch.tensor(self.indices[item], dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SceneTokenPoolV1 on scene risk tensor datasets.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "scene_token_pool_v1.json"))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    train_cfg = config["training"]
    set_seed(int(train_cfg["seed"]))
    device = resolve_device(str(train_cfg["device"]))

    payload = load_npz(config["dataset_npz"])
    split = payload["split"].astype(str)
    label_available = payload["label_available"].astype(bool)
    indices_by_split = {
        name: np.where((split == name) & label_available)[0]
        for name in ["train", "val", "test"]
    }
    norm = compute_normalization(payload, indices_by_split["train"])

    loaders = {
        name: DataLoader(
            SceneDataset(payload, indices, norm),
            batch_size=int(train_cfg["batch_size"]),
            shuffle=(name == "train"),
            num_workers=0,
        )
        for name, indices in indices_by_split.items()
    }

    token_dim = int(payload["token_features"].shape[-1])
    global_dim = int(payload["global_features"].shape[-1])
    model_cfg = SceneTokenPoolConfig(
        token_dim=token_dim,
        global_dim=global_dim,
        token_hidden_dim=int(config["model"]["token_hidden_dim"]),
        global_hidden_dim=int(config["model"]["global_hidden_dim"]),
        fused_hidden_dim=int(config["model"]["fused_hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
        pooling=str(config["model"]["pooling"]),
    )
    model = SceneTokenPoolV1(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    class_weights = build_class_weights(payload, indices_by_split["train"], str(train_cfg["class_weight_mode"]), device)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_score = -math.inf
    patience_left = int(train_cfg["patience"])
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(train_cfg["max_epochs"]) + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, class_weights, config, device)
        val_predictions = predict(model, loaders["val"], payload, config, device)
        val_metrics = evaluate_dataframe(val_predictions)
        val_score = selection_score(val_metrics)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_score": val_score,
                "val_binary_f1": val_metrics["binary"]["f1"],
                "val_4cls_macro_f1": val_metrics["four_class"]["macro_f1"],
                "val_severity_macro_f1": val_metrics["severity"]["macro_f1"],
            }
        )
        if val_score > best_score + 1e-6:
            best_score = val_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(train_cfg["patience"])
        else:
            patience_left -= 1
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} val_score={val_score:.4f} "
            f"bin_f1={val_metrics['binary']['f1']:.4f} 4cls_macro={val_metrics['four_class']['macro_f1']:.4f} "
            f"sev_macro={val_metrics['severity']['macro_f1']:.4f}"
        )
        if patience_left <= 0:
            break

    if best_state is None:
        raise RuntimeError("training produced no best state")
    model.load_state_dict(best_state)

    prediction_frames = [predict(model, loaders[name], payload, config, device) for name in ["train", "val", "test"]]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = {name: evaluate_dataframe(predictions[predictions["split"] == name]) for name in ["train", "val", "test"]}
    metrics["all"] = evaluate_dataframe(predictions)
    metrics["history"] = history
    metrics["best_epoch"] = best_epoch
    metrics["best_val_score"] = best_score
    metrics["selection_score_formula"] = "0.40*binary_f1 + 0.30*four_class_macro_f1 + 0.30*severity_macro_f1"

    outputs = config["outputs"]
    model_dir = Path(outputs["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_path = Path(outputs["prediction_csv"])
    metrics_json = Path(outputs["metrics_json"])
    metrics_md = Path(outputs["metrics_md"])
    manifest_path = Path(outputs["manifest_json"])
    for path in [pred_path, metrics_json, metrics_md, manifest_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "schema_version": "scene_token_pool_v1.checkpoint/v1",
            "model_config": asdict(model_cfg),
            "model_state_dict": model.state_dict(),
            "normalization": {key: value.tolist() for key, value in norm.items()},
            "config": config,
            "best_epoch": best_epoch,
            "best_val_score": best_score,
            "metrics": metrics,
        },
        model_dir / "scene_token_pool_v1_best.pt",
    )
    predictions.to_csv(pred_path, index=False)
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_md.write_text(render_metrics_md(metrics), encoding="utf-8")
    manifest = {
        "run_id": config["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "dataset_npz": config["dataset_npz"],
        "model_path": str(model_dir / "scene_token_pool_v1_best.pt"),
        "prediction_csv": str(pred_path),
        "metrics_json": str(metrics_json),
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "indices_by_split": {name: int(len(indices)) for name, indices in indices_by_split.items()},
        "warning": config.get("warning", "Use val/test for model judgment."),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"best_epoch={best_epoch} best_val_score={best_score:.4f}")
    for name in ["val", "test"]:
        item = metrics[name]
        print(
            f"{name}: binary_f1={item['binary']['f1']:.4f} "
            f"4cls_macro={item['four_class']['macro_f1']:.4f} severity_macro={item['severity']['macro_f1']:.4f}"
        )


def load_npz(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def compute_normalization(payload: dict[str, Any], train_indices: np.ndarray) -> dict[str, np.ndarray]:
    token = payload["token_features"][train_indices].astype(np.float32)
    mask = payload["token_mask"][train_indices].astype(bool)
    valid_tokens = token[mask]
    token_mean = valid_tokens.mean(axis=0)
    token_std = valid_tokens.std(axis=0)
    token_std[token_std < 1e-6] = 1.0
    glob = payload["global_features"][train_indices].astype(np.float32)
    global_mean = glob.mean(axis=0)
    global_std = glob.std(axis=0)
    global_std[global_std < 1e-6] = 1.0
    return {
        "token_mean": token_mean.astype(np.float32),
        "token_std": token_std.astype(np.float32),
        "global_mean": global_mean.astype(np.float32),
        "global_std": global_std.astype(np.float32),
    }


def build_class_weights(payload: dict[str, Any], train_indices: np.ndarray, mode: str, device: torch.device) -> dict[str, torch.Tensor]:
    if mode != "sqrt_balanced":
        raise ValueError(f"unsupported class_weight_mode: {mode}")
    return {
        "four_class": sqrt_balanced_weights(payload["y_4cls"][train_indices], 4, device),
        "binary": sqrt_balanced_weights(payload["y_binary"][train_indices], 2, device),
        "severity": sqrt_balanced_weights(payload["y_severity"][train_indices], 3, device),
    }


def sqrt_balanced_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = labels.astype(np.int64)
    total = float(len(labels))
    weights = []
    for cls in range(num_classes):
        count = max(int((labels == cls).sum()), 1)
        weights.append(math.sqrt(total / (num_classes * count)))
    values = np.asarray(weights, dtype=np.float32)
    values = values * (num_classes / values.sum())
    return torch.tensor(values, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_weights: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    loss_weights = config["training"]["loss_weights"]
    for token, mask, glob, y4, ybin, ysev, _idx in loader:
        token = token.to(device)
        mask = mask.to(device)
        glob = glob.to(device)
        y4 = y4.to(device)
        ybin = ybin.to(device)
        ysev = ysev.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(token, mask, glob)
        loss4 = F.cross_entropy(out["logits_4cls"], y4, weight=class_weights["four_class"])
        lossb = F.cross_entropy(out["logits_binary"], ybin, weight=class_weights["binary"])
        losss = F.cross_entropy(out["logits_severity"], ysev, weight=class_weights["severity"])
        loss = (
            float(loss_weights["four_class"]) * loss4
            + float(loss_weights["binary"]) * lossb
            + float(loss_weights["severity"]) * losss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
        optimizer.step()
        total_loss += float(loss.item()) * int(y4.numel())
        total_count += int(y4.numel())
    return total_loss / max(total_count, 1)


def predict(model: nn.Module, loader: DataLoader, payload: dict[str, Any], config: dict[str, Any], device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    token_feature_columns = [str(x) for x in payload["token_feature_columns"].tolist()]
    global_feature_columns = [str(x) for x in payload["global_feature_columns"].tolist()]
    track_support_idx = global_feature_columns.index("track_support_count") if "track_support_count" in global_feature_columns else None
    with torch.no_grad():
        for token, mask, glob, y4, ybin, ysev, idx in loader:
            token = token.to(device)
            mask = mask.to(device)
            glob = glob.to(device)
            out = model(token, mask, glob)
            prob4 = torch.softmax(out["logits_4cls"], dim=-1).cpu().numpy()
            probb = torch.softmax(out["logits_binary"], dim=-1).cpu().numpy()
            probs = torch.softmax(out["logits_severity"], dim=-1).cpu().numpy()
            pred4_raw = prob4.argmax(axis=1)
            pred_bin = (probb[:, 1] >= float(config["training"]["binary_threshold"])).astype(int)
            pred_sev = probs.argmax(axis=1)
            idx_np = idx.numpy().astype(int)
            global_raw = payload["global_features"][idx_np]
            if track_support_idx is None:
                track_support_count = np.zeros(len(idx_np), dtype=np.float32)
            else:
                track_support_count = global_raw[:, track_support_idx]
            pred4 = pred4_raw.copy()
            if config["postprocess"].get("support_gate_class1_no_track_support", True):
                pred4[(pred4 == 1) & (track_support_count <= 0)] = 0
            if config["postprocess"].get("binary_negative_downgrade_class1", True):
                pred4[(pred4 == 1) & (pred_bin == 0)] = 0
            for local, original_idx in enumerate(idx_np):
                rows.append(
                    {
                        "row_idx": int(original_idx),
                        "split": str(payload["split"][original_idx]),
                        "video_id": str(payload["video_id"][original_idx]),
                        "scene_window_idx": int(payload["scene_window_idx"][original_idx]),
                        "start_sec": float(payload["start_sec"][original_idx]),
                        "end_sec": float(payload["end_sec"][original_idx]),
                        "y_4cls": int(payload["y_4cls"][original_idx]),
                        "y_binary": int(payload["y_binary"][original_idx]),
                        "y_severity": int(payload["y_severity"][original_idx]),
                        "pred_4cls_raw": int(pred4_raw[local]),
                        "pred_4cls": int(pred4[local]),
                        "pred_binary": int(pred_bin[local]),
                        "pred_severity": int(pred_sev[local]),
                        "prob4_0": float(prob4[local, 0]),
                        "prob4_1": float(prob4[local, 1]),
                        "prob4_2": float(prob4[local, 2]),
                        "prob4_3": float(prob4[local, 3]),
                        "prob_binary_positive": float(probb[local, 1]),
                        "prob_severity_0": float(probs[local, 0]),
                        "prob_severity_1": float(probs[local, 1]),
                        "prob_severity_2": float(probs[local, 2]),
                        "track_support_count": float(track_support_count[local]),
                        "token_count": int(payload["token_mask"][original_idx].sum()),
                    }
                )
    _ = token_feature_columns
    return pd.DataFrame(rows)


def evaluate_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    return {
        "binary": binary_metrics(df["y_binary"].to_numpy(), df["pred_binary"].to_numpy()),
        "four_class": multiclass_metrics(df["y_4cls"].to_numpy(), df["pred_4cls"].to_numpy(), 4),
        "four_class_raw": multiclass_metrics(df["y_4cls"].to_numpy(), df["pred_4cls_raw"].to_numpy(), 4),
        "severity": multiclass_metrics(df["y_severity"].to_numpy(), df["pred_severity"].to_numpy(), 3),
        "error_counts": error_counts(df),
        "row_count": int(len(df)),
    }


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, Any]:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= true < num_classes and 0 <= pred < num_classes:
            cm[true, pred] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    for cls in range(num_classes):
        tp = int(cm[cls, cls])
        fp = int(cm[:, cls].sum() - tp)
        fn = int(cm[cls, :].sum() - tp)
        support = int(cm[cls, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[str(cls)] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {
        "macro_f1": float(np.mean([per_class[str(cls)]["f1"] for cls in range(num_classes)])),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def error_counts(df: pd.DataFrame) -> dict[str, int]:
    y = df["y_4cls"].astype(int)
    p = df["pred_4cls"].astype(int)
    return {
        "false_positive_binary": int(((y == 0) & (p > 0)).sum()),
        "false_negative_binary": int(((y > 0) & (p == 0)).sum()),
        "risk1_to_0": int(((y == 1) & (p == 0)).sum()),
        "risk2_to_3": int(((y == 2) & (p == 3)).sum()),
        "risk3_to_2": int(((y == 3) & (p == 2)).sum()),
    }


def selection_score(metrics: dict[str, Any]) -> float:
    return float(
        0.40 * metrics["binary"]["f1"]
        + 0.30 * metrics["four_class"]["macro_f1"]
        + 0.30 * metrics["severity"]["macro_f1"]
    )


def render_metrics_md(metrics: dict[str, Any]) -> str:
    lines = [
        "# SceneTokenPoolV1 Metrics",
        "",
        f"- best_epoch: {metrics['best_epoch']}",
        f"- best_val_score: {metrics['best_val_score']:.4f}",
        "",
        "| split | rows | binary F1 | 4cls macro F1 | severity macro F1 | FP | FN | risk1->0 | risk2->3 | risk3->2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ["train", "val", "test", "all"]:
        item = metrics[split]
        err = item["error_counts"]
        lines.append(
            f"| {split} | {item['row_count']} | {item['binary']['f1']:.4f} | {item['four_class']['macro_f1']:.4f} | "
            f"{item['severity']['macro_f1']:.4f} | {err['false_positive_binary']} | {err['false_negative_binary']} | "
            f"{err['risk1_to_0']} | {err['risk2_to_3']} | {err['risk3_to_2']} |"
        )
    lines.extend(["", "## 4-Class Per-Class F1", ""])
    for split in ["val", "test"]:
        lines.append(f"### {split}")
        lines.append("")
        lines.append("| class | precision | recall | F1 | support |")
        lines.append("|---:|---:|---:|---:|---:|")
        for cls, item in metrics[split]["four_class"]["per_class"].items():
            lines.append(f"| {cls} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} | {item['support']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
