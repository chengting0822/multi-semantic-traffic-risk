from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SceneTokenPoolConfig:
    token_dim: int
    global_dim: int
    token_hidden_dim: int = 64
    global_hidden_dim: int = 64
    fused_hidden_dim: int = 96
    dropout: float = 0.15
    pooling: str = "max_mean"


@dataclass(frozen=True)
class SceneGRUConfig:
    token_dim: int
    global_dim: int
    token_hidden_dim: int = 64
    global_hidden_dim: int = 64
    fused_hidden_dim: int = 96
    gru_hidden_dim: int = 96
    gru_layers: int = 1
    dropout: float = 0.15
    pooling: str = "max_mean"


class SceneTokenPoolV1(nn.Module):
    def __init__(self, config: SceneTokenPoolConfig) -> None:
        super().__init__()
        self.config = config
        self.token_encoder = nn.Sequential(
            nn.LayerNorm(config.token_dim),
            nn.Linear(config.token_dim, config.token_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.token_hidden_dim, config.token_hidden_dim),
            nn.GELU(),
        )
        self.global_encoder = nn.Sequential(
            nn.LayerNorm(config.global_dim),
            nn.Linear(config.global_dim, config.global_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.global_hidden_dim, config.global_hidden_dim),
            nn.GELU(),
        )
        token_pool_dim = config.token_hidden_dim * 2 if config.pooling == "max_mean" else config.token_hidden_dim
        fused_dim = token_pool_dim + config.global_hidden_dim
        self.fused = nn.Sequential(
            nn.Linear(fused_dim, config.fused_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fused_hidden_dim, config.fused_hidden_dim),
            nn.GELU(),
        )
        self.head_4cls = nn.Linear(config.fused_hidden_dim, 4)
        self.head_binary = nn.Linear(config.fused_hidden_dim, 2)
        self.head_severity = nn.Linear(config.fused_hidden_dim, 3)

    def forward(self, token_features: torch.Tensor, token_mask: torch.Tensor, global_features: torch.Tensor) -> dict[str, torch.Tensor]:
        token_embed = self.token_encoder(token_features)
        token_pooled = _pool_tokens(token_embed, token_mask, self.config.pooling)
        global_embed = self.global_encoder(global_features)
        fused = self.fused(torch.cat([token_pooled, global_embed], dim=-1))
        return {
            "logits_4cls": self.head_4cls(fused),
            "logits_binary": self.head_binary(fused),
            "logits_severity": self.head_severity(fused),
        }


class SceneGRUV1(nn.Module):
    def __init__(self, config: SceneGRUConfig) -> None:
        super().__init__()
        self.config = config
        self.token_encoder = nn.Sequential(
            nn.LayerNorm(config.token_dim),
            nn.Linear(config.token_dim, config.token_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.token_hidden_dim, config.token_hidden_dim),
            nn.GELU(),
        )
        self.global_encoder = nn.Sequential(
            nn.LayerNorm(config.global_dim),
            nn.Linear(config.global_dim, config.global_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.global_hidden_dim, config.global_hidden_dim),
            nn.GELU(),
        )
        token_pool_dim = config.token_hidden_dim * 2 if config.pooling == "max_mean" else config.token_hidden_dim
        fused_dim = token_pool_dim + config.global_hidden_dim
        self.frame_encoder = nn.Sequential(
            nn.Linear(fused_dim, config.fused_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fused_hidden_dim, config.fused_hidden_dim),
            nn.GELU(),
        )
        gru_dropout = config.dropout if config.gru_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=config.fused_hidden_dim,
            hidden_size=config.gru_hidden_dim,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=False,
        )
        self.post_gru = nn.Sequential(
            nn.LayerNorm(config.gru_hidden_dim),
            nn.Dropout(config.dropout),
        )
        self.head_4cls = nn.Linear(config.gru_hidden_dim, 4)
        self.head_binary = nn.Linear(config.gru_hidden_dim, 2)
        self.head_severity = nn.Linear(config.gru_hidden_dim, 3)

    def forward(
        self,
        token_features: torch.Tensor,
        token_mask: torch.Tensor,
        global_features: torch.Tensor,
        sequence_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len, top_k, token_dim = token_features.shape
        token_flat = token_features.reshape(batch_size * seq_len, top_k, token_dim)
        mask_flat = token_mask.reshape(batch_size * seq_len, top_k)
        global_flat = global_features.reshape(batch_size * seq_len, global_features.shape[-1])

        token_embed = self.token_encoder(token_flat)
        token_pooled = _pool_tokens(token_embed, mask_flat, self.config.pooling)
        global_embed = self.global_encoder(global_flat)
        frame_embed = self.frame_encoder(torch.cat([token_pooled, global_embed], dim=-1))
        frame_embed = frame_embed.reshape(batch_size, seq_len, -1)

        if sequence_mask is None:
            gru_out, _hidden = self.gru(frame_embed)
        else:
            lengths = sequence_mask.long().sum(dim=1).clamp_min(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                frame_embed,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _hidden = self.gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=seq_len,
            )
        encoded = self.post_gru(gru_out)
        return {
            "logits_4cls": self.head_4cls(encoded),
            "logits_binary": self.head_binary(encoded),
            "logits_severity": self.head_severity(encoded),
        }


def _pool_tokens(token_embed: torch.Tensor, token_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    mask = token_mask.bool()
    masked_embed = token_embed.masked_fill(~mask.unsqueeze(-1), -1e9)
    max_pool = masked_embed.max(dim=1).values
    has_token = mask.any(dim=1, keepdim=True)
    max_pool = torch.where(has_token, max_pool, torch.zeros_like(max_pool))
    if pooling == "max_mean":
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(token_embed.dtype)
        mean_pool = (token_embed * mask.unsqueeze(-1).to(token_embed.dtype)).sum(dim=1) / denom
        return torch.cat([max_pool, mean_pool], dim=-1)
    if pooling == "max":
        return max_pool
    raise ValueError(f"unsupported pooling: {pooling}")
