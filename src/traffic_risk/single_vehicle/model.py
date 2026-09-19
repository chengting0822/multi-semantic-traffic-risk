#!/usr/bin/env python3
"""Model registry for Schema C full_v1 no-cls3 GRU improvement search v2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfigV2:
    model_type: str
    input_dim: int = 15
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.1
    output_dim: int = 3
    layer_norm: bool = False
    mlp_head: bool = False
    residual_input: bool = False
    temporal_attention_head: bool = False
    tcn_channels: int = 64
    tcn_kernel_size: int = 3
    tcn_blocks: int = 3
    bidirectional: bool = False

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return {
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "output_dim": self.output_dim,
            "layer_norm": self.layer_norm,
            "mlp_head": self.mlp_head,
            "residual_input": self.residual_input,
            "temporal_attention_head": self.temporal_attention_head,
            "tcn_channels": self.tcn_channels,
            "tcn_kernel_size": self.tcn_kernel_size,
            "tcn_blocks": self.tcn_blocks,
            "bidirectional": self.bidirectional,
        }


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out.transpose(1, 2)).transpose(1, 2)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + residual


class SmallSemanticTCN(nn.Module):
    def __init__(self, config: ModelConfigV2) -> None:
        super().__init__()
        self.config_dict = config.to_dict()
        self.input_projection = nn.Linear(config.input_dim, config.tcn_channels)
        self.blocks = nn.ModuleList(
            [
                ResidualTCNBlock(
                    channels=config.tcn_channels,
                    kernel_size=config.tcn_kernel_size,
                    dilation=2**block_index,
                    dropout=config.dropout,
                )
                for block_index in range(config.tcn_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(config.tcn_channels)
        self.output_dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.tcn_channels, config.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_projection(x)
        for block in self.blocks:
            out = block(out)
        out = self.output_norm(out)
        out = self.output_dropout(out)
        return self.classifier(out)

    def config(self) -> dict[str, int | float | bool | str]:
        return dict(self.config_dict)


class FlexibleSchemaCGRU(nn.Module):
    def __init__(self, config: ModelConfigV2) -> None:
        super().__init__()
        self.config_dict = config.to_dict()
        if config.bidirectional:
            raise ValueError("bidirectional models are not allowed in the causal v2 mainline")

        self.input_norm = nn.LayerNorm(config.input_dim) if config.layer_norm else nn.Identity()
        rnn_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=False,
        )
        self.residual_projection = (
            nn.Linear(config.input_dim, config.hidden_size) if config.residual_input else None
        )
        self.attention = (
            nn.MultiheadAttention(config.hidden_size, num_heads=1, batch_first=True, dropout=config.dropout)
            if config.temporal_attention_head
            else None
        )
        self.output_norm = nn.LayerNorm(config.hidden_size) if config.layer_norm else nn.Identity()
        self.output_dropout = nn.Dropout(config.dropout)
        if config.mlp_head:
            head_hidden = max(config.hidden_size // 2, 16)
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, head_hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(head_hidden, config.output_dim),
            )
        else:
            self.classifier = nn.Linear(config.hidden_size, config.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(x)
        outputs, _ = self.gru(normalized)
        if self.residual_projection is not None:
            outputs = outputs + self.residual_projection(x)
        if self.attention is not None:
            seq_len = outputs.shape[1]
            attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=outputs.device, dtype=torch.bool),
                diagonal=1,
            )
            attended, _ = self.attention(outputs, outputs, outputs, attn_mask=attn_mask, need_weights=False)
            outputs = outputs + attended
        outputs = self.output_norm(outputs)
        outputs = self.output_dropout(outputs)
        return self.classifier(outputs)

    def config(self) -> dict[str, int | float | bool | str]:
        return dict(self.config_dict)


def build_model(config: ModelConfigV2) -> nn.Module:
    if config.model_type == "gru":
        return FlexibleSchemaCGRU(config)
    if config.model_type == "tcn":
        return SmallSemanticTCN(config)
    raise ValueError(f"unsupported model_type: {config.model_type}")