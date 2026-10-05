from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


class AttentionWeights(NamedTuple):
    edge_index: torch.Tensor
    alpha: torch.Tensor


class SpatioTemporalGNN(nn.Module):
    """Encode each node history, then propagate through learned local attention."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        edge_channels: int,
        temporal_hidden_channels: int = 64,
        graph_hidden_channels: int = 64,
        graph_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or output_channels <= 0 or edge_channels <= 0:
            raise ValueError("input, output, and edge channel counts must be positive")
        if temporal_hidden_channels <= 0 or graph_hidden_channels <= 0:
            raise ValueError("hidden channel counts must be positive")
        if graph_layers <= 0 or attention_heads <= 0:
            raise ValueError("graph_layers and attention_heads must be positive")
        if graph_hidden_channels % attention_heads:
            raise ValueError("graph_hidden_channels must be divisible by attention_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.edge_channels = edge_channels
        self.temporal_encoder = nn.GRU(
            input_size=input_channels,
            hidden_size=temporal_hidden_channels,
            batch_first=True,
        )

        graph_convolutions: list[GATv2Conv] = []
        residual_projections: list[nn.Module] = []
        normalizations: list[nn.LayerNorm] = []
        current_channels = temporal_hidden_channels
        per_head_channels = graph_hidden_channels // attention_heads
        for _ in range(graph_layers):
            graph_convolutions.append(
                GATv2Conv(
                    in_channels=current_channels,
                    out_channels=per_head_channels,
                    heads=attention_heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_channels,
                    add_self_loops=False,
                )
            )
            residual_projections.append(
                nn.Identity()
                if current_channels == graph_hidden_channels
                else nn.Linear(current_channels, graph_hidden_channels, bias=False)
            )
            normalizations.append(nn.LayerNorm(graph_hidden_channels))
            current_channels = graph_hidden_channels

        self.graph_convolutions = nn.ModuleList(graph_convolutions)
        self.residual_projections = nn.ModuleList(residual_projections)
        self.normalizations = nn.ModuleList(normalizations)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Sequential(
            nn.Linear(graph_hidden_channels, graph_hidden_channels // 2),
            nn.ELU(),
            nn.Linear(graph_hidden_channels // 2, output_channels),
        )

    def _validate_data(self, data: Data) -> None:
        if data.x.ndim != 3:
            raise ValueError("data.x must have shape (nodes, time, features)")
        if data.x.shape[-1] != self.input_channels:
            raise ValueError(
                f"data.x has {data.x.shape[-1]} features, expected {self.input_channels}"
            )
        if data.edge_index.ndim != 2 or data.edge_index.shape[0] != 2:
            raise ValueError("data.edge_index must have shape (2, edges)")
        if data.edge_attr.ndim != 2 or data.edge_attr.shape[0] != data.edge_index.shape[1]:
            raise ValueError("data.edge_attr must have one row per edge")
        if data.edge_attr.shape[1] != self.edge_channels:
            raise ValueError(
                f"data.edge_attr has {data.edge_attr.shape[1]} features, "
                f"expected {self.edge_channels}"
            )

    def forward(
        self,
        data: Data,
        *,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[AttentionWeights]]:
        self._validate_data(data)
        _, hidden = self.temporal_encoder(data.x)
        node_state = hidden[-1]
        attention_history: list[AttentionWeights] = []

        for convolution, residual_projection, normalization in zip(
            self.graph_convolutions,
            self.residual_projections,
            self.normalizations,
        ):
            residual = residual_projection(node_state)
            if return_attention_weights:
                convolved, (edge_index, alpha) = convolution(
                    node_state,
                    data.edge_index,
                    data.edge_attr,
                    return_attention_weights=True,
                )
                attention_history.append(AttentionWeights(edge_index=edge_index, alpha=alpha))
            else:
                convolved = convolution(node_state, data.edge_index, data.edge_attr)

            node_state = normalization(residual + self.dropout(self.activation(convolved)))

        prediction = self.output_head(node_state)
        if return_attention_weights:
            return prediction, attention_history
        return prediction


class SensorToGridGNN(nn.Module):
    """Encode sensor histories, attend globally to them, then refine locally."""

    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        edge_channels: int,
        temporal_hidden_channels: int = 64,
        graph_hidden_channels: int = 64,
        attention_heads: int = 4,
        local_refinement_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or output_channels <= 0 or edge_channels <= 0:
            raise ValueError("input, output, and edge channel counts must be positive")
        if graph_hidden_channels % attention_heads:
            raise ValueError("graph_hidden_channels must be divisible by attention_heads")
        if local_refinement_layers < 0:
            raise ValueError("local_refinement_layers must not be negative")
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.edge_channels = edge_channels
        self.temporal_encoder = nn.GRU(
            input_size=input_channels,
            hidden_size=temporal_hidden_channels,
            batch_first=True,
        )
        self.temporal_projection = nn.Linear(temporal_hidden_channels, graph_hidden_channels)
        self.position_encoder = nn.Sequential(
            nn.Linear(2, graph_hidden_channels),
            nn.ELU(),
            nn.Linear(graph_hidden_channels, graph_hidden_channels),
        )
        per_head = graph_hidden_channels // attention_heads
        self.sensor_attention = GATv2Conv(
            in_channels=graph_hidden_channels,
            out_channels=per_head,
            heads=attention_heads,
            concat=True,
            dropout=dropout,
            edge_dim=edge_channels,
            add_self_loops=False,
        )
        self.global_normalization = nn.LayerNorm(graph_hidden_channels)
        self.local_convolutions = nn.ModuleList(
            GATv2Conv(
                in_channels=graph_hidden_channels,
                out_channels=per_head,
                heads=attention_heads,
                concat=True,
                dropout=dropout,
                edge_dim=edge_channels,
                add_self_loops=False,
            )
            for _ in range(local_refinement_layers)
        )
        self.local_normalizations = nn.ModuleList(
            nn.LayerNorm(graph_hidden_channels) for _ in range(local_refinement_layers)
        )
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Sequential(
            nn.Linear(graph_hidden_channels, graph_hidden_channels // 2),
            nn.ELU(),
            nn.Linear(graph_hidden_channels // 2, output_channels),
        )

    def _validate_data(self, data: Data) -> None:
        if data.x.ndim != 3 or data.x.shape[-1] != self.input_channels:
            raise ValueError("data.x has an incompatible shape")
        if not hasattr(data, "sensor_edge_index") or not hasattr(data, "sensor_edge_attr"):
            raise ValueError("data must contain sensor-to-grid edges")

    def forward(
        self,
        data: Data,
        *,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionWeights]:
        self._validate_data(data)
        _, hidden = self.temporal_encoder(data.x)
        temporal_state = self.temporal_projection(hidden[-1])
        query_state = temporal_state + self.position_encoder(data.pos)
        if return_attention_weights:
            global_state, (edge_index, alpha) = self.sensor_attention(
                query_state,
                data.sensor_edge_index,
                data.sensor_edge_attr,
                return_attention_weights=True,
            )
            attention = AttentionWeights(edge_index=edge_index, alpha=alpha)
        else:
            global_state = self.sensor_attention(
                query_state,
                data.sensor_edge_index,
                data.sensor_edge_attr,
            )
        node_state = self.global_normalization(
            query_state + self.dropout(self.activation(global_state))
        )
        for convolution, normalization in zip(
            self.local_convolutions, self.local_normalizations
        ):
            convolved = convolution(node_state, data.edge_index, data.edge_attr)
            node_state = normalization(node_state + self.dropout(self.activation(convolved)))
        prediction = self.output_head(node_state)
        if return_attention_weights:
            return prediction, attention
        return prediction


__all__ = ["AttentionWeights", "SensorToGridGNN", "SpatioTemporalGNN"]
