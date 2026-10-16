"""Vectorized convex interpolation using the same PyG inputs as the GNN."""
import math

import torch
from torch import nn
from torch_geometric.data import Data


class ConvexGeometricKernel(nn.Module):
    def __init__(self, *, mode="neural", cell_size_m=1000., distance_scale_m=1000.,
                 epsilon_m=1., hidden_channels=16, wind_mode="none",
                 wind_speed_scale_m_s=1., initial_wind_strength=.01):
        super().__init__()
        if mode not in {"fixed", "power", "neural"}:
            raise ValueError("mode must be fixed, power or neural")
        if min(cell_size_m, distance_scale_m, epsilon_m) <= 0:
            raise ValueError("distance scales must be positive")
        if wind_mode not in {"none", "directional"}:
            raise ValueError("wind_mode must be none or directional")
        if not math.isfinite(wind_speed_scale_m_s) or wind_speed_scale_m_s <= 0:
            raise ValueError("wind speed scale must be finite and positive")
        if not math.isfinite(initial_wind_strength) or initial_wind_strength <= 0:
            raise ValueError("initial wind strength must be finite and positive")
        self.wind_mode = wind_mode
        self.wind_speed_scale_m_s = float(wind_speed_scale_m_s)
        if wind_mode == "directional":
            self.raw_wind_strength = nn.Parameter(torch.tensor(
                initial_wind_strength + math.log(-math.expm1(-initial_wind_strength))))
        else:
            # Preserve strict loading of existing checkpoints.
            self.register_parameter("raw_wind_strength", None)
        self.mode = mode
        self.cell_size_m = float(cell_size_m)
        self.distance_scale_m = float(distance_scale_m)
        self.epsilon_m = float(epsilon_m)
        self.raw_power = nn.Parameter(torch.tensor(math.log(math.expm1(2.))),
                                      requires_grad=mode != "fixed")
        self.correction = nn.Sequential(nn.Linear(1, hidden_channels), nn.Tanh(),
                                        nn.Linear(hidden_channels, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        for parameter in self.correction.parameters():
            parameter.requires_grad_(mode == "neural")
        self.register_buffer("_geometry_positions", torch.empty(0), persistent=False)
        self.register_buffer("_geometry_log_distance", torch.empty(0), persistent=False)
        self.register_buffer("_geometry_features", torch.empty(0), persistent=False)
        self.register_buffer("_geometry_sources", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_geometry_direction", torch.empty(0), persistent=False)
        self._geometry_node_count = 0

    def prepare_geometry(self, positions, source_nodes):
        """Cache a fixed grid and candidate source cells before training/evaluation."""
        positions = torch.as_tensor(
            positions, dtype=self.raw_power.dtype, device=self.raw_power.device).detach()
        sources = torch.as_tensor(source_nodes, dtype=torch.long, device=positions.device)
        if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0:
            raise ValueError("positions must have shape (nodes, 2)")
        if not torch.isfinite(positions).all():
            raise ValueError("positions must be finite")
        if sources.ndim != 1 or not len(sources) or sources.unique().numel() != sources.numel():
            raise ValueError("source_nodes must be a nonempty unique vector")
        if (sources < 0).any() or (sources >= len(positions)).any():
            raise ValueError("source_nodes outside geometry")
        distances = torch.cdist(
            positions * self.cell_size_m, positions[sources] * self.cell_size_m)
        self._geometry_positions = positions.clone()
        self._geometry_log_distance = torch.log(
            (distances + self.epsilon_m) / self.distance_scale_m)
        self._geometry_features = torch.log1p(
            distances / self.distance_scale_m).unsqueeze(-1)
        if self.wind_mode == "directional":
            displacement = (positions[:, None, :] - positions[sources][None, :, :]) * self.cell_size_m
            self._geometry_direction = displacement / (distances[..., None] + self.epsilon_m)
        self._geometry_sources = sources
        self._geometry_node_count = len(positions)
        return self

    @property
    def power(self):
        return torch.nn.functional.softplus(self.raw_power)

    @property
    def wind_strength(self):
        if self.raw_wind_strength is None:
            return self.raw_power.new_zeros(())
        return torch.nn.functional.softplus(self.raw_wind_strength)

    def _wind_scores(self, data, batch_size):
        nodes = self._geometry_node_count
        wind, mask = getattr(data, "wind_uv", None), getattr(data, "wind_available", None)
        if wind is None or mask is None:
            raise ValueError("directional kernel requires wind_uv and wind_available")
        if wind.shape != (batch_size * nodes, 2) or mask.shape != (batch_size * nodes,):
            raise ValueError("wind must have shape (nodes, 2) with a node availability mask")
        wind = wind.reshape(batch_size, nodes, 2)[:, self._geometry_sources]
        mask = mask.bool().reshape(batch_size, nodes)[:, self._geometry_sources]
        if (mask & ~torch.isfinite(wind).all(dim=-1)).any():
            raise ValueError("available wind vectors must be finite")
        wind = torch.where(mask[..., None], wind, 0.)
        velocity = wind / (torch.linalg.vector_norm(wind, dim=-1, keepdim=True)
                           + self.wind_speed_scale_m_s)
        alignment = torch.einsum("nsd,bsd->bns", self._geometry_direction, velocity)
        return self.wind_strength * alignment

    def forward(self, data: Data, *, return_weights=False):
        """Read final-hour visible AQ and return concatenated (nodes, pollutants)."""
        if not isinstance(data, Data):
            raise TypeError("expected a PyG Data or Batch")
        node_count = self._geometry_node_count
        if not node_count:
            raise ValueError("call prepare_geometry before using the model")
        x, context = data.x, data.context_mask
        if x.ndim != 3 or context.ndim != 3 or x.shape[:2] != context.shape[:2]:
            raise ValueError("x and context_mask must have matching (nodes, time) axes")
        channels = context.shape[-1]
        if not channels or not x.shape[1] or x.shape[-1] < channels:
            raise ValueError("missing pollutant value channels")
        batch_size = getattr(data, "num_graphs", 1)
        if x.shape[0] != batch_size * node_count:
            raise ValueError("all samples must use the same prepared grid")
        ptr = getattr(data, "ptr", None)
        if ptr is not None and not torch.all(ptr.diff() == node_count):
            raise ValueError("all samples must use the same prepared grid")
        positions = data.pos.reshape(batch_size, node_count, 2)
        if not torch.equal(positions, self._geometry_positions[None].expand_as(positions)):
            raise ValueError("sample positions differ from the prepared grid")

        # PyG concatenates complete grids; reshape groups them without a copy.
        values = x[:, -1, :channels].reshape(batch_size, node_count, channels)
        available = context[:, -1, :].bool().reshape(batch_size, node_count, channels)
        target = getattr(data, "target_mask", None)
        if target is not None and (available.reshape(-1, channels) & target.bool()).any():
            raise ValueError("target AQ cannot be visible")
        forbidden = getattr(data, "forbidden_context_mask", None)
        if forbidden is not None and (
            available.reshape(-1, channels) & forbidden.bool().reshape(-1, 1)
        ).any():
            raise ValueError("forbidden AQ cannot be visible")
        source_mask = available[:, self._geometry_sources, :].permute(0, 2, 1)
        if not source_mask.any(dim=-1).all():
            raise ValueError("no current visible AQ sources for a sample/channel")
        if available.sum() != source_mask.sum():
            raise ValueError("visible source outside prepared geometry")
        scores = -self.power * self._geometry_log_distance
        if self.mode == "neural":
            scores = scores + self.correction(self._geometry_features).squeeze(-1)
        scores = scores[None, None, :, :]
        if self.wind_mode == "directional":
            scores = scores + self._wind_scores(data, batch_size)[:, None, :, :]
        logits = scores.masked_fill(~source_mask[:, :, None, :], -torch.inf)
        alpha = logits.softmax(dim=-1)
        source_values = values[:, self._geometry_sources, :].permute(0, 2, 1)
        source_values = torch.where(source_mask, source_values, 0.)
        prediction = (alpha @ source_values.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
        prediction = torch.where(available, values, prediction).reshape(-1, channels)
        if not return_weights:
            return prediction

        # Formatting optional diagnostics only; prediction always uses the tensor path.
        weights = []
        for graph_id in range(batch_size):
            nodes = torch.arange(node_count, device=x.device) + graph_id * node_count
            for channel in range(channels):
                selected = source_mask[graph_id, channel]
                sources = self._geometry_sources[selected] + graph_id * node_count
                weights.append((nodes, sources, channel, alpha[graph_id, channel][:, selected]))
        return prediction, weights
