"""Explicit convex interpolation; labels are never read by the model."""
import math
import torch
from torch import nn


class ConvexGeometricKernel(nn.Module):
    def __init__(self, *, mode="neural", cell_size_m=1000., distance_scale_m=1000.,
                 epsilon_m=1., hidden_channels=16):
        super().__init__()
        if mode not in {"fixed", "power", "neural"}:
            raise ValueError("mode must be fixed, power or neural")
        if min(cell_size_m, distance_scale_m, epsilon_m) <= 0:
            raise ValueError("distance scales must be positive")
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

        self.register_buffer("_geometry_log_distance", torch.empty(0), persistent=False)
        self.register_buffer("_geometry_features", torch.empty(0), persistent=False)
        self.register_buffer("_geometry_sources", torch.empty(0, dtype=torch.long), persistent=False)
        self._geometry_node_count = 0

    def prepare_geometry(self, positions, source_nodes):
        """Precompute fixed geometry once; buffers follow model device/dtype."""
        positions = torch.as_tensor(
            positions, dtype=self.raw_power.dtype, device=self.raw_power.device)
        sources = torch.as_tensor(source_nodes, dtype=torch.long, device=positions.device)
        if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0:
            raise ValueError("positions must have shape (nodes, 2)")
        if sources.ndim != 1 or not len(sources) or sources.unique().numel() != sources.numel():
            raise ValueError("source_nodes must be a nonempty unique vector")
        if (sources < 0).any() or (sources >= len(positions)).any():
            raise ValueError("source_nodes outside geometry")
        distances = torch.cdist(
            positions * self.cell_size_m, positions[sources] * self.cell_size_m)
        self._geometry_log_distance = torch.log(
            (distances + self.epsilon_m) / self.distance_scale_m)
        self._geometry_features = torch.log1p(
            distances / self.distance_scale_m).unsqueeze(-1)
        self._geometry_sources = sources
        self._geometry_node_count = len(positions)
        return self

    @property
    def power(self):
        return torch.nn.functional.softplus(self.raw_power)

    def forward(self, data, *, return_weights=False):
        if data.kernel_values.ndim == 3:
            return self._forward_dense(data, return_weights=return_weights)
        values = data.kernel_values
        available = data.kernel_available.bool()
        membership = getattr(data, "batch", None)
        if membership is None:
            membership = torch.zeros(len(values), dtype=torch.long, device=values.device)
        prediction = torch.zeros_like(values)
        weights = []
        for graph_id in membership.unique():
            nodes = (membership == graph_id).nonzero().flatten()
            positions = data.pos[nodes] * self.cell_size_m
            for channel in range(values.shape[1]):
                sources = nodes[available[nodes, channel]]
                if not len(sources):
                    raise ValueError("no current visible AQ sources for a sample/channel")
                distances = torch.cdist(positions, data.pos[sources] * self.cell_size_m)
                scores = -self.power * torch.log(
                    (distances + self.epsilon_m) / self.distance_scale_m)
                if self.mode == "neural":
                    features = torch.log1p(distances / self.distance_scale_m)
                    scores = scores + self.correction(features.unsqueeze(-1)).squeeze(-1)
                alpha = scores.softmax(dim=1)
                estimates = alpha @ values[sources, channel]
                # Exact reproduction at visible cells; targets have already been removed.
                visible = available[nodes, channel]
                prediction[nodes, channel] = torch.where(
                    visible, values[nodes, channel], estimates)
                if return_weights:
                    weights.append((nodes, sources, channel, alpha))
        return (prediction, weights) if return_weights else prediction


    def _forward_dense(self, data, *, return_weights=False):
        """Evaluate all samples/channels together, sharing learned distance scores."""
        values, available = data.kernel_values, data.kernel_available.bool()
        if self._geometry_node_count != values.shape[1]:
            raise ValueError("call prepare_geometry with the batch grid first")
        if available.shape != values.shape:
            raise ValueError("availability must match values")
        source_mask = available[:, self._geometry_sources, :].permute(0, 2, 1)
        if not source_mask.any(dim=-1).all():
            raise ValueError("no current visible AQ sources for a sample/channel")
        if available.sum() != source_mask.sum():
            raise ValueError("visible source outside prepared geometry")
        scores = -self.power * self._geometry_log_distance
        if self.mode == "neural":
            scores = scores + self.correction(self._geometry_features).squeeze(-1)
        logits = scores[None, None, :, :].masked_fill(
            ~source_mask[:, :, None, :], -torch.inf)
        alpha = logits.softmax(dim=-1)
        source_values = values[:, self._geometry_sources, :].permute(0, 2, 1)
        source_values = torch.where(source_mask, source_values, 0.)
        prediction = (alpha @ source_values.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
        prediction = torch.where(available, values, prediction)
        if not return_weights:
            return prediction
        # Diagnostics only: preserve the original per-sample/per-channel layout.
        weights = []
        node_count = values.shape[1]
        for graph_id in range(values.shape[0]):
            nodes = torch.arange(node_count, device=values.device) + graph_id * node_count
            for channel in range(values.shape[2]):
                selected = source_mask[graph_id, channel]
                sources = self._geometry_sources[selected] + graph_id * node_count
                weights.append((nodes, sources, channel, alpha[graph_id, channel][:, selected]))
        return prediction, weights
