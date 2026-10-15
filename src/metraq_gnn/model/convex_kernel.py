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

    @property
    def power(self):
        return torch.nn.functional.softplus(self.raw_power)

    def forward(self, data, *, return_weights=False):
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
