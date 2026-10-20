"""Current-hour graph reconstruction with learned messages and local fluxes.

Layers are reconstruction iterations, not calibrated physical time steps.
Wind is in m/s; grid positions are in cells. Concentrations are decoded before
transport so advection does not depend on the normalization offset.
"""
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class SensorInitializer(MessagePassing):
    """Shared vector messages with separate diffusion/advection gates."""

    def __init__(self, channels, hidden):
        super().__init__(aggr="add", node_dim=0)
        self.encoder = nn.Sequential(nn.Linear(channels + 1, hidden), nn.SiLU())
        self.score = nn.Sequential(nn.Linear(hidden + 6, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        self.payload = nn.Sequential(nn.Linear(hidden + 6, hidden), nn.SiLU())
        self.output = nn.Linear(hidden, channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, edges, attrs, baseline):
        state = self.encoder(x)
        messages = self.propagate(edges, x=state, edge_attr=attrs, size=(len(x), len(x)))
        return baseline + self.output(messages)

    def message(self, x_j, edge_attr, index, size_i):
        features = torch.cat((x_j, edge_attr), dim=-1)
        scores = self.score(features)
        # Distance is scaled in km. Advection vanishes exactly against wind.
        decay = 1. / (1. + edge_attr[:, 2])
        diffusion = torch.sigmoid(scores[:, 0]) * decay
        advection = torch.sigmoid(scores[:, 1]) * decay * edge_attr[:, 3].relu()
        weights = softmax(torch.log(diffusion.clamp_min(1e-12)), index, num_nodes=size_i)
        return self.payload(features) * (weights * (1. + advection))[:, None]


class AdvectionDiffusionGNN(nn.Module):
    """Independent learned sensor initialization followed by grid transport."""

    def __init__(self, *, channels=1, hidden_channels=32, propagation_steps=4,
                 cell_size_m=1000., normalization_mean=(0.,), normalization_std=(1.,)):
        super().__init__()
        if channels < 1 or hidden_channels < 1 or propagation_steps < 0 or cell_size_m <= 0:
            raise ValueError("invalid model dimensions or geometry")
        self.channels = channels
        self.propagation_steps = propagation_steps
        self.cell_size_m = float(cell_size_m)
        self.initializer = SensorInitializer(channels, hidden_channels)
        self.raw_diffusion = nn.Parameter(torch.tensor(-2.))
        self.raw_advection = nn.Parameter(torch.tensor(-2.))
        self.register_buffer("mean", torch.as_tensor(normalization_mean, dtype=torch.float32).reshape(1, channels))
        self.register_buffer("std", torch.as_tensor(normalization_std, dtype=torch.float32).reshape(1, channels))
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("invalid normalization")

    def local_step(self, concentration, edges, displacement_m, wind, *,
                   diffusion=True, advection=True):
        """Closed-grid conservative flux update with a nonnegative stable step."""
        source, target = edges
        if not source.numel():
            return concentration
        distance = displacement_m.norm(dim=-1)
        direction = displacement_m / distance[:, None]
        velocity = .5 * (wind[source] + wind[target])
        parallel = (velocity * direction).sum(-1)
        # Effective rates; coefficients are learned, not measured diffusivity.
        scaled_distance = distance / self.cell_size_m
        diffusion_rate = torch.nn.functional.softplus(self.raw_diffusion) / scaled_distance.square()
        advection_rate = torch.nn.functional.softplus(self.raw_advection) * parallel.relu() / scaled_distance
        full_rate = diffusion_rate + advection_rate
        rate = diffusion_rate * diffusion + advection_rate * advection
        # Preserve the full model's step when ablating a component: only its
        # flux is removed, avoiding a simultaneous change of integration time.
        outgoing = concentration.new_zeros(len(concentration)).index_add(0, source, full_rate)
        step = .9 / (1. + outgoing.max())
        flux = step * rate[:, None] * concentration[source]
        return concentration.index_add(0, target, flux).index_add(0, source, -flux)

    def forward(self, data, *, initialization="learned", diffusion=True,
                advection=True, trace=None):
        if initialization not in {"learned", "idw"}:
            raise ValueError("initialization must be learned or idw")
        if not hasattr(data, "wind_uv") or not hasattr(data, "wind_available"):
            raise ValueError("wind_uv and wind_available are required")
        graphs = getattr(data, "num_graphs", 1)
        nodes = len(data.pos) // graphs
        if len(data.pos) != nodes * graphs:
            raise ValueError("fixed-size grids are required")
        if hasattr(data, "ptr") and not torch.all(data.ptr.diff() == nodes):
            raise ValueError("fixed-size grids are required")
        outputs = []
        for graph in range(graphs):
            start, stop = graph * nodes, (graph + 1) * nodes
            pos = data.pos[start:stop]
            visible = data.context_mask[start:stop, -1].bool()
            if visible.shape[1] != self.channels or not visible.any(0).all():
                raise ValueError("each pollutant needs visible sources")
            if (visible.any(1) != visible.all(1)).any():
                raise ValueError("this version requires jointly available pollutant channels")
            if (visible & data.target_mask[start:stop]).any():
                raise ValueError("target AQ cannot be visible")
            if hasattr(data, "forbidden_context_mask") and (visible.any(1) & data.forbidden_context_mask[start:stop].bool()).any():
                raise ValueError("forbidden AQ cannot be visible")
            aq = torch.where(visible, data.x[start:stop, -1, :self.channels], 0.)
            sources = visible.all(1).nonzero().flatten()
            delta = (pos[:, None] - pos[sources][None]) * self.cell_size_m
            dist = delta.norm(dim=-1)
            baseline_weights = (dist + 1.).pow(-2)
            baseline_weights = baseline_weights / baseline_weights.sum(-1, keepdim=True)
            baseline = baseline_weights @ aq[sources]

            # Interpolate vectors, never angles. Missing wind means neutral flow.
            available_wind = data.wind_available[start:stop].bool()
            uv = data.wind_uv[start:stop]
            if not torch.isfinite(uv[available_wind]).all():
                raise ValueError("available wind must be finite")
            wind_nodes = available_wind.nonzero().flatten()
            wind = pos.new_zeros((nodes, 2))
            if len(wind_nodes):
                wd = torch.cdist(pos, pos[wind_nodes]) * self.cell_size_m
                ww = (wd + 1.).pow(-2)
                wind = (ww / ww.sum(-1, keepdim=True)) @ uv[wind_nodes]
                wind = torch.where(available_wind[:, None], uv, wind)
            direction = delta / (dist[..., None] + 1.)
            parallel = (direction * wind[sources][None]).sum(-1)
            transverse = direction[..., 0] * wind[sources][None, :, 1] - direction[..., 1] * wind[sources][None, :, 0]
            attrs = torch.cat((delta / self.cell_size_m, (dist / self.cell_size_m)[..., None],
                               parallel[..., None], transverse[..., None],
                               available_wind[sources][None, :, None].expand(nodes, -1, -1)), -1).reshape(-1, 6)
            edges = torch.stack((sources.repeat(nodes), torch.arange(nodes, device=pos.device).repeat_interleave(len(sources))))
            initial = (self.initializer(torch.cat((aq, visible.all(1, keepdim=True).float()), -1), edges, attrs, baseline)
                       if initialization == "learned" else baseline)
            concentration = (initial * self.std + self.mean).clamp_min(0.)
            observed = aq * self.std + self.mean
            concentration = torch.where(visible, observed, concentration)
            if trace is not None:
                trace(graph, 0, concentration, concentration, concentration)
            # Derive eight-neighbour grid edges from positions; no cross-batch edges.
            offset = pos[None] - pos[:, None]
            adjacency = (offset.abs().amax(-1) <= 1.001) & (offset.norm(dim=-1) > 0.)
            local = adjacency.nonzero().T
            displacement = (pos[local[1]] - pos[local[0]]) * self.cell_size_m
            for step_index in range(self.propagation_steps):
                before = concentration
                transported = self.local_step(before, local, displacement, wind,
                                              diffusion=diffusion, advection=advection)
                concentration = torch.where(visible, observed, transported)
                if trace is not None:
                    trace(graph, step_index + 1, before, transported, concentration)
            outputs.append((concentration - self.mean) / self.std)
        return torch.cat(outputs)
