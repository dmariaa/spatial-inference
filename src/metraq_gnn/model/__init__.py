"""Models sharing the PyTorch Geometric node-regression interface."""

from metraq_gnn.model.convex_kernel import ConvexGeometricKernel
from metraq_gnn.model.advection_diffusion import AdvectionDiffusionGNN
from metraq_gnn.model.spatiotemporal import SensorToGridGNN, SpatioTemporalGNN

__all__ = ["AdvectionDiffusionGNN", "ConvexGeometricKernel", "SensorToGridGNN", "SpatioTemporalGNN"]
