"""Datasets for PDE-driven benchmarks."""

from neuroflow.data.burgers import Burgers1dDataset, generate_burgers1d_trajectory
from neuroflow.data.heat2d import Heat2dConfig, Heat2dDataset, generate_heat2d_sample
from neuroflow.data.heat3d import Heat3dConfig, Heat3dDataset, generate_heat3d_sample
from neuroflow.data.integral_op import (
    IntegralOp1dConfig,
    IntegralOp1dDataset,
    generate_integral_op_sample,
)

__all__ = [
    "Burgers1dDataset",
    "generate_burgers1d_trajectory",
    "Heat2dConfig",
    "Heat2dDataset",
    "generate_heat2d_sample",
    "Heat3dConfig",
    "Heat3dDataset",
    "generate_heat3d_sample",
    "IntegralOp1dConfig",
    "IntegralOp1dDataset",
    "generate_integral_op_sample",
]
