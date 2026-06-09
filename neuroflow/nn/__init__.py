"""Neural network modules for neural operators."""

from neuroflow.nn.deeponet import DeepONet, DeepONetConfig
from neuroflow.nn.fno import FNO1d, SpectralConv1d
from neuroflow.nn.fno2d import FNO2d, FNO2dConfig, SpectralConv2d
from neuroflow.nn.fno3d import FNO3d, FNO3dConfig, SpectralConv3d

__all__ = [
    "DeepONet",
    "DeepONetConfig",
    "FNO1d",
    "FNO2d",
    "FNO2dConfig",
    "FNO3d",
    "FNO3dConfig",
    "SpectralConv1d",
    "SpectralConv2d",
    "SpectralConv3d",
]
