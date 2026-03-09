"""PWOM surrogate package.

Physical interpretation:
- This package is reserved for the field-line distribution graph surrogate.
- It represents coarse shell distributions rather than persistent macro-particle trajectories.
"""

# Re-export the main config dataclasses so scripts can import a stable public surface.
from .config import DataConfig, ModelConfig, TrainConfig

__all__ = ["DataConfig", "ModelConfig", "TrainConfig"]
