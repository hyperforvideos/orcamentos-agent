"""Reinforcement Learning module for Spectral Iris.

This module provides the RL-based auto-optimization system for
learning optimal correction parameters over time.
"""

from .environment import SpectralCorrectionEnv
from .agent import RLAgent

__all__ = [
    "SpectralCorrectionEnv",
    "RLAgent",
]
