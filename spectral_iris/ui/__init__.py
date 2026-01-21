"""Spectral Iris User Interface Module.

This module provides the cyber-themed GUI for the Spectral Iris
audio processing application using Dear PyGui.
"""

from .main import SpectralIrisUI, launch_ui
from .theme import THEME, CyberTheme, apply_dpg_theme

__all__ = [
    "SpectralIrisUI",
    "launch_ui",
    "THEME",
    "CyberTheme",
    "apply_dpg_theme",
]
