"""Configuration management for Spectral Iris.

This module provides configuration defaults and constants used throughout
the Spectral Iris audio processing application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class AudioConfig:
    """Audio processing configuration parameters."""

    sample_rate: int = 44100
    bit_depth: int = 24
    n_fft: int = 4096
    hop_length: int = 1024
    window: str = "blackmanharris"

    # True Peak thresholds (in dBFS)
    peak_threshold: float = -0.3
    peak_threshold_min: float = -3.0
    peak_threshold_max: float = -0.1

    # Temporal analysis
    temporal_window_ms: float = 5.0
    min_peak_duration_ms: float = 0.5
    max_peak_duration_ms: float = 50.0


@dataclass
class PeakClassification:
    """Peak type classification thresholds."""

    # Type A: Isolated peaks (>15dB above baseline, <2ms duration)
    type_a_db_threshold: float = 15.0
    type_a_max_duration_ms: float = 2.0

    # Type B: Spectral-temporal clusters (3-5 adjacent bands, 5-20ms)
    type_b_min_bands: int = 3
    type_b_max_bands: int = 5
    type_b_min_duration_ms: float = 5.0
    type_b_max_duration_ms: float = 20.0

    # Type C: Broadband peaks (>1 octave, indicates saturation)
    type_c_min_octaves: float = 1.0

    # Type D: Modulating peaks (oscillating frequency ±5%)
    type_d_freq_tolerance: float = 0.05


@dataclass
class EffectsConfig:
    """Configuration for audio effects processors."""

    # Glitter (HF Air Addition)
    glitter_freq_low: int = 12000
    glitter_freq_high: int = 22000
    glitter_max_boost_db: float = 0.5
    glitter_saturation: float = 0.5

    # Pitty Filter (Dynamic HPF)
    pitty_freq_min: int = 30
    pitty_freq_max: int = 80
    pitty_q: float = 0.707

    # Granular Sutil
    granular_grain_size_ms: float = 2.0
    granular_n_grains: int = 3
    granular_max_mix: float = 0.3


@dataclass
class UIConfig:
    """User interface configuration."""

    window_width: int = 1400
    window_height: int = 900
    min_window_width: int = 1024
    min_window_height: int = 768

    # Cyber color scheme (RGB tuples normalized 0-1)
    bg_primary: Tuple[float, float, float, float] = (0.05, 0.05, 0.05, 1.0)
    bg_secondary: Tuple[float, float, float, float] = (0.08, 0.08, 0.08, 1.0)
    bg_panel: Tuple[float, float, float, float] = (0.12, 0.12, 0.12, 1.0)

    text_primary: Tuple[float, float, float, float] = (0.9, 0.9, 0.9, 1.0)
    text_secondary: Tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)
    text_accent: Tuple[float, float, float, float] = (0.0, 0.8, 0.6, 1.0)

    accent_green: Tuple[float, float, float, float] = (0.0, 0.8, 0.4, 1.0)
    accent_blue: Tuple[float, float, float, float] = (0.0, 0.6, 0.9, 1.0)
    accent_red: Tuple[float, float, float, float] = (0.9, 0.2, 0.2, 1.0)
    accent_yellow: Tuple[float, float, float, float] = (0.9, 0.8, 0.1, 1.0)

    # Terminal-style font (monospace)
    font_size: int = 14
    font_size_large: int = 18
    font_size_small: int = 11


@dataclass
class SpectralIrisConfig:
    """Main configuration container for Spectral Iris."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    peaks: PeakClassification = field(default_factory=PeakClassification)
    effects: EffectsConfig = field(default_factory=EffectsConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    # Application metadata
    app_name: str = "SPECTRAL IRIS"
    app_version: str = "1.0.0"
    app_subtitle: str = "SPECTRAL PROCESSING CORE"


# Global configuration instance
config = SpectralIrisConfig()
