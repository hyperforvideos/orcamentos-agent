"""Spectral Iris Core Audio Processing Module.

This module provides the core audio processing functionality including:
- Audio analysis (LUFS, True Peak, spectral analysis)
- Spectral peak detection and classification
- Peak correction algorithms
- Psychoacoustic masking analysis
- Advanced effects (Glitter, Pitty, Granular)
"""

from .audio_analyzer import (
    AudioMetrics,
    analyze_audio,
    calculate_lufs_integrated,
    calculate_lufs_momentary,
    calculate_lufs_short_term,
    calculate_sample_peak,
    calculate_true_peak,
    compute_magnitude_db,
    compute_stft,
)
from .corrector import (
    CorrectionResult,
    SpectralCorrector,
    apply_true_peak_limiter,
)
from .effects import (
    EffectsChain,
    GlitterProcessor,
    GranularSutil,
    PittyFilter,
)
from .masking import (
    AuditoryMaskingAnalyzer,
    MaskingThreshold,
)
from .peak_detector import (
    PeakType,
    SpectralPeak,
    detect_spectral_peaks,
    find_peaks_above_threshold,
)

__all__ = [
    # Audio Analyzer
    "AudioMetrics",
    "analyze_audio",
    "calculate_lufs_integrated",
    "calculate_lufs_momentary",
    "calculate_lufs_short_term",
    "calculate_sample_peak",
    "calculate_true_peak",
    "compute_magnitude_db",
    "compute_stft",
    # Corrector
    "CorrectionResult",
    "SpectralCorrector",
    "apply_true_peak_limiter",
    # Effects
    "EffectsChain",
    "GlitterProcessor",
    "GranularSutil",
    "PittyFilter",
    # Masking
    "AuditoryMaskingAnalyzer",
    "MaskingThreshold",
    # Peak Detector
    "PeakType",
    "SpectralPeak",
    "detect_spectral_peaks",
    "find_peaks_above_threshold",
]
