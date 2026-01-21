"""Spectral Iris - Advanced Audio Peak Correction System.

A cyber-themed audio processing application for spectral peak detection
and intelligent correction with psychoacoustic awareness.

Features:
- True Peak and LUFS analysis (ITU-R BS.1770-4)
- Spectral peak detection and classification
- Intelligent correction (micro-compression, de-essing, soft saturation)
- Psychoacoustic masking analysis
- Advanced effects (Glitter, Pitty Filter, Granular Sutil)
- Cyber-themed Dear PyGui interface

Usage:
    # Command line
    python -m spectral_iris input.wav -o output.wav

    # With GUI
    python -m spectral_iris --gui

    # Python API
    from spectral_iris import SpectralIrisProcessor
    processor = SpectralIrisProcessor()
    metrics = processor.load_audio("input.wav")
    processed = processor.process()
    processor.save_audio(processed, "output.wav")

© Spectral Iris // Sonic Cybernetics
"""

__version__ = "1.0.0"
__author__ = "Spectral Iris Team"

from .config import SpectralIrisConfig, config
from .__main__ import SpectralIrisProcessor, main

__all__ = [
    "SpectralIrisConfig",
    "SpectralIrisProcessor",
    "config",
    "main",
]
