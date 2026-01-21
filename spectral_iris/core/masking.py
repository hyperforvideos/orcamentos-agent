"""Psychoacoustic masking analysis for Spectral Iris.

This module implements auditory masking models based on Painter & Spanias (2003)
to determine which spectral peaks are perceptually audible versus masked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .peak_detector import SpectralPeak


@dataclass
class MaskingThreshold:
    """Masking threshold data for a spectral frame."""

    frequencies: NDArray[np.floating]  # Center frequencies of bands
    thresholds_db: NDArray[np.floating]  # Masking threshold per band
    absolute_threshold_db: NDArray[np.floating]  # ISO 226 absolute threshold


class AuditoryMaskingAnalyzer:
    """Analyzer for computing auditory masking thresholds.

    Based on psychoacoustic models from:
    - Painter & Spanias (2003)
    - ISO 226:2003 Equal Loudness Contours
    - ERB (Equivalent Rectangular Bandwidth) critical bands
    """

    def __init__(self, sample_rate: int = 44100):
        """Initialize masking analyzer.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.critical_bands = self._calculate_erb_bands()
        self.spreading_function = self._calculate_spreading_function()
        self.absolute_threshold = self._iso226_threshold(self.critical_bands)

    def _calculate_erb_bands(
        self,
        n_bands: int = 64,
    ) -> NDArray[np.floating]:
        """Calculate ERB (Equivalent Rectangular Bandwidth) critical bands.

        ERB scale approximates the frequency resolution of human hearing.

        Args:
            n_bands: Number of bands to calculate

        Returns:
            Array of center frequencies for each band
        """
        # ERB scale from 20Hz to Nyquist
        min_freq = 20.0
        max_freq = self.sample_rate / 2.0

        # ERB scale formula: ERB = 24.7 * (4.37 * f/1000 + 1)
        # Inverse: f = (erb / 24.7 - 1) / 0.00437

        def freq_to_erb(f: float) -> float:
            return 21.4 * np.log10(4.37 * f / 1000 + 1)

        def erb_to_freq(erb: float) -> float:
            return (10 ** (erb / 21.4) - 1) * 1000 / 4.37

        erb_min = freq_to_erb(min_freq)
        erb_max = freq_to_erb(max_freq)

        erb_scale = np.linspace(erb_min, erb_max, n_bands)
        frequencies = np.array([erb_to_freq(erb) for erb in erb_scale])

        return frequencies

    def _calculate_spreading_function(
        self,
        n_bands: int = 64,
    ) -> NDArray[np.floating]:
        """Calculate spreading function for masking.

        Models how masking spreads across critical bands.

        Args:
            n_bands: Number of critical bands

        Returns:
            Spreading function array
        """
        # Spreading function in dB
        # Asymmetric: steeper on high-frequency side

        spread = np.zeros(n_bands * 2 - 1)
        center = n_bands - 1

        for i in range(len(spread)):
            bark_diff = i - center

            if bark_diff < 0:
                # Lower frequencies (upward spread)
                spread[i] = 27.0 * bark_diff  # -27 dB/Bark slope
            else:
                # Higher frequencies (downward spread)
                spread[i] = -10.0 * bark_diff  # -10 dB/Bark slope

            # Limit spread
            spread[i] = max(spread[i], -100)

        # Convert to linear scale
        spread_linear = 10.0 ** (spread / 10.0)

        return spread_linear

    def _iso226_threshold(
        self,
        frequencies: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Calculate absolute threshold of hearing based on ISO 226:2003.

        Args:
            frequencies: Frequency values in Hz

        Returns:
            Absolute threshold in dB SPL (converted to dBFS reference)
        """
        # Simplified absolute threshold formula
        # T_q(f) = 3.64*(f/1000)^-0.8 - 6.5*exp(-0.6*(f/1000-3.3)^2) + 10^-3*(f/1000)^4

        f_khz = frequencies / 1000.0
        f_khz = np.maximum(f_khz, 0.02)  # Avoid division issues

        # Calculate threshold
        threshold = (
            3.64 * (f_khz ** -0.8)
            - 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2)
            + 1e-3 * (f_khz ** 4)
        )

        # Convert from SPL to dBFS (assuming -70 dBFS = threshold of hearing)
        threshold_dbfs = threshold - 90.0

        # Limit to reasonable range
        threshold_dbfs = np.clip(threshold_dbfs, -100, 0)

        return threshold_dbfs

    def calculate_masking_threshold(
        self,
        spectral_frame: NDArray[np.floating],
        frame_frequencies: NDArray[np.floating],
    ) -> MaskingThreshold:
        """Calculate masking threshold for a spectral frame.

        Args:
            spectral_frame: Magnitude spectrum in dB
            frame_frequencies: Corresponding frequencies

        Returns:
            MaskingThreshold with per-band thresholds
        """
        # Interpolate spectral frame to critical bands
        excitation = np.interp(
            self.critical_bands,
            frame_frequencies,
            spectral_frame,
            left=spectral_frame[0],
            right=spectral_frame[-1],
        )

        # Convert to linear for spreading calculation
        excitation_linear = 10.0 ** (excitation / 10.0)

        # Apply spreading function (convolution)
        spread_excitation = np.convolve(
            excitation_linear, self.spreading_function, mode="same"
        )

        # Ensure spread_excitation matches critical bands length
        if len(spread_excitation) != len(self.critical_bands):
            spread_excitation = np.interp(
                np.linspace(0, 1, len(self.critical_bands)),
                np.linspace(0, 1, len(spread_excitation)),
                spread_excitation,
            )

        # Convert back to dB
        masked_threshold_db = 10.0 * np.log10(spread_excitation + 1e-10)

        # Combine with absolute threshold (take maximum)
        final_threshold = np.maximum(masked_threshold_db, self.absolute_threshold)

        return MaskingThreshold(
            frequencies=self.critical_bands,
            thresholds_db=final_threshold,
            absolute_threshold_db=self.absolute_threshold,
        )

    def is_peak_audible(
        self,
        peak_freq: float,
        peak_level_db: float,
        masking_threshold: MaskingThreshold,
        safety_margin_db: float = 2.0,
    ) -> bool:
        """Determine if a spectral peak is audible above masking.

        Args:
            peak_freq: Peak frequency in Hz
            peak_level_db: Peak level in dB
            masking_threshold: Current masking threshold
            safety_margin_db: Additional margin for critical peaks

        Returns:
            True if peak is audible (above masking threshold)
        """
        # Find closest critical band
        band_idx = np.searchsorted(masking_threshold.frequencies, peak_freq)
        band_idx = min(band_idx, len(masking_threshold.thresholds_db) - 1)

        mask_thresh = masking_threshold.thresholds_db[band_idx]

        return peak_level_db > (mask_thresh + safety_margin_db)

    def filter_audible_peaks(
        self,
        peaks: List[SpectralPeak],
        spectral_frame: NDArray[np.floating],
        frame_frequencies: NDArray[np.floating],
    ) -> List[SpectralPeak]:
        """Filter peaks to only those that are perceptually audible.

        Args:
            peaks: List of detected spectral peaks
            spectral_frame: Magnitude spectrum for masking calculation
            frame_frequencies: Corresponding frequencies

        Returns:
            List of peaks that are above masking threshold
        """
        if not peaks:
            return []

        # Calculate masking threshold
        masking = self.calculate_masking_threshold(spectral_frame, frame_frequencies)

        audible_peaks = []
        for peak in peaks:
            is_audible = self.is_peak_audible(
                peak.freq_center, peak.peak_db, masking
            )

            # Update peak audibility flag
            updated_peak = SpectralPeak(
                freq_center=peak.freq_center,
                freq_low=peak.freq_low,
                freq_high=peak.freq_high,
                time_start=peak.time_start,
                time_end=peak.time_end,
                duration_ms=peak.duration_ms,
                peak_db=peak.peak_db,
                baseline_db=peak.baseline_db,
                delta_db=peak.delta_db,
                peak_type=peak.peak_type,
                threshold=peak.threshold,
                frame_idx=peak.frame_idx,
                bin_idx=peak.bin_idx,
                is_audible=is_audible,
            )

            if is_audible:
                audible_peaks.append(updated_peak)

        return audible_peaks

    def get_masking_visualization_data(
        self,
        spectral_frame: NDArray[np.floating],
        frame_frequencies: NDArray[np.floating],
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Get data for visualizing masking threshold.

        Args:
            spectral_frame: Magnitude spectrum in dB
            frame_frequencies: Corresponding frequencies

        Returns:
            Tuple of (frequencies, spectrum, masking_threshold)
        """
        masking = self.calculate_masking_threshold(spectral_frame, frame_frequencies)

        # Interpolate masking back to original frequency resolution
        threshold_interpolated = np.interp(
            frame_frequencies,
            masking.frequencies,
            masking.thresholds_db,
        )

        return frame_frequencies, spectral_frame, threshold_interpolated
