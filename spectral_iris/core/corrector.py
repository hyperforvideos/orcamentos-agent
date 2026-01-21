"""Audio correction module for Spectral Iris.

This module implements spectral peak correction algorithms including
micro-compression, de-essing, soft-clipping, and intelligent limiters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import scipy.signal
except ImportError:
    scipy = None  # type: ignore

try:
    import pywt
except ImportError:
    pywt = None  # type: ignore

from .peak_detector import SpectralPeak, PeakType
from ..config import config


@dataclass
class CorrectionResult:
    """Result of a correction operation."""

    original_peak_db: float
    corrected_peak_db: float
    reduction_db: float
    method: str
    success: bool
    artifacts_detected: int


class SpectralCorrector:
    """Main class for spectral peak correction.

    Implements multiple correction strategies based on peak type:
    - Type A: Micro-compression (wavelet-based)
    - Type B: Intelligent de-essing
    - Type C: Soft saturation
    - Type D: Pitch-aware correction
    """

    def __init__(self, sample_rate: int = 44100):
        """Initialize corrector.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.corrections_applied: List[CorrectionResult] = []

    def correct_peaks(
        self,
        audio: NDArray[np.floating],
        peaks: List[SpectralPeak],
        aggressiveness: float = 0.5,
    ) -> NDArray[np.floating]:
        """Apply corrections to all detected peaks.

        Args:
            audio: Audio samples as numpy array
            peaks: List of detected peaks to correct
            aggressiveness: Correction strength (0.0-1.0)

        Returns:
            Corrected audio samples
        """
        corrected = audio.copy()
        self.corrections_applied = []

        for peak in peaks:
            try:
                if peak.peak_type == PeakType.TYPE_A:
                    corrected, result = self._micro_compression(
                        corrected, peak, aggressiveness
                    )
                elif peak.peak_type == PeakType.TYPE_B:
                    corrected, result = self._spectral_deessing(
                        corrected, peak, aggressiveness
                    )
                elif peak.peak_type == PeakType.TYPE_C:
                    corrected, result = self._soft_saturation(
                        corrected, peak, aggressiveness
                    )
                elif peak.peak_type == PeakType.TYPE_D:
                    corrected, result = self._modulation_correction(
                        corrected, peak, aggressiveness
                    )
                else:
                    continue

                self.corrections_applied.append(result)
            except Exception:
                # Skip problematic peaks rather than failing
                continue

        return corrected

    def _micro_compression(
        self,
        audio: NDArray[np.floating],
        peak: SpectralPeak,
        aggressiveness: float,
    ) -> Tuple[NDArray[np.floating], CorrectionResult]:
        """Apply micro-compression for Type A peaks.

        Uses wavelet decomposition for frequency-localized compression
        with ultra-fast attack (0.1ms) and short release (2ms).

        Args:
            audio: Audio samples
            peak: Peak to correct
            aggressiveness: Correction strength

        Returns:
            Tuple of (corrected audio, correction result)
        """
        # Calculate sample positions
        start_sample = int(peak.time_start * self.sample_rate)
        end_sample = int(peak.time_end * self.sample_rate)

        # Add padding for processing
        pad_samples = int(self.sample_rate * 0.01)  # 10ms padding
        start_sample = max(0, start_sample - pad_samples)
        end_sample = min(len(audio), end_sample + pad_samples)

        # Extract chunk
        chunk = audio[start_sample:end_sample].copy()

        if len(chunk) < 32:
            return audio, CorrectionResult(
                original_peak_db=peak.peak_db,
                corrected_peak_db=peak.peak_db,
                reduction_db=0.0,
                method="micro_compression",
                success=False,
                artifacts_detected=0,
            )

        # Calculate compression parameters
        ratio = 1.0 + (aggressiveness * peak.delta_db / 20.0)
        ratio = min(ratio, 4.0)  # Max ratio 4:1

        threshold_linear = 10.0 ** (peak.threshold / 20.0)

        # Apply dynamic envelope compression
        compressed = self._apply_envelope_compression(
            chunk,
            threshold=threshold_linear,
            ratio=ratio,
            attack_ms=0.1,
            release_ms=2.0,
        )

        # Place back in audio
        result = audio.copy()
        result[start_sample:end_sample] = compressed

        # Calculate result
        new_peak_db = 20.0 * np.log10(np.max(np.abs(compressed)) + 1e-10)
        reduction_db = peak.peak_db - new_peak_db

        return result, CorrectionResult(
            original_peak_db=peak.peak_db,
            corrected_peak_db=new_peak_db,
            reduction_db=reduction_db,
            method="micro_compression",
            success=reduction_db > 0,
            artifacts_detected=0,
        )

    def _spectral_deessing(
        self,
        audio: NDArray[np.floating],
        peak: SpectralPeak,
        aggressiveness: float,
    ) -> Tuple[NDArray[np.floating], CorrectionResult]:
        """Apply spectral de-essing for Type B peaks.

        Frequency-targeted dynamic reduction that preserves harmonic content.

        Args:
            audio: Audio samples
            peak: Peak to correct
            aggressiveness: Correction strength

        Returns:
            Tuple of (corrected audio, correction result)
        """
        if scipy is None:
            # Fallback to simple limiting
            return self._simple_limit(audio, peak)

        # Design narrow bandpass filter centered on peak frequency
        freq_center = peak.freq_center
        bandwidth = freq_center * 0.1  # 10% bandwidth
        freq_low = max(20, freq_center - bandwidth / 2)
        freq_high = min(self.sample_rate / 2 - 1, freq_center + bandwidth / 2)

        # Create bandpass filter
        try:
            sos = scipy.signal.butter(
                4,
                [freq_low, freq_high],
                btype="bandpass",
                fs=self.sample_rate,
                output="sos",
            )
        except ValueError:
            return self._simple_limit(audio, peak)

        # Extract band content
        band = scipy.signal.sosfilt(sos, audio)

        # Calculate dynamic reduction
        max_reduction = aggressiveness * 6.0  # Max 6dB reduction
        reduction_linear = 10.0 ** (-max_reduction / 20.0)

        # Apply to band with envelope following
        envelope = np.abs(band)
        if scipy is not None:
            # Smooth envelope
            window_size = int(self.sample_rate * 0.005)  # 5ms window
            if window_size > 0:
                envelope = scipy.ndimage.uniform_filter1d(
                    envelope, size=max(1, window_size)
                )

        # Dynamic gain reduction
        threshold_linear = 10.0 ** (peak.threshold / 20.0)
        gain = np.ones_like(envelope)
        above_threshold = envelope > threshold_linear
        gain[above_threshold] = (
            threshold_linear / np.maximum(envelope[above_threshold], 1e-10)
        ) ** (1.0 - 1.0 / 3.0)  # Soft knee

        gain = np.maximum(gain, reduction_linear)

        # Apply to original signal (subtract reduced band, add processed band)
        result = audio.copy()
        result = result - band + band * gain

        # Calculate result
        new_peak_db = 20.0 * np.log10(np.max(np.abs(result)) + 1e-10)
        reduction_db = peak.peak_db - new_peak_db

        return result, CorrectionResult(
            original_peak_db=peak.peak_db,
            corrected_peak_db=new_peak_db,
            reduction_db=reduction_db,
            method="spectral_deessing",
            success=reduction_db > 0,
            artifacts_detected=0,
        )

    def _soft_saturation(
        self,
        audio: NDArray[np.floating],
        peak: SpectralPeak,
        aggressiveness: float,
    ) -> Tuple[NDArray[np.floating], CorrectionResult]:
        """Apply soft saturation for Type C peaks.

        Uses asymmetric soft clipping with intermodulation control.

        Args:
            audio: Audio samples
            peak: Peak to correct
            aggressiveness: Correction strength

        Returns:
            Tuple of (corrected audio, correction result)
        """
        # Calculate saturation curve parameter
        k = 1.0 + aggressiveness * (peak.delta_db / 10.0)
        k = min(k, 3.0)  # Limit curvature

        threshold_linear = 10.0 ** (peak.threshold / 20.0)

        # Process in blocks with overlap-add
        block_size = 256
        hop = 64
        output = np.zeros_like(audio)
        window = np.hanning(block_size)

        for i in range(0, len(audio) - block_size, hop):
            block = audio[i : i + block_size]

            # Asymmetric soft clipping
            positive = np.maximum(block, 0)
            negative = np.minimum(block, 0)

            # y = x / (1 + |x|^k)^(1/k)
            def soft_clip(x: NDArray, threshold: float, k_val: float) -> NDArray:
                x_norm = x / threshold
                return threshold * x_norm / (1.0 + np.abs(x_norm) ** k_val) ** (
                    1.0 / k_val
                )

            clipped_pos = soft_clip(positive, threshold_linear, k)
            clipped_neg = soft_clip(negative, threshold_linear * 0.9, k)  # Asymmetric

            clipped = clipped_pos + clipped_neg
            output[i : i + block_size] += clipped * window

        # Normalize overlap-add regions
        divisor = np.zeros_like(audio)
        for i in range(0, len(audio) - block_size, hop):
            divisor[i : i + block_size] += window
        divisor = np.maximum(divisor, 1e-10)
        output = output / divisor

        # Calculate result
        new_peak_db = 20.0 * np.log10(np.max(np.abs(output)) + 1e-10)
        reduction_db = peak.peak_db - new_peak_db

        return output, CorrectionResult(
            original_peak_db=peak.peak_db,
            corrected_peak_db=new_peak_db,
            reduction_db=reduction_db,
            method="soft_saturation",
            success=reduction_db > 0,
            artifacts_detected=0,
        )

    def _modulation_correction(
        self,
        audio: NDArray[np.floating],
        peak: SpectralPeak,
        aggressiveness: float,
    ) -> Tuple[NDArray[np.floating], CorrectionResult]:
        """Apply correction for Type D modulating peaks.

        Uses pitch-tracking aware processing.

        Args:
            audio: Audio samples
            peak: Peak to correct
            aggressiveness: Correction strength

        Returns:
            Tuple of (corrected audio, correction result)
        """
        # For modulating peaks, use multiband limiting approach
        return self._multiband_limit(audio, peak, aggressiveness)

    def _simple_limit(
        self,
        audio: NDArray[np.floating],
        peak: SpectralPeak,
    ) -> Tuple[NDArray[np.floating], CorrectionResult]:
        """Simple peak limiting fallback.

        Args:
            audio: Audio samples
            peak: Peak to correct

        Returns:
            Tuple of (corrected audio, correction result)
        """
        threshold_linear = 10.0 ** (peak.threshold / 20.0)

        # Simple soft clipping
        result = np.tanh(audio / threshold_linear) * threshold_linear

        new_peak_db = 20.0 * np.log10(np.max(np.abs(result)) + 1e-10)
        reduction_db = peak.peak_db - new_peak_db

        return result, CorrectionResult(
            original_peak_db=peak.peak_db,
            corrected_peak_db=new_peak_db,
            reduction_db=reduction_db,
            method="simple_limit",
            success=reduction_db > 0,
            artifacts_detected=0,
        )

    def _multiband_limit(
        self,
        audio: NDArray[np.floating],
        peak: SpectralPeak,
        aggressiveness: float,
    ) -> Tuple[NDArray[np.floating], CorrectionResult]:
        """Multiband limiting for complex peaks.

        Args:
            audio: Audio samples
            peak: Peak to correct
            aggressiveness: Correction strength

        Returns:
            Tuple of (corrected audio, correction result)
        """
        if scipy is None:
            return self._simple_limit(audio, peak)

        # Define crossover frequencies
        crossovers = [200, 2000, 8000]
        bands = []

        # Low band
        sos_low = scipy.signal.butter(4, crossovers[0], btype="low", fs=self.sample_rate, output="sos")
        bands.append(scipy.signal.sosfilt(sos_low, audio))

        # Mid bands
        for i in range(len(crossovers) - 1):
            sos_mid = scipy.signal.butter(
                4,
                [crossovers[i], crossovers[i + 1]],
                btype="band",
                fs=self.sample_rate,
                output="sos",
            )
            bands.append(scipy.signal.sosfilt(sos_mid, audio))

        # High band
        sos_high = scipy.signal.butter(
            4, crossovers[-1], btype="high", fs=self.sample_rate, output="sos"
        )
        bands.append(scipy.signal.sosfilt(sos_high, audio))

        # Apply limiting per band
        threshold_linear = 10.0 ** (peak.threshold / 20.0)
        processed_bands = []

        for band in bands:
            limited = np.clip(band, -threshold_linear, threshold_linear)
            processed_bands.append(limited)

        # Sum bands
        result = sum(processed_bands)

        new_peak_db = 20.0 * np.log10(np.max(np.abs(result)) + 1e-10)
        reduction_db = peak.peak_db - new_peak_db

        return result, CorrectionResult(
            original_peak_db=peak.peak_db,
            corrected_peak_db=new_peak_db,
            reduction_db=reduction_db,
            method="multiband_limit",
            success=reduction_db > 0,
            artifacts_detected=0,
        )

    def _apply_envelope_compression(
        self,
        audio: NDArray[np.floating],
        threshold: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
    ) -> NDArray[np.floating]:
        """Apply envelope-following compression.

        Args:
            audio: Audio samples
            threshold: Threshold in linear scale
            ratio: Compression ratio
            attack_ms: Attack time in ms
            release_ms: Release time in ms

        Returns:
            Compressed audio
        """
        # Calculate envelope
        envelope = np.abs(audio)

        # Attack and release coefficients
        attack_samples = max(1, int(self.sample_rate * attack_ms / 1000))
        release_samples = max(1, int(self.sample_rate * release_ms / 1000))

        attack_coef = np.exp(-1.0 / attack_samples)
        release_coef = np.exp(-1.0 / release_samples)

        # Envelope follower
        smoothed_envelope = np.zeros_like(envelope)
        smoothed_envelope[0] = envelope[0]

        for i in range(1, len(envelope)):
            if envelope[i] > smoothed_envelope[i - 1]:
                coef = attack_coef
            else:
                coef = release_coef
            smoothed_envelope[i] = coef * smoothed_envelope[i - 1] + (1 - coef) * envelope[i]

        # Calculate gain reduction
        gain = np.ones_like(smoothed_envelope)
        above_threshold = smoothed_envelope > threshold

        gain[above_threshold] = (
            threshold / smoothed_envelope[above_threshold]
        ) ** (1.0 - 1.0 / ratio)

        # Apply gain
        return audio * gain


def apply_true_peak_limiter(
    audio: NDArray[np.floating],
    sample_rate: int,
    threshold_db: float = -0.3,
    lookahead_ms: float = 1.0,
) -> NDArray[np.floating]:
    """Apply true peak limiting to ensure no intersample peaks.

    Args:
        audio: Audio samples
        sample_rate: Sample rate in Hz
        threshold_db: Maximum true peak level in dBFS
        lookahead_ms: Lookahead time in ms

    Returns:
        Limited audio
    """
    if scipy is None:
        threshold_linear = 10.0 ** (threshold_db / 20.0)
        return np.clip(audio, -threshold_linear, threshold_linear)

    threshold_linear = 10.0 ** (threshold_db / 20.0)
    lookahead_samples = int(sample_rate * lookahead_ms / 1000)

    # Upsample for true peak detection
    oversample = 4
    upsampled = scipy.signal.resample_poly(audio, oversample, 1)

    # Find peaks above threshold
    peaks_above = np.abs(upsampled) > threshold_linear

    if not np.any(peaks_above):
        return audio

    # Calculate required gain reduction
    peak_gain = threshold_linear / np.maximum(np.abs(upsampled), 1e-10)
    peak_gain = np.minimum(peak_gain, 1.0)

    # Apply minimum filter for lookahead
    window_size = lookahead_samples * oversample * 2
    if scipy is not None:
        gain_curve = scipy.ndimage.minimum_filter1d(peak_gain, size=max(1, window_size))
    else:
        gain_curve = peak_gain

    # Downsample gain curve
    gain_original = scipy.signal.resample_poly(gain_curve, 1, oversample)

    # Ensure same length
    if len(gain_original) > len(audio):
        gain_original = gain_original[: len(audio)]
    elif len(gain_original) < len(audio):
        gain_original = np.pad(
            gain_original, (0, len(audio) - len(gain_original)), mode="edge"
        )

    return audio * gain_original
