"""Spectral peak detection module for Spectral Iris.

This module implements quantum-precision spectral-temporal peak detection
with classification by type (A, B, C, D) based on spectral characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional

import numpy as np
from numpy.typing import NDArray

try:
    import scipy.ndimage
    import scipy.signal
except ImportError:
    scipy = None  # type: ignore

from ..config import config


class PeakType(Enum):
    """Peak classification types based on spectral characteristics."""

    TYPE_A = "isolated"  # >15dB above baseline, <2ms duration
    TYPE_B = "cluster"  # 3-5 adjacent bands, 5-20ms
    TYPE_C = "broadband"  # >1 octave, indicates saturation
    TYPE_D = "modulating"  # Oscillating frequency ±5%


@dataclass
class SpectralPeak:
    """Represents a detected spectral peak with metadata."""

    freq_center: float  # Center frequency in Hz
    freq_low: float  # Lower frequency bound
    freq_high: float  # Upper frequency bound

    time_start: float  # Start time in seconds
    time_end: float  # End time in seconds
    duration_ms: float  # Duration in milliseconds

    peak_db: float  # Peak level in dBFS
    baseline_db: float  # Baseline level in dBFS
    delta_db: float  # Difference from baseline

    peak_type: PeakType  # Classification type
    threshold: float  # Detection threshold used

    frame_idx: int  # Frame index in STFT
    bin_idx: int  # Frequency bin index

    is_audible: bool = True  # Whether peak is above masking threshold


def detect_spectral_peaks(
    audio: NDArray[np.floating],
    sample_rate: int,
    peak_threshold_db: float = -0.3,
    temporal_window_ms: float = 5.0,
    n_fft: int = 4096,
    hop_length: Optional[int] = None,
) -> List[SpectralPeak]:
    """Detect spectral-temporal peaks in audio signal.

    Uses STFT with Blackman-Harris window for high-resolution spectral analysis,
    then identifies peaks relative to an adaptive baseline per critical band.

    Args:
        audio: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        peak_threshold_db: Peak detection threshold in dBFS
        temporal_window_ms: Temporal smoothing window in ms
        n_fft: FFT size for STFT
        hop_length: Hop length (default n_fft // 4)

    Returns:
        List of detected SpectralPeak objects
    """
    if hop_length is None:
        hop_length = n_fft // 4

    # 1. Compute STFT with Blackman-Harris window
    stft_result = _compute_stft(audio, n_fft, hop_length)
    if stft_result is None:
        return []

    D, freqs, times = stft_result

    # 2. Convert to magnitude in dBFS
    S_db = _amplitude_to_db(np.abs(D), ref=1.0, top_db=120.0)

    # 3. Calculate adaptive baseline using smoothing
    freq_smoothing = 3  # Smooth over ~1/3 octave bands
    temporal_frames = max(1, int((sample_rate * temporal_window_ms / 1000) / hop_length))

    baseline = _calculate_baseline(S_db, freq_smoothing, temporal_frames)

    # 4. Detect peaks relative to baseline
    peaks = _find_peaks_relative(
        S_db=S_db,
        baseline=baseline,
        freqs=freqs,
        times=times,
        peak_threshold_db=peak_threshold_db,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )

    # 5. Classify peaks by type
    classified_peaks = [_classify_peak(peak, S_db, sample_rate) for peak in peaks]

    return classified_peaks


def _compute_stft(
    audio: NDArray[np.floating],
    n_fft: int,
    hop_length: int,
) -> Optional[Tuple[NDArray[np.complex128], NDArray[np.floating], NDArray[np.floating]]]:
    """Compute STFT with Blackman-Harris window.

    Returns:
        Tuple of (STFT, frequencies, times) or None if audio is too short
    """
    if len(audio) < n_fft:
        return None

    # Create Blackman-Harris window
    if scipy is not None:
        window = scipy.signal.windows.blackmanharris(n_fft)
    else:
        # Fallback to Hann window
        window = np.hanning(n_fft)

    n_frames = 1 + (len(audio) - n_fft) // hop_length
    if n_frames < 1:
        return None

    D = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)

    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        frame = audio[start:end]
        windowed = frame * window
        D[:, i] = np.fft.rfft(windowed)

    freqs = np.fft.rfftfreq(n_fft, 1.0 / 44100)
    times = np.arange(n_frames) * hop_length / 44100

    return D, freqs, times


def _amplitude_to_db(
    amplitude: NDArray[np.floating],
    ref: float = 1.0,
    top_db: float = 120.0,
) -> NDArray[np.floating]:
    """Convert amplitude spectrogram to dB scale."""
    magnitude = np.maximum(amplitude, 1e-10)
    db = 20.0 * np.log10(magnitude / ref)
    db = np.maximum(db, db.max() - top_db)
    return db


def _calculate_baseline(
    S_db: NDArray[np.floating],
    freq_smoothing: int,
    temporal_smoothing: int,
) -> NDArray[np.floating]:
    """Calculate adaptive baseline using 2D smoothing.

    Args:
        S_db: Magnitude spectrogram in dB
        freq_smoothing: Smoothing kernel size in frequency bins
        temporal_smoothing: Smoothing kernel size in time frames

    Returns:
        Smoothed baseline spectrogram
    """
    if scipy is not None:
        baseline = scipy.ndimage.uniform_filter(
            S_db, size=(freq_smoothing, temporal_smoothing), mode="nearest"
        )
    else:
        # Fallback: simple box filter
        baseline = np.zeros_like(S_db)
        for i in range(S_db.shape[0]):
            for j in range(S_db.shape[1]):
                i_low = max(0, i - freq_smoothing // 2)
                i_high = min(S_db.shape[0], i + freq_smoothing // 2 + 1)
                j_low = max(0, j - temporal_smoothing // 2)
                j_high = min(S_db.shape[1], j + temporal_smoothing // 2 + 1)
                baseline[i, j] = np.mean(S_db[i_low:i_high, j_low:j_high])

    return baseline


def _find_peaks_relative(
    S_db: NDArray[np.floating],
    baseline: NDArray[np.floating],
    freqs: NDArray[np.floating],
    times: NDArray[np.floating],
    peak_threshold_db: float,
    sample_rate: int,
    hop_length: int,
) -> List[SpectralPeak]:
    """Find peaks relative to adaptive baseline.

    Args:
        S_db: Magnitude spectrogram in dB
        baseline: Baseline spectrogram
        freqs: Frequency axis values
        times: Time axis values
        peak_threshold_db: Detection threshold
        sample_rate: Sample rate in Hz
        hop_length: STFT hop length

    Returns:
        List of detected peaks (before classification)
    """
    peaks: List[SpectralPeak] = []

    # Delta spectrogram (signal above baseline)
    delta_db = S_db - baseline

    # Minimum delta to consider as peak (relative to threshold)
    min_delta = 3.0  # At least 3dB above baseline

    for i in range(1, S_db.shape[0] - 1):
        for j in range(1, S_db.shape[1] - 1):
            # Check if above threshold and above baseline
            if S_db[i, j] > peak_threshold_db and delta_db[i, j] > min_delta:
                # Check if local maximum in 3x3 neighborhood
                neighborhood = S_db[i - 1 : i + 2, j - 1 : j + 2]
                if S_db[i, j] >= neighborhood.max():
                    # Calculate temporal span
                    duration_ms = _find_temporal_span(
                        S_db[i, :], j, peak_threshold_db, hop_length, sample_rate
                    )

                    # Filter by duration (0.5ms to 50ms)
                    if 0.5 <= duration_ms <= 50.0:
                        freq = freqs[i] if i < len(freqs) else 0.0
                        time = times[j] if j < len(times) else 0.0

                        # Estimate frequency bounds (1/24 octave resolution)
                        freq_low = freq * (2 ** (-1 / 48))
                        freq_high = freq * (2 ** (1 / 48))

                        peak = SpectralPeak(
                            freq_center=freq,
                            freq_low=freq_low,
                            freq_high=freq_high,
                            time_start=time,
                            time_end=time + duration_ms / 1000.0,
                            duration_ms=duration_ms,
                            peak_db=float(S_db[i, j]),
                            baseline_db=float(baseline[i, j]),
                            delta_db=float(delta_db[i, j]),
                            peak_type=PeakType.TYPE_A,  # Default, will be classified
                            threshold=peak_threshold_db,
                            frame_idx=j,
                            bin_idx=i,
                            is_audible=True,
                        )
                        peaks.append(peak)

    return peaks


def _find_temporal_span(
    S_db_row: NDArray[np.floating],
    center_idx: int,
    threshold_db: float,
    hop_length: int,
    sample_rate: int,
) -> float:
    """Find temporal span of a peak in milliseconds.

    Args:
        S_db_row: Single frequency band over time
        center_idx: Center frame index of peak
        threshold_db: Detection threshold
        hop_length: STFT hop length
        sample_rate: Sample rate in Hz

    Returns:
        Duration of peak in milliseconds
    """
    # Find extent where signal stays above threshold/2
    half_threshold = threshold_db / 2.0

    start_idx = center_idx
    while start_idx > 0 and S_db_row[start_idx - 1] > half_threshold:
        start_idx -= 1

    end_idx = center_idx
    while end_idx < len(S_db_row) - 1 and S_db_row[end_idx + 1] > half_threshold:
        end_idx += 1

    span_frames = end_idx - start_idx + 1
    duration_samples = span_frames * hop_length
    duration_ms = (duration_samples / sample_rate) * 1000.0

    return duration_ms


def _classify_peak(
    peak: SpectralPeak,
    S_db: NDArray[np.floating],
    sample_rate: int,
) -> SpectralPeak:
    """Classify peak type based on spectral-temporal characteristics.

    Classification criteria:
    - Type A: Isolated peaks (>15dB above baseline, <2ms duration)
    - Type B: Spectral-temporal clusters (3-5 adjacent bands, 5-20ms)
    - Type C: Broadband peaks (>1 octave)
    - Type D: Modulating peaks (frequency oscillation ±5%)

    Args:
        peak: Peak to classify
        S_db: Full spectrogram for context
        sample_rate: Sample rate

    Returns:
        Peak with updated type classification
    """
    cfg = config.peaks

    # Type A: Isolated short peaks
    if peak.delta_db > cfg.type_a_db_threshold and peak.duration_ms < cfg.type_a_max_duration_ms:
        peak_type = PeakType.TYPE_A
    # Type B: Temporal clusters
    elif cfg.type_b_min_duration_ms <= peak.duration_ms <= cfg.type_b_max_duration_ms:
        # Check spectral width
        spectral_width = _estimate_spectral_width(peak, S_db)
        if cfg.type_b_min_bands <= spectral_width <= cfg.type_b_max_bands:
            peak_type = PeakType.TYPE_B
        else:
            peak_type = PeakType.TYPE_A
    # Type C: Broadband (check if spans more than 1 octave)
    elif (peak.freq_high / max(peak.freq_low, 1.0)) > 2.0 ** cfg.type_c_min_octaves:
        peak_type = PeakType.TYPE_C
    # Type D: Check for frequency modulation
    elif _detect_frequency_modulation(peak, S_db, cfg.type_d_freq_tolerance):
        peak_type = PeakType.TYPE_D
    else:
        # Default to Type A
        peak_type = PeakType.TYPE_A

    # Create new peak with updated type (since SpectralPeak is not frozen)
    return SpectralPeak(
        freq_center=peak.freq_center,
        freq_low=peak.freq_low,
        freq_high=peak.freq_high,
        time_start=peak.time_start,
        time_end=peak.time_end,
        duration_ms=peak.duration_ms,
        peak_db=peak.peak_db,
        baseline_db=peak.baseline_db,
        delta_db=peak.delta_db,
        peak_type=peak_type,
        threshold=peak.threshold,
        frame_idx=peak.frame_idx,
        bin_idx=peak.bin_idx,
        is_audible=peak.is_audible,
    )


def _estimate_spectral_width(
    peak: SpectralPeak,
    S_db: NDArray[np.floating],
) -> int:
    """Estimate spectral width of peak in frequency bins.

    Args:
        peak: Peak to analyze
        S_db: Full spectrogram

    Returns:
        Number of adjacent frequency bins above threshold
    """
    bin_idx = peak.bin_idx
    frame_idx = peak.frame_idx
    threshold = peak.baseline_db + peak.delta_db / 2.0

    # Count bins above threshold around peak
    count = 1
    # Upward
    for i in range(bin_idx + 1, min(bin_idx + 10, S_db.shape[0])):
        if S_db[i, frame_idx] > threshold:
            count += 1
        else:
            break
    # Downward
    for i in range(bin_idx - 1, max(bin_idx - 10, -1), -1):
        if S_db[i, frame_idx] > threshold:
            count += 1
        else:
            break

    return count


def _detect_frequency_modulation(
    peak: SpectralPeak,
    S_db: NDArray[np.floating],
    tolerance: float,
) -> bool:
    """Detect if peak exhibits frequency modulation.

    Args:
        peak: Peak to analyze
        S_db: Full spectrogram
        tolerance: Frequency deviation tolerance (e.g., 0.05 for ±5%)

    Returns:
        True if frequency modulation detected
    """
    # Look at frequency bin drift across frames
    bin_idx = peak.bin_idx
    frame_start = max(0, peak.frame_idx - 5)
    frame_end = min(S_db.shape[1], peak.frame_idx + 6)

    peak_bins = []
    for j in range(frame_start, frame_end):
        # Find peak bin in this frame around original bin
        search_range = slice(max(0, bin_idx - 3), min(S_db.shape[0], bin_idx + 4))
        local_peak_bin = np.argmax(S_db[search_range, j]) + search_range.start
        peak_bins.append(local_peak_bin)

    if len(peak_bins) < 3:
        return False

    # Check for oscillation
    peak_bins_arr = np.array(peak_bins)
    mean_bin = np.mean(peak_bins_arr)
    deviation = np.std(peak_bins_arr) / max(mean_bin, 1.0)

    return deviation > tolerance


def find_peaks_above_threshold(
    audio: NDArray[np.floating],
    sample_rate: int,
    threshold_db: float = -0.3,
) -> List[SpectralPeak]:
    """Simplified peak detection for True Peak violations.

    Args:
        audio: Audio samples
        sample_rate: Sample rate in Hz
        threshold_db: Peak threshold in dBFS

    Returns:
        List of peaks above threshold
    """
    return detect_spectral_peaks(
        audio=audio,
        sample_rate=sample_rate,
        peak_threshold_db=threshold_db,
        temporal_window_ms=5.0,
    )
