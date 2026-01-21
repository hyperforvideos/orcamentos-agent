"""Audio analysis module for Spectral Iris.

This module provides functions for analyzing audio signals including
LUFS loudness calculation, True Peak detection, and FFT spectral analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore

try:
    import scipy.signal
except ImportError:
    scipy = None  # type: ignore


@dataclass
class AudioMetrics:
    """Container for audio analysis metrics."""

    sample_rate: int
    duration_seconds: float
    channels: int
    bit_depth: int

    # Loudness metrics
    lufs_integrated: float
    lufs_momentary: float
    lufs_short_term: float

    # Peak metrics
    true_peak_db: float
    sample_peak_db: float

    # Spectral metrics
    spectral_centroid: float
    spectral_bandwidth: float


def calculate_rms(audio: NDArray[np.floating]) -> float:
    """Calculate RMS (Root Mean Square) of audio signal.

    Args:
        audio: Audio samples as numpy array

    Returns:
        RMS value in linear scale
    """
    return float(np.sqrt(np.mean(audio**2)))


def amplitude_to_db(amplitude: float, ref: float = 1.0) -> float:
    """Convert amplitude to decibels.

    Args:
        amplitude: Linear amplitude value
        ref: Reference amplitude (default 1.0 for dBFS)

    Returns:
        Value in decibels
    """
    if amplitude <= 0:
        return -120.0
    return 20.0 * np.log10(amplitude / ref)


def db_to_amplitude(db: float, ref: float = 1.0) -> float:
    """Convert decibels to amplitude.

    Args:
        db: Value in decibels
        ref: Reference amplitude (default 1.0 for dBFS)

    Returns:
        Linear amplitude value
    """
    return ref * (10.0 ** (db / 20.0))


def calculate_true_peak(
    audio: NDArray[np.floating],
    sample_rate: int,
    oversample_factor: int = 4,
) -> float:
    """Calculate True Peak level using oversampling.

    True Peak measures intersample peaks by upsampling the signal.
    Based on ITU-R BS.1770-4 recommendation.

    Args:
        audio: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        oversample_factor: Upsampling factor (default 4x)

    Returns:
        True Peak level in dBFS
    """
    if len(audio) == 0:
        return -120.0

    # Upsample using polyphase resampling
    if scipy is not None:
        upsampled = scipy.signal.resample_poly(audio, oversample_factor, 1)
    else:
        # Fallback: linear interpolation
        original_length = len(audio)
        new_length = original_length * oversample_factor
        x_original = np.linspace(0, 1, original_length)
        x_new = np.linspace(0, 1, new_length)
        upsampled = np.interp(x_new, x_original, audio)

    peak_amplitude = float(np.max(np.abs(upsampled)))
    return amplitude_to_db(peak_amplitude)


def calculate_sample_peak(audio: NDArray[np.floating]) -> float:
    """Calculate sample peak level.

    Args:
        audio: Audio samples as numpy array

    Returns:
        Sample peak level in dBFS
    """
    if len(audio) == 0:
        return -120.0

    peak_amplitude = float(np.max(np.abs(audio)))
    return amplitude_to_db(peak_amplitude)


def calculate_lufs_momentary(
    audio: NDArray[np.floating],
    sample_rate: int,
) -> float:
    """Calculate momentary LUFS loudness (400ms window).

    Simplified implementation based on ITU-R BS.1770-4.

    Args:
        audio: Audio samples as numpy array
        sample_rate: Sample rate in Hz

    Returns:
        Momentary LUFS value
    """
    if len(audio) == 0:
        return -70.0

    # Apply K-weighting (simplified high-shelf + high-pass)
    weighted = _apply_k_weighting(audio, sample_rate)

    # Calculate mean square over 400ms window
    window_samples = int(sample_rate * 0.4)
    if len(weighted) < window_samples:
        window_samples = len(weighted)

    # Use last 400ms
    segment = weighted[-window_samples:]
    mean_square = np.mean(segment**2)

    if mean_square <= 0:
        return -70.0

    # LUFS = -0.691 + 10 * log10(mean_square)
    lufs = -0.691 + 10.0 * np.log10(mean_square)
    return float(max(-70.0, lufs))


def calculate_lufs_short_term(
    audio: NDArray[np.floating],
    sample_rate: int,
) -> float:
    """Calculate short-term LUFS loudness (3s window).

    Args:
        audio: Audio samples as numpy array
        sample_rate: Sample rate in Hz

    Returns:
        Short-term LUFS value
    """
    if len(audio) == 0:
        return -70.0

    weighted = _apply_k_weighting(audio, sample_rate)

    window_samples = int(sample_rate * 3.0)
    if len(weighted) < window_samples:
        window_samples = len(weighted)

    segment = weighted[-window_samples:]
    mean_square = np.mean(segment**2)

    if mean_square <= 0:
        return -70.0

    lufs = -0.691 + 10.0 * np.log10(mean_square)
    return float(max(-70.0, lufs))


def calculate_lufs_integrated(
    audio: NDArray[np.floating],
    sample_rate: int,
) -> float:
    """Calculate integrated LUFS loudness over entire signal.

    Uses gated measurement with -70 LUFS absolute gate.

    Args:
        audio: Audio samples as numpy array
        sample_rate: Sample rate in Hz

    Returns:
        Integrated LUFS value
    """
    if len(audio) == 0:
        return -70.0

    weighted = _apply_k_weighting(audio, sample_rate)

    # Block-based measurement (400ms blocks with 75% overlap)
    block_size = int(sample_rate * 0.4)
    hop = block_size // 4

    blocks = []
    for i in range(0, len(weighted) - block_size, hop):
        block = weighted[i : i + block_size]
        mean_square = np.mean(block**2)
        if mean_square > 0:
            block_lufs = -0.691 + 10.0 * np.log10(mean_square)
            if block_lufs > -70.0:  # Absolute gate
                blocks.append(mean_square)

    if not blocks:
        return -70.0

    # Relative gate at -10 LUFS below ungated average
    ungated_avg = np.mean(blocks)
    relative_gate_lufs = -0.691 + 10.0 * np.log10(ungated_avg) - 10.0
    relative_gate_power = 10.0 ** ((relative_gate_lufs + 0.691) / 10.0)

    gated_blocks = [b for b in blocks if b >= relative_gate_power]

    if not gated_blocks:
        return -70.0

    gated_mean = np.mean(gated_blocks)
    lufs = -0.691 + 10.0 * np.log10(gated_mean)
    return float(max(-70.0, lufs))


def _apply_k_weighting(
    audio: NDArray[np.floating],
    sample_rate: int,
) -> NDArray[np.floating]:
    """Apply K-weighting filter for LUFS measurement.

    Simplified implementation of ITU-R BS.1770 pre-filter.

    Args:
        audio: Audio samples as numpy array
        sample_rate: Sample rate in Hz

    Returns:
        K-weighted audio signal
    """
    if scipy is None:
        return audio  # Return unweighted if scipy not available

    # Stage 1: High shelf filter (+4dB above 1.5kHz)
    fc1 = 1500.0
    gain_db = 4.0
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc1 / sample_rate
    sin_w0 = np.sin(w0)
    cos_w0 = np.cos(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / 0.9 - 1.0) + 2.0)

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

    b_shelf = np.array([b0 / a0, b1 / a0, b2 / a0])
    a_shelf = np.array([1.0, a1 / a0, a2 / a0])

    # Stage 2: High-pass filter at 38Hz
    fc2 = 38.0
    Q = 0.5
    w0_hp = 2.0 * np.pi * fc2 / sample_rate
    sin_w0_hp = np.sin(w0_hp)
    cos_w0_hp = np.cos(w0_hp)
    alpha_hp = sin_w0_hp / (2.0 * Q)

    b0_hp = (1 + cos_w0_hp) / 2
    b1_hp = -(1 + cos_w0_hp)
    b2_hp = (1 + cos_w0_hp) / 2
    a0_hp = 1 + alpha_hp
    a1_hp = -2 * cos_w0_hp
    a2_hp = 1 - alpha_hp

    b_hp = np.array([b0_hp / a0_hp, b1_hp / a0_hp, b2_hp / a0_hp])
    a_hp = np.array([1.0, a1_hp / a0_hp, a2_hp / a0_hp])

    # Apply filters
    weighted = scipy.signal.lfilter(b_shelf, a_shelf, audio)
    weighted = scipy.signal.lfilter(b_hp, a_hp, weighted)

    return weighted


def compute_stft(
    audio: NDArray[np.floating],
    n_fft: int = 4096,
    hop_length: int = 1024,
    window: str = "blackmanharris",
    sample_rate: int = 44100,
) -> Tuple[NDArray[np.complex128], NDArray[np.floating], NDArray[np.floating]]:
    """Compute Short-Time Fourier Transform.

    Args:
        audio: Audio samples as numpy array
        n_fft: FFT size
        hop_length: Hop length between frames
        window: Window function name
        sample_rate: Audio sample rate in Hz

    Returns:
        Tuple of (STFT complex array, frequencies, times)
    """
    if librosa is not None:
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window=window)
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        times = librosa.frames_to_time(
            np.arange(D.shape[1]), sr=sample_rate, hop_length=hop_length
        )
        return D, freqs, times

    # Fallback implementation
    if scipy is not None:
        window_func = scipy.signal.get_window(window, n_fft)
    else:
        window_func = np.hanning(n_fft)

    n_frames = 1 + (len(audio) - n_fft) // hop_length
    if n_frames < 1:
        n_frames = 1

    D = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)

    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        if end > len(audio):
            frame = np.zeros(n_fft)
            frame[: len(audio) - start] = audio[start:]
        else:
            frame = audio[start:end]

        windowed = frame * window_func
        fft_result = np.fft.rfft(windowed)
        D[:, i] = fft_result

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    times = np.arange(n_frames) * hop_length / sample_rate

    return D, freqs, times


def compute_magnitude_db(
    stft: NDArray[np.complex128],
    ref: float = 1.0,
    top_db: float = 120.0,
) -> NDArray[np.floating]:
    """Convert STFT to magnitude spectrogram in dB.

    Args:
        stft: Complex STFT array
        ref: Reference amplitude
        top_db: Maximum dynamic range

    Returns:
        Magnitude spectrogram in dB
    """
    magnitude = np.abs(stft)

    # Avoid log of zero
    magnitude = np.maximum(magnitude, 1e-10)

    db = 20.0 * np.log10(magnitude / ref)

    # Clip to top_db range
    db = np.maximum(db, db.max() - top_db)

    return db


def analyze_audio(
    audio: NDArray[np.floating],
    sample_rate: int,
    bit_depth: int = 24,
) -> AudioMetrics:
    """Perform complete audio analysis.

    Args:
        audio: Audio samples as numpy array (mono or first channel)
        sample_rate: Sample rate in Hz
        bit_depth: Bit depth (for metadata)

    Returns:
        AudioMetrics object with all calculated metrics
    """
    # Ensure mono
    if audio.ndim > 1:
        audio_mono = audio[0] if audio.shape[0] < audio.shape[1] else audio[:, 0]
    else:
        audio_mono = audio

    duration = len(audio_mono) / sample_rate

    # Calculate loudness metrics
    lufs_integrated = calculate_lufs_integrated(audio_mono, sample_rate)
    lufs_momentary = calculate_lufs_momentary(audio_mono, sample_rate)
    lufs_short_term = calculate_lufs_short_term(audio_mono, sample_rate)

    # Calculate peak metrics
    true_peak_db = calculate_true_peak(audio_mono, sample_rate)
    sample_peak_db = calculate_sample_peak(audio_mono)

    # Calculate spectral metrics
    if librosa is not None:
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(
            y=audio_mono, sr=sample_rate
        )))
        spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(
            y=audio_mono, sr=sample_rate
        )))
    else:
        spectral_centroid = 0.0
        spectral_bandwidth = 0.0

    channels = 1 if audio.ndim == 1 else audio.shape[0]

    return AudioMetrics(
        sample_rate=sample_rate,
        duration_seconds=duration,
        channels=channels,
        bit_depth=bit_depth,
        lufs_integrated=lufs_integrated,
        lufs_momentary=lufs_momentary,
        lufs_short_term=lufs_short_term,
        true_peak_db=true_peak_db,
        sample_peak_db=sample_peak_db,
        spectral_centroid=spectral_centroid,
        spectral_bandwidth=spectral_bandwidth,
    )
