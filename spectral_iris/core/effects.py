"""Advanced audio effects for Spectral Iris.

This module implements specialized audio effects:
- Glitter: HF air addition post-limiting
- Pitty Filter: Dynamic high-pass for sub control
- Granular Sutil: Micro-granular transient smoothing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

try:
    import scipy.signal
    import scipy.signal.windows
except ImportError:
    scipy = None  # type: ignore

from ..config import config


@dataclass
class EffectState:
    """State container for stateful effects."""

    pitty_filter_state: Optional[NDArray] = None
    glitter_delay_buffer: Optional[NDArray] = None


class GlitterProcessor:
    """High-frequency air addition effect.

    Adds subtle high-frequency "glitter" to compensate for the loss
    of perceived brightness after peak limiting. Uses saturation
    harmonics in the 12-22kHz range.
    """

    def __init__(self, sample_rate: int = 44100):
        """Initialize Glitter processor.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.cfg = config.effects

        # Pre-calculate filter coefficients
        self._setup_bandpass_filter()

    def _setup_bandpass_filter(self) -> None:
        """Set up bandpass filter for HF extraction."""
        if scipy is None:
            self.sos = None
            return

        try:
            # Bandpass filter for 12-22kHz
            freq_low = self.cfg.glitter_freq_low
            freq_high = min(self.cfg.glitter_freq_high, self.sample_rate / 2 - 100)

            if freq_low >= freq_high:
                self.sos = None
                return

            self.sos = scipy.signal.butter(
                8,
                [freq_low, freq_high],
                btype="bandpass",
                fs=self.sample_rate,
                output="sos",
            )
        except ValueError:
            self.sos = None

    def process(
        self,
        audio: NDArray[np.floating],
        peak_reduction_db: float = 0.0,
        amount: float = 0.5,
    ) -> NDArray[np.floating]:
        """Apply glitter effect to audio.

        Args:
            audio: Audio samples
            peak_reduction_db: Amount of peak reduction applied (for adaptive amount)
            amount: Effect amount (0.0-1.0)

        Returns:
            Processed audio with added glitter
        """
        if self.sos is None or amount <= 0:
            return audio

        # Extract high-frequency band
        hf_band = scipy.signal.sosfilt(self.sos, audio)

        # Apply soft saturation for harmonic generation
        saturation = self.cfg.glitter_saturation * amount
        glitter = np.tanh(hf_band * saturation) * (1.0 / saturation)

        # Calculate gain based on peak reduction
        # More reduction = more glitter compensation
        gain = np.clip(abs(peak_reduction_db) * 0.1, 0.0, self.cfg.glitter_max_boost_db)
        gain_linear = 10.0 ** (gain / 20.0) - 1.0  # Convert to additional gain

        # Apply with slight delay (3 samples) to avoid phase issues
        delay = 3
        delayed_glitter = np.zeros_like(glitter)
        delayed_glitter[delay:] = glitter[:-delay] if delay > 0 else glitter

        return audio + (gain_linear * amount * delayed_glitter)


class PittyFilter:
    """Dynamic high-pass filter for subsonic control.

    Only activates during moments of excessive sub-80Hz energy
    to prevent pumping in the rest of the spectrum.
    """

    def __init__(self, sample_rate: int = 44100):
        """Initialize Pitty filter.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.cfg = config.effects

        # State for real-time processing
        self.history = np.zeros(4)
        self._prev_fc = self.cfg.pitty_freq_min

    def calculate_sub_energy(
        self,
        audio: NDArray[np.floating],
        window_size: int = 512,
    ) -> NDArray[np.floating]:
        """Calculate subsonic energy level.

        Args:
            audio: Audio samples
            window_size: Analysis window size

        Returns:
            Normalized sub energy per sample (0-1)
        """
        if scipy is None:
            return np.zeros_like(audio)

        # Extract sub-80Hz content
        sos = scipy.signal.butter(
            4, 80, btype="low", fs=self.sample_rate, output="sos"
        )
        sub_band = scipy.signal.sosfilt(sos, audio)

        # Calculate energy envelope
        energy = sub_band ** 2

        # Smooth with window
        if window_size > 0:
            kernel = np.ones(window_size) / window_size
            smoothed = np.convolve(energy, kernel, mode="same")
        else:
            smoothed = energy

        # Normalize to 0-1 range
        max_energy = np.max(smoothed)
        if max_energy > 0:
            normalized = smoothed / max_energy
        else:
            normalized = np.zeros_like(smoothed)

        return normalized

    def process_sample(
        self,
        sample: float,
        sub_energy: float,
    ) -> float:
        """Process single sample with dynamic HPF.

        Args:
            sample: Input sample
            sub_energy: Current sub energy level (0-1)

        Returns:
            Filtered sample
        """
        # Calculate dynamic cutoff frequency
        fc = self.cfg.pitty_freq_min + (
            (self.cfg.pitty_freq_max - self.cfg.pitty_freq_min)
            * np.clip(sub_energy, 0, 1)
        )

        # Smooth frequency changes
        fc = 0.95 * self._prev_fc + 0.05 * fc
        self._prev_fc = fc

        # Calculate biquad HPF coefficients
        w0 = 2.0 * np.pi * fc / self.sample_rate
        alpha = np.sin(w0) / (2.0 * self.cfg.pitty_q)

        a0 = 1 + alpha
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha
        b0 = (1 + np.cos(w0)) / 2
        b1 = -(1 + np.cos(w0))
        b2 = b0

        # Apply biquad filter
        output = (
            (b0 / a0) * sample
            + (b1 / a0) * self.history[0]
            + (b2 / a0) * self.history[1]
            - (a1 / a0) * self.history[2]
            - (a2 / a0) * self.history[3]
        )

        # Update history
        self.history = np.roll(self.history, 1)
        self.history[0] = sample
        self.history[2] = output

        return output

    def process(
        self,
        audio: NDArray[np.floating],
        amount: float = 0.5,
    ) -> NDArray[np.floating]:
        """Process audio block with Pitty filter.

        Args:
            audio: Audio samples
            amount: Effect amount (0.0-1.0)

        Returns:
            Processed audio
        """
        if amount <= 0:
            return audio

        # Calculate sub energy
        sub_energy = self.calculate_sub_energy(audio)

        # Process each sample
        output = np.zeros_like(audio)
        for i in range(len(audio)):
            filtered = self.process_sample(audio[i], sub_energy[i])
            # Blend based on amount
            output[i] = audio[i] * (1.0 - amount) + filtered * amount

        return output


class GranularSutil:
    """Subtle granular transient smoother.

    Spreads transient energy over 2-5ms using micro-granulation
    to soften aggressive attacks without losing perceived punch.
    """

    def __init__(self, sample_rate: int = 44100):
        """Initialize Granular Sutil processor.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.cfg = config.effects

    def detect_transients(
        self,
        audio: NDArray[np.floating],
        threshold_db: float = -20.0,
    ) -> NDArray[np.bool_]:
        """Detect transient positions in audio.

        Args:
            audio: Audio samples
            threshold_db: Transient detection threshold

        Returns:
            Boolean mask of transient positions
        """
        # Calculate onset strength
        hop_length = int(self.sample_rate * 0.001)  # 1ms hop
        frame_length = hop_length * 4

        # Energy difference
        energy = np.zeros(len(audio))
        for i in range(frame_length, len(audio)):
            frame = audio[i - frame_length : i]
            energy[i] = np.sum(frame ** 2)

        # Onset detection via first difference
        onset_strength = np.zeros_like(energy)
        onset_strength[1:] = np.maximum(0, energy[1:] - energy[:-1])

        # Normalize
        max_onset = np.max(onset_strength)
        if max_onset > 0:
            onset_strength = onset_strength / max_onset

        # Threshold
        threshold_linear = 10.0 ** (threshold_db / 20.0)
        transients = onset_strength > threshold_linear

        return transients

    def process_transient(
        self,
        transient: NDArray[np.floating],
        peak_db: float,
    ) -> NDArray[np.floating]:
        """Apply granular smoothing to transient.

        Args:
            transient: Transient audio segment
            peak_db: Peak level of transient in dB

        Returns:
            Smoothed transient
        """
        grain_size = int(self.sample_rate * self.cfg.granular_grain_size_ms / 1000)
        n_grains = self.cfg.granular_n_grains

        if len(transient) < grain_size or grain_size < 4:
            return transient

        # Create grains
        grains = []
        for i in range(n_grains):
            offset = i * (grain_size // n_grains)
            if offset + grain_size <= len(transient):
                grain = transient[offset : offset + grain_size].copy()

                # Apply Gaussian window
                if scipy is not None:
                    window = scipy.signal.windows.gaussian(grain_size, std=grain_size / 4)
                else:
                    window = np.hanning(grain_size)

                grains.append(grain * window)

        if not grains:
            return transient

        # Overlap-add reconstruction
        output = np.zeros_like(transient)
        overlap = grain_size // 2

        for i, grain in enumerate(grains):
            pos = i * overlap
            end_pos = min(pos + len(grain), len(output))
            grain_len = end_pos - pos
            output[pos:end_pos] += grain[:grain_len] * (0.5 if i > 0 else 1.0)

        # Calculate mix ratio based on peak severity
        mix_ratio = np.clip((peak_db - 3) / 10, 0, self.cfg.granular_max_mix)

        return (1 - mix_ratio) * transient + mix_ratio * output

    def process(
        self,
        audio: NDArray[np.floating],
        amount: float = 0.5,
    ) -> NDArray[np.floating]:
        """Process audio with granular smoothing.

        Args:
            audio: Audio samples
            amount: Effect amount (0.0-1.0)

        Returns:
            Processed audio
        """
        if amount <= 0:
            return audio

        # Detect transients
        transients = self.detect_transients(audio)

        # Find transient regions
        grain_size = int(self.sample_rate * self.cfg.granular_grain_size_ms / 1000)
        padding = grain_size * 2

        output = audio.copy()
        i = 0

        while i < len(audio):
            if transients[i]:
                # Found transient start
                start = max(0, i - padding)
                end = min(len(audio), i + padding + grain_size * 3)

                segment = audio[start:end].copy()
                peak_db = 20.0 * np.log10(np.max(np.abs(segment)) + 1e-10)

                processed = self.process_transient(segment, peak_db)

                # Blend back with crossfade
                fade_len = min(64, len(processed) // 4)
                if fade_len > 0:
                    fade_in = np.linspace(0, 1, fade_len)
                    fade_out = np.linspace(1, 0, fade_len)

                    processed[:fade_len] = (
                        segment[:fade_len] * fade_out + processed[:fade_len] * fade_in
                    )
                    processed[-fade_len:] = (
                        segment[-fade_len:] * fade_in + processed[-fade_len:] * fade_out
                    )

                output[start:end] = audio[start:end] * (1 - amount) + processed * amount

                # Skip processed region
                i = end
            else:
                i += 1

        return output


class EffectsChain:
    """Combined effects chain for Spectral Iris.

    Applies Glitter, Pitty, and Granular in optimal order.
    """

    def __init__(self, sample_rate: int = 44100):
        """Initialize effects chain.

        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.glitter = GlitterProcessor(sample_rate)
        self.pitty = PittyFilter(sample_rate)
        self.granular = GranularSutil(sample_rate)

    def process(
        self,
        audio: NDArray[np.floating],
        peak_reduction_db: float = 0.0,
        glitter_amount: float = 0.5,
        pitty_amount: float = 0.5,
        granular_amount: float = 0.3,
    ) -> NDArray[np.floating]:
        """Apply full effects chain.

        Order: Granular -> Pitty -> Glitter

        Args:
            audio: Audio samples
            peak_reduction_db: Peak reduction for glitter adaptation
            glitter_amount: Glitter effect amount
            pitty_amount: Pitty filter amount
            granular_amount: Granular smoother amount

        Returns:
            Processed audio
        """
        # 1. Granular smoothing on transients
        output = self.granular.process(audio, granular_amount)

        # 2. Pitty dynamic HPF
        output = self.pitty.process(output, pitty_amount)

        # 3. Glitter HF addition
        output = self.glitter.process(output, peak_reduction_db, glitter_amount)

        return output
