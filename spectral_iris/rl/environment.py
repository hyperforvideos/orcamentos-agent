"""Reinforcement Learning Environment for Spectral Correction.

This module implements a gym-compatible environment for training
RL agents to optimize spectral correction parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class CorrectionAction:
    """Represents a correction action taken by the RL agent."""

    # Per-band gain adjustments (64 bands)
    band_gains: NDArray[np.floating]
    # Saturation amount
    saturation: float
    # Glitter amount
    glitter: float
    # Pitty amount
    pitty: float
    # Granular amount
    granular: float


@dataclass
class SpectralState:
    """State representation for the RL environment."""

    # Spectral frame (magnitude spectrum)
    spectral_frame: NDArray[np.floating]
    # Peak locations and magnitudes
    peak_map: NDArray[np.floating]
    # Masking threshold
    masking_threshold: NDArray[np.floating]
    # Historical corrections (rolling buffer)
    correction_history: NDArray[np.floating]


class SpectralCorrectionEnv:
    """Gym-compatible environment for spectral correction RL.

    The agent learns to apply optimal corrections to minimize
    peaks while preserving loudness and minimizing artifacts.

    Observation Space:
        - Spectral frame (256 bins)
        - Peak map (256 bins)
        - Masking threshold (256 bins)
        - Correction history (4 frames x 64 values)

    Action Space:
        - Band gains: 64 values in [-1, 1] -> [-12dB, +6dB]
        - Saturation: 1 value in [0, 1]
        - Glitter: 1 value in [0, 1]
        - Pitty: 1 value in [0, 1]
        - Granular: 1 value in [0, 1]

    Reward:
        Weighted combination of:
        - Peak reduction (positive)
        - Loudness stability (positive)
        - Artifact avoidance (negative for artifacts)
        - Timbre preservation (positive)
    """

    # Gym compatibility
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        sample_rate: int = 44100,
        n_bands: int = 64,
        n_spectral_bins: int = 256,
        history_length: int = 4,
    ):
        """Initialize the environment.

        Args:
            sample_rate: Audio sample rate
            n_bands: Number of frequency bands for control
            n_spectral_bins: Number of spectral bins in state
            history_length: Number of historical frames to include
        """
        self.sample_rate = sample_rate
        self.n_bands = n_bands
        self.n_spectral_bins = n_spectral_bins
        self.history_length = history_length

        # Action space dimensions
        self.action_dim = n_bands + 4  # bands + saturation + glitter + pitty + granular

        # Observation space dimensions
        self.obs_dim = (n_spectral_bins, 3 + history_length)

        # Current state
        self._current_audio: Optional[NDArray] = None
        self._current_frame_idx: int = 0
        self._correction_history: List[NDArray] = []

        # Reward weights
        self.reward_weights = {
            "peak_reduction": 3.0,
            "loudness_stability": 5.0,
            "artifact_avoidance": -10.0,
            "timbre_preservation": 2.0,
        }

    @property
    def action_space_shape(self) -> Tuple[int]:
        """Return action space shape."""
        return (self.action_dim,)

    @property
    def observation_space_shape(self) -> Tuple[int, int]:
        """Return observation space shape."""
        return self.obs_dim

    def reset(
        self,
        audio: Optional[NDArray] = None,
        seed: Optional[int] = None,
    ) -> Tuple[NDArray, Dict[str, Any]]:
        """Reset the environment.

        Args:
            audio: Audio samples to process
            seed: Random seed for reproducibility

        Returns:
            Initial observation and info dict
        """
        if seed is not None:
            np.random.seed(seed)

        if audio is not None:
            self._current_audio = audio.copy()
        elif self._current_audio is None:
            # Generate random test audio
            self._current_audio = np.random.randn(self.sample_rate * 5) * 0.5

        self._current_frame_idx = 0
        self._correction_history = []

        obs = self._get_observation()
        info = {"frame_idx": 0}

        return obs, info

    def step(
        self,
        action: NDArray[np.floating],
    ) -> Tuple[NDArray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Action array of shape (action_dim,)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Parse action
        correction_action = self._parse_action(action)

        # Apply correction
        corrected_audio = self._apply_correction(correction_action)

        # Calculate reward
        reward, reward_components = self._calculate_reward(corrected_audio)

        # Update state
        self._current_frame_idx += 1
        self._correction_history.append(action[:self.n_bands])

        # Check termination
        terminated = self._current_frame_idx >= self._get_n_frames()
        truncated = False

        # Get new observation
        obs = self._get_observation()

        info = {
            "frame_idx": self._current_frame_idx,
            "reward_components": reward_components,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> NDArray:
        """Get current observation from state.

        Returns:
            Observation array of shape (n_spectral_bins, 3 + history_length)
        """
        obs = np.zeros((self.n_spectral_bins, 3 + self.history_length))

        # Get current spectral frame
        spectral_frame = self._get_spectral_frame(self._current_frame_idx)
        obs[:, 0] = spectral_frame

        # Peak map (simplified: spectral frame above -20dB)
        peak_map = np.maximum(0, spectral_frame + 20) / 20
        obs[:, 1] = peak_map

        # Masking threshold (simplified approximation)
        masking = spectral_frame - 10.0
        obs[:, 2] = np.maximum(masking, -80) / 80 + 1

        # Historical corrections (resample to spectral bins)
        for i, hist in enumerate(self._correction_history[-self.history_length:]):
            if i < self.history_length:
                # Resample from n_bands to n_spectral_bins
                resampled = np.interp(
                    np.linspace(0, 1, self.n_spectral_bins),
                    np.linspace(0, 1, len(hist)),
                    hist,
                )
                obs[:, 3 + i] = resampled

        return obs

    def _get_spectral_frame(self, frame_idx: int) -> NDArray:
        """Get spectral frame at given index.

        Args:
            frame_idx: Frame index

        Returns:
            Spectral frame in dB
        """
        if self._current_audio is None:
            return np.zeros(self.n_spectral_bins) - 60

        # Calculate frame position
        n_fft = 4096
        hop_length = n_fft // 4
        start = frame_idx * hop_length

        if start + n_fft > len(self._current_audio):
            return np.zeros(self.n_spectral_bins) - 60

        # Get frame
        frame = self._current_audio[start:start + n_fft]
        frame = frame * np.hanning(n_fft)

        # FFT
        spectrum = np.abs(np.fft.rfft(frame))

        # Resample to n_spectral_bins
        if len(spectrum) != self.n_spectral_bins:
            spectrum = np.interp(
                np.linspace(0, 1, self.n_spectral_bins),
                np.linspace(0, 1, len(spectrum)),
                spectrum,
            )

        # Convert to dB
        db = 20 * np.log10(spectrum + 1e-10)

        return db

    def _get_n_frames(self) -> int:
        """Get total number of frames."""
        if self._current_audio is None:
            return 0
        n_fft = 4096
        hop_length = n_fft // 4
        return (len(self._current_audio) - n_fft) // hop_length

    def _parse_action(self, action: NDArray) -> CorrectionAction:
        """Parse action array into CorrectionAction.

        Args:
            action: Raw action array

        Returns:
            Parsed CorrectionAction
        """
        # Band gains: map [-1, 1] to [-12, 6] dB
        band_gains = action[:self.n_bands] * 9 - 3  # Center at -3dB

        # Effects: map [-1, 1] to [0, 1]
        saturation = (action[self.n_bands] + 1) / 2
        glitter = (action[self.n_bands + 1] + 1) / 2
        pitty = (action[self.n_bands + 2] + 1) / 2
        granular = (action[self.n_bands + 3] + 1) / 2

        return CorrectionAction(
            band_gains=band_gains,
            saturation=saturation,
            glitter=glitter,
            pitty=pitty,
            granular=granular,
        )

    def _apply_correction(self, action: CorrectionAction) -> NDArray:
        """Apply correction action to audio.

        Args:
            action: Correction action

        Returns:
            Corrected audio
        """
        if self._current_audio is None:
            return np.zeros(1024)

        # Get current frame
        n_fft = 4096
        hop_length = n_fft // 4
        start = self._current_frame_idx * hop_length
        end = start + n_fft

        if end > len(self._current_audio):
            return self._current_audio[start:]

        frame = self._current_audio[start:end].copy()

        # Simple gain application (simplified for performance)
        # In production, use spectral processing
        saturation = action.saturation
        if saturation > 0.1:
            frame = np.tanh(frame * (1 + saturation * 2)) / (1 + saturation * 2)

        return frame

    def _calculate_reward(
        self,
        corrected_audio: NDArray,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate reward for correction.

        Args:
            corrected_audio: Corrected audio frame

        Returns:
            Tuple of (total reward, reward components dict)
        """
        if self._current_audio is None:
            return 0.0, {}

        # Get original frame
        n_fft = 4096
        hop_length = n_fft // 4
        start = self._current_frame_idx * hop_length
        end = min(start + n_fft, len(self._current_audio))

        original = self._current_audio[start:end]

        # Peak reduction reward
        original_peak = np.max(np.abs(original))
        corrected_peak = np.max(np.abs(corrected_audio))
        peak_reduction = max(0, original_peak - corrected_peak) * 10
        r_peak = peak_reduction * self.reward_weights["peak_reduction"]

        # Loudness stability (RMS)
        original_rms = np.sqrt(np.mean(original ** 2))
        corrected_rms = np.sqrt(np.mean(corrected_audio ** 2))
        rms_diff = abs(original_rms - corrected_rms)
        r_loudness = np.exp(-rms_diff * 10) * self.reward_weights["loudness_stability"]

        # Artifact detection (simplified: check for sudden changes)
        diff = np.diff(corrected_audio)
        artifact_count = np.sum(np.abs(diff) > 0.5)
        r_artifacts = artifact_count * self.reward_weights["artifact_avoidance"] / len(diff)

        # Timbre preservation (spectral correlation)
        if len(original) == len(corrected_audio):
            correlation = np.corrcoef(original, corrected_audio)[0, 1]
            r_timbre = correlation * self.reward_weights["timbre_preservation"]
        else:
            r_timbre = 0

        components = {
            "peak_reduction": r_peak,
            "loudness_stability": r_loudness,
            "artifact_avoidance": r_artifacts,
            "timbre_preservation": r_timbre,
        }

        total_reward = sum(components.values())

        return total_reward, components

    def render(self, mode: str = "human") -> None:
        """Render the environment (placeholder).

        Args:
            mode: Render mode
        """
        pass

    def close(self) -> None:
        """Clean up environment resources."""
        self._current_audio = None
        self._correction_history = []
