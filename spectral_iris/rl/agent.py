"""Reinforcement Learning Agent for Spectral Correction.

This module provides the RL agent wrapper for training and inference
using stable-baselines3 or a simple fallback agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .environment import SpectralCorrectionEnv


@dataclass
class CorrectionRecord:
    """Record of a correction decision for training."""

    state_hash: str
    action: NDArray[np.floating]
    reward: float
    timestamp: float


class RLAgent:
    """RL Agent for spectral correction optimization.

    Supports both stable-baselines3 (if available) and a simple
    rule-based fallback agent.
    """

    def __init__(
        self,
        n_bands: int = 64,
        model_path: Optional[str] = None,
    ):
        """Initialize the RL agent.

        Args:
            n_bands: Number of frequency bands
            model_path: Optional path to pre-trained model
        """
        self.n_bands = n_bands
        self.action_dim = n_bands + 4

        # Try to load stable-baselines3
        self._sb3_available = False
        self._model = None

        try:
            from stable_baselines3 import PPO
            self._sb3_available = True
            self._PPO = PPO

            if model_path and Path(model_path).exists():
                self._model = PPO.load(model_path)
        except ImportError:
            pass

        # Correction history for self-improvement
        self._correction_history: List[CorrectionRecord] = []
        self._history_max_size = 10000

        # Simple learned parameters (fallback)
        self._learned_weights = np.zeros(self.action_dim)

    @property
    def is_trained(self) -> bool:
        """Check if agent has a trained model."""
        return self._model is not None

    def predict(
        self,
        observation: NDArray[np.floating],
        deterministic: bool = True,
    ) -> NDArray[np.floating]:
        """Predict action for given observation.

        Args:
            observation: State observation
            deterministic: Whether to use deterministic policy

        Returns:
            Action array
        """
        if self._model is not None:
            action, _ = self._model.predict(observation, deterministic=deterministic)
            return action

        # Fallback: rule-based agent with learned adjustments
        return self._fallback_predict(observation)

    def _fallback_predict(
        self,
        observation: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Fallback prediction using simple rules.

        Args:
            observation: State observation

        Returns:
            Action array
        """
        action = np.zeros(self.action_dim)

        # Extract spectral and peak info from observation
        spectral = observation[:, 0] if observation.ndim > 1 else observation

        # Simple rule: reduce gain where peaks are high
        # Normalize spectral frame
        spectral_norm = (spectral - spectral.min()) / (spectral.max() - spectral.min() + 1e-10)

        # Resample to n_bands
        band_response = np.interp(
            np.linspace(0, 1, self.n_bands),
            np.linspace(0, 1, len(spectral_norm)),
            spectral_norm,
        )

        # Action: reduce where peaks are high
        # Map high energy to negative gain (reduction)
        action[:self.n_bands] = -band_response * 0.5  # Max -0.5 (mild reduction)

        # Apply learned adjustments
        action += self._learned_weights * 0.1

        # Clip to valid range
        action = np.clip(action, -1.0, 1.0)

        return action

    def train(
        self,
        env: SpectralCorrectionEnv,
        total_timesteps: int = 10000,
        callback=None,
    ) -> None:
        """Train the agent on the environment.

        Args:
            env: Training environment
            total_timesteps: Number of training steps
            callback: Optional training callback
        """
        if not self._sb3_available:
            print("stable-baselines3 not available, using self-improvement mode")
            self._self_improve()
            return

        # Create and train PPO model
        self._model = self._PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
        )

        self._model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
        )

    def record_correction(
        self,
        state: NDArray,
        action: NDArray,
        reward: float,
    ) -> None:
        """Record a correction for later training.

        Args:
            state: State observation
            action: Action taken
            reward: Reward received
        """
        import time
        import hashlib

        # Create state hash
        state_bytes = state.tobytes()
        state_hash = hashlib.md5(state_bytes).hexdigest()

        record = CorrectionRecord(
            state_hash=state_hash,
            action=action.copy(),
            reward=reward,
            timestamp=time.time(),
        )

        self._correction_history.append(record)

        # Trim history if too large
        if len(self._correction_history) > self._history_max_size:
            self._correction_history = self._correction_history[-self._history_max_size:]

    def _self_improve(self) -> None:
        """Self-improvement using recorded corrections.

        Updates learned weights based on historical performance.
        """
        if len(self._correction_history) < 100:
            return

        # Group corrections by reward
        good_corrections = [r for r in self._correction_history if r.reward > 0]
        bad_corrections = [r for r in self._correction_history if r.reward < 0]

        if not good_corrections:
            return

        # Average good actions
        good_actions = np.mean([r.action for r in good_corrections], axis=0)

        # Update learned weights towards good actions
        learning_rate = 0.01
        self._learned_weights += learning_rate * good_actions

        # Clip weights
        self._learned_weights = np.clip(self._learned_weights, -1.0, 1.0)

    def save(self, path: str) -> None:
        """Save agent to file.

        Args:
            path: Save path
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self._model is not None:
            self._model.save(str(save_path.with_suffix("")))
        else:
            # Save learned weights
            data = {
                "learned_weights": self._learned_weights.tolist(),
                "n_bands": self.n_bands,
            }
            with open(save_path.with_suffix(".json"), "w") as f:
                json.dump(data, f)

    def load(self, path: str) -> None:
        """Load agent from file.

        Args:
            path: Load path
        """
        load_path = Path(path)

        if self._sb3_available and load_path.with_suffix(".zip").exists():
            self._model = self._PPO.load(str(load_path.with_suffix("")))
        elif load_path.with_suffix(".json").exists():
            with open(load_path.with_suffix(".json")) as f:
                data = json.load(f)
            self._learned_weights = np.array(data["learned_weights"])

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent training statistics.

        Returns:
            Dictionary of statistics
        """
        if not self._correction_history:
            return {
                "total_corrections": 0,
                "avg_reward": 0.0,
                "positive_rate": 0.0,
            }

        rewards = [r.reward for r in self._correction_history]
        positive = sum(1 for r in rewards if r > 0)

        return {
            "total_corrections": len(self._correction_history),
            "avg_reward": np.mean(rewards),
            "positive_rate": positive / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
        }
