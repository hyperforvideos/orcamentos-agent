"""Spectral Iris - Audio Processing Application.

Main entry point for the Spectral Iris audio processing application.
This module integrates the core audio processing with the UI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    sf = None  # type: ignore

from .config import config
from .core import (
    AudioMetrics,
    EffectsChain,
    SpectralCorrector,
    analyze_audio,
    detect_spectral_peaks,
)


class SpectralIrisProcessor:
    """Main audio processing engine for Spectral Iris.

    Integrates all audio analysis and correction modules into a
    unified processing pipeline.
    """

    def __init__(self) -> None:
        """Initialize the processor."""
        self.sample_rate: int = 44100
        self.audio_data: Optional[np.ndarray] = None
        self.metrics: Optional[AudioMetrics] = None
        self.corrector: Optional[SpectralCorrector] = None
        self.effects: Optional[EffectsChain] = None

    def load_audio(self, file_path: str) -> AudioMetrics:
        """Load an audio file for processing.

        Args:
            file_path: Path to the audio file

        Returns:
            AudioMetrics with analysis results

        Raises:
            ImportError: If soundfile is not available
            FileNotFoundError: If file doesn't exist
        """
        if not SOUNDFILE_AVAILABLE:
            raise ImportError(
                "soundfile is required for audio loading. "
                "Install with: pip install soundfile"
            )

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Load audio
        audio, sr = sf.read(file_path, dtype="float64")

        # Handle stereo -> mono for analysis
        if audio.ndim > 1:
            audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio

        self.audio_data = audio
        self.sample_rate = sr

        # Analyze
        self.metrics = analyze_audio(audio_mono, sr)

        # Initialize processors
        self.corrector = SpectralCorrector(sr)
        self.effects = EffectsChain(sr)

        return self.metrics

    def detect_peaks(
        self,
        threshold_db: float = -0.3,
    ) -> int:
        """Detect spectral peaks above threshold.

        Args:
            threshold_db: Peak detection threshold in dBFS

        Returns:
            Number of peaks detected
        """
        if self.audio_data is None:
            return 0

        # Ensure mono
        if self.audio_data.ndim > 1:
            audio = np.mean(self.audio_data, axis=1)
        else:
            audio = self.audio_data

        peaks = detect_spectral_peaks(
            audio=audio,
            sample_rate=self.sample_rate,
            peak_threshold_db=threshold_db,
        )

        return len(peaks)

    def process(
        self,
        threshold_db: float = -0.3,
        aggressiveness: float = 0.5,
        glitter_amount: float = 0.3,
        pitty_amount: float = 0.3,
        granular_amount: float = 0.2,
        progress_callback=None,
    ) -> np.ndarray:
        """Process audio with full correction pipeline.

        Args:
            threshold_db: Peak threshold in dBFS
            aggressiveness: Correction aggressiveness 0.0-1.0
            glitter_amount: Glitter effect amount
            pitty_amount: Pitty filter amount
            granular_amount: Granular smoother amount
            progress_callback: Optional callback(progress: float)

        Returns:
            Processed audio array
        """
        if self.audio_data is None:
            raise ValueError("No audio loaded")

        if self.corrector is None or self.effects is None:
            raise ValueError("Processors not initialized")

        # Ensure mono for processing
        if self.audio_data.ndim > 1:
            audio = np.mean(self.audio_data, axis=1)
        else:
            audio = self.audio_data.copy()

        if progress_callback:
            progress_callback(0.1)

        # 1. Detect peaks
        peaks = detect_spectral_peaks(
            audio=audio,
            sample_rate=self.sample_rate,
            peak_threshold_db=threshold_db,
        )

        if progress_callback:
            progress_callback(0.3)

        # 2. Apply corrections
        corrected = self.corrector.correct_peaks(
            audio=audio,
            peaks=peaks,
            aggressiveness=aggressiveness,
        )

        if progress_callback:
            progress_callback(0.6)

        # 3. Calculate peak reduction for effects
        original_peak = np.max(np.abs(audio))
        corrected_peak = np.max(np.abs(corrected))
        peak_reduction_db = 20.0 * np.log10(original_peak / max(corrected_peak, 1e-10))

        # 4. Apply effects chain
        processed = self.effects.process(
            audio=corrected,
            peak_reduction_db=peak_reduction_db,
            glitter_amount=glitter_amount,
            pitty_amount=pitty_amount,
            granular_amount=granular_amount,
        )

        if progress_callback:
            progress_callback(0.9)

        # 5. Stereo: Apply to both channels
        if self.audio_data.ndim > 1:
            n_channels = self.audio_data.shape[1]
            result = np.zeros_like(self.audio_data)
            for ch in range(n_channels):
                ch_audio = self.audio_data[:, ch]
                ch_peaks = detect_spectral_peaks(
                    ch_audio, self.sample_rate, threshold_db
                )
                ch_corrected = self.corrector.correct_peaks(
                    ch_audio, ch_peaks, aggressiveness
                )
                result[:, ch] = self.effects.process(
                    ch_corrected, peak_reduction_db,
                    glitter_amount, pitty_amount, granular_amount
                )
            processed = result

        if progress_callback:
            progress_callback(1.0)

        return processed

    def save_audio(
        self,
        audio: np.ndarray,
        output_path: str,
        bit_depth: int = 24,
    ) -> None:
        """Save processed audio to file.

        Args:
            audio: Audio data to save
            output_path: Output file path
            bit_depth: Output bit depth (16, 24, or 32)
        """
        if not SOUNDFILE_AVAILABLE:
            raise ImportError(
                "soundfile is required for audio saving. "
                "Install with: pip install soundfile"
            )

        subtype_map = {
            16: "PCM_16",
            24: "PCM_24",
            32: "FLOAT",
        }
        subtype = subtype_map.get(bit_depth, "PCM_24")

        sf.write(
            output_path,
            audio,
            self.sample_rate,
            subtype=subtype,
        )


def run_cli() -> None:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Spectral Iris - Audio Peak Correction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  spectral_iris input.wav -o output.wav
  spectral_iris input.wav -t -0.5 -a 0.7 -o output.wav

© Spectral Iris // Sonic Cybernetics
        """,
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Input audio file (WAV, FLAC, etc.)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output audio file",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=-0.3,
        help="Peak threshold in dBFS (default: -0.3)",
    )
    parser.add_argument(
        "-a", "--aggressiveness",
        type=float,
        default=0.5,
        help="Correction aggressiveness 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch graphical interface",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Only analyze, don't process",
    )

    args = parser.parse_args()

    # Launch GUI if requested or no input file
    if args.gui or (args.input is None):
        try:
            from .ui import launch_ui
            ui = launch_ui()
            ui.run()
            ui.cleanup()
        except ImportError as e:
            print(f"GUI not available: {e}")
            print("Use CLI mode with an input file, or install dearpygui")
            sys.exit(1)
        return

    # CLI processing
    if args.input is None:
        parser.print_help()
        sys.exit(1)

    processor = SpectralIrisProcessor()

    print(f"# SPECTRAL IRIS // Sonic Cybernetics")
    print(f"Loading: {args.input}")

    try:
        metrics = processor.load_audio(args.input)
    except Exception as e:
        print(f"Error loading audio: {e}")
        sys.exit(1)

    print(f"\n// Session Manifest")
    print(f"  Sample Rate: {metrics.sample_rate} Hz")
    print(f"  Duration: {metrics.duration_seconds:.2f}s")
    print(f"  Channels: {metrics.channels}")
    print(f"  LUFS Integrated: {metrics.lufs_integrated:.1f}")
    print(f"  True Peak: {metrics.true_peak_db:.1f} dBTP")

    if args.analyze:
        # Analysis only
        peaks = processor.detect_peaks(args.threshold)
        print(f"\n// Analysis Complete")
        print(f"  Peaks above {args.threshold} dBTP: {peaks}")
        return

    # Process
    print(f"\n// Processing...")
    print(f"  Threshold: {args.threshold} dBTP")
    print(f"  Aggressiveness: {args.aggressiveness}")

    def progress_cb(p: float) -> None:
        bar = "=" * int(p * 40) + "-" * (40 - int(p * 40))
        print(f"\r  [{bar}] {int(p * 100):3d}%", end="", flush=True)

    processed = processor.process(
        threshold_db=args.threshold,
        aggressiveness=args.aggressiveness,
        progress_callback=progress_cb,
    )
    print()

    # Save output
    if args.output:
        output_path = args.output
    else:
        # Generate output path
        input_path = Path(args.input)
        output_path = str(input_path.with_stem(f"{input_path.stem}_processed"))

    processor.save_audio(processed, output_path)
    print(f"\n// Output saved: {output_path}")

    # Final analysis
    final_metrics = analyze_audio(
        processed if processed.ndim == 1 else np.mean(processed, axis=1),
        processor.sample_rate,
    )
    print(f"\n// Final Metrics")
    print(f"  LUFS Integrated: {final_metrics.lufs_integrated:.1f}")
    print(f"  True Peak: {final_metrics.true_peak_db:.1f} dBTP")
    print(f"\n© Spectral Iris // Sonic Cybernetics")


def main() -> None:
    """Main entry point."""
    run_cli()


if __name__ == "__main__":
    main()
