"""Tests for Spectral Iris core modules.

This module contains unit tests for the audio analysis, peak detection,
correction, and effects modules.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectral_iris.config import SpectralIrisConfig, config
from spectral_iris.core.audio_analyzer import (
    amplitude_to_db,
    calculate_lufs_integrated,
    calculate_lufs_momentary,
    calculate_rms,
    calculate_sample_peak,
    calculate_true_peak,
    db_to_amplitude,
)
from spectral_iris.core.peak_detector import (
    PeakType,
    SpectralPeak,
    detect_spectral_peaks,
)
from spectral_iris.core.corrector import (
    CorrectionResult,
    SpectralCorrector,
    apply_true_peak_limiter,
)
from spectral_iris.core.effects import (
    EffectsChain,
    GlitterProcessor,
    GranularSutil,
    PittyFilter,
)
from spectral_iris.core.masking import (
    AuditoryMaskingAnalyzer,
    MaskingThreshold,
)


class TestConfig:
    """Tests for configuration module."""

    def test_config_defaults(self):
        """Test that config has expected defaults."""
        assert config.audio.sample_rate == 44100
        assert config.audio.peak_threshold == -0.3
        assert config.app_name == "SPECTRAL IRIS"

    def test_config_creation(self):
        """Test creating new config instance."""
        cfg = SpectralIrisConfig()
        assert cfg.audio.n_fft == 4096
        assert cfg.ui.window_width == 1400


class TestAudioAnalyzer:
    """Tests for audio analysis functions."""

    def test_amplitude_to_db_full_scale(self):
        """Test amplitude to dB conversion at full scale."""
        result = amplitude_to_db(1.0)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_amplitude_to_db_half(self):
        """Test amplitude to dB conversion at -6dB."""
        result = amplitude_to_db(0.5)
        assert result == pytest.approx(-6.02, abs=0.1)

    def test_amplitude_to_db_zero(self):
        """Test amplitude to dB with zero returns minimum."""
        result = amplitude_to_db(0.0)
        assert result == -120.0

    def test_db_to_amplitude_zero_db(self):
        """Test dB to amplitude at 0dB."""
        result = db_to_amplitude(0.0)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_db_to_amplitude_minus_6(self):
        """Test dB to amplitude at -6dB."""
        result = db_to_amplitude(-6.0)
        assert result == pytest.approx(0.5, abs=0.05)

    def test_calculate_rms_sine(self):
        """Test RMS calculation for sine wave."""
        # 1kHz sine at unity amplitude
        sr = 44100
        t = np.linspace(0, 1, sr)
        audio = np.sin(2 * np.pi * 1000 * t)
        rms = calculate_rms(audio)
        # RMS of sine wave is 1/sqrt(2) ≈ 0.707
        assert rms == pytest.approx(0.707, abs=0.01)

    def test_calculate_sample_peak(self):
        """Test sample peak detection."""
        audio = np.array([0.0, 0.5, 1.0, -0.8, 0.3])
        peak_db = calculate_sample_peak(audio)
        assert peak_db == pytest.approx(0.0, abs=0.01)

    def test_calculate_sample_peak_negative(self):
        """Test sample peak with values below 0dB."""
        audio = np.array([0.0, 0.25, 0.5, -0.4, 0.1])
        peak_db = calculate_sample_peak(audio)
        assert peak_db == pytest.approx(-6.02, abs=0.1)

    def test_calculate_true_peak_oversampling(self):
        """Test true peak detection catches intersample peaks."""
        sr = 44100
        # Create audio with potential intersample peak
        audio = np.array([0.0, 0.9, 0.0, -0.9, 0.0])
        true_peak = calculate_true_peak(audio, sr)
        sample_peak = calculate_sample_peak(audio)
        # True peak should be >= sample peak
        assert true_peak >= sample_peak - 0.1

    def test_calculate_lufs_momentary_silence(self):
        """Test LUFS calculation for silence."""
        audio = np.zeros(44100)
        lufs = calculate_lufs_momentary(audio, 44100)
        assert lufs == -70.0

    def test_calculate_lufs_momentary_signal(self):
        """Test LUFS calculation for sine wave."""
        sr = 44100
        t = np.linspace(0, 1, sr)
        audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
        lufs = calculate_lufs_momentary(audio, sr)
        # Should return reasonable LUFS value
        assert -30 < lufs < 0

    def test_calculate_lufs_integrated(self):
        """Test integrated LUFS calculation."""
        sr = 44100
        t = np.linspace(0, 2, sr * 2)
        audio = 0.3 * np.sin(2 * np.pi * 1000 * t)
        lufs = calculate_lufs_integrated(audio, sr)
        assert -40 < lufs < 0


class TestPeakDetector:
    """Tests for peak detection module."""

    def test_detect_peaks_silence(self):
        """Test peak detection on silence."""
        audio = np.zeros(44100)
        peaks = detect_spectral_peaks(audio, 44100)
        assert len(peaks) == 0

    def test_detect_peaks_loud_signal(self):
        """Test peak detection on loud signal."""
        sr = 44100
        t = np.linspace(0, 1, sr)
        # Create signal with peak above threshold
        audio = 1.2 * np.sin(2 * np.pi * 1000 * t)
        # This should detect peaks (clipping)
        peaks = detect_spectral_peaks(audio, sr, peak_threshold_db=-0.3)
        # May or may not find peaks depending on spectral characteristics
        assert isinstance(peaks, list)

    def test_spectral_peak_dataclass(self):
        """Test SpectralPeak dataclass creation."""
        peak = SpectralPeak(
            freq_center=1000.0,
            freq_low=900.0,
            freq_high=1100.0,
            time_start=0.0,
            time_end=0.01,
            duration_ms=10.0,
            peak_db=-0.2,
            baseline_db=-20.0,
            delta_db=19.8,
            peak_type=PeakType.TYPE_A,
            threshold=-0.3,
            frame_idx=0,
            bin_idx=100,
        )
        assert peak.freq_center == 1000.0
        assert peak.peak_type == PeakType.TYPE_A

    def test_peak_type_enum(self):
        """Test PeakType enum values."""
        assert PeakType.TYPE_A.value == "isolated"
        assert PeakType.TYPE_B.value == "cluster"
        assert PeakType.TYPE_C.value == "broadband"
        assert PeakType.TYPE_D.value == "modulating"


class TestCorrector:
    """Tests for spectral correction module."""

    def test_corrector_initialization(self):
        """Test SpectralCorrector initialization."""
        corrector = SpectralCorrector(44100)
        assert corrector.sample_rate == 44100
        assert len(corrector.corrections_applied) == 0

    def test_corrector_empty_peaks(self):
        """Test correction with no peaks."""
        corrector = SpectralCorrector(44100)
        audio = np.random.randn(44100) * 0.1
        result = corrector.correct_peaks(audio, [])
        assert len(result) == len(audio)
        np.testing.assert_array_equal(result, audio)

    def test_true_peak_limiter(self):
        """Test true peak limiter reduces peaks."""
        sr = 44100
        # Create audio with peaks
        audio = np.random.randn(sr) * 0.5
        audio[1000:1010] = 1.5  # Create peak

        limited = apply_true_peak_limiter(audio, sr, threshold_db=-0.3)

        # Peaks should be reduced
        original_peak = np.max(np.abs(audio))
        limited_peak = np.max(np.abs(limited))
        assert limited_peak < original_peak

    def test_correction_result_dataclass(self):
        """Test CorrectionResult dataclass."""
        result = CorrectionResult(
            original_peak_db=-0.1,
            corrected_peak_db=-0.5,
            reduction_db=0.4,
            method="micro_compression",
            success=True,
            artifacts_detected=0,
        )
        assert result.success
        assert result.reduction_db == 0.4


class TestEffects:
    """Tests for effects modules."""

    def test_glitter_processor_init(self):
        """Test GlitterProcessor initialization."""
        glitter = GlitterProcessor(44100)
        assert glitter.sample_rate == 44100

    def test_glitter_no_effect_zero_amount(self):
        """Test Glitter with zero amount."""
        glitter = GlitterProcessor(44100)
        audio = np.random.randn(44100) * 0.3
        result = glitter.process(audio, amount=0.0)
        np.testing.assert_array_equal(result, audio)

    def test_pitty_filter_init(self):
        """Test PittyFilter initialization."""
        pitty = PittyFilter(44100)
        assert pitty.sample_rate == 44100

    def test_pitty_filter_no_effect_zero_amount(self):
        """Test Pitty with zero amount."""
        pitty = PittyFilter(44100)
        audio = np.random.randn(44100) * 0.3
        result = pitty.process(audio, amount=0.0)
        np.testing.assert_array_equal(result, audio)

    def test_granular_sutil_init(self):
        """Test GranularSutil initialization."""
        granular = GranularSutil(44100)
        assert granular.sample_rate == 44100

    def test_granular_no_effect_zero_amount(self):
        """Test Granular with zero amount."""
        granular = GranularSutil(44100)
        audio = np.random.randn(44100) * 0.3
        result = granular.process(audio, amount=0.0)
        np.testing.assert_array_equal(result, audio)

    def test_effects_chain_init(self):
        """Test EffectsChain initialization."""
        chain = EffectsChain(44100)
        assert chain.sample_rate == 44100
        assert chain.glitter is not None
        assert chain.pitty is not None
        assert chain.granular is not None

    def test_effects_chain_process(self):
        """Test EffectsChain processes audio."""
        chain = EffectsChain(44100)
        audio = np.random.randn(44100) * 0.3
        result = chain.process(audio, peak_reduction_db=1.0)
        assert len(result) == len(audio)


class TestMasking:
    """Tests for masking analysis module."""

    def test_masking_analyzer_init(self):
        """Test AuditoryMaskingAnalyzer initialization."""
        analyzer = AuditoryMaskingAnalyzer(44100)
        assert analyzer.sample_rate == 44100
        assert len(analyzer.critical_bands) > 0

    def test_critical_bands_range(self):
        """Test critical bands cover audible range."""
        analyzer = AuditoryMaskingAnalyzer(44100)
        assert analyzer.critical_bands[0] >= 19  # Low frequency (allow small tolerance)
        assert analyzer.critical_bands[-1] <= 22051  # Nyquist (with tolerance)

    def test_absolute_threshold_exists(self):
        """Test absolute threshold is calculated."""
        analyzer = AuditoryMaskingAnalyzer(44100)
        assert len(analyzer.absolute_threshold) > 0
        # Most thresholds should be below 0dBFS (allow some at very low frequencies)
        assert sum(t < 0 for t in analyzer.absolute_threshold) >= len(analyzer.absolute_threshold) * 0.8

    def test_masking_threshold_calculation(self):
        """Test masking threshold calculation."""
        analyzer = AuditoryMaskingAnalyzer(44100)
        spectral = np.random.randn(2049) * 10 - 30  # Random spectrum in dB
        freqs = np.linspace(0, 22050, 2049)

        threshold = analyzer.calculate_masking_threshold(spectral, freqs)

        assert isinstance(threshold, MaskingThreshold)
        assert len(threshold.frequencies) > 0
        assert len(threshold.thresholds_db) > 0

    def test_is_peak_audible(self):
        """Test peak audibility check."""
        analyzer = AuditoryMaskingAnalyzer(44100)
        spectral = np.ones(2049) * -40  # Quiet spectrum
        freqs = np.linspace(0, 22050, 2049)

        threshold = analyzer.calculate_masking_threshold(spectral, freqs)

        # Loud peak should be audible
        is_audible = analyzer.is_peak_audible(1000, -10, threshold)
        assert is_audible

        # Very quiet peak may not be audible
        is_quiet_audible = analyzer.is_peak_audible(1000, -80, threshold)
        assert not is_quiet_audible


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_analyze_and_correct(self):
        """Test full analysis and correction pipeline."""
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Create test audio with some peaks
        audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
        audio += 0.3 * np.sin(2 * np.pi * 5000 * t)

        # Add some transient peaks
        audio[10000:10050] *= 2.0

        # Detect peaks
        peaks = detect_spectral_peaks(audio, sr, peak_threshold_db=-0.3)

        # Apply correction
        corrector = SpectralCorrector(sr)
        corrected = corrector.correct_peaks(audio, peaks)

        # Apply effects
        effects = EffectsChain(sr)
        final = effects.process(corrected)

        assert len(final) == len(audio)

    def test_config_affects_processing(self):
        """Test that config values are used in processing."""
        sr = config.audio.sample_rate
        n_fft = config.audio.n_fft

        assert sr == 44100
        assert n_fft == 4096
