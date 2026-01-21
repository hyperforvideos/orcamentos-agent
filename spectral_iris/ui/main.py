"""Main UI application for Spectral Iris.

This module provides the main window and navigation for the
Spectral Iris audio processing application with a cyber-themed interface.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    import dearpygui.dearpygui as dpg
    DPG_AVAILABLE = True
except ImportError:
    DPG_AVAILABLE = False
    dpg = None  # type: ignore

from ..config import config
from .theme import THEME, apply_dpg_theme, CYBER_LABELS, AGENT_LABELS


@dataclass
class SessionData:
    """Current session data for the UI."""

    file_path: str = ""
    sample_rate: int = 44100
    bit_depth: int = 24
    channels: int = 2
    duration_seconds: float = 0.0
    lufs_initial: float = -23.0
    peak_initial: float = -0.3
    lufs_current: float = -23.0
    peak_current: float = -0.3
    lufs_momentary: float = -23.0
    correction_progress: float = 0.0
    peaks_reduced: int = 0
    lufs_change: float = 0.0
    artifacts_detected: int = 0
    is_processing: bool = False
    is_loaded: bool = False


@dataclass
class VisualizationData:
    """Data for real-time visualizations."""

    # Spectrogram data (time x freq)
    spectrogram: Optional[NDArray] = None
    # Spectrum analyzer data (freq bins)
    spectrum: Optional[NDArray] = None
    # Frequency labels
    frequencies: Optional[NDArray] = None
    # Time labels
    times: Optional[NDArray] = None
    # LUFS history
    lufs_history: List[float] = field(default_factory=list)
    # Peak history
    peak_history: List[float] = field(default_factory=list)


class SpectralIrisUI:
    """Main UI application for Spectral Iris.

    Implements a cyber-themed interface with multiple panels:
    - Login/Status (AI-20 style)
    - Cyroscope (real-time metrics)
    - Sound Chart (spectrogram/spectrum)
    - Core (agent processing status)
    - Micro Control (granular adjustments)
    """

    def __init__(self) -> None:
        """Initialize the Spectral Iris UI."""
        if not DPG_AVAILABLE:
            raise ImportError(
                "Dear PyGui is required for the UI. "
                "Install with: pip install dearpygui"
            )

        self.session = SessionData()
        self.viz_data = VisualizationData()

        # UI element IDs
        self.window_ids: Dict[str, int] = {}
        self.widget_ids: Dict[str, int] = {}

        # Callbacks
        self.on_file_load: Optional[Callable[[str], None]] = None
        self.on_process_start: Optional[Callable[[], None]] = None
        self.on_process_cancel: Optional[Callable[[], None]] = None

        # Threading for UI updates
        self._update_thread: Optional[threading.Thread] = None
        self._running = False

    def setup(self) -> None:
        """Set up Dear PyGui context and create windows."""
        dpg.create_context()

        # Configure viewport
        dpg.create_viewport(
            title="SPECTRAL IRIS // Sonic Cybernetics",
            width=config.ui.window_width,
            height=config.ui.window_height,
            min_width=config.ui.min_window_width,
            min_height=config.ui.min_window_height,
        )

        # Apply theme
        theme_id = apply_dpg_theme()
        dpg.bind_theme(theme_id)

        # Create main window
        self._create_main_window()

        dpg.setup_dearpygui()
        dpg.show_viewport()

    def _create_main_window(self) -> None:
        """Create the main application window with all panels."""
        with dpg.window(label="Main", tag="main_window", no_title_bar=True,
                        no_resize=True, no_move=True, no_scrollbar=True):
            # Header
            self._create_header()

            # Main content area with panels
            with dpg.group(horizontal=True):
                # Left column: Status + Cyroscope
                with dpg.child_window(width=400, height=-50, border=True):
                    self._create_status_panel()
                    dpg.add_separator()
                    self._create_cyroscope_panel()

                # Center column: Visualizations
                with dpg.child_window(width=-300, height=-50, border=True):
                    self._create_visualization_panel()

                # Right column: Control + Health
                with dpg.child_window(width=-1, height=-50, border=True):
                    self._create_control_panel()
                    dpg.add_separator()
                    self._create_health_panel()

            # Footer
            self._create_footer()

        # Set main window as primary
        dpg.set_primary_window("main_window", True)

    def _create_header(self) -> None:
        """Create the header section."""
        with dpg.group(horizontal=True):
            dpg.add_text(CYBER_LABELS["title"], color=THEME.accent_green)
            dpg.add_spacer(width=20)
            dpg.add_text(
                CYBER_LABELS["subtitle"],
                color=THEME.text_secondary
            )

        dpg.add_separator()
        dpg.add_spacer(height=5)

    def _create_status_panel(self) -> None:
        """Create the session status panel (AI-20 style)."""
        dpg.add_text("// SESSION MANIFEST", color=THEME.accent_blue)
        dpg.add_spacer(height=5)

        # Status indicators
        with dpg.group(horizontal=True):
            dpg.add_text("STATUS:", color=THEME.text_dim)
            self.widget_ids["status_indicator"] = dpg.add_text(
                "AWAITING FILE",
                color=THEME.status_warning
            )

        # Scanning/progress bar
        with dpg.group(horizontal=True):
            dpg.add_text("SCANNING:", color=THEME.text_dim)
            self.widget_ids["scan_progress"] = dpg.add_progress_bar(
                default_value=0.0,
                width=200
            )
            self.widget_ids["scan_percent"] = dpg.add_text("0%", color=THEME.text_secondary)

        dpg.add_spacer(height=10)

        # Session manifest table
        with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True,
                       borders_innerV=True, borders_outerV=True,
                       row_background=True, width=-1):
            dpg.add_table_column(label="PARAMETER", width_fixed=True, init_width_or_weight=120)
            dpg.add_table_column(label="VALUE")

            # File path
            with dpg.table_row():
                dpg.add_text("File Path", color=THEME.text_secondary)
                self.widget_ids["manifest_path"] = dpg.add_text("---")

            # Sample rate
            with dpg.table_row():
                dpg.add_text("Sample Rate", color=THEME.text_secondary)
                self.widget_ids["manifest_sr"] = dpg.add_text("---")

            # Bit depth
            with dpg.table_row():
                dpg.add_text("Bit Depth", color=THEME.text_secondary)
                self.widget_ids["manifest_bits"] = dpg.add_text("---")

            # Channels
            with dpg.table_row():
                dpg.add_text("Channels", color=THEME.text_secondary)
                self.widget_ids["manifest_channels"] = dpg.add_text("---")

            # Duration
            with dpg.table_row():
                dpg.add_text("Duration", color=THEME.text_secondary)
                self.widget_ids["manifest_duration"] = dpg.add_text("---")

            # Initial LUFS
            with dpg.table_row():
                dpg.add_text("LUFS Initial", color=THEME.text_secondary)
                self.widget_ids["manifest_lufs"] = dpg.add_text("---")

            # Initial Peak
            with dpg.table_row():
                dpg.add_text("Peak Initial", color=THEME.text_secondary)
                self.widget_ids["manifest_peak"] = dpg.add_text("---")

        dpg.add_spacer(height=10)

        # Action buttons
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="LOAD FILE",
                width=100,
                callback=self._on_load_file_click
            )
            dpg.add_button(
                label="PERSISTENT",
                width=100,
                callback=self._on_persistent_click
            )
            dpg.add_button(
                label="CANCEL",
                width=80,
                callback=self._on_cancel_click
            )

    def _create_cyroscope_panel(self) -> None:
        """Create the Cyroscope monitoring panel."""
        dpg.add_text(CYBER_LABELS["cyroscope_header"], color=THEME.accent_green)
        dpg.add_spacer(height=10)

        # Magnetic (always 0 - digital system)
        with dpg.group(horizontal=True):
            dpg.add_text("Magnetic:", color=THEME.text_dim, indent=10)
            dpg.add_text("0.000000000", color=THEME.accent_cyan)

        # Sound vibration (LUFS Momentary)
        with dpg.group(horizontal=True):
            dpg.add_text("Sound vibration:", color=THEME.text_dim, indent=10)
            self.widget_ids["cyro_lufs"] = dpg.add_text(
                "-23.0 LUFS",
                color=THEME.accent_green
            )

        # Core state (True Peak)
        with dpg.group(horizontal=True):
            dpg.add_text("Core state:", color=THEME.text_dim, indent=10)
            self.widget_ids["cyro_peak"] = dpg.add_text(
                "-0.3 dBTP",
                color=THEME.accent_yellow
            )

        dpg.add_spacer(height=15)

        # LUFS History plot
        with dpg.plot(label="LUFS Momentary", height=100, width=-1, no_menus=True):
            dpg.add_plot_axis(dpg.mvXAxis, label="", no_tick_labels=True)
            with dpg.plot_axis(dpg.mvYAxis, label="LUFS"):
                self.widget_ids["lufs_plot"] = dpg.add_line_series(
                    [], [], label="LUFS"
                )
                dpg.set_axis_limits(dpg.last_item(), -60, 0)

    def _create_visualization_panel(self) -> None:
        """Create the main visualization panel with spectrogram and spectrum."""
        # Sound chart header (Spectrogram)
        dpg.add_text(CYBER_LABELS["sound_chart_header"], color=THEME.accent_green)
        dpg.add_text("// Temporal Activity", color=THEME.text_dim)

        # Placeholder for spectrogram
        with dpg.drawlist(width=-1, height=200, tag="spectrogram_canvas"):
            # Background
            dpg.draw_rectangle(
                (0, 0), (dpg.get_item_width("spectrogram_canvas") or 600, 200),
                fill=THEME.bg_darkest
            )
            # Placeholder text
            dpg.draw_text(
                (10, 90), "// Awaiting audio data...",
                color=THEME.text_dim, size=14
            )

        dpg.add_spacer(height=10)

        # Spectrum analyzer header
        dpg.add_text(CYBER_LABELS["spectrum_header"], color=THEME.accent_green)
        dpg.add_text("// Frequency Analysis", color=THEME.text_dim)

        # Spectrum analyzer plot
        with dpg.plot(label="", height=180, width=-1, no_menus=True):
            dpg.add_plot_axis(dpg.mvXAxis, label="Frequency (Hz)", log_scale=True)
            dpg.set_axis_limits(dpg.last_item(), 20, 20000)

            with dpg.plot_axis(dpg.mvYAxis, label="dBFS"):
                dpg.set_axis_limits(dpg.last_item(), -80, 0)
                self.widget_ids["spectrum_plot"] = dpg.add_line_series(
                    [20, 100, 1000, 10000, 20000],
                    [-60, -50, -40, -50, -60],
                    label="Spectrum"
                )

        dpg.add_spacer(height=10)

        # Core / Agent Processing
        dpg.add_text(CYBER_LABELS["core_header"], color=THEME.accent_green)
        dpg.add_text("## Spectral Agent Processing", color=THEME.text_secondary)

        # Agent status table
        with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True,
                       borders_innerV=True, borders_outerV=True, width=-1):
            dpg.add_table_column(label="AGENT", width_fixed=True, init_width_or_weight=150)
            dpg.add_table_column(label="STATUS", width_fixed=True, init_width_or_weight=80)
            dpg.add_table_column(label="LOAD")

            for agent_name, status in AGENT_LABELS:
                with dpg.table_row():
                    dpg.add_text(agent_name, color=THEME.text_primary)
                    status_color = THEME.status_online if status == "online" else THEME.status_offline
                    dpg.add_text(f"[{status}]", color=status_color)
                    # Load bar (placeholder)
                    dpg.add_progress_bar(default_value=0.3, width=-1)

    def _create_control_panel(self) -> None:
        """Create the micro control panel with frequency band sliders."""
        dpg.add_text(CYBER_LABELS["control_header"], color=THEME.accent_green)
        dpg.add_text("// Frequency Band Control", color=THEME.text_dim)
        dpg.add_spacer(height=10)

        # Frequency band sliders (1/3 octave from 20Hz to 20kHz)
        freq_bands = [
            "20", "25", "31.5", "40", "50", "63", "80", "100",
            "125", "160", "200", "250", "315", "400", "500", "630",
            "800", "1k", "1.25k", "1.6k", "2k", "2.5k", "3.15k", "4k",
            "5k", "6.3k", "8k", "10k", "12.5k", "16k", "20k"
        ]

        # Show subset for UI space
        display_bands = freq_bands[::3]  # Every 3rd band

        for freq in display_bands:
            with dpg.group(horizontal=True):
                dpg.add_text(f"{freq:>6}:", color=THEME.text_dim)
                dpg.add_slider_float(
                    default_value=0.0,
                    min_value=-12.0,
                    max_value=6.0,
                    width=-50,
                    format="%.1f dB"
                )

        dpg.add_spacer(height=10)

        # Global controls
        dpg.add_text("// Global Parameters", color=THEME.text_dim)

        with dpg.group(horizontal=True):
            dpg.add_text("Threshold:", color=THEME.text_secondary)
            self.widget_ids["threshold_slider"] = dpg.add_slider_float(
                default_value=-0.3,
                min_value=-3.0,
                max_value=-0.1,
                width=-1,
                format="%.1f dBTP"
            )

        with dpg.group(horizontal=True):
            dpg.add_text("Aggressive:", color=THEME.text_secondary)
            self.widget_ids["aggression_slider"] = dpg.add_slider_float(
                default_value=0.5,
                min_value=0.0,
                max_value=1.0,
                width=-1,
                format="%.0f%%"
            )

    def _create_health_panel(self) -> None:
        """Create the health status panel."""
        dpg.add_text(CYBER_LABELS["health_header"], color=THEME.accent_blue)
        dpg.add_spacer(height=5)

        # Correction status
        with dpg.group(horizontal=True):
            dpg.add_text("Status:", color=THEME.text_dim)
            self.widget_ids["health_status"] = dpg.add_text(
                "idle",
                color=THEME.text_secondary
            )

        # Progress bar
        self.widget_ids["correction_progress"] = dpg.add_progress_bar(
            default_value=0.0,
            width=-1
        )

        dpg.add_spacer(height=5)

        # Metrics
        with dpg.group():
            with dpg.group(horizontal=True):
                dpg.add_text("Peaks Reduced:", color=THEME.text_dim)
                self.widget_ids["health_peaks"] = dpg.add_text("0", color=THEME.accent_green)

            with dpg.group(horizontal=True):
                dpg.add_text("LUFS Change:", color=THEME.text_dim)
                self.widget_ids["health_lufs"] = dpg.add_text("+0.0", color=THEME.text_primary)

            with dpg.group(horizontal=True):
                dpg.add_text("Artifacts:", color=THEME.text_dim)
                self.widget_ids["health_artifacts"] = dpg.add_text("0", color=THEME.accent_green)

        dpg.add_spacer(height=10)

        # Process button
        dpg.add_button(
            label="START PROCESSING",
            width=-1,
            height=40,
            callback=self._on_process_click
        )

    def _create_footer(self) -> None:
        """Create the footer section."""
        dpg.add_separator()
        dpg.add_spacer(height=2)

        with dpg.group(horizontal=True):
            dpg.add_text(CYBER_LABELS["footer"], color=THEME.text_dim)
            dpg.add_spacer()
            self.widget_ids["footer_time"] = dpg.add_text(
                time.strftime("%H:%M:%S"),
                color=THEME.text_dim
            )

    def _on_load_file_click(self, sender: int, app_data: Any) -> None:
        """Handle load file button click."""
        # Show file dialog
        if dpg.does_item_exist("file_dialog"):
            dpg.delete_item("file_dialog")

        with dpg.file_dialog(
            directory_selector=False,
            show=True,
            callback=self._on_file_selected,
            tag="file_dialog",
            width=700,
            height=400
        ):
            dpg.add_file_extension(".wav", color=(0, 255, 0, 255))
            dpg.add_file_extension(".flac", color=(0, 200, 255, 255))
            dpg.add_file_extension(".mp3", color=(255, 200, 0, 255))
            dpg.add_file_extension(".aiff", color=(0, 255, 200, 255))

    def _on_file_selected(self, sender: int, app_data: Dict) -> None:
        """Handle file selection."""
        if app_data and "file_path_name" in app_data:
            file_path = app_data["file_path_name"]
            self.load_file(file_path)

    def _on_persistent_click(self, sender: int, app_data: Any) -> None:
        """Handle persistent/save button click."""
        # TODO: Implement profile saving
        pass

    def _on_cancel_click(self, sender: int, app_data: Any) -> None:
        """Handle cancel button click."""
        if self.on_process_cancel:
            self.on_process_cancel()
        self.session.is_processing = False
        self._update_status("CANCELLED")

    def _on_process_click(self, sender: int, app_data: Any) -> None:
        """Handle process button click."""
        if self.session.is_loaded and self.on_process_start:
            self.on_process_start()

    def load_file(self, file_path: str) -> None:
        """Load an audio file and update session data.

        Args:
            file_path: Path to the audio file
        """
        self.session.file_path = file_path
        self._update_status("LOADING...")

        if self.on_file_load:
            self.on_file_load(file_path)

    def update_session(self, **kwargs: Any) -> None:
        """Update session data and refresh UI.

        Args:
            **kwargs: Session data fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self.session, key):
                setattr(self.session, key, value)

        self._refresh_manifest()

    def _update_status(self, status: str) -> None:
        """Update the status indicator."""
        if "status_indicator" in self.widget_ids:
            color = THEME.status_online
            if "ERROR" in status or "CANCEL" in status:
                color = THEME.status_error
            elif "AWAIT" in status or "LOAD" in status:
                color = THEME.status_warning
            elif "PROCESS" in status:
                color = THEME.status_processing

            dpg.set_value(self.widget_ids["status_indicator"], status)
            dpg.configure_item(self.widget_ids["status_indicator"], color=color)

    def _refresh_manifest(self) -> None:
        """Refresh the session manifest display."""
        s = self.session

        if "manifest_path" in self.widget_ids:
            # Truncate path for display
            display_path = s.file_path[-40:] if len(s.file_path) > 40 else s.file_path
            dpg.set_value(self.widget_ids["manifest_path"], display_path or "---")

        if "manifest_sr" in self.widget_ids:
            dpg.set_value(self.widget_ids["manifest_sr"], f"{s.sample_rate} Hz")

        if "manifest_bits" in self.widget_ids:
            dpg.set_value(self.widget_ids["manifest_bits"], f"{s.bit_depth} bit")

        if "manifest_channels" in self.widget_ids:
            dpg.set_value(self.widget_ids["manifest_channels"], str(s.channels))

        if "manifest_duration" in self.widget_ids:
            mins = int(s.duration_seconds // 60)
            secs = s.duration_seconds % 60
            dpg.set_value(self.widget_ids["manifest_duration"], f"{mins}:{secs:05.2f}")

        if "manifest_lufs" in self.widget_ids:
            dpg.set_value(self.widget_ids["manifest_lufs"], f"{s.lufs_initial:.1f} LUFS")

        if "manifest_peak" in self.widget_ids:
            dpg.set_value(self.widget_ids["manifest_peak"], f"{s.peak_initial:.1f} dBTP")

        if s.is_loaded:
            self._update_status("READY")

    def update_realtime_metrics(
        self,
        lufs_momentary: float,
        true_peak: float,
    ) -> None:
        """Update real-time metrics display.

        Args:
            lufs_momentary: Current momentary LUFS
            true_peak: Current true peak in dBTP
        """
        self.session.lufs_momentary = lufs_momentary
        self.session.peak_current = true_peak

        if "cyro_lufs" in self.widget_ids:
            dpg.set_value(
                self.widget_ids["cyro_lufs"],
                f"{lufs_momentary:.1f} LUFS"
            )

        if "cyro_peak" in self.widget_ids:
            peak_color = THEME.accent_green
            if true_peak > -1.0:
                peak_color = THEME.accent_yellow
            if true_peak > -0.3:
                peak_color = THEME.accent_red

            dpg.set_value(
                self.widget_ids["cyro_peak"],
                f"{true_peak:.1f} dBTP"
            )
            dpg.configure_item(self.widget_ids["cyro_peak"], color=peak_color)

        # Update LUFS history plot
        self.viz_data.lufs_history.append(lufs_momentary)
        if len(self.viz_data.lufs_history) > 100:
            self.viz_data.lufs_history = self.viz_data.lufs_history[-100:]

        if "lufs_plot" in self.widget_ids:
            x_data = list(range(len(self.viz_data.lufs_history)))
            dpg.set_value(
                self.widget_ids["lufs_plot"],
                [x_data, self.viz_data.lufs_history]
            )

    def update_progress(
        self,
        progress: float,
        peaks_reduced: int = 0,
        lufs_change: float = 0.0,
        artifacts: int = 0,
    ) -> None:
        """Update correction progress.

        Args:
            progress: Progress value 0.0-1.0
            peaks_reduced: Number of peaks reduced
            lufs_change: LUFS change
            artifacts: Number of artifacts detected
        """
        self.session.correction_progress = progress
        self.session.peaks_reduced = peaks_reduced
        self.session.lufs_change = lufs_change
        self.session.artifacts_detected = artifacts

        if "correction_progress" in self.widget_ids:
            dpg.set_value(self.widget_ids["correction_progress"], progress)

        if "health_status" in self.widget_ids:
            status = "correcting..." if progress < 1.0 else "complete"
            dpg.set_value(self.widget_ids["health_status"], status)

        if "health_peaks" in self.widget_ids:
            dpg.set_value(self.widget_ids["health_peaks"], str(peaks_reduced))

        if "health_lufs" in self.widget_ids:
            sign = "+" if lufs_change >= 0 else ""
            dpg.set_value(self.widget_ids["health_lufs"], f"{sign}{lufs_change:.1f}")

        if "health_artifacts" in self.widget_ids:
            color = THEME.accent_green if artifacts == 0 else THEME.accent_red
            dpg.set_value(self.widget_ids["health_artifacts"], str(artifacts))
            dpg.configure_item(self.widget_ids["health_artifacts"], color=color)

        if "scan_progress" in self.widget_ids:
            dpg.set_value(self.widget_ids["scan_progress"], progress)

        if "scan_percent" in self.widget_ids:
            dpg.set_value(self.widget_ids["scan_percent"], f"{int(progress * 100)}%")

    def run(self) -> None:
        """Run the main event loop."""
        self._running = True

        # Start update thread for clock
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()

        # Main render loop
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()

        self._running = False

    def _update_loop(self) -> None:
        """Background update loop for UI elements."""
        while self._running:
            # Update clock
            if "footer_time" in self.widget_ids:
                try:
                    dpg.set_value(
                        self.widget_ids["footer_time"],
                        time.strftime("%H:%M:%S")
                    )
                except Exception:
                    pass

            time.sleep(1.0)

    def cleanup(self) -> None:
        """Clean up Dear PyGui context."""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=2.0)
        dpg.destroy_context()


def launch_ui() -> SpectralIrisUI:
    """Launch the Spectral Iris UI application.

    Returns:
        The UI instance
    """
    ui = SpectralIrisUI()
    ui.setup()
    return ui
