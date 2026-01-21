"""UI Theme configuration for Spectral Iris.

This module provides the cyber-themed color scheme and styling
for the Dear PyGui interface, matching the AI-20/Cyroscope reference design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

# Color type: RGBA tuple with values 0-255
Color = Tuple[int, int, int, int]


@dataclass
class CyberTheme:
    """Cyber-themed color palette for Spectral Iris UI.

    Based on the reference images (AI-20, Cyroscope, Sound chart, Core, DBTL Control)
    with dark backgrounds, terminal-green/blue accents, and monospace typography.
    """

    # Primary backgrounds
    bg_darkest: Color = (10, 10, 10, 255)
    bg_primary: Color = (15, 15, 15, 255)
    bg_secondary: Color = (20, 20, 22, 255)
    bg_panel: Color = (25, 28, 30, 255)
    bg_elevated: Color = (35, 38, 42, 255)

    # Borders and lines
    border_dark: Color = (40, 42, 45, 255)
    border_light: Color = (60, 65, 70, 255)
    border_accent: Color = (0, 180, 120, 255)

    # Text colors
    text_primary: Color = (220, 225, 230, 255)
    text_secondary: Color = (140, 145, 150, 255)
    text_dim: Color = (80, 85, 90, 255)
    text_bright: Color = (255, 255, 255, 255)

    # Accent colors (terminal/cyber style)
    accent_green: Color = (0, 200, 100, 255)
    accent_green_dim: Color = (0, 120, 60, 255)
    accent_blue: Color = (0, 160, 220, 255)
    accent_blue_dim: Color = (0, 80, 140, 255)
    accent_cyan: Color = (0, 220, 200, 255)
    accent_yellow: Color = (220, 200, 30, 255)
    accent_red: Color = (220, 60, 60, 255)
    accent_orange: Color = (230, 140, 40, 255)
    accent_magenta: Color = (200, 80, 200, 255)

    # Status colors
    status_online: Color = (0, 200, 100, 255)
    status_offline: Color = (120, 120, 120, 255)
    status_warning: Color = (220, 180, 30, 255)
    status_error: Color = (220, 60, 60, 255)
    status_processing: Color = (0, 160, 220, 255)

    # Visualization colors
    spectrum_cold: Color = (0, 100, 200, 255)
    spectrum_neutral: Color = (0, 180, 100, 255)
    spectrum_warm: Color = (220, 180, 30, 255)
    spectrum_hot: Color = (220, 60, 60, 255)

    # Chart/graph colors
    graph_line1: Color = (0, 200, 150, 255)
    graph_line2: Color = (0, 150, 220, 255)
    graph_line3: Color = (200, 100, 200, 255)
    graph_fill: Color = (0, 100, 80, 80)

    # Interactive elements
    button_bg: Color = (35, 40, 45, 255)
    button_bg_hover: Color = (50, 55, 60, 255)
    button_bg_active: Color = (0, 140, 100, 255)
    slider_bg: Color = (30, 32, 35, 255)
    slider_grab: Color = (0, 180, 120, 255)
    input_bg: Color = (20, 22, 25, 255)


# Default theme instance
THEME = CyberTheme()


def color_to_float(color: Color) -> Tuple[float, float, float, float]:
    """Convert 0-255 color to 0-1 float tuple for Dear PyGui.

    Args:
        color: RGBA tuple with values 0-255

    Returns:
        RGBA tuple with values 0.0-1.0
    """
    return (
        color[0] / 255.0,
        color[1] / 255.0,
        color[2] / 255.0,
        color[3] / 255.0,
    )


def get_spectrum_color(
    value: float,
    min_val: float = -60.0,
    max_val: float = 0.0,
) -> Color:
    """Get color for spectrum visualization based on dB value.

    Args:
        value: Value in dB
        min_val: Minimum value (cold)
        max_val: Maximum value (hot)

    Returns:
        RGBA color tuple
    """
    # Normalize to 0-1
    normalized = (value - min_val) / (max_val - min_val)
    normalized = max(0.0, min(1.0, normalized))

    # Color gradient: blue -> green -> yellow -> red
    if normalized < 0.33:
        # Blue to green
        t = normalized / 0.33
        r = int(THEME.spectrum_cold[0] * (1 - t) + THEME.spectrum_neutral[0] * t)
        g = int(THEME.spectrum_cold[1] * (1 - t) + THEME.spectrum_neutral[1] * t)
        b = int(THEME.spectrum_cold[2] * (1 - t) + THEME.spectrum_neutral[2] * t)
    elif normalized < 0.66:
        # Green to yellow
        t = (normalized - 0.33) / 0.33
        r = int(THEME.spectrum_neutral[0] * (1 - t) + THEME.spectrum_warm[0] * t)
        g = int(THEME.spectrum_neutral[1] * (1 - t) + THEME.spectrum_warm[1] * t)
        b = int(THEME.spectrum_neutral[2] * (1 - t) + THEME.spectrum_warm[2] * t)
    else:
        # Yellow to red
        t = (normalized - 0.66) / 0.34
        r = int(THEME.spectrum_warm[0] * (1 - t) + THEME.spectrum_hot[0] * t)
        g = int(THEME.spectrum_warm[1] * (1 - t) + THEME.spectrum_hot[1] * t)
        b = int(THEME.spectrum_warm[2] * (1 - t) + THEME.spectrum_hot[2] * t)

    return (r, g, b, 255)


def apply_dpg_theme() -> int:
    """Apply the cyber theme to Dear PyGui.

    Returns:
        Theme ID that can be bound to the viewport
    """
    try:
        import dearpygui.dearpygui as dpg
    except ImportError:
        return 0

    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            # Window styling
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, THEME.bg_primary)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, THEME.bg_secondary)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, THEME.bg_elevated)
            dpg.add_theme_color(dpg.mvThemeCol_Border, THEME.border_dark)

            # Text
            dpg.add_theme_color(dpg.mvThemeCol_Text, THEME.text_primary)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, THEME.text_dim)

            # Frames and inputs
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, THEME.input_bg)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, THEME.bg_elevated)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, THEME.bg_elevated)

            # Titles
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, THEME.bg_darkest)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, THEME.bg_panel)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed, THEME.bg_darkest)

            # Tabs
            dpg.add_theme_color(dpg.mvThemeCol_Tab, THEME.bg_panel)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, THEME.accent_green_dim)
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, THEME.accent_green)

            # Buttons
            dpg.add_theme_color(dpg.mvThemeCol_Button, THEME.button_bg)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, THEME.button_bg_hover)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, THEME.button_bg_active)

            # Headers
            dpg.add_theme_color(dpg.mvThemeCol_Header, THEME.bg_elevated)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, THEME.accent_green_dim)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, THEME.accent_green)

            # Sliders
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, THEME.slider_grab)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, THEME.accent_green)

            # Progress bar
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, THEME.accent_green)

            # Checkboxes
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, THEME.accent_green)

            # Separators
            dpg.add_theme_color(dpg.mvThemeCol_Separator, THEME.border_dark)

            # Scrollbars
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, THEME.bg_darkest)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, THEME.border_light)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, THEME.accent_green_dim)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, THEME.accent_green)

            # Style adjustments
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 2)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 4)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 4)

    return global_theme


# Monospace-style labels for cyber aesthetic
CYBER_LABELS = {
    "title": "# SPECTRAL IRIS",
    "subtitle": "SPECTRAL PROCESSING CORE v1.0",
    "login_header": "// SYSTEM ACCESS",
    "cyroscope_header": "# 01. Cyroscope",
    "sound_chart_header": "# 02. Sound chart",
    "spectrum_header": "# 03. Sound Spectrum",
    "core_header": "# 04. Core",
    "control_header": "# MICRO CONTROL",
    "health_header": "HEALTH STATUS",
    "footer": "© Spectral Iris // Sonic Cybernetics",
}


# Agent status labels
AGENT_LABELS: List[Tuple[str, str]] = [
    ("01_SpecAnalyzer", "online"),
    ("02_DynDeEsser", "online"),
    ("03_PeakHunter", "online"),
    ("04_RL_Optimizer", "online"),
    ("05_Logger", "online"),
]
