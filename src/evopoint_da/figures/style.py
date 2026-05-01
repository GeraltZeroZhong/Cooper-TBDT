"""Shared visual style hooks for Cooper-TBDT figures."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FigureStyle:
    name: str
    palette: dict[str, str] = field(default_factory=dict)
    font_family: str = "DejaVu Sans"
    base_font_size: float = 8.0
    axis_label_size: float = 8.5
    tick_label_size: float = 7.0
    title_size: float = 9.0
    legend_size: float = 7.0
    panel_label_size: float = 11.0
    line_width: float = 1.15
    grid_width: float = 0.65
    grid_alpha: float = 0.78
    point_size: float = 22.0
    dpi: int = 300


STYLES: dict[str, FigureStyle] = {
    "cooper": FigureStyle(
        name="cooper",
        palette={
            "navy": "#19324D",
            "peacock": "#0F6C7A",
            "aurora": "#2FA7C9",
            "celadon": "#A8C8C1",
            "tech": "#3B6FF5",
            "glacier": "#DCEFFC",
            "sakura": "#F6C8D7",
            "instagram": "#E66AA3",
            "blossom": "#FF9BB8",
            "raw": "#19324D",
            "primary": "#3B6FF5",
            "model_blend": "#3B6FF5",
            "model_seed": "#2FA7C9",
            "blend": "#2FA7C9",
            "baseline": "#0F6C7A",
            "gold": "#3B6FF5",
            "silver": "#A8C8C1",
            "bronze": "#F6C8D7",
            "train": "#0F6C7A",
            "val": "#A8C8C1",
            "test": "#E66AA3",
            "improved": "#0F6C7A",
            "worsened": "#E66AA3",
            "neutral": "#A8C8C1",
            "ci": "#DCEFFC",
            "grid": "#DCEFFC",
            "text": "#000000",
            "reference": "#000000",
            "anm": "#2FA7C9",
            "gnm": "#A8C8C1",
            "iupred": "#A8C8C1",
            "p2rank": "#FF9BB8",
            "fpocket": "#F6C8D7",
            "protcross": "#E66AA3",
            "rsa": "#A8C8C1",
            "paper": "#FFFFFF",
            "panel_bg": "#FFFFFF",
            "muted_bg": "#DCEFFC",
            "soft_gray": "#DCEFFC",
            "bin_lt_0p5": "#DCEFFC",
            "bin_0p5_to_1": "#A8C8C1",
            "bin_1_to_2": "#2FA7C9",
            "bin_2_to_5": "#3B6FF5",
            "bin_ge_5": "#E66AA3",
        },
    )
}


def get_style(name: str) -> FigureStyle:
    try:
        return STYLES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(STYLES))
        raise ValueError(f"Unknown figure style {name!r}. Available styles: {choices}") from exc


def apply_style(style: FigureStyle) -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": style.font_family,
            "font.size": style.base_font_size,
            "axes.labelsize": style.axis_label_size,
            "axes.titlesize": style.title_size,
            "axes.titleweight": "semibold",
            "axes.titlepad": 7.0,
            "axes.linewidth": style.line_width,
            "axes.edgecolor": style.palette["text"],
            "axes.labelcolor": style.palette["text"],
            "axes.facecolor": style.palette["panel_bg"],
            "xtick.labelsize": style.tick_label_size,
            "ytick.labelsize": style.tick_label_size,
            "xtick.color": style.palette["text"],
            "ytick.color": style.palette["text"],
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "legend.fontsize": style.legend_size,
            "legend.frameon": False,
            "figure.facecolor": style.palette["paper"],
            "figure.dpi": style.dpi,
            "savefig.dpi": style.dpi,
            "savefig.facecolor": style.palette["paper"],
            "savefig.edgecolor": style.palette["paper"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_panel_label(ax: object, label: str, style: FigureStyle, *, x: float = -0.12, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=style.panel_label_size,
        fontweight="bold",
        color=style.palette["text"],
        va="bottom",
        ha="left",
    )


def clean_axis(ax: object, style: FigureStyle, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=style.palette["grid"], linewidth=style.grid_width, alpha=style.grid_alpha)
    ax.set_axisbelow(True)


def add_note_box(
    ax: object,
    text: str,
    style: FigureStyle,
    *,
    x: float = 0.02,
    y: float = 0.96,
    ha: str = "left",
    va: str = "top",
    size: float | None = None,
) -> None:
    """Add a small unobtrusive note box inside an axes."""

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=style.legend_size if size is None else size,
        color=style.palette["text"],
        bbox={
            "facecolor": style.palette["paper"],
            "edgecolor": "none",
            "alpha": 0.82,
            "pad": 2.4,
        },
    )


def figure_output_dir(root: str | Path, out_name: str) -> Path:
    """Return and create the canonical directory for a figure."""

    out_dir = Path(root) / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def parse_formats(value: str) -> list[str]:
    """Parse a comma-separated output-format string."""

    formats = [item.strip().lower().lstrip(".") for item in value.split(",") if item.strip()]
    if not formats:
        raise ValueError("At least one output format is required")
    return formats


def lighten_color(hex_color: str, amount: float = 0.5) -> str:
    """Return a lighter version of a hex color."""

    color = hex_color.lstrip("#")
    red, green, blue = (int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    lightness = 1.0 - amount * (1.0 - lightness)
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{round(red * 255):02X}{round(green * 255):02X}{round(blue * 255):02X}"


def save_figure(fig: object, out_stem: str | Path, *, formats: list[str], dpi: int) -> list[Path]:
    stem = Path(out_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        suffix = fmt.lower().lstrip(".")
        out_path = stem.with_suffix(f".{suffix}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        written.append(out_path)
    return written
