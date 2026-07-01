"""Shared plotting style + palettes for the FLKD3D figure suite."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from flkd_results import FL_ORDER, KD_ORDER  # noqa: E402

# --------------------------------------------------------------------------- #
# Colour system                                                                 #
# --------------------------------------------------------------------------- #
# Family colours — saturated, high-contrast, colour-blind friendly-ish.
FAMILY_COLOR = {
    "baseline": "#EF476F",   # rose
    "fl": "#118AB2",         # cerulean
    "kd": "#06D6A0",         # mint
    "combined": "#7B2FF7",   # violet
}

# A 13-colour qualitative palette for FL strategies (stable mapping).
_FL_HEX = [
    "#E6194B", "#F58231", "#FFD93D", "#BFEF45", "#3CB44B", "#1ABC9C",
    "#42D4F4", "#4363D8", "#911EB4", "#F032E6", "#FABED4", "#9A6324",
    "#808000",
]
FL_COLOR = {m: _FL_HEX[i % len(_FL_HEX)] for i, m in enumerate(FL_ORDER)}

# A 10-colour qualitative palette for KD methods (stable mapping).
_KD_HEX = [
    "#2E86DE", "#10AC84", "#EE5253", "#F368E0", "#FF9F43",
    "#A29BFE", "#01A3A4", "#C44569", "#576574", "#FECA57",
]
KD_COLOR = {m: _KD_HEX[i % len(_KD_HEX)] for i, m in enumerate(KD_ORDER)}

# Sequential colormap for accuracy heatmaps (dark-violet -> hot-orange).
ACC_CMAP = LinearSegmentedColormap.from_list(
    "viv_acc", ["#2B0B5E", "#6A1B9A", "#C2185B", "#F4511E", "#FFC400"])
# Diverging colormap for synergy (anti-synergy red <- 0 -> synergy green).
SYN_CMAP = LinearSegmentedColormap.from_list(
    "viv_syn", ["#B71C1C", "#E57373", "#F5F5F5", "#66BB6A", "#1B5E20"])


def fl_color(m: str) -> str:
    return FL_COLOR.get(m, "#555555")


def kd_color(m: str) -> str:
    return KD_COLOR.get(m, "#555555")


# --------------------------------------------------------------------------- #
# Global rcParams — presentation grade, embeddable PDF text                     #
# --------------------------------------------------------------------------- #
def setup_style():
    # prefer a clean sans if available, otherwise fall back silently
    for fam in ("DejaVu Sans", "Arial", "Helvetica"):
        if any(fam in f.name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = fam
            break
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 21,
        "axes.titleweight": "bold",
        "axes.labelsize": 17,
        "axes.labelweight": "medium",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.titlesize": 24,
        "figure.titleweight": "bold",
        "axes.linewidth": 1.1,
        "lines.linewidth": 2.8,
        "lines.markersize": 8,
        "savefig.dpi": 300,
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "#FBFBFE",
        "axes.edgecolor": "#444444",
        "text.color": "#1A1A1A",
        "axes.labelcolor": "#1A1A1A",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    })


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def save(fig, outdir: Path, name: str, also_pdf: bool = True):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", pad_inches=0.06,
                facecolor=fig.get_facecolor())
    if also_pdf:
        fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.06,
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved {name}.png" + (" / .pdf" if also_pdf else ""))


def grad_bars(ax, bars, base_color, lighten=0.45):
    """Give each bar a subtle vertical gradient for a cleaner, modern look."""
    import matplotlib.colors as mcolors
    rgb = np.array(mcolors.to_rgb(base_color))
    top = rgb + (1 - rgb) * lighten
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    grad = np.repeat(grad, 4, axis=1)
    for bar in bars:
        bar.set_facecolor("none")
        x, y = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()
        cmap = mcolors.LinearSegmentedColormap.from_list("g", [top, rgb])
        ax.imshow(grad, extent=[x, x + w, y, y + h], aspect="auto",
                  cmap=cmap, origin="lower", zorder=2,
                  clip_path=bar, clip_on=True)
