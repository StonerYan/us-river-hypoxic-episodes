"""Shared figure style and helpers.

One restrained palette: neutral grey for context, one warm family for the
water-temperature driver, one cool family for the wind driver, one amber
accent for radiation, one violet accent for the random-forest panel.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIG, OUT, SI_FIG, SRC

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "legend.frameon": False,
    "figure.dpi": 140,
})

ANN = 8.4      # in-panel annotation
NOTE = 8.0     # sample-size note

C = {
    "grey": "#5A5A5A",
    "grey_l": "#BFBFBF",
    "grey_f": "#E6E6E6",
    "heat": "#B4442E",
    "heat_l": "#E7A392",
    "wind": "#1F6F8B",
    "wind_l": "#8FBFD0",
    "light": "#D19A2E",
    "ml": "#6E5192",
    "zero": "#9A9A9A",
}
ENTRY, EXIT = C["heat"], C["wind"]

MM = 1 / 25.4
SINGLE, DOUBLE = 89 * MM, 183 * MM


def load(name: str):
    p = OUT / name
    if name.endswith(".json"):
        return json.loads(p.read_text(encoding="utf-8"))
    return pd.read_parquet(p)


def panel_tag(ax, letter: str, dx: float = -0.16, dy: float = 1.06, size: float = 11.5):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="top", ha="left")


def dumbbell(ax, y, x0, x1, c0, c1, lw: float = 1.1, ms: float = 6.0):
    """Two paired states joined by a rule: reads as a shift, not two bars."""
    ax.plot([x0, x1], [y, y], color=C["grey_l"], lw=lw, zorder=1,
            solid_capstyle="round")
    ax.plot(x0, y, "o", color=c0, ms=ms, mec="white", mew=0.8, zorder=3)
    ax.plot(x1, y, "o", color=c1, ms=ms, mec="white", mew=0.8, zorder=3)


def lollipop(ax, y, v, lo, hi, color, ms: float = 6.0, lw: float = 1.4):
    ax.hlines(y, 0, v, color=color, lw=lw, alpha=0.55, zorder=2)
    ax.hlines(y, lo, hi, color=color, lw=1.0, zorder=3)
    ax.plot(v, y, "o", color=color, ms=ms, mec="white", mew=0.8, zorder=4)


def raincloud(ax, x, values, color, width: float = 0.34, side: int = 1,
              ms: float = 1.6, jitter: float = 0.10, seed: int = 3):
    """Half violin with the raw episodes beside it, so the reader sees the
    distribution rather than a summary bar."""
    from scipy.stats import gaussian_kde
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    grid = np.linspace(np.percentile(v, 0.5), np.percentile(v, 99.5), 200)
    dens = gaussian_kde(v)(grid)
    dens = width * dens / dens.max()
    ax.fill_betweenx(grid, x, x + side * dens, color=color, alpha=0.35,
                     linewidth=0, zorder=2)
    ax.plot(x + side * dens, grid, color=color, lw=0.8, zorder=3)
    rng = np.random.default_rng(seed)
    keep = v if len(v) <= 900 else rng.choice(v, 900, replace=False)
    ax.plot(x - side * (0.035 + jitter * rng.random(len(keep))), keep, "o",
            ms=ms, color=color, alpha=0.35, mec="none", zorder=1)


def zero_line(ax, orient: str = "v", val: float = 0.0):
    if orient == "v":
        ax.axvline(val, color=C["zero"], lw=0.6, ls=(0, (3, 2)), zorder=0)
    else:
        ax.axhline(val, color=C["zero"], lw=0.6, ls=(0, (3, 2)), zorder=0)


def save(fig, stem: str, si: bool = False, src: pd.DataFrame | None = None):
    d = SI_FIG if si else FIG
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"{stem}.{ext}", bbox_inches="tight",
                    dpi=600 if ext == "png" else None)
    if src is not None:
        SRC.mkdir(parents=True, exist_ok=True)
        src.to_csv(SRC / f"{stem}.csv", index=False)
    plt.close(fig)
    print("  figure", d.name + "/" + stem, flush=True)


def us_axes(ax):
    """Light continental United States backdrop for a scatter map."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    ax.set_extent([-125, -66.5, 24.5, 49.5], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F4F3F1", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#DCE6EA",
                   edgecolor="none", zorder=0.5)
    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#D2D0CC",
                   linewidth=0.35, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#9E9C98",
                   linewidth=0.45, zorder=1)
    ax.spines["geo"].set_visible(False)
    return ax


def or_ci(term: dict) -> tuple[float, float, float]:
    return term["or"], term["or_lo"], term["or_hi"]


def bar_ci(ax, x, y, lo, hi, color, width=0.62, lw=0.8):
    ax.bar(x, y, width=width, color=color, edgecolor="none", zorder=2)
    ax.vlines(x, lo, hi, color="#2B2B2B", lw=lw, zorder=3)


def fmt_ci(v: dict, digits: int = 2, key: str = "b") -> str:
    return f"{v[key]:.{digits}f} ({v['lo']:.{digits}f} to {v['hi']:.{digits}f})"
