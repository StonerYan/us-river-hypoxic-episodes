"""Figure 3. Wind acts on the oxygen floor as a gas-transfer flux."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from config import SRC
from fig_style import (ANN, C, DOUBLE, EXIT, NOTE, load, lollipop, panel_tag,
                       save, zero_line)


def panel_a(ax, bins: pd.DataFrame):
    """Wind sensitivity of the next-day change in the daily oxygen minimum,
    within quintiles of the same-day saturation deficit."""
    x = np.arange(len(bins))
    ax.errorbar(x, bins["bU"], yerr=[bins["bU"] - bins["lo"], bins["hi"] - bins["bU"]],
                color=EXIT, lw=1.2, marker="o", ms=4.2, mfc="white", mew=1.0,
                capsize=1.8, elinewidth=0.8)
    zero_line(ax, "h")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.1f}" for v in bins["def_mid"]], fontsize=8.4)
    ax.set_xlabel("saturation deficit (mg L$^{-1}$, quintile median)")
    ax.set_ylabel("oxygen gained per 1 s.d. of wind\n(mg L$^{-1}$ d$^{-1}$)")
    ax.set_ylim(0, 0.052)
    ax.text(0.34, 0.40, "wind adds almost no oxygen when the\n"
            "water is already near saturation, and\nmore as the deficit grows",
            transform=ax.transAxes, va="top", fontsize=ANN, color=C["grey"])
    ax.text(0.97, 0.97, f"n = {bins['n'].sum():,} river-days", ha="right",
            va="top", transform=ax.transAxes, fontsize=NOTE, color=C["grey"])


def panel_b(ax, mech: dict):
    """The deficit scaling appears at the daily minimum, not at the daily maximum:
    the floor of the daily cycle and its ceiling answer differently."""
    keys = [("d_do_min", "daily minimum\n(the floor)", EXIT),
            ("d_do_max", "daily maximum\n(the ceiling)", C["grey"])]
    rows, pts = [], []
    for k, lab, col in keys:
        v = mech[k]["terms"]["UxD"]
        pts.append((v["b"], col))
        ax.plot([0.0, 0.0], [v["lo"], v["hi"]], color=col, lw=1.1, zorder=3)
        ax.plot(0.0, v["b"], "o", color=col, ms=7.0, mec="white", mew=1.0, zorder=4)
        ax.text(0.16, v["b"], lab, color=col, fontsize=ANN, va="center", ha="left")
        rows.append({"target": lab.split("\n")[0], "wind_x_deficit": v["b"],
                     "lo": v["lo"], "hi": v["hi"], "p": v["p"]})
    ax.plot([0.0, 0.0], [pts[0][0], pts[1][0]], color=C["grey_l"], lw=1.2, zorder=2)
    zero_line(ax, "h")
    ax.set_xticks([])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("wind $\\times$ deficit interaction")
    ax.set_ylim(-0.016, 0.017)
    ax.spines["bottom"].set_visible(False)
    ax.text(0.5, 0.015, "the flux raises the floor,\nnot the ceiling",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=ANN,
            color=C["grey"], linespacing=1.35)
    return pd.DataFrame(rows)


def panel_c(ax, mech: dict):
    """Wind does not change the diel oxygen swing that metabolism sets."""
    terms = mech["amplitude"]["terms"]
    keys = [("aU_z", "wind speed", C["wind"]),
            ("aT_z", "water temperature", C["heat"]),
            ("aL_z", "solar radiation", C["light"])]
    y = np.arange(len(keys))[::-1]
    rows = []
    for i, (k, lab, col) in enumerate(keys):
        v = terms[k]
        lollipop(ax, y[i], v["b"], v["lo"], v["hi"], col, ms=6.4)
        rows.append({"driver": lab, "b": v["b"], "lo": v["lo"], "hi": v["hi"],
                     "p": v["p"]})
    zero_line(ax, "v")
    ax.set_yticks(y)
    ax.set_yticklabels([lab.replace(" ", "\n") for _, lab, _ in keys])
    ax.set_xlabel("diel swing response\n(s.d. per s.d. of driver)")
    ax.set_xlim(-0.03, 0.27)
    ax.set_ylim(-1.45, len(keys) - 0.4)
    ax.spines["left"].set_bounds(-0.4, len(keys) - 0.6)
    ax.text(0.52, 0.03, "the interval for wind includes zero:\n"
            "the swing belongs to metabolism", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=ANN, color=C["grey"],
            linespacing=1.35)
    ax.text(0.98, 0.97, f"n = {mech['amplitude']['n']:,} river-days",
            transform=ax.transAxes, ha="right", va="top", fontsize=NOTE,
            color=C["grey"])
    return pd.DataFrame(rows)


def panel_d(ax, hz: pd.DataFrame):
    """Observed exit probability against episode age, split by same-day wind."""
    style = {"calm": (C["heat"], "calm third"), "median": (C["grey"], "middle third"),
             "windy": (EXIT, "windy third")}
    for t, (col, lab) in style.items():
        g = hz[hz["terc"] == t].sort_values("spell")
        ax.plot(g["spell"], g["exit"], color=col, lw=1.2, marker="o", ms=3.2,
                mfc="white", mew=0.9, label=lab)
        ax.fill_between(g["spell"], g["lo"], g["hi"], color=col, alpha=0.13,
                        linewidth=0)
    ax.set_xlabel("episode age (days since onset)")
    ax.set_ylabel("probability of exit\non the next day")
    ax.set_xticks([1, 2, 4, 6, 8])
    ax.set_xticklabels(["1", "2", "4", "6", "8+"])
    ax.set_ylim(0, 0.52)
    ax.legend(loc="upper right", title="same-day wind anomaly",
              title_fontsize=NOTE, handlelength=1.3, fontsize=NOTE)
    ax.text(0.03, 0.03, "the windy third exits faster at every age",
            transform=ax.transAxes, fontsize=ANN, color=C["grey"])


def main() -> int:
    mech = load("mechanism.json")
    bins = load("wind_gain_by_deficit.parquet")
    hz = load("hazard_curves.parquet")

    fig = plt.figure(figsize=(DOUBLE, 116 / 25.4))
    gs = GridSpec(2, 12, figure=fig, hspace=0.72, wspace=2.4)
    axa = fig.add_subplot(gs[0, 0:7])
    panel_a(axa, bins)
    panel_tag(axa, "a", dx=-0.13, dy=1.10)
    axb = fig.add_subplot(gs[0, 8:12])
    b = panel_b(axb, mech)
    panel_tag(axb, "b", dx=-0.42, dy=1.16)
    axc = fig.add_subplot(gs[1, 0:5])
    c = panel_c(axc, mech)
    panel_tag(axc, "c", dx=-0.34, dy=1.10)
    axd = fig.add_subplot(gs[1, 6:12])
    panel_d(axd, hz)
    panel_tag(axd, "d", dx=-0.15, dy=1.10)

    save(fig, "fig3_mechanism", src=bins)
    b.to_csv(SRC / "fig3b_deficit_interaction.csv", index=False)
    c.to_csv(SRC / "fig3c_amplitude.csv", index=False)
    hz.to_csv(SRC / "fig3d_hazard_by_wind.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
