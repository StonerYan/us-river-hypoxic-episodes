"""Figure 4. Day-scale wind sets episode length, and seasonal averages hide it."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.gridspec import GridSpec

from config import SRC
from fig_style import (ANN, C, DOUBLE, EXIT, NOTE, load, panel_tag, save,
                       zero_line)


def survival(hz: dict, u: float, tmax: int = 40) -> np.ndarray:
    t = hz["recovery_hyp2"]["terms"]
    days = np.arange(1, tmax + 1)
    h = 1 / (1 + np.exp(-(t["const"]["b"] + t["aU_z"]["b"] * u
                          + t["lspell"]["b"] * np.log(days))))
    return np.cumprod(1 - h)


def panel_a(ax, hz: dict):
    """Two counterfactual winds, and the gap between the two survival curves as
    the length that wind adds or removes."""
    rows, curves = [], {}
    for u, col, lab in [(-1.0, C["heat"], "1 s.d. calmer than normal"),
                        (0.0, C["grey"], "normal wind"),
                        (1.0, EXIT, "1 s.d. windier than normal")]:
        s = survival(hz, u)
        x = np.arange(1, len(s) + 1)
        curves[u] = (x, s)
        ax.plot(x, s, color=col, lw=1.6, label=lab, zorder=3)
        rows += [{"wind_sd": u, "day": int(d), "still_hypoxic": float(v)}
                 for d, v in zip(x, s)]
    ax.fill_between(curves[1.0][0], curves[1.0][1], curves[-1.0][1],
                    color=C["grey_l"], alpha=0.30, linewidth=0, zorder=1)
    day = 12
    s_calm = float(curves[-1.0][1][day - 1])
    s_wind = float(curves[1.0][1][day - 1])
    ax.annotate("", xy=(day, s_wind), xytext=(day, s_calm),
                arrowprops=dict(arrowstyle="<|-|>,head_width=0.16,head_length=0.4",
                                color=C["grey"], lw=0.9), zorder=5)
    ax.annotate(f"{s_calm / s_wind:.1f} times as many\ncalm episodes are\n"
                f"still running at day {day}",
                xy=(day, (s_calm + s_wind) / 2), xytext=(0.98, 0.30),
                textcoords="axes fraction", fontsize=ANN, color=C["grey"],
                va="center", ha="right", linespacing=1.35,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C["grey"],
                                shrinkA=3, shrinkB=3))
    ax.set_xlabel("days after onset")
    ax.set_ylabel("probability the episode\nis still running")
    ax.set_xlim(1, 25)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", handlelength=1.4, fontsize=NOTE,
              labelspacing=0.3, borderpad=0.15)
    d = hz["duration"]
    ax.text(0.30, 0.60, f"expected length {d['aU=-1']['mean_days']:.1f} d when calm,"
            f"\n{d['aU=+1']['mean_days']:.1f} d when windy", transform=ax.transAxes,
            fontsize=ANN, color=C["grey"], va="top", linespacing=1.35)
    return pd.DataFrame(rows)


def panel_b(ax, hz: dict):
    """What a seasonal-average analysis of the same rivers would report."""
    items = [("hyp_days_z", "hypoxic days\nper summer", hz["decompose"]),
             ("n_ep_z", "number of\nepisodes", hz["decompose"]),
             ("mean_dur_z", "mean episode\nlength", hz["decompose"]),
             ("do_mean_z", "summer mean\noxygen", hz["seasonal"])]
    y = np.arange(len(items))[::-1]
    rows = []
    for i, (k, lab, src) in enumerate(items):
        v = src[k]["terms"]["U_z"]
        sig = not (v["lo"] < 0 < v["hi"])
        col = EXIT if sig else C["grey_l"]
        ax.plot([v["lo"], v["hi"]], [y[i]] * 2, color=col, lw=1.8,
                solid_capstyle="round", zorder=2)
        ax.plot(v["b"], y[i], "o" if sig else "s", color=col, ms=5.6, mec="white",
                mew=0.9, zorder=3)
        rows.append({"outcome": lab.replace("\n", " "), "b": v["b"], "lo": v["lo"],
                     "hi": v["hi"], "p": v["p"], "n": src[k]["n"]})
    zero_line(ax, "v")
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab, _ in items])
    ax.set_xlabel("response to a 1 s.d. windier summer\n(s.d. of the outcome)")
    ax.set_xlim(-0.12, 0.215)
    ax.set_ylim(-1.15, len(items) - 0.42)
    ax.spines["left"].set_bounds(-0.4, len(items) - 0.6)
    ax.text(0.02, 0.02, "a windier summer brings more\nepisodes, not shorter ones",
            transform=ax.transAxes, fontsize=ANN, color=C["grey"],
            linespacing=1.35)
    return pd.DataFrame(rows)


def panel_c(ax, win: pd.DataFrame):
    """The same model refitted with wind averaged over longer windows."""
    x = win["window_days"].to_numpy(dtype=float)
    ax.fill_between(x, win["lo"], win["hi"], color=EXIT, alpha=0.15, linewidth=0)
    ax.plot(x, win["b"], color=EXIT, lw=1.4, marker="o", ms=3.6, mfc="white",
            mew=0.9)
    zero_line(ax, "h")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 7, 10, 15, 30])
    ax.set_xticklabels(["1", "2", "3", "5", "7", "10", "15", "30"])
    ax.minorticks_off()
    ax.set_xlabel("window used to measure wind (days)")
    ax.set_ylabel("effect of wind on exit\n(log-odds per s.d.)")
    ax.set_ylim(-0.19, 0.33)
    cross = win.loc[(win["lo"] < 0) & (win["hi"] > 0), "window_days"].min()
    ax.axvline(cross, color=C["grey"], lw=0.6, ls=(0, (2, 2)))
    ax.text(cross * 1.10, 0.31, f"interval includes\nzero from {cross:.0f} days",
            fontsize=ANN, color=C["grey"], va="top", linespacing=1.35)
    ax.text(0.03, 0.04, "beyond two weeks\nthe sign reverses",
            transform=ax.transAxes, fontsize=ANN, color=C["grey"],
            linespacing=1.35)
    return win


def main() -> int:
    hz = load("hazard.json")
    win = load("wind_window.parquet")

    fig = plt.figure(figsize=(DOUBLE, 74 / 25.4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.60, width_ratios=[1.06, 1.0, 1.0])
    axa = fig.add_subplot(gs[0, 0])
    a = panel_a(axa, hz)
    panel_tag(axa, "a", dx=-0.24, dy=1.10)
    axb = fig.add_subplot(gs[0, 1])
    b = panel_b(axb, hz)
    panel_tag(axb, "b", dx=-0.40, dy=1.10)
    axc = fig.add_subplot(gs[0, 2])
    c = panel_c(axc, win)
    panel_tag(axc, "c", dx=-0.26, dy=1.10)

    save(fig, "fig4_consequence", src=a)
    b.to_csv(SRC / "fig4b_seasonal_scale.csv", index=False)
    c.to_csv(SRC / "fig4c_wind_window.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
