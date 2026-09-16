"""Supplementary figures S1-S8."""
from __future__ import annotations

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from fig_style import (C, DOUBLE, ENTRY, EXIT, SINGLE, load, panel_tag, save,
                       us_axes, zero_line)

LAB = {"aT_z": "water temperature", "aU_z": "wind speed", "aL_z": "solar radiation",
       "aQ_z": "discharge", "lspell": "log episode age", "deficit_z": "saturation deficit",
       "UxD": "wind $\\times$ deficit"}


def s1_network():
    p = load("daily_panel.parquet")
    t = p.groupby("site_id").agg(lat=("lat", "first"), lon=("lon", "first"),
                                 n_days=("valid_do", "sum"),
                                 n_years=("year", "nunique")).reset_index()
    fig = plt.figure(figsize=(DOUBLE, 74 / 25.4))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.7, 0.8, 0.8], wspace=0.42)
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.LambertConformal(
        central_longitude=-96, central_latitude=38))
    us_axes(ax)
    sc = ax.scatter(t["lon"], t["lat"], s=13, c=t["n_years"], cmap="viridis",
                    vmin=8, vmax=15, edgecolor="#555", linewidth=0.25,
                    transform=ccrs.PlateCarree(), zorder=4)
    cax = ax.inset_axes([0.64, 0.05, 0.31, 0.033])
    cb = plt.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("summers retained", fontsize=8.1, labelpad=1.5)
    cb.ax.tick_params(labelsize=6.0, width=0.5, length=1.8, pad=1.2)
    panel_tag(ax, "a", dx=-0.04, dy=1.03)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(t["n_years"], bins=np.arange(7.5, 16.5), color=C["grey_l"],
             edgecolor="white", linewidth=0.5)
    ax2.set_xlabel("summers per river")
    ax2.set_ylabel("number of rivers")
    panel_tag(ax2, "b", dx=-0.30, dy=1.08)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(t["n_days"] / 1e3, bins=18, color=C["grey_l"], edgecolor="white",
             linewidth=0.5)
    ax3.set_xlabel("valid summer days (thousands)")
    ax3.set_ylabel("number of rivers")
    panel_tag(ax3, "c", dx=-0.30, dy=1.08)
    save(fig, "figS1_network", si=True, src=t)


def _forest(ax, terms: dict, keys: list[str], color, title: str, n: int, sites: int):
    y = np.arange(len(keys))[::-1]
    for i, k in enumerate(keys):
        v = terms[k]
        ax.plot([v["lo"], v["hi"]], [y[i]] * 2, color=color, lw=1.5)
        ax.plot(v["b"], y[i], "o", color=color, ms=4.2, mec="white", mew=0.8)
    zero_line(ax, "v")
    ax.set_yticks(y)
    ax.set_yticklabels([LAB.get(k, k) for k in keys], fontsize=8.5)
    ax.set_title(title, fontsize=8.9, pad=3)
    ax.set_xlabel("log-odds per s.d.")
    ax.text(0.02, 0.03, f"n = {n:,} days, {sites} rivers", transform=ax.transAxes,
            fontsize=8, color=C["grey"])


def s5_robustness(hz: dict):
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 52 / 25.4))
    _forest(axes[0], hz["onset_hyp2_Q"]["terms"], ["aT_z", "aU_z", "aL_z", "aQ_z"],
            ENTRY, "entry, discharge controlled", hz["onset_hyp2_Q"]["n"],
            hz["onset_hyp2_Q"]["sites"])
    _forest(axes[1], hz["recovery_hyp2_Q"]["terms"],
            ["aT_z", "aU_z", "aL_z", "aQ_z", "lspell"], EXIT,
            "exit, discharge controlled", hz["recovery_hyp2_Q"]["n"],
            hz["recovery_hyp2_Q"]["sites"])
    _forest(axes[2], hz["placebo"]["terms"], ["aT_z", "aU_z", "aL_z", "lspell"],
            C["grey"], "exit, wind taken from a distant river",
            hz["placebo"]["n"], hz["placebo"]["sites"])
    for a, t in zip(axes, "abc"):
        panel_tag(a, t, dx=-0.36, dy=1.12)
    fig.subplots_adjust(wspace=0.75)
    save(fig, "figS5_robustness", si=True)


def s2_rf(rf: dict):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 54 / 25.4))
    ax = axes[0]
    feats = ["U_mean", "tw_mean", "ssrd_MJ", "aU_z", "aT_z", "aL_z"]
    nice = {"U_mean": "wind speed", "tw_mean": "water temperature",
            "ssrd_MJ": "solar radiation", "aU_z": "wind anomaly",
            "aT_z": "temperature anomaly", "aL_z": "radiation anomaly"}
    y = np.arange(len(feats))[::-1]
    for i, f in enumerate(feats):
        for key, col, off in [("onset", ENTRY, 0.17),
                              ("recovery_weather_only", EXIT, -0.17)]:
            v = rf[key]["importance"][f]
            ax.plot([max(v["mean"] - v["sd"], 0), v["mean"] + v["sd"]],
                    [y[i] + off] * 2, color=col, lw=1.3)
            ax.plot(v["mean"], y[i] + off, "o", color=col, ms=3.8, mec="white",
                    mew=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([nice[f] for f in feats], fontsize=8.4)
    ax.set_xlabel("loss in held-out AUC when permuted")
    ax.set_title("absolute levels and anomalies together", fontsize=8.9, pad=3)
    ax.legend(handles=[Line2D([], [], color=ENTRY, marker="o", lw=1.3, ms=3.8,
                              label="entry"),
                       Line2D([], [], color=EXIT, marker="o", lw=1.3, ms=3.8,
                              label="exit")], loc="lower right")
    panel_tag(ax, "a", dx=-0.40, dy=1.12)

    ax = axes[1]
    for key, col, lab in [("onset_anom", ENTRY, "entry"),
                          ("recovery_anom", EXIT, "exit")]:
        p = rf[key]["pdp"]["aU_z"]
        x, yv = np.array(p["x"]), np.array(p["y"])
        ax.plot(x, yv / yv.mean(), color=col, lw=1.4, label=lab)
    ax.set_xlabel("wind anomaly (s.d.)")
    ax.set_ylabel("partial dependence\n(relative to the mean)")
    ax.set_title("forest response to wind", fontsize=8.9, pad=3)
    ax.legend(loc="lower left", handlelength=1.3, borderpad=0.15)
    zero_line(ax, "h", 1.0)
    panel_tag(ax, "b", dx=-0.28, dy=1.12)
    fig.subplots_adjust(wspace=0.55)
    save(fig, "figS2_forests", si=True)


def s3_threshold(hz: dict):
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 50 / 25.4))
    _forest(axes[0], hz["onset_str4"]["terms"], ["aT_z", "aU_z", "aL_z"], ENTRY,
            "entry below 4 mg L$^{-1}$", hz["onset_str4"]["n"],
            hz["onset_str4"]["sites"])
    _forest(axes[1], hz["recovery_str4"]["terms"],
            ["aT_z", "aU_z", "aL_z", "lspell"], EXIT,
            "exit above 4 mg L$^{-1}$", hz["recovery_str4"]["n"],
            hz["recovery_str4"]["sites"])
    for a, t in zip(axes, "ab"):
        panel_tag(a, t, dx=-0.36, dy=1.12)
    fig.subplots_adjust(wspace=0.7)
    save(fig, "figS3_threshold4", si=True)


def s4_budget(mech: dict):
    fig, ax = plt.subplots(figsize=(SINGLE, 54 / 25.4))
    keys = ["aU_z", "aT_z", "aL_z", "sat_def"]
    nice = {"aU_z": "wind speed", "aT_z": "water temperature",
            "aL_z": "solar radiation", "sat_def": "saturation deficit"}
    y = np.arange(len(keys))[::-1]
    for i, k in enumerate(keys):
        for tgt, col, off, lab in [("d_do_min", EXIT, 0.16, "daily minimum"),
                                   ("d_do_max", C["heat"], -0.16, "daily maximum")]:
            v = mech[tgt]["terms"][k]
            ax.plot([v["lo"], v["hi"]], [y[i] + off] * 2, color=col, lw=1.4)
            ax.plot(v["b"], y[i] + off, "o", color=col, ms=4.0, mec="white", mew=0.7)
    zero_line(ax, "v")
    ax.set_yticks(y)
    ax.set_yticklabels([nice[k] for k in keys], fontsize=8.5)
    ax.set_xlabel("next-day change in oxygen (mg L$^{-1}$ d$^{-1}$)")
    ax.legend(handles=[Line2D([], [], color=EXIT, marker="o", lw=1.4, ms=4,
                              label="daily minimum"),
                       Line2D([], [], color=C["heat"], marker="o", lw=1.4, ms=4,
                              label="daily maximum")], loc="lower right")
    ax.text(0.02, 0.03, f"n = {mech['d_do_min']['n']:,} river-days, "
            f"{mech['d_do_min']['sites']} rivers", transform=ax.transAxes,
            fontsize=8, color=C["grey"])
    save(fig, "figS4_oxygen_change", si=True)


def s6_within_episode():
    d = load("within_episode_contrast.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 52 / 25.4))
    ax = axes[0]
    ax.hist(d["aU_z"], bins=np.arange(-3, 3.2, 0.25), color=C["wind_l"],
            edgecolor="white", linewidth=0.4)
    zero_line(ax, "v")
    ax.axvline(d["aU_z"].mean(), color=EXIT, lw=1.3)
    ax.set_xlabel("wind contrast, exit day minus persistence days (s.d.)")
    ax.set_ylabel("number of episodes")
    ax.text(0.03, 0.95, f"{100 * (d['aU_z'] > 0).mean():.0f}% of episodes\n"
            f"are positive\nn = {len(d):,}", transform=ax.transAxes, va="top",
            fontsize=8.2, color=C["grey"])
    panel_tag(ax, "a", dx=-0.26, dy=1.10)
    ax = axes[1]
    d = d.copy()
    d["cls"] = pd.cut(d["n_days"], [2, 4, 7, 14, 400],
                      labels=["3-4", "5-7", "8-14", "15+"])
    rows = []
    for c, g in d.groupby("cls", observed=True):
        m = g["aU_z"].mean()
        se = g["aU_z"].std(ddof=1) / np.sqrt(len(g))
        rows.append((str(c), m, 1.96 * se, len(g)))
    r = pd.DataFrame(rows, columns=["cls", "m", "e", "n"])
    ax.errorbar(np.arange(len(r)), r["m"], yerr=r["e"], fmt="o", color=EXIT,
                ms=4.2, mfc="white", mew=1.0, capsize=2.0, elinewidth=0.9)
    zero_line(ax, "h")
    ax.set_xticks(np.arange(len(r)))
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(r["cls"], r["n"])],
                       fontsize=8.2)
    ax.set_xlabel("episode length class (days)")
    ax.set_ylabel("wind contrast (s.d.)")
    panel_tag(ax, "b", dx=-0.26, dy=1.10)
    fig.subplots_adjust(wspace=0.45)
    save(fig, "figS6_within_episode", si=True, src=d.drop(columns=["site_id"]))


def s7_window():
    w = load("wind_window.parquet")
    fig, ax = plt.subplots(figsize=(SINGLE, 52 / 25.4))
    ax.fill_between(w["window_days"], w["lo"], w["hi"], color=EXIT, alpha=0.15,
                    linewidth=0)
    ax.plot(w["window_days"], w["b"], color=EXIT, lw=1.4, marker="o", ms=3.6,
            mfc="white", mew=0.9)
    zero_line(ax, "h")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 7, 10, 15, 30])
    ax.set_xticklabels(["1", "2", "3", "5", "7", "10", "15", "30"], fontsize=8.3)
    ax.minorticks_off()
    ax.set_xlabel("window used to measure wind (days)")
    ax.set_ylabel("effect of wind on exit (log-odds per s.d.)")
    for _, r in w.iterrows():
        ax.text(r["window_days"], -0.155, f"{r['or']:.2f}", ha="center",
                fontsize=7.5, color=C["grey"])
    ax.text(0.5, -0.20, "odds ratio", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=7.9, color=C["grey"])
    save(fig, "figS7_wind_window", si=True, src=w)


def main() -> int:
    hz = load("hazard.json")
    mech = load("mechanism.json")
    rf = load("rf.json")
    s1_network()
    s2_rf(rf)
    s3_threshold(hz)
    s4_budget(mech)
    s5_robustness(hz)
    s6_within_episode()
    s7_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
