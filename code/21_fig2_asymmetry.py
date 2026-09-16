"""Figure 2. Entry and exit of river hypoxia answer to different drivers.

Each panel uses a different display on purpose: paired intervals for the two
hazards, an arrow for the interaction that tests them against each other, the
raw episode distribution for the within-episode control, a paired range plot
for the forests, and a replication scatter against the milder threshold.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from config import SRC
from fig_style import (ANN, C, DOUBLE, ENTRY, EXIT, NOTE, dumbbell, load,
                       panel_tag, raincloud, save, zero_line)

LAB = {"aT_z": "water\ntemperature", "aU_z": "wind\nspeed", "aL_z": "solar\nradiation"}
SHORT = {"aT_z": "water temperature", "aU_z": "wind speed", "aL_z": "solar radiation"}
COL = {"aT_z": C["heat"], "aU_z": C["wind"], "aL_z": C["light"]}
ORDER = ["aT_z", "aU_z", "aL_z"]


def panel_a(ax, hz: dict):
    """Odds ratio per 1 s.d. of within-site daily anomaly, entry against exit."""
    on, rc = hz["onset_hyp2"]["terms"], hz["recovery_hyp2"]["terms"]
    rows = []
    y = np.arange(len(ORDER))[::-1]
    for i, k in enumerate(ORDER):
        for src, col, lab, off in [(on, ENTRY, "entry", 0.18),
                                   (rc, EXIT, "exit", -0.18)]:
            v = src[k]
            ax.plot([v["or_lo"], v["or_hi"]], [y[i] + off] * 2, color=col, lw=1.8,
                    solid_capstyle="round", zorder=2)
            ax.plot(v["or"], y[i] + off, "o", color=col, ms=5.4, mec="white",
                    mew=0.9, zorder=3)
            rows.append({"driver": SHORT[k], "transition": lab,
                         "odds_ratio": v["or"], "lo": v["or_lo"], "hi": v["or_hi"],
                         "p": v["p"]})
    zero_line(ax, "v", 1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([LAB[k] for k in ORDER])
    ax.set_xlabel("odds ratio per 1 s.d. of daily anomaly")
    ax.set_xlim(0.72, 1.46)
    ax.set_ylim(-0.80, len(ORDER) - 0.34)
    ax.annotate("", xy=(1.34, -0.66), xytext=(1.13, -0.66),
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C["grey"]))
    ax.text(1.125, -0.56, "makes the transition more likely", fontsize=ANN,
            color=C["grey"])
    ax.legend(handles=[Line2D([], [], color=ENTRY, marker="o", lw=1.8, ms=5.0,
                              label="entry into hypoxia"),
                       Line2D([], [], color=EXIT, marker="o", lw=1.8, ms=5.0,
                              label="exit from hypoxia")],
              loc="lower right", bbox_to_anchor=(1.04, 0.06))
    ax.text(0.015, 0.98, f"entry: n = {hz['onset_hyp2']['n']:,} days, "
            f"{hz['onset_hyp2']['sites']} rivers\n"
            f"exit: n = {hz['recovery_hyp2']['n']:,} days, "
            f"{hz['recovery_hyp2']['sites']} rivers",
            transform=ax.transAxes, fontsize=NOTE, color=C["grey"], va="top")
    return pd.DataFrame(rows)


def panel_b(ax, hz: dict):
    """The interaction as a displacement: how far a driver's action moves once
    the river is already hypoxic."""
    tr = hz["transition_hyp2"]
    y = np.arange(len(ORDER))[::-1]
    rows = []
    for i, k in enumerate(ORDER):
        v = tr["terms"][k + "xH"]
        ax.fill_betweenx([y[i] - 0.16, y[i] + 0.16], v["lo"], v["hi"],
                         color=COL[k], alpha=0.20, linewidth=0, zorder=1)
        ax.annotate("", xy=(v["b"], y[i]), xytext=(0.0, y[i]),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.45",
                                    color=COL[k], lw=1.9, shrinkA=0, shrinkB=0),
                    zorder=3)
        rows.append({"driver": SHORT[k], "interaction": v["b"], "lo": v["lo"],
                     "hi": v["hi"], "p": v["p"]})
    zero_line(ax, "v")
    ax.set_yticks(y)
    ax.set_yticklabels([LAB[k] for k in ORDER])
    ax.set_xlabel("shift in log-odds once\nthe river is hypoxic")
    ax.set_xlim(-0.42, 0.30)
    ax.set_ylim(-1.55, len(ORDER) - 0.36)
    ax.spines["left"].set_bounds(-0.4, len(ORDER) - 0.6)
    j = tr["joint_interaction"]
    pstr = "P < 0.001" if j["p"] < 1e-3 else f"P = {j['p']:.3f}"
    ax.text(0.5, 0.02, "the drivers act differently on the\ntwo transitions: "
            f"$\\chi^2$ = {j['chi2']:.0f}, d.f. = 3, {pstr}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=ANN,
            color=C["grey"], linespacing=1.35,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.text(0.98, 0.99, f"n = {tr['n']:,} days", transform=ax.transAxes,
            fontsize=NOTE, color=C["grey"], va="top", ha="right")
    return pd.DataFrame(rows)


def panel_c(ax, mech: dict, con: pd.DataFrame):
    """Every episode as its own control: the distribution of exit-day minus
    persistence-day anomalies, one value per episode."""
    we = mech["within_episode"]
    x = np.arange(len(ORDER))
    rows = []
    for i, k in enumerate(ORDER):
        raincloud(ax, x[i] + 0.10, con[k].to_numpy(), COL[k], width=0.32)
        v = we["terms"][k]
        ax.plot([x[i] + 0.10] * 2, [v["lo"], v["hi"]], color="#1A1A1A", lw=1.1,
                zorder=6)
        ax.plot(x[i] + 0.10, v["mean"], "D", color="#1A1A1A", ms=3.8, zorder=7)
        rows.append({"driver": SHORT[k], "contrast_sd": v["mean"], "lo": v["lo"],
                     "hi": v["hi"]})
    zero_line(ax, "h")
    ax.legend(handles=[Line2D([], [], color=COL[k], marker="o", lw=0, ms=5.0,
                              label=SHORT[k]) for k in ORDER],
              loc="upper right", handletextpad=0.3, labelspacing=0.25,
              borderpad=0.15, fontsize=NOTE, bbox_to_anchor=(1.04, 1.03))
    ax.set_xticks([])
    ax.set_ylabel("exit minus persistence days\n(s.d. of daily anomaly)")
    ax.set_ylim(-3.1, 5.9)
    ax.set_xlim(-0.55, len(ORDER) - 0.28)
    ax.spines["left"].set_bounds(-3.0, 3.0)
    ax.set_yticks([-2, 0, 2])
    ax.set_title(f"exit days are +{we['terms']['aU_z']['mean']:.2f} s.d. windier",
                 fontsize=ANN, color=C["grey"], pad=4)
    return pd.DataFrame(rows)


def panel_d(ax, rf: dict):
    """Forests evaluated on rivers withheld from training: the skill each driver carries
    at entry against the skill it carries at exit."""
    y = np.arange(len(ORDER))[::-1]
    rows = []
    for i, k in enumerate(ORDER):
        e, x = rf["onset_anom"]["importance"][k], rf["recovery_anom"]["importance"][k]
        ax.plot([e["lo"], e["hi"]], [y[i] + 0.12] * 2, color=ENTRY, lw=0.9, zorder=2)
        ax.plot([x["lo"], x["hi"]], [y[i] - 0.12] * 2, color=EXIT, lw=0.9, zorder=2)
        dumbbell(ax, y[i], e["mean"], x["mean"], ENTRY, EXIT, lw=1.3, ms=5.6)
        for tag, v in (("entry", e), ("exit", x)):
            rows.append({"driver": SHORT[k], "transition": tag,
                         "delta_auc": v["mean"], "fold_min": v["lo"],
                         "fold_max": v["hi"], "fold_sd": v["sd"]})
    ax.set_yticks(y)
    ax.set_yticklabels([LAB[k] for k in ORDER])
    ax.set_xlabel("loss in held-out AUC\nwhen the driver is permuted")
    ax.set_xlim(0, 0.072)
    ax.set_ylim(-1.30, len(ORDER) - 0.36)
    ax.set_xticks([0, 0.02, 0.04, 0.06])
    ax.spines["left"].set_bounds(-0.4, len(ORDER) - 0.6)
    ratio = (rf["recovery_anom"]["importance"]["aU_z"]["mean"]
             / rf["recovery_anom"]["importance"]["aT_z"]["mean"])
    ax.text(0.5, 0.02, f"at exit, wind carries {ratio:.1f} times\n"
            "the skill of temperature",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=ANN,
            color=C["grey"])
    return pd.DataFrame(rows)


def panel_e(ax, hz: dict):
    """Replication: each coefficient at 2 mg/L against the same coefficient at
    the milder 4 mg/L line."""
    mark = {"aT_z": "o", "aU_z": "s", "aL_z": "^"}
    rows = []
    for k in ORDER:
        for a_key, b_key, col, tag in [
                ("onset_hyp2", "onset_str4", ENTRY, "entry"),
                ("recovery_hyp2", "recovery_str4", EXIT, "exit")]:
            a, b = hz[a_key]["terms"][k], hz[b_key]["terms"][k]
            ax.plot([a["or_lo"], a["or_hi"]], [b["or"]] * 2, color=col, lw=0.8,
                    alpha=0.5, zorder=2)
            ax.plot([a["or"]] * 2, [b["or_lo"], b["or_hi"]], color=col, lw=0.8,
                    alpha=0.5, zorder=2)
            ax.plot(a["or"], b["or"], mark[k], color=col, ms=5.6, mec="white",
                    mew=0.9, zorder=4)
            rows.append({"driver": SHORT[k], "transition": tag,
                         "or_2mg": a["or"], "or_4mg": b["or"],
                         "lo_4mg": b["or_lo"], "hi_4mg": b["or_hi"]})
    ax.legend(handles=[Line2D([], [], color=C["grey"], marker=mark[k], lw=0,
                              ms=5.0, label=SHORT[k].split()[-1])
                       for k in ORDER],
              loc="lower right", handletextpad=0.3, labelspacing=0.25,
              borderpad=0.15, fontsize=NOTE, bbox_to_anchor=(1.04, 0.02))
    lim = (0.74, 1.46)
    ax.plot(lim, lim, color=C["zero"], lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.axhline(1.0, color=C["grey_l"], lw=0.6, zorder=0)
    ax.axvline(1.0, color=C["grey_l"], lw=0.6, zorder=0)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("odds ratio at 2 mg L$^{-1}$")
    ax.set_ylabel("odds ratio at 4 mg L$^{-1}$")
    ax.set_title("the asymmetry replicates at 4 mg L$^{-1}$", fontsize=ANN,
                 color=C["grey"], pad=4)
    return pd.DataFrame(rows)


def main() -> int:
    hz = load("hazard.json")
    mech = load("mechanism.json")
    rf = load("rf.json")
    con = load("within_episode_contrast.parquet")

    fig = plt.figure(figsize=(DOUBLE, 132 / 25.4))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.04],
                  hspace=0.62, wspace=0.52)
    axa = fig.add_subplot(gs[0, :2])
    a = panel_a(axa, hz)
    panel_tag(axa, "a", dx=-0.105, dy=1.08)
    axb = fig.add_subplot(gs[0, 2])
    b = panel_b(axb, hz)
    panel_tag(axb, "b", dx=-0.36, dy=1.08)
    axc = fig.add_subplot(gs[1, 0])
    c = panel_c(axc, mech, con)
    panel_tag(axc, "c", dx=-0.40, dy=1.10)
    axd = fig.add_subplot(gs[1, 1])
    d = panel_d(axd, rf)
    panel_tag(axd, "d", dx=-0.34, dy=1.10)
    axe = fig.add_subplot(gs[1, 2])
    e = panel_e(axe, hz)
    panel_tag(axe, "e", dx=-0.30, dy=1.10)

    save(fig, "fig2_asymmetry", src=a)
    for nm, df in [("fig2b_interaction", b), ("fig2c_within_episode", c),
                   ("fig2d_rf_importance", d), ("fig2e_threshold4", e)]:
        df.to_csv(SRC / f"{nm}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
