"""Figure 1. River hypoxia arrives as short episodes with two distinct transitions."""
from __future__ import annotations

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from fig_style import C, SINGLE, DOUBLE, load, panel_tag, save, us_axes


def site_table() -> pd.DataFrame:
    panel = load("daily_panel.parquet")
    ep = load("episodes_hyp2.parquet")
    base = panel.groupby("site_id").agg(
        lat=("lat", "first"), lon=("lon", "first"),
        n_summers=("year", "nunique"), n_days=("valid_do", "sum")).reset_index()
    agg = ep.groupby("site_id").agg(n_ep=("dur", "size"), med_dur=("dur", "median"),
                                    max_dur=("dur", "max"), ep_days=("dur", "sum")).reset_index()
    t = base.merge(agg, on="site_id", how="left")
    t[["n_ep", "ep_days"]] = t[["n_ep", "ep_days"]].fillna(0.0)
    t["ep_per_summer"] = t["n_ep"] / t["n_summers"]
    t["frac_days"] = 100 * t["ep_days"] / t["n_days"]
    return t


def panel_a(ax, t: pd.DataFrame):
    us_axes(ax)
    none = t[t["n_ep"] == 0]
    some = t[t["n_ep"] > 0].sort_values("ep_per_summer")
    ax.scatter(none["lon"], none["lat"], s=5, c="white", edgecolor=C["grey_l"],
               linewidth=0.4, transform=ccrs.PlateCarree(), zorder=3,
               label=f"no hypoxic day (n = {len(none)})")
    sz = 8 + 26 * np.clip(some["med_dur"].to_numpy(), 1, 12) / 12
    sc = ax.scatter(some["lon"], some["lat"], s=sz, c=some["ep_per_summer"],
                    cmap="YlGnBu", vmin=0, vmax=6, edgecolor="#3A3A3A",
                    linewidth=0.3, transform=ccrs.PlateCarree(), zorder=4,
                    label=f"hypoxic episodes (n = {len(some)})")
    cax = ax.inset_axes([0.66, 0.045, 0.30, 0.030])
    cb = plt.colorbar(sc, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("episodes per summer", fontsize=8.3, labelpad=1.5)
    cb.ax.tick_params(labelsize=6.1, width=0.5, length=1.8, pad=1.2)
    cb.outline.set_linewidth(0.4)
    for d, lab in [(2, "2 d"), (7, "7 d"), (12, "12 d")]:
        ax.scatter([], [], s=8 + 26 * d / 12, c="none", edgecolor="#3A3A3A",
                   linewidth=0.5, label=f"median length {lab}")
    ax.legend(loc="lower left", handletextpad=0.4, labelspacing=0.30,
              borderpad=0.2, fontsize=8, bbox_to_anchor=(-0.01, -0.02))


def panel_b(ax, ep: pd.DataFrame):
    d = np.sort(ep["dur"].to_numpy())
    x = np.unique(d)
    surv = np.array([(d >= v).mean() for v in x])
    ax.step(x, surv, where="post", color=C["grey"], lw=1.2)
    ax.fill_between(x, surv, step="post", color=C["grey_f"], zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("episode length (days)")
    ax.set_ylabel("fraction of episodes\nat least this long")
    med = np.median(d)
    ax.axvline(med, color=C["wind"], lw=0.7, ls=(0, (3, 2)))
    ax.text(med * 1.5, 0.030, f"median {med:.0f} d", fontsize=8.2, color=C["wind"],
            va="center")
    ax.text(0.97, 0.94, f"n = {len(d)} episodes\n{ep.site_id.nunique()} rivers\n"
            f"longest {d.max():.0f} d", transform=ax.transAxes, ha="right",
            va="top", fontsize=8.2, color=C["grey"])
    ax.set_ylim(5e-4, 1.6)


def panel_c(ax, rc: pd.DataFrame):
    d = rc.copy()
    d["k"] = d["spell_day"].clip(upper=10)
    rows = []
    for k, g in d.groupby("k"):
        p = g["y"].mean()
        se = np.sqrt(p * (1 - p) / len(g))
        rows.append((int(k), p, 1.96 * se, len(g)))
    r = pd.DataFrame(rows, columns=["k", "p", "e", "n"])
    ax.fill_between(r["k"], r["p"] - r["e"], r["p"] + r["e"], color=C["wind"],
                    alpha=0.16, linewidth=0, zorder=1)
    ax.plot(r["k"], r["p"], color=C["wind"], lw=1.5, zorder=2)
    ax.plot(r["k"], r["p"], "o", color=C["wind"], ms=4.0, mfc="white", mew=1.1,
            zorder=3)
    ax.set_xlabel("episode age (days since onset)")
    ax.set_ylabel("probability of exit\non the next day")
    ax.set_ylim(0, 0.47)
    ax.set_xticks([1, 3, 5, 7, 9])
    ax.set_xticklabels(["1", "3", "5", "7", "9+"])
    ax.text(0.97, 0.96, "older episodes are\nless likely to end",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.4,
            color=C["grey"], linespacing=1.35)
    return r


def panel_d(ax, ep: pd.DataFrame):
    start = ep["start"].dt.dayofyear.to_numpy()
    end = (ep["end"] + pd.Timedelta(days=1)).dt.dayofyear.to_numpy()
    bins = np.arange(120, 310, 10)
    for v, col, lab in [(start, C["heat"], "entries"), (end, C["wind"], "exits")]:
        h, e = np.histogram(v, bins=bins)
        ax.step(e[:-1], h, where="post", color=col, lw=1.1, label=lab)
        ax.fill_between(e[:-1], h, step="post", color=col, alpha=0.12)
    ax.set_xlabel("day of year")
    ax.set_ylabel("number of transitions")
    ticks = [121, 152, 182, 213, 244, 274]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["May", "Jun", "Jul", "Aug", "Sep", "Oct"], fontsize=8.4)
    ax.legend(loc="upper left", ncol=2, columnspacing=0.9,
              handlelength=1.1, bbox_to_anchor=(-0.02, 1.05))
    ax.set_ylim(0, 430)


def panel_e(ax, ep: pd.DataFrame):
    """Exposure is concentrated in the long tail: the cumulative share of hypoxic
    days against episodes ranked longest first."""
    d = np.sort(ep["dur"].to_numpy())[::-1]
    fe = 100 * np.arange(1, len(d) + 1) / len(d)
    fd = 100 * np.cumsum(d) / d.sum()
    ax.plot([0, 100], [0, 100], color=C["zero"], lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.fill_between(fe, fd, color=C["wind"], alpha=0.13, linewidth=0, zorder=2)
    ax.plot(fe, fd, color=C["wind"], lw=1.7, zorder=3)
    m = d >= 8
    x8, y8 = 100 * m.mean(), 100 * d[m].sum() / d.sum()
    ax.plot([x8, x8], [0, y8], color=C["grey"], lw=0.7, ls=(0, (2, 2)), zorder=2)
    ax.plot([0, x8], [y8, y8], color=C["grey"], lw=0.7, ls=(0, (2, 2)), zorder=2)
    ax.plot(x8, y8, "o", color=C["heat"], ms=5.4, mec="white", mew=0.9, zorder=5)
    ax.annotate(f"episodes of 8 days or more:\n{x8:.0f}% of the episodes,"
                f"\n{y8:.0f}% of the hypoxic days",
                xy=(x8, y8), xytext=(0.97, 0.05), textcoords="axes fraction",
                fontsize=8.2, color=C["grey"], va="bottom", ha="right",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C["grey"],
                                shrinkA=2, shrinkB=4))
    ax.text(72, 66, "equal-length episodes", fontsize=8.0, color=C["zero"],
            rotation=37, ha="center", va="center", rotation_mode="anchor")
    ax.set_xlabel("episodes ranked longest first (%)")
    ax.set_ylabel("cumulative share of\nhypoxic days (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 103)
    step = max(1, len(d) // 400)
    return pd.DataFrame({"episodes_pct": fe[::step], "hypoxic_days_pct": fd[::step]})


def main() -> int:
    t = site_table()
    ep = load("episodes_hyp2.parquet")
    rc = load("recovery_hyp2.parquet")

    fig = plt.figure(figsize=(DOUBLE, 132 / 25.4))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.18, 1.0],
                  hspace=0.44, wspace=0.42)
    axa = fig.add_subplot(gs[0, :2], projection=ccrs.LambertConformal(
        central_longitude=-96, central_latitude=38))
    panel_a(axa, t)
    panel_tag(axa, "a", dx=0.005, dy=0.99)

    axb = fig.add_subplot(gs[0, 2])
    panel_b(axb, ep)
    panel_tag(axb, "b", dx=-0.30, dy=1.10)
    axc = fig.add_subplot(gs[1, 0])
    share = panel_e(axc, ep)
    panel_tag(axc, "c", dx=-0.28, dy=1.12)
    share.to_csv(__import__("config").SRC / "fig1c_day_concentration.csv", index=False)
    axd = fig.add_subplot(gs[1, 1])
    hz = panel_c(axd, rc)
    panel_tag(axd, "d", dx=-0.24, dy=1.14)
    axe = fig.add_subplot(gs[1, 2])
    panel_d(axe, ep)
    panel_tag(axe, "e", dx=-0.24, dy=1.14)

    save(fig, "fig1_episodes", src=t)
    hz.to_csv(__import__("config").SRC / "fig1d_exit_hazard.csv", index=False)

    print("sites", len(t), "with episodes", int((t.n_ep > 0).sum()),
          "median episodes/summer (episodic sites)",
          round(t.loc[t.n_ep > 0, "ep_per_summer"].median(), 2),
          "median % of days hypoxic",
          round(t.loc[t.n_ep > 0, "frac_days"].median(), 2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
