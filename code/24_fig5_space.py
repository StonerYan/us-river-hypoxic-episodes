"""Figure 5. Where wind governs the exit, and what makes a river responsive."""
from __future__ import annotations

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.gridspec import GridSpec

from config import SRC
from fig_style import (ANN, C, DOUBLE, EXIT, NOTE, load, panel_tag, save,
                       us_axes, zero_line)


def site_context() -> pd.DataFrame:
    p = load("daily_panel.parquet")
    ctx = p[p["valid_do"]].groupby("site_id").agg(
        U_bar=("U_mean", "mean"), tw_bar=("tw_mean", "mean"),
        q_bar=("q_mean", "median"), do_bar=("do_mean", "mean"),
        sat_bar=("sat_frac", "mean"), lat=("lat", "first"), lon=("lon", "first"))
    return ctx.reset_index()


def panel_a(ax, s: pd.DataFrame):
    us_axes(ax)
    sig = s[(s["bU"] - 1.96 * s["seU"]) > 0]
    rest = s[~s.index.isin(sig.index)]
    v = np.clip(s["bU"], -0.5, 0.9)
    sc = ax.scatter(s["lon"], s["lat"], s=54, c=v, cmap="RdYlBu_r",
                    vmin=-0.5, vmax=0.9, edgecolor="#606060", linewidth=0.35,
                    transform=ccrs.PlateCarree(), zorder=4)
    ax.scatter(sig["lon"], sig["lat"], s=112, facecolor="none", edgecolor="#1A1A1A",
               linewidth=0.8, transform=ccrs.PlateCarree(), zorder=5)
    cax = ax.inset_axes([0.66, 0.07, 0.30, 0.035])
    cb = plt.colorbar(sc, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("wind effect on exit (log-odds per s.d.)", fontsize=NOTE,
                 labelpad=1.5)
    cb.ax.tick_params(labelsize=7.0, width=0.5, length=1.8, pad=1.2)
    cb.outline.set_linewidth(0.4)
    ax.scatter([], [], s=40, facecolor="none", edgecolor="#1A1A1A", linewidth=0.7,
               label="95% interval above zero")
    ax.legend(loc="lower left", fontsize=NOTE, borderpad=0.2,
              bbox_to_anchor=(0.02, 0.04))
    print(f"  site map: {len(s)} rivers, {100 * (s['bU'] > 0).mean():.0f}% positive, "
          f"{100 * len(sig) / len(s):.0f}% individually above zero", flush=True)


def panel_b(ax, s: pd.DataFrame):
    """The whole distribution of river-by-river slopes, split at zero, with the
    smoothed density over it."""
    from scipy.stats import gaussian_kde
    v = s["bU"].dropna().to_numpy(dtype=float)
    edges = np.arange(-0.8, 1.36, 0.15)
    n, _, patches = ax.hist(v, bins=edges, edgecolor="white", linewidth=0.6,
                            zorder=2)
    for e, p in zip(edges[:-1], patches):
        p.set_facecolor(C["wind_l"] if e >= 0 else C["heat_l"])
    grid = np.linspace(edges[0], edges[-1], 300)
    dens = gaussian_kde(v)(grid) * len(v) * 0.15
    ax.plot(grid, dens, color=C["grey"], lw=1.1, zorder=3)
    zero_line(ax, "v")
    med = float(np.median(v))
    ax.plot(med, 0, "^", color=EXIT, ms=6.4, mec="white", mew=0.8, clip_on=False,
            zorder=4)
    ax.set_xlabel("site-level wind effect on exit\n(log-odds per s.d.)")
    ax.set_ylabel("number of rivers")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, 1.30 * n.max())
    ax.text(0.02, 0.96, f"{100 * (v > 0).mean():.0f}% of rivers positive\n"
            f"median {med:+.2f}, n = {len(v)} rivers", transform=ax.transAxes,
            ha="left", va="top", fontsize=ANN, color=C["grey"], linespacing=1.35,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


def panel_c(ax, s: pd.DataFrame):
    """Rivers that sit further from saturation respond more to wind, as a
    flux control predicts."""
    d = s.dropna(subset=["bU", "sat_bar"]).copy()
    ax.scatter(100 * d["sat_bar"], d["bU"], s=17, c=np.clip(d["bU"], -0.5, 0.9),
               cmap="RdYlBu_r", vmin=-0.5, vmax=0.9, alpha=0.85,
               edgecolor="white", linewidth=0.25, zorder=3)
    X = sm.add_constant(d[["sat_bar"]].astype(float))
    m = sm.OLS(d["bU"].astype(float), X).fit()
    xs = np.linspace(d["sat_bar"].min(), d["sat_bar"].max(), 40)
    pr = m.get_prediction(sm.add_constant(pd.DataFrame({"sat_bar": xs})))
    fr = pr.summary_frame(alpha=0.05)
    ax.plot(100 * xs, fr["mean"], color="#1A1A1A", lw=1.4, zorder=5)
    ax.fill_between(100 * xs, fr["mean_ci_lower"], fr["mean_ci_upper"],
                    color=C["grey"], alpha=0.16, linewidth=0, zorder=2)
    zero_line(ax, "h")
    ci = m.conf_int()
    ax.set_xlabel("mean summer oxygen saturation (%)")
    ax.set_ylabel("wind effect on exit\n(log-odds per s.d.)")
    ax.set_ylim(-1.32, 1.22)
    ax.text(0.03, 0.03, f"slope {m.params['sat_bar']:+.2f} per unit saturation "
            f"(95% CI {ci.loc['sat_bar', 0]:+.2f} to {ci.loc['sat_bar', 1]:+.2f}), "
            f"n = {len(d)}", transform=ax.transAxes, fontsize=ANN,
            color=C["grey"])
    return {"b": float(m.params["sat_bar"]), "lo": float(ci.loc["sat_bar", 0]),
            "hi": float(ci.loc["sat_bar", 1]), "p": float(m.pvalues["sat_bar"]),
            "n": int(len(d))}


def main() -> int:
    s = load("site_exit_slopes.parquet").merge(
        site_context().drop(columns=["lat", "lon"]), on="site_id", how="left")

    fig = plt.figure(figsize=(DOUBLE, 166 / 25.4))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.72, 1.0],
                  hspace=0.30, wspace=0.30)
    axa = fig.add_subplot(gs[0, :], projection=ccrs.LambertConformal(
        central_longitude=-96, central_latitude=38))
    panel_a(axa, s)
    panel_tag(axa, "a", dx=0.005, dy=0.99)
    axb = fig.add_subplot(gs[1, 0])
    panel_b(axb, s)
    panel_tag(axb, "b", dx=-0.16, dy=1.10)
    axc = fig.add_subplot(gs[1, 1])
    fit = panel_c(axc, s)
    panel_tag(axc, "c", dx=-0.16, dy=1.10)

    save(fig, "fig5_space", src=s.drop(columns=["site_id"]))
    print("saturation control on the wind effect:", fit, flush=True)
    pd.DataFrame([fit]).to_csv(SRC / "fig5c_saturation_control.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
