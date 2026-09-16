"""Entry and exit hazards of river hypoxia, and a formal test that they differ.

Three model families:
  1. separate discrete-time logistic hazards for onset and for recovery;
  2. one stacked state-transition model whose interaction terms test directly
     whether a driver acts differently on entry than on exit;
  3. robustness: 4 mg L-1 threshold, discharge control, site fixed effects,
     spatially displaced wind as a falsification.
Standard errors are clustered on site throughout.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import OUT, ensure

BASE = ["aT_z", "aU_z", "aL_z"]


def fit(d: pd.DataFrame, cols: list[str], label: str) -> dict:
    d = d.dropna(subset=cols + ["y"])
    X = sm.add_constant(d[cols].astype(float))
    m = sm.Logit(d["y"].astype(float), X).fit(disp=0, cov_type="cluster",
                                              cov_kwds={"groups": d["site_id"]})
    ci = m.conf_int()
    out = {
        "label": label,
        "n": int(len(d)),
        "events": int(d["y"].sum()),
        "sites": int(d["site_id"].nunique()),
        "base_rate": float(d["y"].mean()),
        "terms": {},
    }
    for k in m.params.index:
        out["terms"][k] = {
            "b": float(m.params[k]),
            "lo": float(ci.loc[k, 0]),
            "hi": float(ci.loc[k, 1]),
            "p": float(m.pvalues[k]),
            "or": float(np.exp(m.params[k])),
            "or_lo": float(np.exp(ci.loc[k, 0])),
            "or_hi": float(np.exp(ci.loc[k, 1])),
        }
    return out


def stacked_transition(onset: pd.DataFrame, rec: pd.DataFrame) -> dict:
    """P(hypoxic tomorrow) over all days, interacted with today's state.

    H = 1 when the site is hypoxic today. The interaction coefficient on a
    driver is the difference between its action on persistence and on entry,
    which is the asymmetry the paper reports.
    """
    a = onset.copy()
    a["H"] = 0.0
    a["hyp_next"] = a["y"].astype(float)          # entered hypoxia
    b = rec.copy()
    b["H"] = 1.0
    b["hyp_next"] = 1.0 - b["y"].astype(float)    # stayed hypoxic
    d = pd.concat([a, b], ignore_index=True).dropna(subset=BASE)
    for c in BASE:
        d[c + "xH"] = d[c] * d["H"]
    # duration dependence only exists once a site is already hypoxic
    d["Hxlspell"] = np.where(d["H"] > 0, np.log(d["spell_day"].fillna(1.0)), 0.0)
    cols = BASE + ["H"] + [c + "xH" for c in BASE] + ["Hxlspell"]
    X = sm.add_constant(d[cols].astype(float))
    m = sm.Logit(d["hyp_next"].astype(float), X).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": d["site_id"]})
    ci = m.conf_int()
    res = {"n": int(len(d)), "sites": int(d["site_id"].nunique()), "terms": {}}
    for k in m.params.index:
        res["terms"][k] = {"b": float(m.params[k]), "lo": float(ci.loc[k, 0]),
                           "hi": float(ci.loc[k, 1]), "p": float(m.pvalues[k])}
    k = len(m.params)
    R = np.zeros((3, k))
    for i, c in enumerate(BASE):
        R[i, list(m.params.index).index(c + "xH")] = 1.0
    joint = m.wald_test(R, scalar=True)
    res["joint_interaction"] = {"chi2": float(joint.statistic), "p": float(joint.pvalue), "df": 3}
    return res


def expected_duration(rec_fit: dict, aU_values=(-1.0, 0.0, 1.0), tmax: int = 200) -> dict:
    """Mean and median episode length implied by the fitted exit hazard."""
    t = rec_fit["terms"]
    b0, bU, bs = t["const"]["b"], t["aU_z"]["b"], t["lspell"]["b"]
    out = {}
    for u in aU_values:
        days = np.arange(1, tmax + 1)
        h = 1.0 / (1.0 + np.exp(-(b0 + bU * u + bs * np.log(days))))
        surv = np.cumprod(1.0 - h)
        mean_dur = 1.0 + surv.sum()
        med = int(days[np.argmax(surv <= 0.5)]) if (surv <= 0.5).any() else tmax
        out[f"aU={u:+.0f}"] = {"mean_days": float(mean_dur), "median_days": med,
                               "hazard_day1": float(h[0])}
    return out


def site_exit_slopes(rec: pd.DataFrame, min_days: int = 40, min_ev: int = 6) -> pd.DataFrame:
    rows = []
    for s, g in rec.dropna(subset=BASE).groupby("site_id"):
        if len(g) < min_days or g["y"].sum() < min_ev or g["y"].sum() == len(g):
            continue
        try:
            X = sm.add_constant(g[["aU_z", "aT_z"]].astype(float))
            m = sm.Logit(g["y"].astype(float), X).fit(disp=0)
            if not np.isfinite(m.bse["aU_z"]) or m.bse["aU_z"] > 5:
                continue
            rows.append({"site_id": s, "lat": g["lat"].iloc[0], "lon": g["lon"].iloc[0],
                         "bU": float(m.params["aU_z"]), "seU": float(m.bse["aU_z"]),
                         "bT": float(m.params["aT_z"]),
                         "n": len(g), "ev": int(g["y"].sum()),
                         "exit_rate": float(g["y"].mean())})
        except Exception:
            continue
    return pd.DataFrame(rows)


def seasonal_scale(panel: pd.DataFrame) -> dict:
    """What a seasonal-mean study would have seen with the same sites."""
    p = panel[panel["valid_do"]].copy()
    sy = p.groupby(["site_id", "year"], as_index=False).agg(
        hyp_days=("hyp2", "sum"), do_mean=("do_mean", "mean"),
        U=("U_mean", "mean"), tw=("tw_mean", "mean"), n=("do_min", "size"),
        lat=("lat", "first"), lon=("lon", "first"))
    for c in ["U", "tw", "hyp_days", "do_mean"]:
        g = sy.groupby("site_id")[c]
        sy[c + "_z"] = (sy[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    res = {}
    for y in ["hyp_days_z", "do_mean_z"]:
        d = sy.dropna(subset=[y, "U_z", "tw_z"])
        X = sm.add_constant(d[["U_z", "tw_z"]].astype(float))
        m = sm.OLS(d[y].astype(float), X).fit(cov_type="cluster",
                                              cov_kwds={"groups": d["site_id"]})
        ci = m.conf_int()
        res[y] = {"n": int(len(d)), "sites": int(d["site_id"].nunique()),
                  "terms": {k: {"b": float(m.params[k]), "lo": float(ci.loc[k, 0]),
                                "hi": float(ci.loc[k, 1]), "p": float(m.pvalues[k])}
                            for k in m.params.index}}
    sy.to_parquet(OUT / "site_summer.parquet", index=False)
    return res


def displaced_wind(rec: pd.DataFrame, panel: pd.DataFrame, seed: int = 11) -> dict:
    """Falsification: replace each site's wind with a distant site's same-day wind.

    Donor series come from the full daily panel, so the risk set is preserved and
    only the spatial pairing of wind to oxygen is broken.
    """
    rng = np.random.default_rng(seed)
    don = panel[["site_id", "date", "aU", "lat", "lon"]].dropna(subset=["aU"]).copy()
    g = don.groupby("site_id")["aU"]
    don["aU_z"] = (don["aU"] - g.transform("mean")) / g.transform("std")
    sites = np.array(sorted(rec["site_id"].unique()))
    coords = rec.groupby("site_id")[["lat", "lon"]].first()
    donors = {}
    pool = np.array(sorted(don["site_id"].unique()))
    pool_xy = don.groupby("site_id")[["lat", "lon"]].first().loc[pool]
    for s in sites:
        la, lo = coords.loc[s]
        far = pool[(np.abs(pool_xy["lat"].to_numpy() - la) > 8)
                   | (np.abs(pool_xy["lon"].to_numpy() - lo) > 20)]
        donors[s] = rng.choice(far) if len(far) else s
    key = don[["site_id", "date", "aU_z"]].rename(
        columns={"site_id": "donor", "aU_z": "aU_fake"})
    d = rec.copy()
    d["donor"] = d["site_id"].map(donors)
    d = d.merge(key, on=["donor", "date"], how="left")
    d = d.drop(columns=["aU_z"]).rename(columns={"aU_fake": "aU_z"})
    d = d.dropna(subset=["aU_z", "aT_z", "aL_z"])
    return fit(d, ["aT_z", "aU_z", "aL_z", "lspell"], "recovery, wind from a distant site")


def decompose_season(panel: pd.DataFrame, ep: pd.DataFrame) -> dict:
    """Hypoxic days per summer = number of episodes x mean episode length.

    Regressing each channel on the same seasonal wind anomaly shows whether a
    seasonal-mean signal comes from exposure or from persistence.
    """
    p = panel[panel["valid_do"]].copy()
    sy = p.groupby(["site_id", "year"], as_index=False).agg(
        hyp_days=("hyp2", "sum"), U=("U_mean", "mean"), tw=("tw_mean", "mean"),
        lat=("lat", "first"), lon=("lon", "first"))
    cnt = ep.groupby(["site_id", "year"], as_index=False).agg(
        n_ep=("dur", "size"), mean_dur=("dur", "mean"), max_dur=("dur", "max"))
    sy = sy.merge(cnt, on=["site_id", "year"], how="left")
    sy[["n_ep", "mean_dur"]] = sy[["n_ep", "mean_dur"]].fillna({"n_ep": 0.0})
    # keep sites that actually get hypoxia, so that the channels are defined
    ok = sy.groupby("site_id")["hyp_days"].sum() > 0
    sy = sy[sy["site_id"].isin(ok[ok].index)]
    for c in ["U", "tw", "hyp_days", "n_ep", "mean_dur", "max_dur"]:
        g = sy.groupby("site_id")[c]
        sy[c + "_z"] = (sy[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    sy.to_parquet(OUT / "site_summer_episodes.parquet", index=False)
    res = {}
    for y in ["hyp_days_z", "n_ep_z", "mean_dur_z", "max_dur_z"]:
        d = sy.dropna(subset=[y, "U_z", "tw_z"])
        X = sm.add_constant(d[["U_z", "tw_z"]].astype(float))
        m = sm.OLS(d[y].astype(float), X).fit(cov_type="cluster",
                                              cov_kwds={"groups": d["site_id"]})
        ci = m.conf_int()
        res[y] = {"n": int(len(d)), "sites": int(d["site_id"].nunique()),
                  "terms": {k: {"b": float(m.params[k]), "lo": float(ci.loc[k, 0]),
                                "hi": float(ci.loc[k, 1]), "p": float(m.pvalues[k])}
                            for k in m.params.index}}
    return res


def main() -> int:
    ensure()
    panel = pd.read_parquet(OUT / "daily_panel.parquet")
    panel["hyp2"] = panel["valid_do"] & (panel["do_min"] <= 2.0)

    res: dict = {}
    for tag in ["hyp2", "str4"]:
        on = pd.read_parquet(OUT / f"onset_{tag}.parquet")
        rc = pd.read_parquet(OUT / f"recovery_{tag}.parquet")
        rc["lspell"] = np.log(rc["spell_day"])
        res[f"onset_{tag}"] = fit(on, BASE, f"onset {tag}")
        res[f"recovery_{tag}"] = fit(rc, BASE + ["lspell"], f"recovery {tag}")
        res[f"transition_{tag}"] = stacked_transition(on, rc)
        if tag == "hyp2":
            res["recovery_hyp2_Q"] = fit(rc, BASE + ["aQ_z", "lspell"], "recovery + discharge")
            res["onset_hyp2_Q"] = fit(on, BASE + ["aQ_z"], "onset + discharge")
            res["recovery_hyp2_deficit"] = fit(
                rc.assign(UxD=rc["aU_z"] * rc["deficit_z"]),
                BASE + ["deficit_z", "UxD", "lspell"], "recovery + deficit interaction")
            res["duration"] = expected_duration(res["recovery_hyp2"])
            res["placebo"] = displaced_wind(rc, panel)
            slopes = site_exit_slopes(rc)
            slopes.to_parquet(OUT / "site_exit_slopes.parquet", index=False)
            res["site_exit"] = {
                "n_sites": int(len(slopes)),
                "median_bU": float(slopes["bU"].median()),
                "frac_positive": float((slopes["bU"] > 0).mean()),
                "frac_sig_positive": float(((slopes["bU"] - 1.96 * slopes["seU"]) > 0).mean()),
            }
            # regional split on episode-rich sites
            for name, q in [("south", slopes["lat"] < 35), ("north", slopes["lat"] >= 35)]:
                sub = slopes[q]
                if len(sub) >= 10:
                    res["site_exit"][name] = {"n": int(len(sub)),
                                              "median_bU": float(sub["bU"].median())}

    res["seasonal"] = seasonal_scale(panel)
    res["decompose"] = decompose_season(panel, pd.read_parquet(OUT / "episodes_hyp2.parquet"))
    (OUT / "hazard.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    def show(k):
        r = res[k]
        print(f"\n[{k}] n={r['n']} ev={r.get('events')} sites={r['sites']}")
        for t, v in r["terms"].items():
            if t == "const":
                continue
            print(f"   {t:>10s}  b={v['b']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}] "
                  f"OR={v.get('or', float('nan')):.3f} p={v['p']:.2e}")

    for k in ["onset_hyp2", "recovery_hyp2", "onset_str4", "recovery_str4",
              "recovery_hyp2_Q", "placebo"]:
        show(k)
    tr = res["transition_hyp2"]["terms"]
    print("\n[transition hyp2] interaction terms (persistence minus entry)")
    for c in BASE:
        v = tr[c + "xH"]
        print(f"   {c}xH  b={v['b']:+.3f} [{v['lo']:+.3f},{v['hi']:+.3f}] p={v['p']:.2e}")
    print("   joint", res["transition_hyp2"]["joint_interaction"])
    print("\n[duration]", json.dumps(res["duration"], indent=2))
    print("[site exit]", json.dumps(res["site_exit"], indent=2))
    print("\n[seasonal + decomposition] coefficient on seasonal wind anomaly")
    for k, r in list(res["seasonal"].items()) + list(res["decompose"].items()):
        v = r["terms"]["U_z"]
        w = r["terms"]["tw_z"]
        print(f"   {k:>12s} n={r['n']:5d} U:{v['b']:+.3f}[{v['lo']:+.3f},{v['hi']:+.3f}] "
              f"p={v['p']:.1e}   tw:{w['b']:+.3f}[{w['lo']:+.3f},{w['hi']:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
