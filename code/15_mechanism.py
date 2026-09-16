"""Mechanism and event-scale evidence behind the entry-exit asymmetry.

  A. daily oxygen budget: does the wind gain in the daily minimum scale with the
     saturation deficit, as a gas-transfer flux must?
  B. diel amplitude: does wind change the metabolic swing, or only the floor?
  C. within-episode contrast: exit day versus the persistence days of the same
     episode, which removes site and episode as confounders by construction.
  D. empirical exit hazard against episode age and same-day wind.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import OUT, ensure

SEED = 7


def tidy(m, keys=None) -> dict:
    ci = m.conf_int()
    ks = keys or list(m.params.index)
    return {k: {"b": float(m.params[k]), "lo": float(ci.loc[k, 0]),
                "hi": float(ci.loc[k, 1]), "p": float(m.pvalues[k])} for k in ks}


def budget(panel: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    p = panel.sort_values(["site_id", "date"]).reset_index(drop=True).copy()
    for c in ["aT", "aU", "aL", "aAmp"]:
        g = p.groupby("site_id")[c]
        p[c + "_z"] = (p[c] - g.transform("mean")) / g.transform("std")
    s = p["site_id"].to_numpy()
    day = p["date"].values.astype("datetime64[D]").astype(int)
    cont = np.zeros(len(p), bool)
    cont[:-1] = (s[1:] == s[:-1]) & (day[1:] - day[:-1] == 1)
    for c in ["do_min", "do_mean", "do_max"]:
        v = p[c].to_numpy(dtype=float)
        p["d_" + c] = np.where(cont, np.r_[v[1:], np.nan] - v, np.nan)
    p["sat_def"] = p["do_sat"] - p["do_mean"]

    res = {}
    cols = ["aU_z", "aT_z", "aL_z", "sat_def", "UxD"]
    for y in ["d_do_min", "d_do_mean", "d_do_max"]:
        d = p.dropna(subset=[y, "aU_z", "aT_z", "aL_z", "sat_def"]).copy()
        d["UxD"] = d["aU_z"] * d["sat_def"]
        X = sm.add_constant(d[cols].astype(float))
        m = sm.OLS(d[y].astype(float), X).fit(cov_type="cluster",
                                              cov_kwds={"groups": d["site_id"]})
        res[y] = {"n": int(len(d)), "sites": int(d["site_id"].nunique()),
                  "terms": tidy(m)}

    # binned wind sensitivity of the daily minimum by saturation deficit
    d = p.dropna(subset=["d_do_min", "aU_z", "sat_def", "aT_z", "aL_z"]).copy()
    d["qdef"] = pd.qcut(d["sat_def"], 5, labels=False, duplicates="drop")
    rows = []
    for q, g in d.groupby("qdef"):
        X = sm.add_constant(g[["aU_z", "aT_z", "aL_z"]].astype(float))
        m = sm.OLS(g["d_do_min"].astype(float), X).fit(
            cov_type="cluster", cov_kwds={"groups": g["site_id"]})
        ci = m.conf_int()
        rows.append({"q": int(q), "def_mid": float(g["sat_def"].median()),
                     "n": int(len(g)),
                     "bU": float(m.params["aU_z"]), "lo": float(ci.loc["aU_z", 0]),
                     "hi": float(ci.loc["aU_z", 1])})
    bins = pd.DataFrame(rows)

    # C: amplitude
    a = p.dropna(subset=["aAmp_z", "aU_z", "aT_z", "aL_z"])
    X = sm.add_constant(a[["aU_z", "aT_z", "aL_z"]].astype(float))
    m = sm.OLS(a["aAmp_z"].astype(float), X).fit(cov_type="cluster",
                                                 cov_kwds={"groups": a["site_id"]})
    res["amplitude"] = {"n": int(len(a)), "sites": int(a["site_id"].nunique()),
                        "terms": tidy(m)}
    return res, bins


def episode_ids(rc: pd.DataFrame) -> pd.DataFrame:
    d = rc.sort_values(["site_id", "date"]).copy()
    s = d["site_id"].to_numpy()
    day = d["date"].values.astype("datetime64[D]").astype(int)
    new = np.ones(len(d), bool)
    new[1:] = (s[1:] != s[:-1]) | (day[1:] - day[:-1] != 1)
    d["ep"] = np.cumsum(new)
    return d


def within_episode(rc: pd.DataFrame, min_days: int = 3, n_boot: int = 4000) -> dict:
    d = episode_ids(rc.dropna(subset=["aU_z", "aT_z", "aL_z"]))
    sz = d.groupby("ep").size()
    d = d[d["ep"].isin(sz[sz >= min_days].index)]
    rows = []
    for ep, g in d.groupby("ep"):
        ex, ns = g[g["y"] == 1], g[g["y"] == 0]
        if ex.empty or ns.empty:
            continue
        r = {k: float(ex[k].mean() - ns[k].mean()) for k in ["aU_z", "aT_z", "aL_z"]}
        r["ep"] = ep
        r["site_id"] = g["site_id"].iloc[0]
        r["n_days"] = len(g)
        rows.append(r)
    diff = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    out = {"n_episodes": int(len(diff)), "sites": int(diff["site_id"].nunique()),
           "min_days": min_days, "terms": {}}
    for c in ["aU_z", "aT_z", "aL_z"]:
        v = diff[c].to_numpy()
        bs = np.array([np.nanmean(rng.choice(v, v.size, replace=True))
                       for _ in range(n_boot)])
        out["terms"][c] = {"mean": float(np.nanmean(v)),
                           "lo": float(np.percentile(bs, 2.5)),
                           "hi": float(np.percentile(bs, 97.5))}
    diff.to_parquet(OUT / "within_episode_contrast.parquet", index=False)
    return out


def hazard_curves(rc: pd.DataFrame) -> pd.DataFrame:
    d = rc.dropna(subset=["aU_z"]).copy()
    d["terc"] = pd.qcut(d["aU_z"], 3, labels=["calm", "median", "windy"])
    rows = []
    for (t, k), g in d.groupby(["terc", d["spell_day"].clip(upper=8)], observed=True):
        if len(g) < 40:
            continue
        p = g["y"].mean()
        se = np.sqrt(p * (1 - p) / len(g))
        rows.append({"terc": str(t), "spell": int(k), "n": int(len(g)),
                     "exit": float(p), "lo": float(max(p - 1.96 * se, 0)),
                     "hi": float(min(p + 1.96 * se, 1))})
    return pd.DataFrame(rows)


def main() -> int:
    ensure()
    panel = pd.read_parquet(OUT / "daily_panel.parquet")
    rc = pd.read_parquet(OUT / "recovery_hyp2.parquet")

    res, bins = budget(panel)
    bins.to_parquet(OUT / "wind_gain_by_deficit.parquet", index=False)
    res["within_episode"] = within_episode(rc)
    hz = hazard_curves(rc)
    hz.to_parquet(OUT / "hazard_curves.parquet", index=False)
    (OUT / "mechanism.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    for k in ["d_do_min", "d_do_max", "amplitude"]:
        r = res[k]
        print(f"\n[{k}] n={r['n']} sites={r['sites']}")
        for t, v in r["terms"].items():
            if t == "const":
                continue
            print(f"   {t:>8s} b={v['b']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] p={v['p']:.2e}")
    print("\n[wind gain in daily minimum by saturation deficit quintile]")
    print(bins.round(4).to_string(index=False))
    print("\n[within-episode contrast, exit day minus persistence days]")
    print(json.dumps(res["within_episode"], indent=2))
    print("\n[empirical exit hazard]")
    print(hz.pivot(index="spell", columns="terc", values="exit").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
