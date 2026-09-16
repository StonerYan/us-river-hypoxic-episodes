"""Daily river panel with diel amplitude, oxygen deficit and weather anomalies.

Screen: May-Oct 2010-2024, >=70% valid days in >=8 summers (rivers).
Anomalies are within-site day-of-year departures, so seasonality is removed
before any driver enters a model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import INPUTS, OUT, ensure, era5_daily_path

SUMMER_DAYS = 184
MIN_SUMMERS = 8
MIN_FRAC = 0.70
DOY_WIN = 15  # +/- days for the day-of-year climatology


def sid(s) -> str:
    t = str(s).strip()
    return t.lstrip("0") or "0"


def do_sat(tc: np.ndarray) -> np.ndarray:
    """Benson & Krause (1984) freshwater saturation, mg L-1 at 1 atm."""
    tk = tc + 273.15
    ln_c = (
        -139.34411
        + 1.575701e5 / tk
        - 6.642308e7 / tk**2
        + 1.243800e10 / tk**3
        - 8.621949e11 / tk**4
    )
    return np.exp(ln_c)


def doy_anomaly(df: pd.DataFrame, col: str) -> pd.Series:
    """Within-site departure from a smoothed day-of-year climatology."""
    out = np.full(len(df), np.nan)
    doy = df["doy"].to_numpy()
    vals = df[col].to_numpy(dtype=float)
    sites = df["site_id"].to_numpy()
    order = np.argsort(sites, kind="stable")
    for start, stop in _runs(sites[order]):
        idx = order[start:stop]
        d = doy[idx]
        v = vals[idx]
        clim = np.full(len(idx), np.nan)
        for k in range(len(idx)):
            m = np.abs(d - d[k]) <= DOY_WIN
            if m.sum() >= 5:
                clim[k] = np.nanmean(v[m])
        out[idx] = v - clim
    return pd.Series(out, index=df.index)


def _runs(arr):
    if len(arr) == 0:
        return []
    change = np.flatnonzero(arr[1:] != arr[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(arr)]))
    return list(zip(bounds[:-1], bounds[1:]))


def main() -> int:
    ensure()
    dv = pd.read_parquet(INPUTS / "usgs_dv_summer.parquet")
    dv["site_id"] = dv["site_id"].map(sid)
    dv["date"] = pd.to_datetime(dv["date"])

    riv = pd.read_parquet(INPUTS / "rivers_candidates.parquet")
    riv = riv[riv["source"] == "usgs_nwis"][
        ["site_id", "lat", "lon", "era5_lat", "era5_lon", "site_type"]
    ].copy()
    riv["site_id"] = riv["site_id"].map(sid)
    riv = riv.drop_duplicates("site_id")
    riv = riv[riv["lat"].between(26, 49) & riv["lon"].between(-125, -67)]

    df = dv.merge(riv, on="site_id", how="inner")
    df["year"] = df["date"].dt.year
    df = df[df["year"].between(2010, 2024)]

    # fouling: flat daily trace away from zero
    flat = df["do_max"].notna() & df["do_min"].notna() & (df["do_max"] == df["do_min"]) & (df["do_min"] > 0.2)
    df["valid_do"] = df["do_min"].notna() & ~flat

    g = df.groupby(["site_id", "year"], as_index=False).agg(n_ok=("valid_do", "sum"))
    g["frac"] = g["n_ok"] / SUMMER_DAYS
    keep_sy = g[g["frac"] >= MIN_FRAC][["site_id", "year"]]
    n_sum = keep_sy.groupby("site_id").size()
    keep_sites = n_sum[n_sum >= MIN_SUMMERS].index
    df = df.merge(keep_sy, on=["site_id", "year"]).query("site_id in @keep_sites").copy()
    print("screened rivers", df.site_id.nunique(), "site-summers", len(keep_sy.query("site_id in @keep_sites")), flush=True)

    era = pd.read_parquet(era5_daily_path())
    era["date"] = pd.to_datetime(era["date"])
    df = df.merge(era, on=["era5_lat", "era5_lon", "date"], how="left")

    df["doy"] = df["date"].dt.dayofyear
    df["amp"] = df["do_max"] - df["do_min"]
    sat = np.full(len(df), np.nan)
    tw = df["tw_mean"].to_numpy(dtype=float)
    ok = np.isfinite(tw)
    sat[ok] = do_sat(tw[ok])
    df["do_sat"] = sat
    df["deficit"] = df["do_sat"] - df["do_min"]
    df["sat_frac"] = df["do_min"] / df["do_sat"]
    df["logq"] = np.log10(df["q_mean"].clip(lower=0.01))
    df["ssrd_MJ"] = df["ssrd_sum"] / 1e6

    df = df.sort_values(["site_id", "date"]).reset_index(drop=True)
    for col, name in [
        ("tw_mean", "aT"),
        ("U_mean", "aU"),
        ("ssrd_MJ", "aL"),
        ("logq", "aQ"),
        ("amp", "aAmp"),
    ]:
        df[name] = doy_anomaly(df, col)
        print("anomaly", name, "n", int(df[name].notna().sum()), flush=True)

    df.to_parquet(OUT / "daily_panel.parquet", index=False)
    print("wrote daily_panel", df.shape, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
