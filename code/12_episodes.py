"""Hypoxic episodes and the two discrete-time risk sets.

An episode is a run of consecutive calendar days with daily minimum oxygen at or
below the threshold. A missing or fouled day breaks the run and censors it.

onset risk set     : oxygen above threshold today, next calendar day observed
recovery risk set  : oxygen at or below threshold today, next calendar day observed
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import HYP, OUT, STRESS, ensure

PRED = ["aT", "aU", "aL"]


def label_episodes(df: pd.DataFrame, hypcol: str) -> pd.DataFrame:
    d = df.sort_values(["site_id", "date"]).copy()
    hyp = d[hypcol].to_numpy()
    valid = d["valid_do"].to_numpy()
    site = d["site_id"].to_numpy()
    day = d["date"].values.astype("datetime64[D]").astype(int)
    n = len(d)
    ep_id = np.full(n, -1)
    spell = np.full(n, np.nan)
    cur = -1
    for i in range(n):
        cont = (
            i > 0
            and site[i] == site[i - 1]
            and day[i] - day[i - 1] == 1
            and valid[i - 1]
            and hyp[i - 1]
        )
        if valid[i] and hyp[i]:
            if cont and cur >= 0:
                spell[i] = spell[i - 1] + 1
            else:
                cur += 1
                spell[i] = 1
            ep_id[i] = cur
    d["ep_id"] = ep_id
    d["spell_day"] = spell
    return d


def risk_sets(d: pd.DataFrame, hypcol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = d.sort_values(["site_id", "date"]).copy()
    site = d["site_id"].to_numpy()
    day = d["date"].values.astype("datetime64[D]").astype(int)
    hyp = d[hypcol].to_numpy()
    valid = d["valid_do"].to_numpy()
    nxt_ok = np.zeros(len(d), bool)
    nxt_hyp = np.zeros(len(d), bool)
    nxt_ok[:-1] = (site[1:] == site[:-1]) & (day[1:] - day[:-1] == 1) & valid[1:]
    nxt_hyp[:-1] = hyp[1:]
    d["nxt_ok"] = nxt_ok
    d["nxt_hyp"] = nxt_hyp
    base = d["valid_do"] & d["nxt_ok"] & d[PRED].notna().all(axis=1)
    onset = d[base & ~d[hypcol]].copy()
    onset["y"] = onset["nxt_hyp"].astype(int)
    rec = d[base & d[hypcol]].copy()
    rec["y"] = (~rec["nxt_hyp"]).astype(int)
    return onset, rec


def episode_table(d: pd.DataFrame) -> pd.DataFrame:
    e = d[d["ep_id"] >= 0]
    if e.empty:
        return pd.DataFrame()
    t = e.groupby("ep_id").agg(
        site_id=("site_id", "first"),
        lat=("lat", "first"),
        lon=("lon", "first"),
        year=("year", "first"),
        start=("date", "min"),
        end=("date", "max"),
        dur=("date", "size"),
        do_min=("do_min", "min"),
        tw=("tw_mean", "mean"),
        U=("U_mean", "mean"),
        aU=("aU", "mean"),
        aT=("aT", "mean"),
    ).reset_index()
    return t


def within_z(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        g = d.groupby("site_id")[c]
        d[c + "_z"] = (d[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return d


def main() -> int:
    ensure()
    p = pd.read_parquet(OUT / "daily_panel.parquet")
    p["hyp"] = p["valid_do"] & (p["do_min"] <= HYP)
    p["str4"] = p["valid_do"] & (p["do_min"] <= STRESS)
    # one common within-site scale for every model, so that onset and recovery
    # coefficients are read on the same unit
    p = within_z(p, ["aT", "aU", "aL", "aQ", "deficit"])

    for tag, col in [("hyp2", "hyp"), ("str4", "str4")]:
        d = label_episodes(p, col)
        ep = episode_table(d)
        ep.to_parquet(OUT / f"episodes_{tag}.parquet", index=False)
        onset, rec = risk_sets(d, col)
        keep = [
            "site_id", "date", "year", "lat", "lon", "doy", "y", "spell_day",
            "do_min", "do_mean", "amp", "deficit", "sat_frac", "do_sat",
            "tw_mean", "U_mean", "U_min", "ssrd_MJ", "logq",
            "aT", "aU", "aL", "aQ", "aAmp",
            "aT_z", "aU_z", "aL_z", "aQ_z", "deficit_z",
        ]
        onset = onset[keep]
        rec = rec[keep]
        onset.to_parquet(OUT / f"onset_{tag}.parquet", index=False)
        rec.to_parquet(OUT / f"recovery_{tag}.parquet", index=False)
        print(
            tag,
            "episodes", len(ep),
            "sites with episodes", ep.site_id.nunique() if len(ep) else 0,
            "median dur", float(ep.dur.median()) if len(ep) else np.nan,
            "| onset rows", len(onset), "events", int(onset.y.sum()),
            "| recovery rows", len(rec), "events", int(rec.y.sum()),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
