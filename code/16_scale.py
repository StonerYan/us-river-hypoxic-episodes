"""How long an averaging window survives: the exit signal is a day-scale signal.

The exit hazard is refitted with the wind anomaly replaced by its trailing mean
over windows of 1 to 30 days. Everything else in the model is unchanged, so the
only thing that varies is the time scale at which wind is measured.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import OUT, ensure

WINDOWS = [1, 2, 3, 5, 7, 10, 15, 21, 30]


def main() -> int:
    ensure()
    p = pd.read_parquet(OUT / "daily_panel.parquet")[
        ["site_id", "date", "aU", "aT", "aL"]].sort_values(["site_id", "date"])
    rc = pd.read_parquet(OUT / "recovery_hyp2.parquet")
    rc["lspell"] = np.log(rc["spell_day"])

    roll = p[["site_id", "date"]].copy()
    for w in WINDOWS:
        roll[f"U{w}"] = (p.groupby("site_id")["aU"]
                         .transform(lambda s: s.rolling(w, min_periods=w).mean()))
    for c in [f"U{w}" for w in WINDOWS]:
        g = roll.groupby("site_id")[c]
        roll[c] = (roll[c] - g.transform("mean")) / g.transform("std")

    d = rc.merge(roll, on=["site_id", "date"], how="left")
    rows = []
    for w in WINDOWS:
        col = f"U{w}"
        s = d.dropna(subset=[col, "aT_z", "aL_z"])
        X = sm.add_constant(s[[col, "aT_z", "aL_z", "lspell"]].astype(float))
        m = sm.Logit(s["y"].astype(float), X).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": s["site_id"]})
        ci = m.conf_int()
        rows.append({"window_days": w, "n": int(len(s)),
                     "b": float(m.params[col]), "lo": float(ci.loc[col, 0]),
                     "hi": float(ci.loc[col, 1]), "p": float(m.pvalues[col]),
                     "or": float(np.exp(m.params[col]))})
    r = pd.DataFrame(rows)
    r.to_parquet(OUT / "wind_window.parquet", index=False)
    (OUT / "scale.json").write_text(
        json.dumps({"windows": rows}, indent=2), encoding="utf-8")
    print(r.round(4).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
