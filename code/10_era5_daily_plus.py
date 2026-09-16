"""Daily ERA5 per 0.25 deg cell: wind speed moments, 2 m temperature, shortwave.

Adds surface solar radiation downwards (ssrd) so that light can enter the same
driver set as temperature, flow and wind.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CELLS, OUT, ensure


def main() -> int:
    ensure()
    if not CELLS.exists():
        raise SystemExit(
            "Hourly ERA5 cell extracts are not shipped. Set ERA5_HOURLY_CELLS "
            "to a directory of cell_*.parquet files, or use the shipped "
            "data/derived/era5_daily_plus.parquet."
        )
    files = sorted(CELLS.glob("cell_*.parquet"))
    print(f"cells {len(files)}", flush=True)
    dest = OUT / "era5_daily_plus.parquet"
    chunks = []
    for i, fp in enumerate(files, 1):
        stem = fp.stem.replace("cell_", "")
        lat_s, lon_s = stem.split("_", 1)
        cols = ["time", "u10", "v10", "t2m", "ssrd"]
        try:
            df = pd.read_parquet(fp, columns=cols)
        except Exception:
            df = pd.read_parquet(fp)
            for c in cols:
                if c not in df.columns:
                    df[c] = np.nan
            df = df[cols]
        df["time"] = pd.to_datetime(df["time"])
        u = df["u10"].to_numpy(dtype="float32")
        v = df["v10"].to_numpy(dtype="float32")
        df["U"] = np.sqrt(u * u + v * v)
        df["date"] = df["time"].dt.floor("D")
        g = df.groupby("date", as_index=False).agg(
            U_mean=("U", "mean"),
            U_min=("U", "min"),
            U_max=("U", "max"),
            t2m_mean=("t2m", "mean"),
            ssrd_sum=("ssrd", "sum"),
        )
        g["era5_lat"] = float(lat_s)
        g["era5_lon"] = float(lon_s)
        chunks.append(g)
        if i % 400 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    daily = pd.concat(chunks, ignore_index=True)
    daily.to_parquet(dest, index=False)
    print("wrote", dest, len(daily), "cells",
          daily.groupby(["era5_lat", "era5_lon"]).ngroups, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
