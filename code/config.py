"""Repository paths. Defaults are relative to the repository root."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
INPUTS = DATA / "inputs"
OUT = DATA / "derived"
FIG = ROOT / "figures"
SI_FIG = FIG / "si"
SRC = FIG / "source_data"

# Hourly ERA5 point extracts are not shipped. Set this only to rebuild
# data/derived/era5_daily_plus.parquet from local cell files.
_cells = os.environ.get("ERA5_HOURLY_CELLS", "")
CELLS = Path(_cells) if _cells else ROOT / "data" / "era5_hourly_cells"

HYP = 2.0        # hypoxia line, mg L-1
STRESS = 4.0     # milder line used for replication
MIN_GAP = 1      # days of non-hypoxia needed to close an episode


def ensure() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (FIG, SI_FIG, SRC):
        p.mkdir(parents=True, exist_ok=True)
    return OUT


def era5_daily_path() -> Path:
    """Return the daily ERA5 table, joining the two shipped parts if needed."""
    whole = OUT / "era5_daily_plus.parquet"
    if whole.exists():
        return whole
    p0 = OUT / "era5_daily_plus_part0.parquet"
    p1 = OUT / "era5_daily_plus_part1.parquet"
    if p0.exists() and p1.exists():
        pd.concat(
            [pd.read_parquet(p0), pd.read_parquet(p1)],
            ignore_index=True,
        ).to_parquet(whole, index=False)
        return whole
    raise FileNotFoundError(
        "Missing data/derived/era5_daily_plus_part0.parquet and part1. "
        "Rebuild with 10_era5_daily_plus.py if you have hourly cell extracts."
    )
