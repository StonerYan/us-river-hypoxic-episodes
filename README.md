# Entry and exit of hypoxic episodes in United States rivers

Code, analysis-ready tables and figure source data for a study of low-oxygen
spells in monitored rivers of the continental United States. Each spell is
treated as an episode with two edges. The daily probability of entering one
and the daily probability of leaving it are estimated separately, with the
same weather anomalies, on the same scale, at the same rivers.

Across 358 rivers and fifteen warm seasons (2010–2024, 1 May to 31 October),
entry follows a warm anomaly in water temperature, while exit follows a
windier-than-usual day. The wind response on the daily oxygen minimum grows
with the saturation deficit. The exit signal is strongest at a one-day wind
window and is no longer recoverable once wind is averaged over a week.

## Layout

```
code/                 analysis and figure scripts, run in numerical order
data/inputs/          screened USGS summer daily values and site coordinates
data/derived/         daily panel, episode tables, risk sets, model estimates
figures/              rendered figures
figures/si/           supplementary figures
figures/source_data/  one CSV per panel
```

## Requirements

Python 3.10 or later.

```
pip install -r requirements.txt
```

## Reproduce

The files under `data/derived/` are enough to re-run the estimates from the
episode tables onward and to redraw every figure. Scripts run from `code/`.

| Script | What it produces |
|---|---|
| `10_era5_daily_plus.py` | daily wind, air temperature and shortwave per 0.25° cell (optional; needs local hourly ERA5 extracts) |
| `11_build_panel.py` | screened daily river panel with saturation, deficit, diel amplitude and within-site day-of-year anomalies |
| `12_episodes.py` | episode tables and the two disjoint risk sets, at 2 and 4 mg L⁻¹ |
| `13_hazard.py` | entry and exit hazards, stacked interaction test, robustness and spatial-displacement fits |
| `14_rf.py` | random forests with river-blocked cross-validation and permutation importance |
| `15_mechanism.py` | daily oxygen budget, wind × deficit scaling, diel amplitude, within-episode contrast |
| `16_scale.py` | exit model refitted over trailing wind windows of 1–30 days |
| `20`–`24_fig*.py` | main figures 1–5 |
| `25_figs_si.py` | supplementary figures S1–S7 |

`fig_style.py` holds the shared figure style. Every panel writes its own
source-data CSV under `figures/source_data/`.

The daily ERA5 table is stored as two parts under `data/derived/`. The first
script that needs it joins them automatically. To rebuild that table from
hourly point extracts, set `ERA5_HOURLY_CELLS` to a directory of
`cell_*.parquet` files. That hourly archive is not shipped.

```
cd code
python 13_hazard.py
python 20_fig1_object.py
```

## Data

Primary observations are public. Daily dissolved oxygen, water temperature
and discharge are from the National Water Information System of the United
States Geological Survey (https://waterdata.usgs.gov/nwis). Hourly 10 m wind
components, 2 m air temperature and surface downwelling shortwave radiation
are from the ERA5 single-level reanalysis of the Copernicus Climate Change
Service (https://doi.org/10.24381/cds.adbb2d47). Oxygen solubility follows
the freshwater equilibrium expression of Benson and Krause (1984).

This repository ships the screened summer daily-values table, the site list,
the daily panel, the episode and risk-set tables at both thresholds, the
model estimates, and the source data for every figure panel. Hourly ERA5
cell extracts and the raw USGS instantaneous records are not included.
