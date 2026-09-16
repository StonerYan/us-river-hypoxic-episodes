# Data

## `inputs/`

| File | Contents |
|---|---|
| `usgs_dv_summer.parquet` | USGS daily-values oxygen, water temperature and discharge for warm-season days used to build the panel |
| `rivers_candidates.parquet` | site coordinates and the ERA5 cell assigned to each river |

## `derived/`

| File | Contents |
|---|---|
| `era5_daily_plus_part0.parquet`, `era5_daily_plus_part1.parquet` | daily mean, minimum and maximum 10 m wind, 2 m air temperature and shortwave total by 0.25° cell, stored as two parts; the first call joins them |
| `daily_panel.parquet` | screened river-day panel with saturation, deficit, diel amplitude and within-site anomalies |
| `episodes_hyp2.parquet`, `episodes_str4.parquet` | episode tables at 2 and 4 mg L⁻¹ |
| `onset_hyp2.parquet`, `onset_str4.parquet` | entry risk sets |
| `recovery_hyp2.parquet`, `recovery_str4.parquet` | exit risk sets |
| `hazard.json` | pooled hazard estimates, stacked interaction test and robustness fits |
| `site_exit_slopes.parquet` | river-level exit wind coefficients |
| `site_summer.parquet`, `site_summer_episodes.parquet` | summer aggregates |
| `rf.json` | random-forest held-out skill and permutation importance |
| `mechanism.json`, `wind_gain_by_deficit.parquet`, `within_episode_contrast.parquet`, `hazard_curves.parquet` | daily oxygen-budget and within-episode results |
| `wind_window.parquet`, `scale.json` | exit wind coefficient by averaging window |

Hourly ERA5 point extracts are not stored here. Rebuild `era5_daily_plus.parquet`
only if you have a local copy of those extracts and set `ERA5_HOURLY_CELLS`.
