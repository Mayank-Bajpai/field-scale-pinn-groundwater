# Input data

The field data used in the study are **not distributed** with this repository. Groundwater-head and river-stage
observations are held by the Central Ground Water Board (Government of India) and the Smart Lab on Clean Rivers (SLCR),
IIT (BHU) Varanasi; access requires a research collaboration with SLCR. Exact observation coordinates are withheld.

To test the workflow without the restricted data, generate a synthetic dataset with the same schema:

```bash
python examples/make_synthetic_data.py --out data
```

## Files and schema

Place the files below in this folder (or point `load_and_preprocess_data(data_dir=...)`, `--data-dir`, or the
`PINN_DATA_DIR` environment variable at another folder). All X/Y coordinates must be in the **same projected coordinate
system (metres)**. Dates are parsed with `pandas.to_datetime`; ISO 8601 (`YYYY-MM-DD`) is recommended.

| File | Required | Columns | Description |
|---|---|---|---|
| `Observed_GWT.csv` | yes | `ObsWell, X, Y, Date, GWT` | Observed groundwater head at monitoring wells (`GWT`, m above mean sea level). |
| `Observed_stage.csv` | yes | `ObsPoint, Date, X, Y, Stage_masl` | Observed river stage at surveyed cross-sections (m above mean sea level); imposed as a Dirichlet condition. |
| `Well_extraction_child.csv` | yes | `Wellid, X, Y, Date, Q_md` | Abstraction rate per pumping well and date. Records with `Q_md` below `well_q_threshold` (default −150) are dropped. An unnamed index column, if present, is ignored. |
| `ET_RCH_VARUNA_ML.csv` | yes | `sno, X, Y, Date, RCH, ET` | Gridded recharge (`RCH`) and evapotranspiration (`ET`) time series (in the study, from a calibrated SWAT model resampled to a 50 m grid). |
| `Z_top_extn_depth.csv` | yes | `OBJECTID, ID, I, J, K, X, Y, Ztop, Dz` | Aquifer geometry on a grid: top elevation (`Ztop`) and thickness/extension depth (`Dz`). |
| `varuna_points_latlong.csv` | recommended | `FID, Longitude, Latitude` | Ordered vertices of the river polyline, used to interpolate stage along the river. **Despite the column names, values must be projected X/Y coordinates** in the same system as the other files. |
| `derivative_scaling_factors.csv` | optional | `parameter, scale_factor` (read by `load_and_preprocess_data`) | Optional scaling of the gradient-penalty search ranges. If absent, default ranges are used. |
| `kriging_results/kriging_2022-05-25.tif` | optional | single-band GeoTIFF | Kriged head surface for the initial date, sampled to impose an initial-condition loss in `optimize_pinn`. If absent, the initial-condition term is skipped. |

The column names can be adapted in `fit_transform_with_midrange_head` (automatic detection of common alternatives such as
`Easting`/`Northing`, `head`, `Q`) and in the loader functions of `pinn_unified.py`.

## Preprocessing (performed by the code)

- X, Y and time are min–max scaled to [0, 1] with a scaler shared across all datasets.
- Heads and river stage are scaled as `(h - h_mid) / L_h` with `h_mid = (h_max + h_min) / 2` and
  `L_h = 0.6 (h_max - h_min)`.
- Abstraction rates are min–max scaled; recharge, ET and abstraction are made continuous in space and time by radial-basis
  interpolation.

Files placed in this folder are excluded from version control by `.gitignore` (except this README).
