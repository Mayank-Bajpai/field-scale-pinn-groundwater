"""
pinn_unified.py
===============

Physics-informed neural networks (PINNs) for field-scale groundwater modelling:
joint prediction of hydraulic head h(x, y, t) and inverse estimation of
anisotropic hydraulic conductivity (Kx, Ky) and aquifer storage.

Three configurations share this module:
  1. Baseline (simple) PINN with fixed hyperparameters   -> run_pinn_iterations()
  2. PINN with Bayesian hyperparameter optimisation      -> optimize_pinn(use_film=False)
  3. FiLM-conditioned PINN with Bayesian optimisation    -> optimize_pinn(use_film=True)

Typical use
-----------
    import pinn_unified as pu
    stage_s, gwt_s, well_s, stats_df = pu.load_and_preprocess_data(data_dir="data")
    study, retrain = pu.optimize_pinn(n_trials=30, split_type="random")

Command line:  python pinn_unified.py --data-dir data --split-type random --n-trials 30

Input data are not distributed with this repository; see data/README.md for the
required files and column schema.

SPDX-License-Identifier: MIT
"""
import os
os.environ.setdefault("DDE_BACKEND", "pytorch")  # must be set before deepxde is imported
import sys
import time
import math
import json
import gc
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import deepxde as dde
from typing import Optional, Tuple, Dict, Any, List, Iterable, Union
from dataclasses import dataclass, asdict
from pathlib import Path

# Global constants restored
FIXED_MAIN_NUM_ITERS = 15000
FIXED_ADAM_ITERS = 4000
FIXED_HYBRID_N_ANCHOR = 7
DEFAULT_OUTPUT_DIR = "optuna_pinn_results_random_split_no_film"

# Directory holding the input files (see data/README.md). Override with the
# PINN_DATA_DIR environment variable or the data_dir argument of load_and_preprocess_data().
DATA_DIR = os.environ.get("PINN_DATA_DIR", "data")

def data_path(name: str) -> str:
    """Return the path of an input file inside DATA_DIR."""
    return os.path.join(DATA_DIR, name)

def _ensure_df(obj, name: str) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, (str, Path)):
        try:
            return pd.read_csv(obj)
        except Exception as e:
            raise ValueError(f"Failed to read CSV for '{name}' from {obj}: {e}")
    raise TypeError(f"Expected DataFrame or path for '{name}', got {type(obj)}")

def pick_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def drop_unnamed_and_empty(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop(columns=[c for c in out.columns if c.startswith("Unnamed")], errors="ignore")
    obj_cols = out.select_dtypes(include=["object"]).columns
    if len(obj_cols):
        out[obj_cols] = out[obj_cols].replace(r"^\s*$", np.nan, regex=True)
    out = out.dropna(axis=0, how="any")
    return out

def ensure_datetime(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = df.copy()
    if date_col not in out.columns:
        raise ValueError(f"Expected a '{date_col}' column. Available: {list(out.columns)}")
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    return out

def date_to_float_days(dt: pd.Series) -> np.ndarray:
    s = pd.to_datetime(dt, errors="coerce")
    ns = s.values.view("int64")  # nanoseconds since epoch
    arr = ns.astype(float)
    arr[~s.notna().values] = np.nan
    return arr / 86400e9  # ns -> days

# ---------------- Invertible 1D scalers ----------------

@dataclass
class MinMaxScaler1D:
    vmin: float = 0.0
    vmax: float = 1.0

    def fit(self, x: np.ndarray):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            self.vmin, self.vmax = 0.0, 1.0
        else:
            self.vmin, self.vmax = float(np.min(x)), float(np.max(x))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        rng = self.vmax - self.vmin
        if not np.isfinite(rng) or rng == 0.0:
            return np.where(np.isfinite(x), 0.0, np.nan)
        return (x - self.vmin) / rng

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, float)
        return z * (self.vmax - self.vmin) + self.vmin

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HeadMidRangeScaler:
    """
    Custom head scaler:
      h_scaled = (h - h_mid) / Lh
      where h_mid = 0.5 * (hmax + hmin), and Lh = 0.5 * 1.2 * (hmax - hmin) = 0.6 * range.
    """
    hmin: float = 0.0
    hmax: float = 1.0
    hmid: float = 0.5
    Lh: float = 0.5

    def fit(self, x: np.ndarray):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            self.hmin, self.hmax = 0.0, 1.0
        else:
            self.hmin, self.hmax = float(np.min(x)), float(np.max(x))
        rng = self.hmax - self.hmin
        self.hmid = 0.5 * (self.hmin + self.hmax)
        self.Lh = 0.5 * 1.2 * rng  # 0.6 * range
        if not np.isfinite(self.Lh) or self.Lh == 0.0:
            self.Lh = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        return (x - self.hmid) / self.Lh

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, float)
        return self.hmid + self.Lh * z

    def to_dict(self) -> Dict[str, Any]:
        return {"hmin": self.hmin, "hmax": self.hmax, "hmid": self.hmid, "Lh": self.Lh}

# ---------------- Shared fitting/apply ----------------

@dataclass
class ScalerBundle:
    # shared for coordinates/time across all frames
    x_mm: MinMaxScaler1D
    y_mm: MinMaxScaler1D
    t_mm: MinMaxScaler1D
    # value scalers
    gwt_head: HeadMidRangeScaler
    stage_head: HeadMidRangeScaler
    well_val_mm: MinMaxScaler1D
    # names used
    cols: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x_mm": self.x_mm.to_dict(),
            "y_mm": self.y_mm.to_dict(),
            "t_mm": self.t_mm.to_dict(),
            "gwt_head": self.gwt_head.to_dict(),
            "stage_head": self.stage_head.to_dict(),
            "well_val_mm": self.well_val_mm.to_dict(),
            "cols": self.cols,
        }

def fit_shared_minmax_for_xyz(stage: pd.DataFrame, gwt: pd.DataFrame, well: pd.DataFrame,
                              x_col: str, y_col: str, t_col: str) -> Tuple[MinMaxScaler1D, MinMaxScaler1D, MinMaxScaler1D]:
    xs = np.concatenate([
        pd.to_numeric(stage[x_col], errors="coerce").to_numpy(float) if x_col in stage.columns else np.array([]),
        pd.to_numeric(gwt[x_col], errors="coerce").to_numpy(float) if x_col in gwt.columns else np.array([]),
        pd.to_numeric(well[x_col], errors="coerce").to_numpy(float) if x_col in well.columns else np.array([]),
    ])
    ys = np.concatenate([
        pd.to_numeric(stage[y_col], errors="coerce").to_numpy(float) if y_col in stage.columns else np.array([]),
        pd.to_numeric(gwt[y_col], errors="coerce").to_numpy(float) if y_col in gwt.columns else np.array([]),
        pd.to_numeric(well[y_col], errors="coerce").to_numpy(float) if y_col in well.columns else np.array([]),
    ])
    td_stage = date_to_float_days(stage[t_col]) if t_col in stage.columns else np.array([])
    td_gwt   = date_to_float_days(gwt[t_col]) if t_col in gwt.columns else np.array([])
    td_well  = date_to_float_days(well[t_col]) if t_col in well.columns else np.array([])
    ts = np.concatenate([td_stage, td_gwt, td_well])

    x_mm = MinMaxScaler1D().fit(xs)
    y_mm = MinMaxScaler1D().fit(ys)
    t_mm = MinMaxScaler1D().fit(ts)
    return x_mm, y_mm, t_mm

def transform_frames_with_scalers(
    stage: pd.DataFrame, gwt: pd.DataFrame, well: pd.DataFrame,
    x_mm: MinMaxScaler1D, y_mm: MinMaxScaler1D, t_mm: MinMaxScaler1D,
    gwt_head: HeadMidRangeScaler, stage_head: HeadMidRangeScaler, well_val_mm: MinMaxScaler1D,
    x_col: str, y_col: str, t_col: str, stage_val_col: str, gwt_val_col: str, well_val_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage_o = stage.copy()
    gwt_o   = gwt.copy()
    well_o  = well.copy()

    # shared x,y,t (same min-max across frames)
    if x_col in stage_o.columns: stage_o["x_scaled"] = x_mm.transform(stage_o[x_col].to_numpy(float))
    if y_col in stage_o.columns: stage_o["y_scaled"] = y_mm.transform(stage_o[y_col].to_numpy(float))
    if t_col in stage_o.columns: stage_o["Date_scaled"] = t_mm.transform(date_to_float_days(stage_o[t_col]))

    if x_col in gwt_o.columns: gwt_o["x_scaled"] = x_mm.transform(gwt_o[x_col].to_numpy(float))
    if y_col in gwt_o.columns: gwt_o["y_scaled"] = y_mm.transform(gwt_o[y_col].to_numpy(float))
    if t_col in gwt_o.columns: gwt_o["Date_scaled"] = t_mm.transform(date_to_float_days(gwt_o[t_col]))

    if x_col in well_o.columns: well_o["x_scaled"] = x_mm.transform(well_o[x_col].to_numpy(float))
    if y_col in well_o.columns: well_o["y_scaled"] = y_mm.transform(well_o[y_col].to_numpy(float))
    if t_col in well_o.columns: well_o["Date_scaled"] = t_mm.transform(date_to_float_days(well_o[t_col]))

    # values
    gwt_vals   = pd.to_numeric(gwt_o[gwt_val_col], errors="coerce").to_numpy(float)
    stage_vals = pd.to_numeric(stage_o[stage_val_col], errors="coerce").to_numpy(float)
    well_vals  = pd.to_numeric(well_o[well_val_col], errors="coerce").to_numpy(float)

    gwt_o["GWT_scaled_val"]     = gwt_head.transform(gwt_vals)
    stage_o["Stage_scaled_val"] = stage_head.transform(stage_vals)
    well_o["Q_scaled_val"]      = well_val_mm.transform(well_vals)

    return stage_o, gwt_o, well_o

def build_scaler_stats_dataframe(bundle: ScalerBundle) -> pd.DataFrame:
    """
    Collect min, max, range for all variables based on the fitted scalers.
    Adds:
      - H_mid, Lh from GWT head scaler (primary head scaling for PINN)
      - Stage_mid, Stage_Lh from stage head scaler
    """
    rows = []
    rows.append(("x",    bundle.x_mm.vmin,    bundle.x_mm.vmax,    bundle.x_mm.vmax - bundle.x_mm.vmin))
    rows.append(("y",    bundle.y_mm.vmin,    bundle.y_mm.vmax,    bundle.y_mm.vmax - bundle.y_mm.vmin))
    rows.append(("Date", bundle.t_mm.vmin,    bundle.t_mm.vmax,    bundle.t_mm.vmax - bundle.t_mm.vmin))

    # Original head min/max/range (for reference)
    gwt_key = bundle.cols["gwt_val"]
    stage_key = bundle.cols["stage_val"]
    well_key = bundle.cols["well_val"]

    rows.append((gwt_key,   bundle.gwt_head.hmin,   bundle.gwt_head.hmax,   bundle.gwt_head.hmax - bundle.gwt_head.hmin))
    rows.append((stage_key, bundle.stage_head.hmin, bundle.stage_head.hmax, bundle.stage_head.hmax - bundle.stage_head.hmin))
    rows.append((well_key,  bundle.well_val_mm.vmin, bundle.well_val_mm.vmax, bundle.well_val_mm.vmax - bundle.well_val_mm.vmin))

    # Alias rows for convenience (head mid-range scaling)
    rows.append(("H_mid", bundle.gwt_head.hmid, np.nan, bundle.gwt_head.Lh))
    rows.append(("Lh",    bundle.gwt_head.Lh,  np.nan, bundle.gwt_head.Lh))
    rows.append(("Stage_mid", bundle.stage_head.hmid, np.nan, bundle.stage_head.Lh))
    rows.append(("Stage_Lh",  bundle.stage_head.Lh,   np.nan, bundle.stage_head.Lh))

    stats_df = pd.DataFrame(rows, columns=["attribute", "min", "max", "range"]).set_index("attribute").sort_index()
    return stats_df

# ---------------- High-level API ----------------

def fit_transform_with_midrange_head(
    stage_in, gwt_in, well_in,
    x_col: Optional[str] = None, y_col: Optional[str] = None, t_col: str = "Date",
    stage_val_col: Optional[str] = None, gwt_val_col: Optional[str] = None, well_val_col: Optional[str] = None,
    well_q_threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ScalerBundle, pd.DataFrame]:
    """
    Fit and transform with custom head scaling:
      h_scaled = (h - h_mid) / (0.5 * 1.2 * h_range)

    Enhancements:
      - Compute Hmid_spatial(x,y) = mean_t H(x,y,t) (physical units) over all available times.
      - Compute Hrange_t = H(actual) - Hmid_spatial (per record).
      - Min-max scale Hmid_spatial to Hmid_spatial_scaled.
      - Print and record the absolute maximum |Hrange_t|.
      - All three new columns are added to gwt_s.
    """
    # Accept DataFrame or path
    stage = _ensure_df(stage_in, "stage")
    gwt   = _ensure_df(gwt_in, "gwt")
    well  = _ensure_df(well_in, "well")

    # Clean and ensure Date parsed
    stage = ensure_datetime(drop_unnamed_and_empty(stage), t_col)
    gwt   = ensure_datetime(drop_unnamed_and_empty(gwt),   t_col)
    well  = ensure_datetime(drop_unnamed_and_empty(well),  t_col)

    if well_q_threshold is not None:
        cand = pick_col(well, ("Q_md", "Q", "Demand", "q_md"))
        if cand is None:
            raise ValueError("Could not find a well demand column to threshold (looked for Q_md/Q/Demand/q_md).")
        well[cand] = pd.to_numeric(well[cand], errors="coerce")
        before = len(well)
        well = well[well[cand] >= well_q_threshold].copy()
        well = well.dropna(subset=[cand])
        print(f"Well demand: dropped {before - len(well)} rows with {cand} < {well_q_threshold}")

    # Infer columns if not provided
    x_col = x_col or pick_col(stage, ("X", "x", "x_coord", "Easting", "x_scaled")) or \
                    pick_col(gwt,   ("X", "x", "x_coord", "Easting", "x_scaled")) or \
                    pick_col(well,  ("X", "x", "x_coord", "Easting", "x_scaled"))
    y_col = y_col or pick_col(stage, ("Y", "y", "y_coord", "Northing", "y_scaled")) or \
                    pick_col(gwt,   ("Y", "y", "y_coord", "Northing", "y_scaled")) or \
                    pick_col(well,  ("Y", "y", "y_coord", "Northing", "y_scaled"))
    if x_col is None or y_col is None:
        raise ValueError("Could not infer x/y columns. Pass x_col and y_col explicitly.")

    stage_val_col = stage_val_col or pick_col(stage, ("Stage_masl", "stage_masl", "stage", "Stage", "h", "head"))
    gwt_val_col   = gwt_val_col   or pick_col(gwt,   ("GWT", "GWT_masl", "h", "head"))
    well_val_col  = well_val_col  or pick_col(well,  ("Q_md", "Q", "Demand", "q_md"))

    if stage_val_col is None or gwt_val_col is None or well_val_col is None:
        raise ValueError("Could not infer one of the value columns (stage/gwt/well). Provide explicit names.")

    # Shared min-max for x,y,t
    x_mm, y_mm, t_mm = fit_shared_minmax_for_xyz(stage, gwt, well, x_col, y_col, t_col)

    # Head mid-range scalers (PER dataset)
    gwt_head   = HeadMidRangeScaler().fit(pd.to_numeric(gwt[gwt_val_col], errors="coerce").to_numpy(float))
    stage_head = HeadMidRangeScaler().fit(pd.to_numeric(stage[stage_val_col], errors="coerce").to_numpy(float))
    # Well demand min-max
    well_val_mm  = MinMaxScaler1D().fit(pd.to_numeric(well[well_val_col], errors="coerce").to_numpy(float))

    # Apply transforms
    stage_s, gwt_s, well_s = transform_frames_with_scalers(
        stage, gwt, well,
        x_mm, y_mm, t_mm,
        gwt_head, stage_head, well_val_mm,
        x_col, y_col, t_col, stage_val_col, gwt_val_col, well_val_col
    )

    # ---------------- NEW SECTION: Hmid_spatial, Hrange_t, Hmid_spatial_scaled ----------------
    # Compute spatial mid value Hmid_spatial(x,y) over time using physical H from gwt_s[gwt_val_col].
    # Grouping by scaled coordinates is robust (identical per location after shared scaling).
    # If scaled cols missing (shouldn't), fallback to original x,y.
    group_keys = ["x_scaled", "y_scaled"] if ("x_scaled" in gwt_s.columns and "y_scaled" in gwt_s.columns) else [x_col, y_col]

    gwt_phys = pd.to_numeric(gwt_s[gwt_val_col], errors="coerce")
    hm_spatial_df = (
        gwt_s.assign(_H=gwt_phys)
             .groupby(group_keys, as_index=False)["_H"].mean()
             .rename(columns={"_H": "Hmid_spatial"})
    )

    # Merge back to gwt_s
    gwt_s = gwt_s.merge(hm_spatial_df, on=group_keys, how="left")

    # Hrange_t = H(actual) - Hmid_spatial
    gwt_s["Hrange_t"] = gwt_phys - gwt_s["Hmid_spatial"]

    # Min-max scale Hmid_spatial
    hm_mm = MinMaxScaler1D().fit(gwt_s["Hmid_spatial"].to_numpy(float))
    gwt_s["Hmid_spatial_scaled"] = hm_mm.transform(gwt_s["Hmid_spatial"].to_numpy(float))

    # Absolute single maximum of |Hrange_t|
    hr_arr = np.abs(gwt_s["Hrange_t"].to_numpy(float))
    hrange_abs_max = float(np.nanmax(hr_arr)) if np.isfinite(hr_arr).any() else float("nan")
    print(f"[INFO] Max absolute |Hrange_t| = {hrange_abs_max:.6g} (physical units)")

    # ---------------- Bundle & stats ----------------
    bundle = ScalerBundle(
        x_mm=x_mm, y_mm=y_mm, t_mm=t_mm,
        gwt_head=gwt_head, stage_head=stage_head, well_val_mm=well_val_mm,
        cols={"x": x_col, "y": y_col, "t": t_col, "stage_val": stage_val_col, "gwt_val": gwt_val_col, "well_val": well_val_col}
    )

    stats_df = build_scaler_stats_dataframe(bundle)
    # Append Hmid_spatial stats and |Hrange_t|_abs_max to stats_df
    stats_df.loc["Hmid_spatial", ["min", "max", "range"]] = [
        hm_mm.vmin, hm_mm.vmax, hm_mm.vmax - hm_mm.vmin
    ]
    stats_df.loc["Hrange_t_abs_max", ["min", "max", "range"]] = [hrange_abs_max, np.nan, np.nan]
    stats_df = stats_df.sort_index()

    return stage_s, gwt_s, well_s, bundle, stats_df

# ---------------- Plotting ----------------

def plot_scaled_value_histograms(gwt_s: pd.DataFrame, well_s: pd.DataFrame, stage_s: pd.DataFrame,
                                 bins: int = 60, title: str = "Histograms of scaled values (custom head scaling)") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(gwt_s["GWT_scaled_val"].dropna().to_numpy(float), bins=bins, color="#1f77b4", alpha=0.85)
    axes[0].set_title("GWT (mid-range)"); axes[0].set_xlabel("GWT_scaled_val"); axes[0].set_ylabel("count")

    axes[1].hist(well_s["Q_scaled_val"].dropna().to_numpy(float), bins=bins, color="#ff7f0e", alpha=0.85)
    axes[1].set_title("Well demand (min-max)"); axes[1].set_xlabel("Q_scaled_val"); axes[1].set_ylabel("count")

    axes[2].hist(stage_s["Stage_scaled_val"].dropna().to_numpy(float), bins=bins, color="#2ca02c", alpha=0.85)
    axes[2].set_title("River stage (mid-range)"); axes[2].set_xlabel("Stage_scaled_val"); axes[2].set_ylabel("count")

    plt.suptitle(title, y=1.02, fontsize=12)
    plt.tight_layout()
    # plt.show()

def plot_histograms_xyz_scaled(stage_s: pd.DataFrame, gwt_s: pd.DataFrame, well_s: pd.DataFrame, bins: int = 50,
                               title: str = "Histograms of x/y/t (shared min-max)") -> None:
    xs = np.concatenate([
        stage_s.get("x_scaled", pd.Series([], dtype=float)).dropna().to_numpy(float),
        gwt_s.get("x_scaled", pd.Series([], dtype=float)).dropna().to_numpy(float),
        well_s.get("x_scaled", pd.Series([], dtype=float)).dropna().to_numpy(float),
    ])
    ys = np.concatenate([
        stage_s.get("y_scaled", pd.Series([], dtype=float)).dropna().to_numpy(float),
        gwt_s.get("y_scaled", pd.Series([], dtype=float)).dropna().to_numpy(float),
        well_s.get("y_scaled", pd.Series([], dtype=float)).dropna().to_numpy(float),
    ])
    ts = np.concatenate([
        stage_s.get("Date_scaled", pd.Series([], dtype=float)).dropna().to_numpy(float),
        gwt_s.get("Date_scaled", pd.Series([], dtype=float)).dropna().to_numpy(float),
        well_s.get("Date_scaled", pd.Series([], dtype=float)).dropna().to_numpy(float),
    ])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].hist(xs, bins=bins, color="#1f77b4", alpha=0.85); axes[0].set_title("x_scaled"); axes[0].set_xlabel("x"); axes[0].set_ylabel("count")
    axes[1].hist(ys, bins=bins, color="#ff7f0e", alpha=0.85); axes[1].set_title("y_scaled"); axes[1].set_xlabel("y"); axes[1].set_ylabel("count")
    axes[2].hist(ts, bins=bins, color="#2ca02c", alpha=0.85); axes[2].set_title("Date_scaled"); axes[2].set_xlabel("t"); axes[2].set_ylabel("count")
    plt.suptitle(title, y=1.02, fontsize=12)
    plt.tight_layout()
    # plt.show()

# ---------------- Example usage ----------------
# Junk block 1 removed
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def pick_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def default_colmap(
    gwt_scaled: pd.DataFrame,
    stage_scaled: pd.DataFrame,
    well_demand_scaled: pd.DataFrame,
) -> Dict[str, Dict[str, str]]:
    gwt_map = {
        "x": pick_col(gwt_scaled, ("x_scaled", "X_scaled", "x", "X")),
        "y": pick_col(gwt_scaled, ("y_scaled", "Y_scaled", "y", "Y")),
        "t": pick_col(gwt_scaled, ("Date_scaled", "t_scaled", "time_scaled", "Date", "time", "t")),
        # prefer new midrange-scaled column names
        "h": pick_col(gwt_scaled, ("GWT_scaled_val", "h_scaled", "head_scaled", "h", "head", "GWT")),
    }
    stg_map = {
        "x": pick_col(stage_scaled, ("x_scaled", "X_scaled", "x", "X")),
        "y": pick_col(stage_scaled, ("y_scaled", "Y_scaled", "y", "Y")),
        "t": pick_col(stage_scaled, ("Date_scaled", "t_scaled", "time_scaled", "Date", "time", "t")),
        "h": pick_col(stage_scaled, ("Stage_scaled_val", "stage_scaled", "Stage_masl", "Stage", "stage")),
    }
    wel_map = {
        "x": pick_col(well_demand_scaled, ("x_scaled", "X_scaled", "x", "X")),
        "y": pick_col(well_demand_scaled, ("y_scaled", "Y_scaled", "y", "Y")),
        "t": pick_col(well_demand_scaled, ("Date_scaled", "t_scaled", "time_scaled", "Date", "time", "t")),
        "Q": pick_col(well_demand_scaled, ("Q_scaled_val", "Q_scaled", "q_scaled", "Q_md", "Q", "q", "Demand_scaled", "Demand")),
    }
    return {"gwt": gwt_map, "stage": stg_map, "well": wel_map}


def default_colmap_scaled(
    gwt_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    well_df: pd.DataFrame,
) -> Dict[str, Dict[str, str]]:
    """
    Column mapping for the new scaled schema (with midrange head scaling).
    """
    def _has(df, col): return col in df.columns
    reqs = [
        ("gwt", gwt_df, ["x_scaled", "y_scaled", "Date_scaled", "GWT_scaled_val"]),
        ("stage", stage_df, ["x_scaled", "y_scaled", "Date_scaled", "Stage_scaled_val"]),
        ("well", well_df, ["x_scaled", "y_scaled", "Date_scaled", "Q_scaled_val"]),
    ]
    for name, df, cols in reqs:
        missing = [c for c in cols if not _has(df, c)]
        if missing:
            raise ValueError(f"{name} dataframe missing required columns {missing}. Available: {list(df.columns)}")

    return {
        "gwt":   {"x": "x_scaled", "y": "y_scaled", "t": "Date_scaled", "h": "GWT_scaled_val"},
        "stage": {"x": "x_scaled", "y": "y_scaled", "t": "Date_scaled", "h": "Stage_scaled_val"},
        "well":  {"x": "x_scaled", "y": "y_scaled", "t": "Date_scaled", "Q": "Q_scaled_val"},
    }


def _safe_min_max_range(arr_like) -> Tuple[float, float, float]:
    """Compute min, max, range while safely ignoring NaNs."""
    a = np.asarray(arr_like, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (np.nan, np.nan, np.nan)
    amin = float(np.min(a))
    amax = float(np.max(a))
    return amin, amax, float(amax - amin)


def build_scaler_stats_dataframe(bundle, gwt_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Enhanced version of build_scaler_stats_dataframe for midrange head scaling.

    Returns a single DataFrame that includes, for each variable:
      - min, max, range (as before)
    And adds convenience/alias rows:
      - X_min, X_max, Lx
      - Y_min, Y_max, Ly
      - Date_min, Date_max, Lt
      - H_mid, Lh  (from the groundwater head midrange scaler)
      - Q_min, Q_max, Q_range (well demand)
      - Stage_mid, Stage_Lh (from stage head midrange scaler)

    Additionally, if gwt_df is provided and contains the new features, append:
      - Hmid_spatial:   min/max/range in physical units
      - Hmid_spatial_scaled: min/max/range (typically 0..1)
      - Hrange_t:       min/max/range in physical units
      - Hrange_t_abs_max: single absolute maximum of |Hrange_t| (stored under 'min')
    """
    rows = []

    # Shared x/y/t
    x_min, x_max = bundle.x_mm.vmin, bundle.x_mm.vmax
    y_min, y_max = bundle.y_mm.vmin, bundle.y_mm.vmax
    t_min, t_max = bundle.t_mm.vmin, bundle.t_mm.vmax
    Lx = x_max - x_min
    Ly = y_max - y_min
    Lt = t_max - t_min

    rows.append(("x",    x_min, x_max, Lx))
    rows.append(("y",    y_min, y_max, Ly))
    rows.append(("Date", t_min, t_max, Lt))

    # Value variables keep their original (unscaled) column names
    gwt_key   = bundle.cols["gwt_val"]
    well_key  = bundle.cols["well_val"]
    stage_key = bundle.cols["stage_val"]

    # Head (midrange) stats
    gwt_min, gwt_max = bundle.gwt_head.hmin, bundle.gwt_head.hmax
    stage_min, stage_max = bundle.stage_head.hmin, bundle.stage_head.hmax
    well_min, well_max = bundle.well_val_mm.vmin, bundle.well_val_mm.vmax

    rows.append((gwt_key,   gwt_min,   gwt_max,   gwt_max - gwt_min))
    rows.append((well_key,  well_min,  well_max,  well_max - well_min))
    rows.append((stage_key, stage_min, stage_max, stage_max - stage_min))

    # ---- Alias/convenience rows ----
    # Coordinates/time aliases
    rows.append(("X_min", x_min, np.nan, Lx))
    rows.append(("X_max", x_max, np.nan, Lx))
    rows.append(("Lx",    Lx,    np.nan, Lx))

    rows.append(("Y_min", y_min, np.nan, Ly))
    rows.append(("Y_max", y_max, np.nan, Ly))
    rows.append(("Ly",    Ly,    np.nan, Ly))

    rows.append(("Date_min", t_min, np.nan, Lt))
    rows.append(("Date_max", t_max, np.nan, Lt))
    rows.append(("Lt",       Lt,    np.nan, Lt))

    # Head/stage aliases (mid-range scaling)
    rows.append(("H_mid", bundle.gwt_head.hmid, np.nan, bundle.gwt_head.Lh))
    rows.append(("Lh",    bundle.gwt_head.Lh,   np.nan, bundle.gwt_head.Lh))

    rows.append(("Stage_mid", bundle.stage_head.hmid, np.nan, bundle.stage_head.Lh))
    rows.append(("Stage_Lh",  bundle.stage_head.Lh,   np.nan, bundle.stage_head.Lh))

    # Wells
    rows.append(("Q_min", well_min, np.nan, well_max - well_min))
    rows.append(("Q_max", well_max, np.nan, well_max - well_min))
    rows.append(("Q_range", well_max - well_min, np.nan, well_max - well_min))

    # ---- NEW FEATURE STATS (if available in gwt_df) ----
    if gwt_df is not None:
        if "Hmid_spatial" in gwt_df.columns:
            mn, mx, rn = _safe_min_max_range(gwt_df["Hmid_spatial"])
            rows.append(("Hmid_spatial", mn, mx, rn))
        if "Hmid_spatial_scaled" in gwt_df.columns:
            mn, mx, rn = _safe_min_max_range(gwt_df["Hmid_spatial_scaled"])
            rows.append(("Hmid_spatial_scaled", mn, mx, rn))
        if "Hrange_t" in gwt_df.columns:
            mn, mx, rn = _safe_min_max_range(gwt_df["Hrange_t"])
            rows.append(("Hrange_t", mn, mx, rn))
            # absolute single maximum |Hrange_t|
            abs_max = np.nan
            try:
                vals = np.abs(pd.to_numeric(gwt_df["Hrange_t"], errors="coerce").to_numpy(float))
                if np.isfinite(vals).any():
                    abs_max = float(np.nanmax(vals))
            except Exception:
                pass
            rows.append(("Hrange_t_abs_max", abs_max, np.nan, np.nan))

    stats_df = (
        pd.DataFrame(rows, columns=["attribute", "min", "max", "range"])
        .set_index("attribute")
        .sort_index()
    )
    return stats_df


def stats_to_scales_dict(stats_df: pd.DataFrame, colmap: Dict[str, Dict[str, str]]) -> Dict[str, float]:
    """
    Convert stats_df into a scales dictionary with the keys:
      {
        "X_min", "Y_min", "Lx", "Ly", "Lt",
        "H_mid", "Lh",
        "Q_min", "Q_range"
      }
    """
    def _get(attr, field="min", default=None):
        return float(stats_df.loc[attr, field]) if attr in stats_df.index else default

    # Prefer explicit alias rows if present; otherwise fall back to base rows.
    X_min = _get("X_min", "min", _get("x", "min", 0.0))
    Y_min = _get("Y_min", "min", _get("y", "min", 0.0))
    Lx    = _get("Lx", "range", _get("x", "range", 1.0))
    Ly    = _get("Ly", "range", _get("y", "range", 1.0))
    Lt    = _get("Lt", "range", _get("Date", "range", 1.0))

    # Groundwater head midrange
    H_mid = _get("H_mid", "min", None)
    Lh    = _get("Lh", "min", None)
    if H_mid is None or Lh is None:
        # fallback to the gwt value key if aliases are missing
        gwt_key = colmap["gwt"]["h"]
        hmin = _get(gwt_key, "min", 0.0)
        hmax = _get(gwt_key, "max", 1.0)
        H_mid = 0.5 * (hmin + hmax)
        Lh    = 0.5 * 1.2 * (hmax - hmin)

    # Wells
    Q_min   = _get("Q_min", "min", None)
    Q_range = _get("Q_range", "range", None)
    if Q_min is None or Q_range is None:
        q_key = colmap["well"]["Q"]
        qmin = _get(q_key, "min", 0.0)
        qmax = _get(q_key, "max", 1.0)
        Q_min   = qmin
        Q_range = qmax - qmin

    return dict(
        X_min=X_min, Y_min=Y_min, Lx=Lx, Ly=Ly, Lt=Lt,
        H_mid=H_mid, Lh=Lh,
        Q_min=Q_min, Q_range=Q_range,
    )


# Example (expects `bundle`, `gwt_s`, `stage_s`, `well_s` in scope):
# Junk block 2 removed
"""
Hybrid / Anisotropic Single‑Head PINN (Simple single-branch head; h_scaled = (h_phys - H_mid)/H_range)
Modification: remove square-root scaling on all penalty weights (including phys_weight and lambda_*).
"""
import os, math
os.environ.setdefault("DDE_BACKEND", "pytorch")

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import deepxde as dde
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt

try:
    from scipy.spatial import cKDTree
    _HAS_KDTREE = True
except ImportError:
    _HAS_KDTREE = False

import importlib
if not callable(np.concatenate):
    importlib.reload(np)
print("np.concatenate restored:", callable(np.concatenate))

_ENV_LOW_MEM_CKPT = os.getenv("PINN_LOW_MEM_CHECKPOINT", "0") == "1"
_ENV_LOW_MEM_AMP  = os.getenv("PINN_LOW_MEM_AMP", "0") == "1"
_ENV_MEM_LOG      = os.getenv("PINN_MEM_LOG", "0") == "1"
_ENV_OFFLOAD_AFTER_ITER = os.getenv("PINN_OFFLOAD_AFTER_ITER", "0") == "1"

torch.backends.cudnn.benchmark = False
dde.config.set_default_float("float32")
torch.set_default_dtype(torch.float32)

# --------------------------------------------------------
# Memory Utility
# --------------------------------------------------------
def memory_report(tag=""):
    if not _ENV_MEM_LOG:
        return
    if not torch.cuda.is_available():
        print(f"[MEM] {tag}: (CPU only)")
        return
    alloc = torch.cuda.memory_allocated()/1024**2
    res   = torch.cuda.memory_reserved()/1024**2
    print(f"[MEM] {tag}: alloc={alloc:.1f}MB reserved={res:.1f}MB")

# --------------------------------------------------------
# Helpers
# --------------------------------------------------------
def drop_unnamed_and_empty(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop(columns=[c for c in out.columns if c.startswith("Unnamed")], errors="ignore")
    obj_cols = out.select_dtypes(include=["object"]).columns
    if len(obj_cols):
        out[obj_cols] = out[obj_cols].replace(r"^\s*$", np.nan, regex=True)
    out = out.dropna(axis=0, how="any")
    return out

def ensure_datetime(df: pd.DataFrame, date_col="Date"):
    out=df.copy()
    out[date_col]=pd.to_datetime(out[date_col],errors="coerce")
    out=out.dropna(subset=[date_col])
    return out

def date_to_float_days(dt: pd.Series) -> np.ndarray:
    s=pd.to_datetime(dt,errors="coerce")
    ns=s.astype("int64").astype(float)
    ns[~s.notna().values]=np.nan
    return ns / 86400e9

def default_colmap_scaled(gwt_df, stage_df, well_df):
    reqs=[
        ("gwt", gwt_df, ["x_scaled","y_scaled","Date_scaled","GWT_scaled_val"]),
        ("stage", stage_df, ["x_scaled","y_scaled","Date_scaled","Stage_scaled_val"]),
        ("well", well_df, ["x_scaled","y_scaled","Date_scaled","Q_scaled_val"]),
    ]
    for name,df,cols in reqs:
        miss=[c for c in cols if c not in df.columns]
        if miss: raise ValueError(f"{name} missing {miss}")
    return {
        "gwt":{"x":"x_scaled","y":"y_scaled","t":"Date_scaled","h":"GWT_scaled_val"},
        "stage":{"x":"x_scaled","y":"y_scaled","t":"Date_scaled","h":"Stage_scaled_val"},
        "well":{"x":"x_scaled","y":"y_scaled","t":"Date_scaled","Q":"Q_scaled_val"},
    }

def stats_to_scales_dict(stats_df: pd.DataFrame, colmap) -> Dict[str,float]:
    def _get(attr, field="min", default=None):
        return float(stats_df.loc[attr, field]) if attr in stats_df.index else default
    return dict(
        X_min=_get("x","min",0.0),
        Y_min=_get("y","min",0.0),
        T_min=_get("Date","min",0.0),
        Lx=_get("x","range",1.0),
        Ly=_get("y","range",1.0),
        Lt=_get("Date","range",1.0),
        H_mid=_get("H_mid","min",0.0),
        Lh=_get("Lh","min",1.0),
        Q_min=_get(colmap["well"]["Q"],"min",0.0),
        Q_range=_get(colmap["well"]["Q"],"range",1.0),
    )

# --------------------------------------------------------
# Fourier Features
# --------------------------------------------------------
class FourierFeatures(nn.Module):
    def __init__(self, in_dim, freq_list=(1,2,4,8,16), learnable=False):
        super().__init__()
        freqs = torch.tensor(freq_list, dtype=torch.float32)
        self.freqs = nn.Parameter(freqs) if learnable else freqs
    def forward(self, x):
        feats=[x]
        for f in self.freqs:
            arg=2*math.pi*f*x
            feats.append(torch.sin(arg)); feats.append(torch.cos(arg))
        return torch.cat(feats,dim=1)

# --------------------------------------------------------
# FiLM Layer
# --------------------------------------------------------
class FiLMLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 2 * out_dim)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, context):
        # x: [N, out_dim], context: [N, in_dim]
        params = self.fc(context)
        gamma, beta = torch.chunk(params, 2, dim=1)
        return (1 + gamma) * x + beta

# --------------------------------------------------------
# Single-Branch Head (with optional FiLM)
# --------------------------------------------------------
class SingleHeadDecomposedNet(nn.Module):
    def __init__(self, base_hidden=128, depth_xy=3, depth_t=3,
                 freq_list_t=(1,2,4,8,16), use_checkpointing=False,
                 H_mid=0.0, Lh=1.0, use_film=False, **_):
        super().__init__()
        self.use_checkpointing = use_checkpointing or _ENV_LOW_MEM_CKPT
        self.use_film = use_film
        
        # For DeepXDE compatibility (required for dde.Model)
        self.regularizer = None
        
        # Buffers
        self.register_buffer("H_mid_buf", torch.tensor(float(H_mid), dtype=torch.float32))
        self.register_buffer("Lh_buf", torch.tensor(float(Lh), dtype=torch.float32))

        # Architecture
        self.ff = FourierFeatures(in_dim=3, freq_list=freq_list_t, learnable=False)
        
        # Dimensions
        # If FiLM: input to main is just X,Y (2 dims + fourier). Context is T (1 dim + fourier).
        # Normal: input is X,Y,T (3 dims + fourier).
        
        input_dim = 3 + 2 * len(freq_list_t) * 3
        depth = max(1, max(depth_xy, depth_t))
        dims = [input_dim] + [base_hidden] * depth + [1]
        
        if use_film:
            # Overwrite for FiLM structure: Main net takes (x,y), generator takes (t)
            # Spatial Embedding: (x,y) -> [2 + 2*freq*2]
            self.ff_xy = FourierFeatures(in_dim=2, freq_list=freq_list_t, learnable=False)
            dim_xy = 2 + 2 * len(freq_list_t) * 2
            
            # Temporal Embedding: (t) -> [1 + 2*freq*1]
            self.ff_t = FourierFeatures(in_dim=1, freq_list=freq_list_t, learnable=False)
            dim_t = 1 + 2 * len(freq_list_t) * 1
            
            # Main Network (Spatial)
            self.layers = nn.ModuleList()
            self.film_layers = nn.ModuleList()
            
            curr_dim = dim_xy
            for i in range(len(dims)-2): # hidden layers
                out_dim = dims[i+1]
                l = nn.Linear(curr_dim, out_dim)
                nn.init.xavier_uniform_(l.weight, gain=0.8); nn.init.zeros_(l.bias)
                self.layers.append(l)
                
                # FiLM Layer for this block
                self.film_layers.append(FiLMLayer(in_dim=dim_t, out_dim=out_dim))
                
                curr_dim = out_dim

            # Output layer
            l_out = nn.Linear(dims[-2], 1)
            nn.init.xavier_uniform_(l_out.weight, gain=0.8); nn.init.zeros_(l_out.bias)
            self.layers.append(l_out)
            
        else:
            # Standard MLP
            self.layers = nn.ModuleList()
            for i in range(len(dims)-2):
                l = nn.Linear(dims[i], dims[i+1])
                nn.init.xavier_uniform_(l.weight, gain=0.8); nn.init.zeros_(l.bias)
                self.layers.append(l)
                self.layers.append(nn.Tanh())
            
            l_out = nn.Linear(dims[-2], dims[-1])
            nn.init.xavier_uniform_(l_out.weight, gain=0.8); nn.init.zeros_(l_out.bias)
            self.layers.append(l_out)

        self.post_act = nn.Softsign()

    def forward(self, X):
        # X is (x, y, t)
        if self.use_film:
            xy = X[:, 0:2]
            t = X[:, 2:3]
            
            feat_xy = self.ff_xy(xy)
            feat_t = self.ff_t(t)
            
            x = feat_xy
            # Apply layers with FiLM
            # layers has linear layers. film_layers has FiLM gen/apply.
            # Sequence: Linear -> FiLM(context) -> Activation
            
            n_hidden = len(self.film_layers)
            for i in range(n_hidden):
                linear = self.layers[i]
                film = self.film_layers[i]
                
                x = linear(x) # Linear
                x = film(x, feat_t) # Modulation
                x = torch.tanh(x) # Activation
            
            # Last layer
            out = self.layers[-1](x)
            return self.post_act(out)
            
        else:
            feat = self.ff(X)
            x = feat
            for m in self.layers:
                if isinstance(m, nn.Linear) and self.use_checkpointing:
                    x = checkpoint(lambda inp, layer=m: layer(inp), x)
                else:
                    x = m(x)
            return self.post_act(x)

# --------------------------------------------------------
# Hybrid fields (unchanged architecture)
# --------------------------------------------------------
class SpatialFieldMLP(nn.Module):
    def __init__(self, hidden=32, depth=2, activation="tanh", use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing or _ENV_LOW_MEM_CKPT
        act=nn.Tanh if activation=="tanh" else nn.ReLU
        dims=[2]+[hidden]*depth+[2]
        mods=[]
        for i in range(len(dims)-2):
            l=nn.Linear(dims[i],dims[i+1])
            nn.init.xavier_uniform_(l.weight,gain=0.8); nn.init.zeros_(l.bias)
            mods.append(l); mods.append(act())
        lf=nn.Linear(dims[-2],dims[-1])
        nn.init.xavier_uniform_(lf.weight,gain=0.8); nn.init.zeros_(lf.bias)
        mods.append(lf)
        self.layers=nn.ModuleList(mods)
    def forward(self,x):
        for m in self.layers:
            if isinstance(m, nn.Linear) and self.use_checkpointing:
                x = checkpoint(lambda inp, layer=m: layer(inp), x)
            else:
                x = m(x)
        return x

class AnisotropicHybridK(nn.Module):
    def __init__(self, n_anchor=7, mlp_hidden=48, mlp_depth=2, activation="tanh",
                 separate_alpha=False, K_min=1e-2, K_max=5e2,
                 ratio_min=0.1, ratio_max=10.0, enforce_ratio=True,
                 use_checkpointing=False):
        super().__init__()
        assert ratio_min>0 and ratio_max>ratio_min
        self.n_anchor=n_anchor
        self.raw_latent_x=nn.Parameter(torch.zeros(n_anchor,n_anchor))
        self.raw_latent_y=nn.Parameter(torch.zeros(n_anchor,n_anchor))
        self.mlp=SpatialFieldMLP(hidden=mlp_hidden, depth=mlp_depth,
                                 activation=activation,
                                 use_checkpointing=use_checkpointing)
        if separate_alpha:
            self.alpha_param_x=nn.Parameter(torch.tensor(0.0))
            self.alpha_param_y=nn.Parameter(torch.tensor(0.0))
        else:
            self.alpha_param=nn.Parameter(torch.tensor(0.0))
            self.alpha_param_x=self.alpha_param_y=None
        self.separate_alpha=separate_alpha
        self.K_min=float(K_min); self.K_max=float(K_max)
        self.ratio_min=float(ratio_min); self.ratio_max=float(ratio_max)
        self.enforce_ratio=enforce_ratio
        log_range=math.log(self.K_max/self.K_min)
        hi = math.log(self.ratio_max)
        self.diff_half_max = 0.5 * hi / log_range
    def _alpha_x(self): return torch.sigmoid(self.alpha_param_x if self.separate_alpha else self.alpha_param)
    def _alpha_y(self): return torch.sigmoid(self.alpha_param_y if self.separate_alpha else self.alpha_param)
    def _bilinear(self, xy, latent):
        A=self.n_anchor
        g=torch.sigmoid(latent)
        x=xy[:,0].clamp(0,0.999999)*(A-1)
        y=xy[:,1].clamp(0,0.999999)*(A-1)
        x0=torch.floor(x).long(); x1=(x0+1).clamp(max=A-1)
        y0=torch.floor(y).long(); y1=(y0+1).clamp(max=A-1)
        sx=x-x0.float(); sy=y-y0.float()
        g00=g[y0,x0]; g01=g[y1,x0]; g10=g[y0,x1]; g11=g[y1,x1]
        gx0=g00*(1-sy)+g01*sy
        gx1=g10*(1-sy)+g11*sy
        gxy=gx0*(1-sx)+gx1*sx
        return gxy.unsqueeze(1)
    def forward(self, xy):
        mlp_out=torch.sigmoid(self.mlp(xy))
        z_grid_x=self._bilinear(xy,self.raw_latent_x)
        z_grid_y=self._bilinear(xy,self.raw_latent_y)
        alpha_x=self._alpha_x(); alpha_y=self._alpha_y()
        Kx_u = alpha_x*z_grid_x + (1-alpha_x)*mlp_out[:,0:1]
        Ky_u = alpha_y*z_grid_y + (1-alpha_y)*mlp_out[:,1:2]
        if self.enforce_ratio:
            mean = 0.5*(Kx_u + Ky_u)
            diff = 0.5*(Kx_u - Ky_u)
            diff_clamped = torch.clamp(diff, -self.diff_half_max, self.diff_half_max)
            Kx_u = (mean + diff_clamped).clamp(0.0,1.0)
            Ky_u = (mean - diff_clamped).clamp(0.0,1.0)
        return Kx_u, Ky_u
    def latent_variance(self):
        return torch.var(torch.sigmoid(self.raw_latent_x)), torch.var(torch.sigmoid(self.raw_latent_y))
    def latent_roughness(self):
        def rough(lat):
            z=torch.sigmoid(lat)
            diffs=[(z[:,1:]-z[:,:-1])**2,(z[1:,:]-z[:-1,:])**2]
            return torch.cat([d.reshape(-1) for d in diffs]).mean()
        return rough(self.raw_latent_x), rough(self.raw_latent_y)
    def mlp_l2(self):
        s=0.0; n=0
        for p in self.mlp.parameters():
            s+=(p**2).sum(); n+=p.numel()
        return s/(n+1e-12)

class HybridFieldScalar(nn.Module):
    def __init__(self, n_anchor=7, mlp_hidden=48, mlp_depth=2, activation="tanh",
                 use_checkpointing=False):
        super().__init__()
        self.n_anchor=n_anchor
        self.raw_latent=nn.Parameter(torch.zeros(n_anchor,n_anchor))
        act = nn.Tanh if activation=="tanh" else nn.ReLU
        dims=[2]+[mlp_hidden]*mlp_depth+[1]
        layers=[]
        for i in range(len(dims)-2):
            l=nn.Linear(dims[i],dims[i+1])
            nn.init.xavier_uniform_(l.weight,gain=0.8); nn.init.zeros_(l.bias)
            layers += [l, act()]
        lf=nn.Linear(dims[-2],dims[-1])
        nn.init.xavier_uniform_(lf.weight,gain=0.8); nn.init.zeros_(lf.bias)
        layers.append(lf)
        self.layers=nn.ModuleList(layers)
        self.alpha_param=nn.Parameter(torch.tensor(0.0))
        self.use_checkpointing=use_checkpointing or _ENV_LOW_MEM_CKPT
    def _alpha(self): return torch.sigmoid(self.alpha_param)
    def _bilinear(self, xy):
        A=self.n_anchor
        g=torch.sigmoid(self.raw_latent)
        x=xy[:,0].clamp(0,0.999999)*(A-1)
        y=xy[:,1].clamp(0,0.999999)*(A-1)
        x0=torch.floor(x).long(); x1=(x0+1).clamp(max=A-1)
        y0=torch.floor(y).long(); y1=(y0+1).clamp(max=A-1)
        sx=x-x0.float(); sy=y-y0.float()
        g00=g[y0,x0]; g01=g[y1,x0]; g10=g[y0,x1]; g11=g[y1,x1]
        gx0=g00*(1-sy)+g01*sy
        gx1=g10*(1-sy)+g11*sy
        gxy=gx0*(1-sx)+gx1*sx
        return gxy.unsqueeze(1)
    def forward(self, xy):
        z_grid=self._bilinear(xy)
        z=xy
        for layer in self.layers:
            if isinstance(layer, nn.Linear) and self.use_checkpointing:
                z=checkpoint(lambda inp, layer=layer: layer(inp), z)
            else:
                z=layer(z)
        z_mlp=torch.sigmoid(z)
        return self._alpha()*z_grid + (1-self._alpha())*z_mlp
    def latent_variance(self):
        return torch.var(torch.sigmoid(self.raw_latent))
    def latent_roughness(self):
        z=torch.sigmoid(self.raw_latent)
        diffs=[(z[:,1:]-z[:,:-1])**2,(z[1:,:]-z[:-1,:])**2]
        return torch.cat([d.reshape(-1) for d in diffs]).mean()
    def mlp_l2(self):
        s=0.0; n=0
        for p in self.layers:
            if hasattr(p,"weight"):
                s+=(p.weight**2).sum(); n+=p.weight.numel()
        return s/(n+1e-12)

# --------------------------------------------------------
# Mapping helpers
# --------------------------------------------------------
def map_unit_to_range_exp(z_unit: torch.Tensor, pmin: float, pmax: float):
    log_min = math.log(pmin); log_max = math.log(pmax)
    return torch.exp(log_min + z_unit*(log_max-log_min))

# --------------------------------------------------------
# River CSV interpolation
# --------------------------------------------------------
def build_river_dataset_from_csv(
    csv_path: str,
    stage_df: pd.DataFrame,
    colmap,
    scales,
    river_spacing_filter: Optional[float]=None,
    xcol_csv="Longitude",
    ycol_csv="Latitude",
    tolerance_outside=1e-3
) -> Dict[str,Any]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)
    riv = pd.read_csv(csv_path)
    if "FID" in riv.columns:
        riv = riv.sort_values("FID")
    xr = riv[xcol_csv].to_numpy(np.float64)
    yr = riv[ycol_csv].to_numpy(np.float64)
    dx = np.diff(xr); dy = np.diff(yr)
    seg_len = np.sqrt(dx*dx + dy*dy)
    dist = np.concatenate(([0], np.cumsum(seg_len)))
    if dist[-1] == 0:
        raise ValueError("River polyline has zero length.")
    s_norm = dist / dist[-1]
    if river_spacing_filter is not None and river_spacing_filter > 0:
        keep=[0]
        for i in range(1,len(xr)):
            if dist[i]-dist[keep[-1]] >= river_spacing_filter:
                keep.append(i)
        if keep[-1] != len(xr)-1:
            keep.append(len(xr)-1)
        xr=xr[keep]; yr=yr[keep]; dist=dist[keep]; s_norm=s_norm[keep]
    xs_scaled=(xr - scales["X_min"]) / scales["Lx"]
    ys_scaled=(yr - scales["Y_min"]) / scales["Ly"]
    mask = (
        (xs_scaled >= -tolerance_outside) & (xs_scaled <= 1 + tolerance_outside) &
        (ys_scaled >= -tolerance_outside) & (ys_scaled <= 1 + tolerance_outside)
    )
    xs_scaled=xs_scaled[mask]; ys_scaled=ys_scaled[mask]; s_norm=s_norm[mask]
    if len(xs_scaled)<2:
        raise ValueError("Too few river points inside domain after scaling.")
    st_x=stage_df[colmap["stage"]["x"]].to_numpy(np.float32)
    st_y=stage_df[colmap["stage"]["y"]].to_numpy(np.float32)
    st_t=stage_df[colmap["stage"]["t"]].to_numpy(np.float32)
    st_h=stage_df[colmap["stage"]["h"]].to_numpy(np.float32)
    if _HAS_KDTREE:
        kdt=cKDTree(np.stack([xs_scaled,ys_scaled],axis=1))
        _,idxs=kdt.query(np.stack([st_x,st_y],axis=1),k=1)
    else:
        rp=np.stack([xs_scaled,ys_scaled],axis=1)
        diff=rp[None,:,:]-np.stack([st_x,st_y],axis=1)[:,:,None]
        dist2=(diff**2).sum(1)
        idxs=np.argmin(dist2,axis=1)
    s_obs=s_norm[idxs]
    unique_t=np.sort(np.unique(st_t))
    rows=[]
    for tval in unique_t:
        m=(st_t==tval)
        s_local=s_obs[m]; h_local=st_h[m]
        if len(s_local)==0: continue
        o=np.argsort(s_local)
        s_local=s_local[o]; h_local=h_local[o]
        s_u,inv=np.unique(s_local,return_inverse=True)
        if len(s_u)!=len(s_local):
            acc=np.zeros_like(s_u); cnt=np.zeros_like(s_u)
            for i,v in enumerate(inv):
                acc[v]+=h_local[i]; cnt[v]+=1
            h_local=acc/cnt; s_local=s_u
        if len(s_local)==1:
            h_interp=np.full_like(s_norm,h_local[0])
        else:
            h_interp=np.interp(s_norm,s_local,h_local,left=h_local[0],right=h_local[-1])
        rows.append(pd.DataFrame({
            "x_scaled":xs_scaled.astype(np.float32),
            "y_scaled":ys_scaled.astype(np.float32),
            "Date_scaled":np.full_like(xs_scaled,tval,dtype=np.float32),
            "Stage_scaled_val":h_interp.astype(np.float32),
            "s_norm":s_norm.astype(np.float32),
        }))
    river_stage_df=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    return dict(river_stage_df=river_stage_df, xs_scaled=xs_scaled.astype(np.float32), ys_scaled=ys_scaled.astype(np.float32))


# --------------------------------------------------------
# RBF pumping (with temporal kernel)
# --------------------------------------------------------
def subsample_dataframe(df: pd.DataFrame, nmax: Optional[int]) -> pd.DataFrame:
    if nmax is None or len(df)<=nmax: return df
    return df.sample(n=nmax, random_state=1234)

def make_rbf_source_from_wells_torch(
    well_df, xcol, ycol, tcol, qcol,
    Q_min, Q_range, Lh,
    rbf_lengthscales=(0.05, 0.05, 0.05),
    temporal_kernel_scale: Optional[float] = None,
    max_centers=800
):
    wells = subsample_dataframe(well_df[[xcol, ycol, tcol, qcol]].dropna(), max_centers)
    C = wells[[xcol, ycol, tcol]].to_numpy(np.float32)
    q_scaled = wells[qcol].to_numpy(np.float32).reshape(-1,1)
    w_phys_scaled = (Q_min + Q_range * q_scaled)/Lh
    centers = torch.tensor(C, dtype=torch.float32)
    weights = torch.tensor(w_phys_scaled, dtype=torch.float32)
    ls = torch.tensor(np.asarray(rbf_lengthscales,np.float32).reshape(1,1,3), dtype=torch.float32)
    if temporal_kernel_scale is not None and temporal_kernel_scale <= 0:
        temporal_kernel_scale=None
    def q_fun(x: torch.Tensor):
        dev=x.device
        c=centers.to(dev, non_blocking=True); w=weights.to(dev, non_blocking=True)
        ls_local=ls.to(dev, non_blocking=True)
        d=(x.unsqueeze(1)-c.unsqueeze(0))/ls_local
        r2=(d*d).sum(-1)
        k=torch.exp(-r2)
        if temporal_kernel_scale is not None:
            dt=x[:,2:3]-c[:,2].unsqueeze(0)
            k=k*torch.exp(-0.5*(dt/temporal_kernel_scale)**2)
        return k@w
    return q_fun

# --------------------------------------------------------
# ET / RCH Interpolators
# --------------------------------------------------------
def preprocess_et_rch_csv_days(
    csv_path, scales, xcol="X", ycol="Y", datecol="Date", etcol="ET", rchcol="RCH",
    well_time_window_scaled=None, time_pad=0.0
):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)
    df=ensure_datetime(drop_unnamed_and_empty(pd.read_csv(csv_path)),date_col=datecol)
    t_days=date_to_float_days(df[datecol])
    T0=scales["T_min"]; T1=T0+scales["Lt"]
    m=np.isfinite(t_days)&(t_days>=T0)&(t_days<=T1)
    df=df.loc[m].copy()
    if df.empty: raise ValueError("No ET/RCH rows in time range.")
    t_days=t_days[m]
    xs=((pd.to_numeric(df[xcol],errors="coerce").to_numpy(np.float32)-scales["X_min"])/scales["Lx"])
    ys=((pd.to_numeric(df[ycol],errors="coerce").to_numpy(np.float32)-scales["Y_min"])/scales["Ly"])
    ts=((t_days-T0)/scales["Lt"]).astype(np.float32)
    if well_time_window_scaled is not None:
        tmin,tmax=well_time_window_scaled
        m2=(ts>=(tmin-time_pad))&(ts<=(tmax+time_pad))
        xs,ys,ts=xs[m2],ys[m2],ts[m2]
        df=df.loc[m2].copy()
        if df.empty: raise ValueError("No ET/RCH rows after window filter.")
    return pd.DataFrame({
        "x_s":xs.astype(np.float32),
        "y_s":ys.astype(np.float32),
        "t_s":ts,
        "ET":pd.to_numeric(df[etcol],errors="coerce").to_numpy(np.float32),
        "RCH":pd.to_numeric(df[rchcol],errors="coerce").to_numpy(np.float32)
    }).dropna()

def make_rbf_interpolator_from_df_3d(data_df,val_col,ls_xyz=(0.08,0.08,0.05),max_centers=15000):
    if data_df.empty:
        def z(x): return torch.zeros((x.shape[0],1),dtype=torch.float32,device=x.device)
        return z
    df=data_df[["x_s","y_s","t_s",val_col]].dropna()
    if max_centers and len(df)>max_centers:
        df=df.sample(n=max_centers,random_state=1234)
    C=df[["x_s","y_s","t_s"]].to_numpy(np.float32)
    v=df[[val_col]].to_numpy(np.float32)
    centers=torch.tensor(C,dtype=torch.float32)
    values=torch.tensor(v,dtype=torch.float32)
    ls=torch.tensor(np.asarray(ls_xyz,np.float32).reshape(1,1,3),dtype=torch.float32)
    def f(x):
        d=(x.unsqueeze(1)-centers.unsqueeze(0))/ls
        r2=(d*d).sum(-1)
        k=torch.exp(-r2)
        wsum=torch.clamp(k.sum(1,keepdim=True),1e-8)
        return (k@values)/wsum
    return f

def make_rbf_interpolator_from_csv_2d(csv_path,xcol,ycol,vcol,scales,rbf_lengthscales_xy=(0.02,0.02),max_centers=8000):
    if not os.path.isfile(csv_path):
        def z(xy): return torch.zeros((xy.shape[0],1),dtype=torch.float32,device=xy.device)
        return z
    df=pd.read_csv(csv_path).dropna(subset=[xcol,ycol,vcol])
    xs=((pd.to_numeric(df[xcol],errors="coerce").to_numpy(np.float32)-scales["X_min"])/scales["Lx"]).reshape(-1,1)
    ys=((pd.to_numeric(df[ycol],errors="coerce").to_numpy(np.float32)-scales["Y_min"])/scales["Ly"]).reshape(-1,1)
    vs=pd.to_numeric(df[vcol],errors="coerce").to_numpy(np.float32).reshape(-1,1)
    data=pd.DataFrame({"x":xs.ravel(),"y":ys.ravel(),"v":vs.ravel()})
    data=subsample_dataframe(data,max_centers)
    C=data[["x","y"]].to_numpy(np.float32); v=data[["v"]].to_numpy(np.float32)
    centers=torch.tensor(C,dtype=torch.float32)
    values=torch.tensor(v,dtype=torch.float32)
    ls=torch.tensor(np.asarray(rbf_lengthscales_xy,np.float32).reshape(1,1,2),dtype=torch.float32)
    def f(xy):
        d=(xy.unsqueeze(1)-centers.unsqueeze(0))/ls
        r2=(d*d).sum(-1)
        k=torch.exp(-r2)
        wsum=torch.clamp(k.sum(1,keepdim=True),1e-8)
        return (k@values)/wsum
    return f



# --------------------------------------------------------
# Training (single iteration) - weights without sqrt
# --------------------------------------------------------
def random_train_val_test_split(df: pd.DataFrame, frac_train: float = 0.7, frac_val: float = 0.15, seed: int = 1234):
    """
    RANDOM SPLIT (3-Way): Train, Validation, and Test.
    Includes test index tracking to prevent data leakage (WS-2).
    """
    global GLOBAL_TEST_INDICES
    if len(df) == 0: return df.copy(), df.copy(), df.copy()
    
    rng = np.random.default_rng(seed)
    N = len(df)
    perm = rng.permutation(N)
    
    n_tr = int(round(frac_train * N))
    n_val = int(round(frac_val * N))
    
    train_idx = perm[:n_tr]
    val_idx = perm[n_tr:n_tr+n_val]
    test_idx = perm[n_tr+n_val:]
    
    train = df.iloc[train_idx].copy()
    val = df.iloc[val_idx].copy()
    test = df.iloc[test_idx].copy()
    
    GLOBAL_TEST_INDICES = set(test.index)
    print(f"[INFO] 3-Way Random Split: {len(train)} train, {len(val)} val, {len(test)} test rows.")
    return train, val, test

def make_pointset_bc(df,xcol,ycol,tcol,vcol):
    X=np.stack([df[xcol].to_numpy(np.float32),
                df[ycol].to_numpy(np.float32),
                df[tcol].to_numpy(np.float32)],axis=1)
    Y=df[vcol].to_numpy(np.float32).reshape(-1,1)
    return dde.icbc.PointSetBC(X,Y,component=0)

def evaluate_metrics(head_net, gwt_test, colmap, scales):
    if gwt_test is None or len(gwt_test)==0:
        return dict(rmse=float("nan"),mae=float("nan"),r2=float("nan"))
    H_mid,Lh=scales["H_mid"],scales["Lh"]
    xcol,ycol,tcol,hcol=colmap["gwt"]["x"],colmap["gwt"]["y"],colmap["gwt"]["t"],colmap["gwt"]["h"]
    X=np.stack([gwt_test[xcol].to_numpy(np.float32),
                gwt_test[ycol].to_numpy(np.float32),
                gwt_test[tcol].to_numpy(np.float32)],axis=1)
    with torch.no_grad():
        out=head_net(torch.tensor(X,dtype=torch.float32))
        hs=out[:,0:1].cpu().numpy()
    tru=gwt_test[hcol].to_numpy(np.float32).reshape(-1,1)
    y_true=H_mid+Lh*tru
    y_pred=H_mid+Lh*hs
    diff=y_pred-y_true
    rmse=float(np.sqrt((diff**2).mean()))
    mae=float(np.abs(diff).mean())
    ss_res=float((diff**2).sum())
    ss_tot=float(((y_true-y_true.mean())**2).sum())
    r2=float(1-ss_res/ss_tot) if ss_tot>0 else float("nan")
    return dict(rmse=rmse,mae=mae,r2=r2)
    

def train_one_iteration(
    stage_s, gwt_s, well_s, stats_df,
    et_fun_phys, rch_fun_phys, ztop_fun, dz_fun,
    include_forcings,
    prev_net,
    river_info=None,
    ic_df=None,
    *,
    num_domain=12000,
    adam_iters=4000,
    layers=5,
    units=64,
    lr=1e-3,
    phys_weight=1.0,
    gwt_train_frac=0.7,
    cluster_bins=4,
    oversample_factor=1,
    hybrid_n_anchor=7,
    mlp_hidden=48,
    mlp_depth=2,
    mlp_activation="tanh",
    device_model="cuda",
    skip_lbfgs=True,
    lambda_smooth=0.0,
    lambda_link=0.0,
    lambda_latent_var=0.0,
    lambda_latent_rough=0.0,
    target_latent_var=0.05,
    lambda_mlp_l2=0.0,
    lambda_grad_raw=0.0,
    lambda_grad_t_raw=0.0,
    lambda_h_tt=0.0,
    w_bc=1.0,
    w_ic=1.0,
    temporal_grad_clip=None,
    lambda_temporal_clip=0.0,
    target_s_var=0.0,
    lambda_s_var_shortfall=0.0,
    lambda_latent_rough_S=0.0,
    ratio_min=0.1,
    ratio_max=10.0,
    temporal_kernel_scale=None,
    use_fourier_head=True,
    fourier_freqs=(1,2,4,8,16),
    split_seed=1234,
    warmup_all_data=False,
    return_penalty_history=False,
    loss_save_frequency=100,
    use_film=False,
):
    """
    Train PINN for one iteration with detailed penalty logging. 
    
    Returns:
        tuple: (head_net, K_aniso, S_field, scales, colmap, gwt_te, metrics, penalty_history)
    """
    colmap = default_colmap_scaled(gwt_s, stage_s, well_s)
    scales = stats_to_scales_dict(stats_df, colmap)

    # Split data
    if warmup_all_data:
        gwt_tr = gwt_s. copy()
        gwt_te = gwt_s.iloc[0:0]. copy()
    else:
        gwt_tr = gwt_s.copy(); gwt_te = gwt_s.iloc[0:0].copy()

    # Oversample GWT to target 10k
    TARGET = 10000
    if len(gwt_tr) == 0:
        gwt_tr_os = gwt_tr.copy()
    else:
        rep = int(math.ceil(TARGET / max(1, len(gwt_tr))))
        gwt_tr_os = pd.concat([gwt_tr] * rep, ignore_index=True)\
                      .sample(n=TARGET, random_state=split_seed).reset_index(drop=True)

    # Initialize hybrid fields
    K_aniso = AnisotropicHybridK(
        n_anchor=hybrid_n_anchor, mlp_hidden=mlp_hidden, mlp_depth=mlp_depth,
        activation=mlp_activation, K_min=1e-2, K_max=5e2,
        ratio_min=ratio_min, ratio_max=ratio_max, enforce_ratio=True
    )
    S_field = HybridFieldScalar(
        n_anchor=hybrid_n_anchor, mlp_hidden=mlp_hidden, mlp_depth=mlp_depth,
        activation=mlp_activation
    )

    # Initialize head network
    H_mid = scales["H_mid"]
    Lh = scales["Lh"]
    head_net = SingleHeadDecomposedNet(
        base_hidden=units, depth_xy=layers // 2, depth_t=layers // 2,
        freq_list_t=fourier_freqs, use_checkpointing=_ENV_LOW_MEM_CKPT,
        H_mid=H_mid, Lh=Lh, use_film=use_film
    )

    # Move to device
    if device_model == "cuda" and torch.cuda.is_available():
        head_net.cuda()
        K_aniso.cuda()
        S_field.cuda()

    # Load previous weights if available
    if prev_net is not None:
        try:
            head_net.load_state_dict(prev_net.state_dict(), strict=False)
        except Exception:
            pass

    # Physical constants
    Lx, Ly, Lt = scales["Lx"], scales["Ly"], scales["Lt"]
    inv_Lt = 1.0 / Lt
    inv_Lx2 = 1.0 / (Lx * Lx)
    inv_Ly2 = 1.0 / (Ly * Ly)
    K_min, K_max = 1e-2, 5e2
    S_min, S_max = 0.02, 0.35  # Specific Yield (WS-1)

    # Placeholder functions (can be replaced with actual implementations)
    q_wells = lambda x: torch.zeros((x.shape[0], 1), device=x.device)

    @torch.no_grad()
    def et_actual(x):
        return torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)

    @torch.no_grad()
    def rch_eff(x):
        return torch. zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)

    def _head_l2():
        s = 0.0
        n = 0
        for p in head_net.parameters():
            s += (p ** 2).sum()
            n += p.numel()
        return s / (n + 1e-12)

    # PDE definition
    def pde(x, y_all):
        h_sc = y_all[: , 0:1]
        h_t = dde. grad. jacobian(h_sc, x, i=0, j=2)
        h_x = dde.grad.jacobian(h_sc, x, i=0, j=0)
        h_y = dde.grad.jacobian(h_sc, x, i=0, j=1)
        h_xx = dde.grad.hessian(h_sc, x, i=0, j=0)
        h_yy = dde.grad.hessian(h_sc, x, i=0, j=1)

        xy = x[: , : 2]
        Kx_u, Ky_u = K_aniso(xy)
        S_u = S_field(xy)
        Kx = map_unit_to_range_exp(Kx_u, K_min, K_max) + 1e-12
        Ky = map_unit_to_range_exp(Ky_u, K_min, K_max) + 1e-12
        S_xy = map_unit_to_range_exp(S_u, S_min, S_max) + 1e-12
        Kx_x = dde.grad.jacobian(Kx, x, i=0, j=0)
        Ky_y = dde.grad.jacobian(Ky, x, i=0, j=1)
        diff = inv_Lx2 * (Kx * h_xx + Kx_x * h_x) + inv_Ly2 * (Ky * h_yy + Ky_y * h_y)

        q_w = q_wells(x)
        et_a = et_actual(x)
        rch_e = rch_eff(x)
        q_dist = (rch_e - et_a) / scales["Lh"]

        phys_res = phys_weight * (S_xy * inv_Lt * h_t - diff - (q_dist + q_w))
        residuals = [phys_res]

        if lambda_grad_raw > 0:
            residuals. append(lambda_grad_raw * h_x)
            residuals.append(lambda_grad_raw * h_y)
        if lambda_grad_t_raw > 0:
            residuals.append(lambda_grad_t_raw * h_t)
        if lambda_h_tt > 0:
            h_tt = dde.grad.hessian(h_sc, x, i=0, j=2)
            residuals.append(lambda_h_tt * h_tt)
        if temporal_grad_clip is not None and temporal_grad_clip > 0 and lambda_temporal_clip > 0:
            h_t_clamped = torch.clamp(h_t, -temporal_grad_clip, temporal_grad_clip)
            residuals.append(lambda_temporal_clip * (h_t - h_t_clamped))

        N = x.shape[0]
        reg = []
        if lambda_latent_var > 0:
            varKx, varKy = K_aniso. latent_variance()
            reg.append((varKx - target_latent_var) * lambda_latent_var)
            reg.append((varKy - target_latent_var) * lambda_latent_var)
        if lambda_latent_rough > 0:
            rKx, rKy = K_aniso.latent_roughness()
            reg.append(rKx * lambda_latent_rough)
            reg.append(rKy * lambda_latent_rough)
        varS = S_field.latent_variance()
        if lambda_s_var_shortfall > 0 and target_s_var > 0:
            shortfall = torch.relu(target_s_var - varS)
            reg.append(shortfall * lambda_s_var_shortfall)
        elif lambda_latent_var > 0:
            reg. append((varS - target_latent_var) * lambda_latent_var)
        if lambda_latent_rough_S > 0:
            rS = S_field.latent_roughness()
            reg.append(rS * lambda_latent_rough_S)
        if lambda_mlp_l2 > 0:
            reg. append(K_aniso.mlp_l2() * lambda_mlp_l2)
            reg.append(S_field. mlp_l2() * lambda_mlp_l2)
            reg.append(_head_l2() * lambda_mlp_l2)

        for p in reg:
            if not torch.is_tensor(p):
                p = torch.tensor(p, dtype=x.dtype, device=x.device)
            residuals.append(p.view(1, 1).expand(N, 1) / math.sqrt(N))
        return torch.cat(residuals, dim=1)

    # Build boundary conditions
    stage_bc = make_pointset_bc(stage_s, colmap["stage"]["x"], colmap["stage"]["y"],
                                 colmap["stage"]["t"], colmap["stage"]["h"])
    gwt_bc = make_pointset_bc(gwt_tr_os, colmap["gwt"]["x"], colmap["gwt"]["y"],
                               colmap["gwt"]["t"], colmap["gwt"]["h"])
    bcs = [stage_bc, gwt_bc]

    if ic_df is not None and not ic_df.empty:
        # Use simple PointSetBC for IC (technically a snapshot constraint)
        ic_bc = make_pointset_bc(ic_df, colmap["gwt"]["x"], colmap["gwt"]["y"],
                                 colmap["gwt"]["t"], colmap["gwt"]["h"])
        bcs.append(ic_bc)
        # print(f"[INFO] Added IC points: {len(ic_df)}")

    # Create geometry and data
    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    time_dom = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, time_dom)
    data = dde.data.TimePDE(geomtime, pde, bcs, num_domain, 0, 0)
    model = dde.Model(data, head_net)

    ext_vars = list(K_aniso.parameters()) + list(S_field.parameters())

    # ============== DETAILED LOSS LOGGING ==============
    epoch_loss_records = []
    class DetailedLossCallback(dde.callbacks. Callback):
        def __init__(self, period):
            super().__init__()
            self.period = period
    
        def on_epoch_end(self):
            current_epoch = self.model.train_state.step  # Use . step instead of deprecated .epoch
            if current_epoch % self.period == 0 or current_epoch == 0:
                # Manually compute all losses with full penalty breakdown
                head_net. eval()
                K_aniso. eval()
                S_field.eval()
    
                try:
                    # Get training data from model's train_state
                    # Sample domain points for PDE evaluation
                    n_eval = min(1000, num_domain)
                    domain_pts = np.random.rand(n_eval, 3).astype(np.float32)
                    domain_pts = torch.tensor(domain_pts, dtype=torch.float32,
                                             device=head_net.layers[0].weight.device)
    
                    # Enable gradients for PDE computation
                    domain_pts_grad = domain_pts.clone().detach().requires_grad_(True)
    
                    # Forward pass
                    y_pred = head_net(domain_pts_grad)
    
                    # Compute PDE residual components
                    h_sc = y_pred[:, 0:1]
                    h_t = dde.grad. jacobian(h_sc, domain_pts_grad, i=0, j=2)
                    h_x = dde.grad.jacobian(h_sc, domain_pts_grad, i=0, j=0)
                    h_y = dde.grad.jacobian(h_sc, domain_pts_grad, i=0, j=1)
                    h_xx = dde. grad.hessian(h_sc, domain_pts_grad, i=0, j=0)
                    h_yy = dde.grad.hessian(h_sc, domain_pts_grad, i=0, j=1)
    
                    xy = domain_pts_grad[:, :2]
                    Kx_u, Ky_u = K_aniso(xy)
                    S_u = S_field(xy)
                    Kx = map_unit_to_range_exp(Kx_u, K_min, K_max) + 1e-12
                    Ky = map_unit_to_range_exp(Ky_u, K_min, K_max) + 1e-12
                    S_xy = map_unit_to_range_exp(S_u, S_min, S_max) + 1e-12
                    Kx_x = dde.grad.jacobian(Kx, domain_pts_grad, i=0, j=0)
                    Ky_y = dde.grad.jacobian(Ky, domain_pts_grad, i=0, j=1)
                    diff = inv_Lx2 * (Kx * h_xx + Kx_x * h_x) + inv_Ly2 * (Ky * h_yy + Ky_y * h_y)
    
                    q_w = q_wells(domain_pts_grad)
                    et_a = et_actual(domain_pts_grad)
                    rch_e = rch_eff(domain_pts_grad)
                    q_dist = (rch_e - et_a) / scales["Lh"]
    
                    # Main physics residual
                    phys_res = phys_weight * (S_xy * inv_Lt * h_t - diff - (q_dist + q_w))
                    phys_loss = (phys_res ** 2).mean()
    
                    # Initialize loss dictionary
                    loss_dict = {
                        "epoch": current_epoch,
                        "pde_residual": float(phys_loss. detach().cpu().item())
                    }
    
                    # Gradient penalties
                    if lambda_grad_raw > 0:
                        grad_x_loss = lambda_grad_raw * ((h_x) ** 2).mean()
                        grad_y_loss = lambda_grad_raw * ((h_y) ** 2).mean()
                        loss_dict["penalty_grad_x"] = float(grad_x_loss.detach().cpu().item())
                        loss_dict["penalty_grad_y"] = float(grad_y_loss.detach().cpu().item())
    
                    if lambda_grad_t_raw > 0:
                        grad_t_loss = lambda_grad_t_raw * ((h_t) ** 2).mean()
                        loss_dict["penalty_grad_t"] = float(grad_t_loss. detach().cpu().item())
    
                    if lambda_h_tt > 0:
                        h_tt = dde.grad.hessian(h_sc, domain_pts_grad, i=0, j=2)
                        h_tt_loss = lambda_h_tt * ((h_tt) ** 2).mean()
                        loss_dict["penalty_h_tt"] = float(h_tt_loss. detach().cpu().item())
    
                    if temporal_grad_clip is not None and temporal_grad_clip > 0 and lambda_temporal_clip > 0:
                        h_t_clamped = torch.clamp(h_t, -temporal_grad_clip, temporal_grad_clip)
                        clip_loss = lambda_temporal_clip * ((h_t - h_t_clamped) ** 2).mean()
                        loss_dict["penalty_temporal_clip"] = float(clip_loss.detach().cpu().item())
    
                    # Latent field penalties (no gradients needed)
                    with torch.no_grad():
                        if lambda_latent_var > 0:
                            varKx, varKy = K_aniso. latent_variance()
                            loss_Kx_var = lambda_latent_var * ((varKx - target_latent_var) ** 2)
                            loss_Ky_var = lambda_latent_var * ((varKy - target_latent_var) ** 2)
                            loss_dict["penalty_latent_var_Kx"] = float(loss_Kx_var.cpu().item())
                            loss_dict["penalty_latent_var_Ky"] = float(loss_Ky_var. cpu().item())
    
                        if lambda_latent_rough > 0:
                            rKx, rKy = K_aniso.latent_roughness()
                            loss_Kx_rough = lambda_latent_rough * rKx
                            loss_Ky_rough = lambda_latent_rough * rKy
                            loss_dict["penalty_latent_rough_Kx"] = float(loss_Kx_rough.cpu().item())
                            loss_dict["penalty_latent_rough_Ky"] = float(loss_Ky_rough.cpu().item())
    
                        varS = S_field.latent_variance()
                        if lambda_s_var_shortfall > 0 and target_s_var > 0:
                            shortfall = torch.relu(target_s_var - varS)
                            loss_S_var = lambda_s_var_shortfall * (shortfall ** 2)
                            loss_dict["penalty_S_var_shortfall"] = float(loss_S_var.cpu().item())
                        elif lambda_latent_var > 0:
                            loss_S_var = lambda_latent_var * ((varS - target_latent_var) ** 2)
                            loss_dict["penalty_latent_var_S"] = float(loss_S_var.cpu().item())
    
                        if lambda_latent_rough_S > 0:
                            rS = S_field.latent_roughness()
                            loss_S_rough = lambda_latent_rough_S * rS
                            loss_dict["penalty_latent_rough_S"] = float(loss_S_rough. cpu().item())
    
                        # MLP L2 penalties
                        if lambda_mlp_l2 > 0:
                            loss_K_l2 = lambda_mlp_l2 * K_aniso.mlp_l2()
                            loss_S_l2 = lambda_mlp_l2 * S_field.mlp_l2()
                            loss_head_l2 = lambda_mlp_l2 * _head_l2()
                            loss_dict["penalty_mlp_l2_K"] = float(loss_K_l2.cpu().item())
                            loss_dict["penalty_mlp_l2_S"] = float(loss_S_l2.cpu().item())
                            loss_dict["penalty_mlp_l2_head"] = float(loss_head_l2.cpu().item())
    
                        # BC losses - construct BC data from our original BCs
                        try:
                            # Stage BC
                            stage_x = torch.tensor(stage_bc. points, dtype=torch.float32,
                                                  device=head_net.layers[0].weight.device)
                            stage_y = torch.tensor(stage_bc.values, dtype=torch.float32,
                                                  device=head_net.layers[0].weight. device)
                            stage_pred = head_net(stage_x)
                            bc_stage_loss = ((stage_pred - stage_y) ** 2).mean()
                            loss_dict["bc_stage"] = float(bc_stage_loss.cpu().item())
    
                            # GWT BC
                            gwt_x = torch.tensor(gwt_bc.points, dtype=torch.float32,
                                                device=head_net.layers[0]. weight.device)
                            gwt_y = torch.tensor(gwt_bc.values, dtype=torch.float32,
                                                device=head_net. layers[0].weight.device)
                            gwt_pred = head_net(gwt_x)
                            bc_gwt_loss = ((gwt_pred - gwt_y) ** 2).mean()
                            loss_dict["bc_gwt"] = float(bc_gwt_loss.cpu().item())
    
                            # Total BC
                            loss_dict["bc_total"] = loss_dict["bc_stage"] + loss_dict["bc_gwt"]
    
                        except Exception as bc_err:
                            # Fallback if BC data not accessible
                            loss_dict["bc_stage"] = 0.0
                            loss_dict["bc_gwt"] = 0.0
                            loss_dict["bc_total"] = 0.0
    
                    # Total loss (sum of all components)
                    total = sum([v for k, v in loss_dict.items() if
                                k != "epoch" and isinstance(v, (int, float))])
                    loss_dict["total_loss"] = total
    
                    epoch_loss_records.append(loss_dict)
    
                except Exception as e:
                    print(f"[WARN] Loss computation failed at epoch {current_epoch}: {e}")
    
                head_net.train()
                K_aniso.train()
                S_field.train()
                
    # Compile model
    
    # Dynamically build loss_weights
    K = 1 # phys_res
    if lambda_grad_raw > 0: K += 2
    if lambda_grad_t_raw > 0: K += 1
    if lambda_h_tt > 0: K += 1
    if temporal_grad_clip is not None and temporal_grad_clip > 0 and lambda_temporal_clip > 0: K += 1
    
    # Regularization residuals
    num_reg = 0
    if lambda_latent_var > 0: num_reg += 2
    if lambda_latent_rough > 0: num_reg += 2
    if lambda_s_var_shortfall > 0 and target_s_var > 0: num_reg += 1
    elif lambda_latent_var > 0: num_reg += 1
    if lambda_latent_rough_S > 0: num_reg += 1
    if lambda_mlp_l2 > 0: num_reg += 3
    K += num_reg

    # bcs is [stage_bc, gwt_bc] and maybe ic_bc
    # Weights for PDE outputs (1.0 for phys and all penalties)
    # The PDE returns a single concatenated tensor for all physics/regularization components.
    # Therefore, DeepXDE sees exactly 1 PDE output.
    lw = [1.0]  # Weight for the combined PDE block
    lw.append(w_bc)   # stage_bc
    lw.append(1.0)    # gwt_bc (L_obs)
    if ic_df is not None and not ic_df.empty:
        lw.append(w_ic) # ic_bc

    model.compile("adam", lr=lr, external_trainable_variables=ext_vars, loss_weights=lw)


    # Setup callback
    loss_callback = None
    if return_penalty_history:
        loss_callback = DetailedLossCallback(period=loss_save_frequency)
        callbacks_list = [loss_callback]
    else:
        callbacks_list = None

    # Train
    losshistory = None
    train_state = None
    try:
        losshistory, train_state = model.train(
            iterations=adam_iters,
            display_every=1000,
            callbacks=callbacks_list
        )
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        penalty_history = pd.DataFrame({"epoch": [0], "total_loss": [0.0]})
        return head_net, K_aniso, S_field, scales, colmap, gwt_te, {
            "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")
        }, penalty_history

    # Optional L-BFGS
    if not skip_lbfgs:
        try:
            model.compile("L-BFGS", external_trainable_variables=ext_vars)
            lbfgs_lh, lbfgs_ts = model.train()
        except Exception as e:
            print("[WARN] LBFGS skipped:", e)

    # Evaluate metrics
    metrics = evaluate_metrics(model. net, gwt_te, colmap, scales)

    # Build penalty_history from collected records
    penalty_history = None
    if return_penalty_history:
        try:
            if epoch_loss_records:
                penalty_history = pd.DataFrame(epoch_loss_records)
                print(f"[INFO] Captured {len(penalty_history)} detailed loss records "
                      f"(every {loss_save_frequency} epochs)")
            else:
                # Fallback to losshistory
                if losshistory is not None and hasattr(losshistory, 'loss_train') and losshistory.loss_train is not None:
                    loss_train = np.array(losshistory. loss_train)
                    penalty_history = pd.DataFrame({
                        "epoch": np. arange(len(loss_train)) * 1000,
                        "total_loss": loss_train. astype(float) if loss_train.ndim == 1 else loss_train. sum(
                            axis=1).astype(float)
                    })
                    print(f"[INFO] Fallback:  Captured {len(penalty_history)} loss records from losshistory")
                else:
                    penalty_history = pd.DataFrame({"epoch": [0], "total_loss": [0.0]})
        except Exception as e:
            print(f"[WARN] Failed to build penalty_history: {e}")
            import traceback
            traceback.print_exc()
            penalty_history = pd.DataFrame({"epoch": [0], "total_loss": [0.0]})
    else:
        penalty_history = pd.DataFrame({"epoch": [0], "total_loss": [0.0]})

    # Finalize
    head_final = model.net
    head_final.eval()
    del model
    del data
    del loss_callback
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    return head_final, K_aniso, S_field, scales, colmap, gwt_te, metrics, penalty_history
    
# --------------------------------------------------------
# Artifacts
# --------------------------------------------------------
@dataclass
class IterationArtifact:
    iteration: int
    net_path:str
    k_path:str
    s_path:str
    meta_path:str

def save_artifact(it, save_dir, net, K_field, S_field, scales, colmap, cfg):
    os.makedirs(save_dir,exist_ok=True)
    net_p=os.path.join(save_dir,f"iter{it}_net.pt")
    k_p=os.path.join(save_dir,f"iter{it}_K_aniso.pt")
    s_p=os.path.join(save_dir,f"iter{it}_S_field.pt")
    meta_p=os. path.join(save_dir,f"iter{it}_meta.json")
    torch.save(net.state_dict(),net_p)
    torch.save(K_field.state_dict(),k_p)
    torch.save(S_field.state_dict(),s_p)
    with open(meta_p,"w") as f:
        json.dump(dict(iteration=it,scales=scales,colmap=colmap,config=cfg),f,indent=2)
    return IterationArtifact(it,net_p,k_p,s_p,meta_p)
    
# --------------------------------------------------------
# Orchestrator (unchanged)
# --------------------------------------------------------
def run_pinn_iterations(
    stage_s, gwt_s, well_s, stats_df,
    et_rch_csv_path, ztop_extn_csv_path,
    ic_tif_path=None,
    *,
    river_csv_path="varuna_points_latlong.csv",
    augment_river_dirichlet=True,
    warmup_iters=1,
    num_iters=2,
    warmup_adam_iters=1500,
    adam_iters=4000,
    num_domain=12000,
    lr=1e-3,
    layers=5,
    units=64,
    gwt_train_frac=0.7,
    cluster_bins=4,
    oversample_factor=3,
    hybrid_n_anchor=7,
    mlp_hidden=48,
    mlp_depth=2,
    lambda_smooth=0.0,
    lambda_link=0.0,
    phys_weight=0.7,
    lambda_latent_var=0.05,
    lambda_latent_rough=0.02,
    lambda_latent_rough_S=0.0,
    lambda_mlp_l2=0.0,
    lambda_grad_raw=0.3,
    lambda_grad_t_raw=0.5,
    lambda_h_tt=0.2,
    temporal_grad_clip=0.15,
    lambda_temporal_clip=0.3,
    target_latent_var=0.05,
    target_s_var=0.06,
    lambda_s_var_shortfall=0.5,
    temporal_kernel_scale=0.01,
    ratio_min=0.1,
    ratio_max=10.0,
    use_fourier_head=True,
    fourier_freqs=(1,2,4,8,16,32),
    device_model="cuda",
    include_forcings: bool = True,
    skip_lbfgs=True,
    save_dir="pinn_hybrid_aniso_single_head_simple",
    return_penalty_history=False,  # NEW
    w_bc=1.0,
    w_ic=1.0,
    use_film=False,
):
    print("[INFO] Staged PINN training start (simple single-branch head).")
    colmap=default_colmap_scaled(gwt_s, stage_s, well_s)
    scales=stats_to_scales_dict(stats_df,colmap)

    ic_df = None
    if ic_tif_path:
        print(f"[INFO] Sampling IC from {ic_tif_path} (N={num_domain})")
        ic_df = sample_ic_from_tif(ic_tif_path, num_domain, scales)
        if ic_df is not None:
             print(f"[INFO] IC sampled successfully: {len(ic_df)} points")

    river_info=None
    if river_csv_path and os.path.isfile(river_csv_path):
        try:
            rivdata=build_river_dataset_from_csv(river_csv_path,stage_s,colmap,scales)
            river_info=rivdata
            if augment_river_dirichlet and rivdata["river_stage_df"] is not None and not rivdata["river_stage_df"].empty:
                stage_s=pd.concat([stage_s,
                                   rivdata["river_stage_df"][["x_scaled","y_scaled","Date_scaled","Stage_scaled_val"]]],
                                  ignore_index=True)
                stage_s. drop_duplicates(subset=["x_scaled","y_scaled","Date_scaled"],inplace=True)
                print(f"[INFO] Added river Dirichlet points:  {len(rivdata['river_stage_df'])}")
        except Exception as e: 
            print(f"[WARN] River CSV processing failed: {e}")
    else:
        print("[INFO] River CSV not found; skipping augmentation.")

    w_tmin=float(well_s[colmap["well"]["t"]].min())
    w_tmax=float(well_s[colmap["well"]["t"]].max())
    etrch_df=preprocess_et_rch_csv_days(et_rch_csv_path, scales,
                                        well_time_window_scaled=(w_tmin,w_tmax),
                                        time_pad=0.0)
    et_fun_phys = make_rbf_interpolator_from_df_3d(etrch_df,"ET",ls_xyz=(0.07,0.07,0.05))
    rch_fun_phys= make_rbf_interpolator_from_df_3d(etrch_df,"RCH",ls_xyz=(0.07,0.07,0.05))
    ztop_fun = make_rbf_interpolator_from_csv_2d(ztop_extn_csv_path,"X","Y","Ztop",scales)
    dz_fun   = make_rbf_interpolator_from_csv_2d(ztop_extn_csv_path,"X","Y","Dz",scales)

    artifacts=[]; metrics=[]
    prev_net=None
    last=dict(net=None,K=None,S=None,colmap=None,scales=None,gwt_test=None)
    iter_counter=0
    last_penalty_history = None  # Track last iteration's penalty

    for w in range(warmup_iters):
        iter_counter+=1
        print("\n"+"="*80); print(f"[WARMUP {w+1}/{warmup_iters}]"); print("="*80)
        head_net, K_field, S_field, sc_it, cm_it, gwt_te, metr, pen_hist = train_one_iteration(
            stage_s,gwt_s,well_s,stats_df,
            et_fun_phys,rch_fun_phys,ztop_fun,dz_fun,
            include_forcings=True,
            prev_net=prev_net,
            river_info=river_info,
            ic_df=ic_df,
            num_domain=num_domain,
            adam_iters=warmup_adam_iters,
            layers=layers, units=units, lr=lr,
            phys_weight=phys_weight,
            gwt_train_frac=1.0,
            cluster_bins=cluster_bins,
            oversample_factor=oversample_factor,
            hybrid_n_anchor=hybrid_n_anchor,
            mlp_hidden=mlp_hidden,
            mlp_depth=mlp_depth,
            mlp_activation="tanh",
            device_model=device_model,
            skip_lbfgs=skip_lbfgs,
            lambda_smooth=lambda_smooth,
            lambda_link=lambda_link,
            lambda_latent_var=lambda_latent_var,
            lambda_latent_rough=lambda_latent_rough,
            lambda_latent_rough_S=lambda_latent_rough_S,
            target_latent_var=target_latent_var,
            lambda_mlp_l2=lambda_mlp_l2,
            lambda_grad_raw=lambda_grad_raw,
            lambda_grad_t_raw=lambda_grad_t_raw,
            lambda_h_tt=lambda_h_tt,
        w_bc=w_bc,
        w_ic=w_ic,
            temporal_grad_clip=temporal_grad_clip,
            lambda_temporal_clip=lambda_temporal_clip,
            target_s_var=target_s_var,
            lambda_s_var_shortfall=lambda_s_var_shortfall,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            temporal_kernel_scale=temporal_kernel_scale,
            use_fourier_head=use_fourier_head,
            fourier_freqs=fourier_freqs,
            split_seed=1234+w,
            warmup_all_data=True,
            return_penalty_history=return_penalty_history,
            use_film=use_film,
        )
        metrics.append(metr)
        if pen_hist is not None:
            last_penalty_history = pen_hist
        art=save_artifact(iter_counter,save_dir,head_net,K_field,S_field,sc_it,cm_it,dict(stage="warmup"))
        artifacts.append(art)
        prev_net=head_net
        last. update(net=head_net,K=K_field,S=S_field,colmap=cm_it,scales=sc_it,gwt_test=gwt_te)

    for it in range(num_iters):
        iter_counter+=1
        print("\n"+"="*80); print(f"[MAIN {it+1}/{num_iters}]"); print("="*80)
        head_net, K_field, S_field, sc_it, cm_it, gwt_te, metr, pen_hist = train_one_iteration(
            stage_s,gwt_s,well_s,stats_df,
            et_fun_phys,rch_fun_phys,ztop_fun,dz_fun,
            include_forcings=include_forcings,
            prev_net=prev_net,
            river_info=river_info,
            ic_df=ic_df,
            num_domain=num_domain,
            adam_iters=adam_iters,
            layers=layers, units=units, lr=lr,
            phys_weight=phys_weight,
            gwt_train_frac=gwt_train_frac,
            cluster_bins=cluster_bins,
            oversample_factor=oversample_factor,
            hybrid_n_anchor=hybrid_n_anchor,
            mlp_hidden=mlp_hidden,
            mlp_depth=mlp_depth,
            mlp_activation="tanh",
            device_model=device_model,
            skip_lbfgs=skip_lbfgs,
            lambda_smooth=lambda_smooth,
            lambda_link=lambda_link,
            lambda_latent_var=lambda_latent_var,
            lambda_latent_rough=lambda_latent_rough,
            lambda_latent_rough_S=lambda_latent_rough_S,
            target_latent_var=target_latent_var,
            lambda_mlp_l2=lambda_mlp_l2,
            lambda_grad_raw=lambda_grad_raw,
            lambda_grad_t_raw=lambda_grad_t_raw,
            lambda_h_tt=lambda_h_tt,
        w_bc=w_bc,
        w_ic=w_ic,
            temporal_grad_clip=temporal_grad_clip,
            lambda_temporal_clip=lambda_temporal_clip,
            target_s_var=target_s_var,
            lambda_s_var_shortfall=lambda_s_var_shortfall,
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            temporal_kernel_scale=temporal_kernel_scale,
            use_fourier_head=use_fourier_head,
            fourier_freqs=fourier_freqs,
            split_seed=999+it,
            warmup_all_data=False,
            return_penalty_history=return_penalty_history,
            use_film=use_film,
        )
        metrics.append(metr)
        if pen_hist is not None:
            last_penalty_history = pen_hist
        print(f"[METRICS] Iter {iter_counter}: {metr}")
        art=save_artifact(iter_counter,save_dir,head_net,K_field,S_field,sc_it,cm_it,dict(stage="main"))
        artifacts.append(art)
        prev_net=head_net
        last.update(net=head_net,K=K_field,S=S_field,colmap=cm_it,scales=sc_it,gwt_test=gwt_te)

    return dict(
        artifacts=artifacts,
        metrics=metrics,
        final_net=last["net"],
        final_K_field=last["K"],
        final_S_field=last["S"],
        scales=last["scales"],
        colmap=last["colmap"],
        gwt_test=last["gwt_test"],
        penalty_history=last_penalty_history,  # NEW:  return penalty history
    )

# --------------------------------------------------------
# Sampling (unchanged)
# --------------------------------------------------------
def sample_K_S_maps(K_field, S_field,
                    K_range=(1e-2,5e2), S_range=(0.02, 0.35),
                    nx=121, ny=101, device=None):
    K_field.eval(); S_field.eval()
    xs=np.linspace(0,1,nx,dtype=np.float32)
    ys=np.linspace(0,1,ny,dtype=np.float32)
    Xg,Yg=np.meshgrid(xs,ys)
    xy=torch.tensor(np.stack([Xg.ravel(),Yg.ravel()],axis=1),
                    dtype=torch.float32, device=K_field.raw_latent_x.device)
    with torch.no_grad():
        Kx_u, Ky_u = K_field(xy)
        S_u = S_field(xy)
        Kx = map_unit_to_range_exp(Kx_u,*K_range).cpu().numpy().reshape(ny,nx)
        Ky = map_unit_to_range_exp(Ky_u,*K_range).cpu().numpy().reshape(ny,nx)
        S  = map_unit_to_range_exp(S_u ,*S_range).cpu().numpy().reshape(ny,nx)
    return Kx,Ky,S

# --------------------------------------------------------
# MAIN (example usage)
# --------------------------------------------------------
if False: # __name__ == "__main__":
    TIMES_TO_PLOT=[0.0,0.5,1.0]
    needed=["stage_s","gwt_s","well_s","stats_df"]
    if not all(n in globals() for n in needed):
        print("Define stage_s, gwt_s, well_s, stats_df first.")
        raise SystemExit(0)

    results=run_pinn_iterations(
        stage_s=stage_s,
        gwt_s=gwt_s,
        well_s=well_s,
        stats_df=stats_df,
        et_rch_csv_path="ET_RCH_VARUNA_ML.csv",
        ztop_extn_csv_path="Z_top_extn_depth.csv",
        river_csv_path="varuna_points_latlong.csv",
        augment_river_dirichlet=True,
        warmup_iters=1,
        num_iters=3,
        warmup_adam_iters=1000,
        adam_iters=3000,
        gwt_train_frac=0.7,
        cluster_bins=4,
        oversample_factor=1,
        hybrid_n_anchor=8,
        mlp_hidden=80,
        mlp_depth=3,
        lambda_smooth=0.0,
        lambda_link=0.0,
        phys_weight=0.695,
        lambda_latent_var=0.1071,
        lambda_latent_rough=0.0521,
        lambda_latent_rough_S=0.02593,
        lambda_mlp_l2=1e-4,
        lambda_grad_raw=0.6253,
        lambda_grad_t_raw=0.373,
        lambda_h_tt=0.4864,
        temporal_grad_clip=0.113,
        lambda_temporal_clip=0.1537,
        target_latent_var=0.15,
        target_s_var=0.0824,
        lambda_s_var_shortfall=0.69448,
        temporal_kernel_scale=0.013346,
        ratio_min=0.1,
        ratio_max=10.0,
        use_fourier_head=True,
        fourier_freqs=(1,2,4,8,16,32),
        device_model="cuda",
        include_forcings=True,
        skip_lbfgs=True,
        save_dir="pinn_hybrid_aniso_single_head_simple"
    )

    print("\n[INFO] Iteration metrics (single-head h vs obs):")
    for i,m in enumerate(results["metrics"],1):
        print(f" Iter {i}: RMSE={m['rmse']:.3f} MAE={m['mae']:.3f} R2={m['r2']:.3f}")

    if results["final_net"] is not None:
        net_final=results["final_net"]
        scales=results["scales"]; colmap=results["colmap"]; gwt_test=results["gwt_test"]
        m=evaluate_metrics(net_final,gwt_test,colmap,scales)
        print(f"[FINAL] metrics: {m}")
        Kx_map,Ky_map,S_map=sample_K_S_maps(results["final_K_field"],results["final_S_field"])
        print(f"Kx stats mean={Kx_map.mean():.3f} std={Kx_map.std():.3f}")
        print(f"Ky stats mean={Ky_map.mean():.3f} std={Ky_map.std():.3f}")
        print(f"S  stats mean={S_map.mean():.3e} std={S_map.std():.3e}")

    print("\nArtifacts saved.")

"""
Optuna HPO driver for the latest SINGLE-BRANCH PINN (no branching / no dot product)
with a FIXED train/test split held constant across ALL Bayesian (Optuna) trials.

CHANGES (2026-01-14):
- Added penalty monitoring utilities:
    * Persist per-epoch penalty history to CSV.
    * Plot penalty curves and total loss curves per trial.
- These utilities are resilient to upstream implementations:
    * If run_pinn_iterations returns a `penalty_history` (list[dict] or DataFrame) we save/plot it.
    * If it returns file paths (`penalty_csv`, `penalty_plot`, `penalty_loss_plot`) we copy/reuse them.
    * If none are returned, we emit a warning and continue.
"""

import os
import json
import argparse
import shutil
from datetime import datetime
from typing import Any, Dict, Tuple, Optional, Iterable, List

import random
import numpy as np
import pandas as pd
import gc
import torch
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.25)
    print("[INFO] GPU memory fraction set to 0.25")
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import rasterio
from rasterio.transform import rowcol

# --------------------------------------------------------
# --------------------------------------------------------
# Kriging IC Sampling
# --------------------------------------------------------
def sample_ic_from_tif(tif_path: str, n_samples: int,
                       scales: Dict[str, float],
                       t_date_str: str = "2022-05-25") -> pd.DataFrame:
    """
    Load TIF, sample n_samples points, scale coordinates and values,
    and return as a DataFrame matching the training data schema.
    Uses scales dict for transformation: (val - min) / range
    """
    if not os.path.exists(tif_path):
        print(f"[WARN] IC TIF not found: {tif_path}")
        return None

    try:
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            # Create meshgrid of coordinates
            height, width = data.shape
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            xs, ys = rasterio.transform.xy(src.transform, rows, cols)
            xs = np.array(xs).flatten()
            ys = np.array(ys).flatten()
            vals = data.flatten()

            # Filter NoData
            valid = np.isfinite(vals) & (vals > -1e9) & (vals < 1e9)
            xs = xs[valid]
            ys = ys[valid]
            vals = vals[valid]

            if len(vals) < n_samples:
                idx = np.arange(len(vals))
            else:
                idx = np.random.choice(len(vals), n_samples, replace=False)

            x_sel = xs[idx]
            y_sel = ys[idx]
            val_sel = vals[idx]

            # SCALE using dictionary values
            # x_scaled = (x - X_min) / Lx
            x_scaled = (x_sel - scales["X_min"]) / scales["Lx"]
            y_scaled = (y_sel - scales["Y_min"]) / scales["Ly"]

            # Time
            t_dt = pd.to_datetime(t_date_str)
            t_days = date_to_float_days(pd.Series([t_dt]*len(x_sel)))
            t_scaled = (t_days - scales["T_min"]) / scales["Lt"]

            # Value (Head): (h - H_mid) / Lh
            h_scaled = (val_sel - scales["H_mid"]) / scales["Lh"]

            df = pd.DataFrame({
                "x_scaled": x_scaled,
                "y_scaled": y_scaled,
                "Date_scaled": t_scaled,
                "GWT_scaled_val": h_scaled
            })
            return df

    except Exception as e:
        print(f"[WARN] Failed to sample IC from TIF: {e}")
        return None

# ============== GLOBAL CONSTANTS =================
FIXED_ADAM_ITERS = 7000
FIXED_HYBRID_N_ANCHOR = 8
FIXED_MAIN_NUM_ITERS = 3

DEFAULT_OUTPUT_DIR = "optuna_pinn_results_random_split"
# We will set it dynamically in main_cli

GLOBAL_GWT_TRAIN: Optional[pd.DataFrame] = None
GLOBAL_GWT_TEST: Optional[pd.DataFrame] = None

# ============== DEPENDENCY CHECK =================
def _require_globals():
    missing = [n for n in [
        "stage_s", "gwt_s", "well_s", "stats_df",
        "run_pinn_iterations",
        "map_unit_to_range_exp", "AnisotropicHybridK", "HybridFieldScalar"
    ] if n not in globals()]
    if missing:
        raise RuntimeError(
            "Missing globals: {}.\n"
            "Import required symbols and define DataFrames before running.\n"
            "Example:\n"
            "from pinn_hybrid_aniso_single_head_simple import run_pinn_iterations, AnisotropicHybridK, HybridFieldScalar, map_unit_to_range_exp".format(missing)
        )

# ============== REPRO SEED =======================
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def verify_backend():
    if dde.backend.backend_name != "pytorch":
        raise RuntimeError(
            f"DeepXDE backend is {dde.backend.backend_name}, expected 'pytorch'. "
            "Restart the Python session AFTER setting os.environ['DDE_BACKEND']='pytorch' "
            "BEFORE importing deepxde."
        )
verify_backend()

# ============== METRICS ==========================
def _safe_std(x): return float(np.std(x, ddof=1)) if x.size > 1 else 0.0
def _safe_mean(x): return float(np.mean(x)) if x.size else 0.0
def _safe_corr(x, y):
    if x.size < 2 or y.size < 2: return 0.0
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    if sx == 0 or sy == 0: return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def compute_kge(pred: np.ndarray, obs: np.ndarray) -> float:
    if pred.size == 0 or obs.size == 0: return -np.inf
    r = _safe_corr(pred, obs)
    alpha = (_safe_std(pred) / _safe_std(obs)) if _safe_std(obs) != 0 else 0.0
    beta = (_safe_mean(pred) / _safe_mean(obs)) if _safe_mean(obs) != 0 else 0.0
    return float(1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))

def compute_basic_metrics(pred, obs):
    if pred.size == 0 or obs.size == 0:
        return dict(rmse=np.nan, mae=np.nan, r2=np.nan, kge=np.nan)
    diff = pred - obs
    rmse = float(np.sqrt((diff**2).mean()))
    mae  = float(np.abs(diff).mean())
    ss_res = float((diff**2).sum())
    ss_tot = float(((obs - obs.mean())**2).sum())
    r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else np.nan
    kge = compute_kge(pred, obs)
    return dict(rmse=rmse, mae=mae, r2=r2, kge=kge)

# ============== PENALTY PERSISTENCE ==============
def _plot_penalty_df(df:  pd.DataFrame, out_png: str, title_prefix: str = ""):
    # Ensure numeric columns
    numeric_cols = []
    for c in df.columns:
        if c != "epoch": 
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                if df[c].abs().sum() > 0:  # Check if column has non-zero values
                    numeric_cols.append(c)
            except Exception: 
                continue
    
    if not numeric_cols:
        print(f"[WARN] No valid numeric penalty data to plot for {out_png}.")
        return
    
    n = len(numeric_cols)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    axes = axes.flatten()
    
    for i, c in enumerate(numeric_cols):
        axes[i].plot(df["epoch"], df[c], lw=1.2)
        axes[i].set_title(c)
        axes[i].set_xlabel("epoch")
        axes[i].grid(alpha=0.3)
    
    for j in range(len(numeric_cols), len(axes)):
        axes[j].axis("off")
    
    plt.suptitle(f"{title_prefix} Penalties", y=1.02)
    plt.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Penalty plot saved:  {out_png}")

def _plot_loss_df(df: pd.DataFrame, out_png: str, title_prefix: str = ""):
    if "total_loss" not in df:
        print(f"[WARN] total_loss column missing for {out_png}")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["total_loss"], lw=1.5, color="C0")
    ax.set_xlabel("epoch"); ax.set_ylabel("total_loss")
    ax.set_title(f"{title_prefix} Total Loss")
    ax.grid(alpha=0.3); ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Loss plot saved: {out_png}")

def _generate_plots_if_csv(csv_path: str, pen_plot: str, loss_plot: str, title_prefix: str):
    if not os.path.isfile(csv_path):
        return
    try:
        df = pd.read_csv(csv_path)
        if "epoch" not in df.columns:
            df.insert(0, "epoch", np.arange(len(df)))
        _plot_penalty_df(df, pen_plot, title_prefix)
        _plot_loss_df(df, loss_plot, title_prefix)
    except Exception as e:
        print(f"[WARN] Could not generate plots from CSV {csv_path}: {e}")


def _persist_penalties_from_results(results: Dict[str, Any], trial_dir: str, trial_num: int):
    """
    Persist penalty history to CSV/PNG for a trial, robust to multiple return shapes.
    Returns dict with keys: penalty_csv, penalty_plot, loss_plot (paths or None).
    """
    pen_csv = os.path.join(trial_dir, f"trial_{trial_num}_penalties.csv")
    pen_plot = os.path.join(trial_dir, f"trial_{trial_num}_penalties.png")
    loss_plot = os.path.join(trial_dir, f"trial_{trial_num}_loss.png")

    # 1) Copy already-produced files if present on results
    for key in ("penalty_csv", "penalty_plot", "penalty_loss_plot"):
        if key in results and isinstance(results[key], str) and os.path.isfile(results[key]):
            tgt = pen_csv if "csv" in key else (loss_plot if "loss" in key else pen_plot)
            try:
                shutil.copyfile(results[key], tgt)
                print(f"[INFO] Copied existing {key} -> {tgt}")
            except Exception as e:
                print(f"[WARN] Failed to copy {key}: {e}")

    # 1b) Look into artifacts list for penalty files if not found above
    if not os.path.isfile(pen_csv) or not os.path.isfile(pen_plot) or not os.path.isfile(loss_plot):
        arts = results.get("artifacts", [])
        for art in arts:
            for attr_name in ("penalty_csv", "penalty_plot", "penalty_loss_plot", "loss_plot"):
                path = getattr(art, attr_name, None) if hasattr(art, attr_name) else None
                if isinstance(path, str) and os.path.isfile(path):
                    tgt = pen_csv if "penalty_csv" in attr_name else (loss_plot if "loss" in attr_name else pen_plot)
                    try:
                        shutil.copyfile(path, tgt)
                        print(f"[INFO] Copied artifact {attr_name} -> {tgt}")
                    except Exception as e:
                        print(f"[WARN] Failed to copy artifact {attr_name}: {e}")

    # 1c) If still missing, glob for any penalty/loss files in trial_dir
    if not os.path.isfile(pen_csv):
        found_csvs = glob.glob(os.path.join(trial_dir, "**", "*penalt*.csv"), recursive=True)
        if found_csvs:
            try:
                shutil.copyfile(found_csvs[0], pen_csv)
                print(f"[INFO] Globbed penalty CSV -> {pen_csv}")
            except Exception as e:
                print(f"[WARN] Failed to copy globbed penalty CSV: {e}")
    if not os.path.isfile(pen_plot):
        found_plots = glob.glob(os.path.join(trial_dir, "**", "*penalt*.[pj][np]g"), recursive=True)
        if found_plots:
            try:
                shutil.copyfile(found_plots[0], pen_plot)
                print(f"[INFO] Globbed penalty plot -> {pen_plot}")
            except Exception as e:
                print(f"[WARN] Failed to copy globbed penalty plot: {e}")
    if not os.path.isfile(loss_plot):
        found_loss = glob.glob(os.path.join(trial_dir, "**", "*loss*.[pj][np]g"), recursive=True)
        if found_loss:
            try:
                shutil.copyfile(found_loss[0], loss_plot)
                print(f"[INFO] Globbed loss plot -> {loss_plot}")
            except Exception as e:
                print(f"[WARN] Failed to copy globbed loss plot: {e}")

    # 2) If penalty_history provided directly, save/plot it
    pen_hist = results.get("penalty_history")
    if pen_hist is not None:
        try:
            if isinstance(pen_hist, pd.DataFrame):
                df = pen_hist.copy()
            else:
                df = pd.DataFrame(pen_hist)
            if "epoch" not in df.columns:
                df.insert(0, "epoch", np.arange(len(df)))
            df.to_csv(pen_csv, index=False)
            print(f"[INFO] Penalty history saved: {pen_csv}")
            _plot_penalty_df(df, pen_plot, title_prefix=f"trial_{trial_num}")
            _plot_loss_df(df, loss_plot, title_prefix=f"trial_{trial_num}")
        except Exception as e:
            print(f"[WARN] Could not persist penalty history for trial {trial_num}: {e}")

    # 3) If we have CSV but missing plots, generate them
    if os.path.isfile(pen_csv):
        if not os.path.isfile(pen_plot) or not os.path.isfile(loss_plot):
            _generate_plots_if_csv(pen_csv, pen_plot, loss_plot, title_prefix=f"trial_{trial_num}")

    # 4) Final warning if nothing was produced
    if not os.path.isfile(pen_csv) and not os.path.isfile(pen_plot):
        print(f"[WARN] No penalty artifacts found/created for trial {trial_num} in {trial_dir}.")

    return dict(
        penalty_csv=pen_csv if os.path.isfile(pen_csv) else None,
        penalty_plot=pen_plot if os.path.isfile(pen_plot) else None,
        loss_plot=loss_plot if os.path.isfile(loss_plot) else None,
    )

# ============== DERIVATIVE SCALING =================
DERIV_SCALES = {}
try:
    if os.path.exists(data_path("derivative_scaling_factors.csv")):
        df_scales = pd.read_csv(data_path("derivative_scaling_factors.csv"))
        for _, row in df_scales.iterrows():
            DERIV_SCALES[row['derivative']] = row['w_max']
        print(f"[INFO] Loaded derivative scales: {DERIV_SCALES}")
    else:
        print("[WARN] derivative_scaling_factors.csv not found. Using defaults.")
except Exception as e:
    print(f"[WARN] Failed to load derivative scales: {e}")

# ============== SEARCH SPACE =====================
def build_search_space(trial: optuna.Trial, use_film: bool = False) -> Dict[str, Any]:
    p = {}
    p["phys_weight"]   = trial.suggest_float("phys_weight", 0.4, 1.2, log=True)
    
    # Helper for scaling
    def get_range(name, default_max, min_factor=1e-4, max_factor=10.0):
        if name in DERIV_SCALES and DERIV_SCALES[name] > 0:
            base = 1.0 / (DERIV_SCALES[name] + 1e-9)
            # Center the search around 'base'. Log scale.
            # e.g. base * 0.01 to base * 100
            low = base * min_factor
            high = base * 200.0 # wider upper bound
            return low, high
        else:
            return 1e-8, default_max

    # Gradient penalties - Scaled ranges
    l_grad_low, l_grad_high = get_range("dk_dx", 40.0) # approx for spatial
    p["lambda_grad_raw"]   = trial.suggest_float("lambda_grad_raw", l_grad_low, l_grad_high, log=True)
    
    l_grad_t_low, l_grad_t_high = get_range("dk_dt", 0.1)
    p["lambda_grad_t_raw"] = trial.suggest_float("lambda_grad_t_raw", l_grad_t_low, l_grad_t_high, log=True)
    
    l_htt_low, l_htt_high = get_range("d2k_dt2", 1.0) # Using d2k_dt2 as proxy or just use default?
    # Actually h_tt is about head, derivative_scaling is about K. 
    # But usually we scale physics terms. Let's keep h_tt wide or scaled if we had h scaling.
    p["lambda_h_tt"]       = trial.suggest_float("lambda_h_tt", 1e-8, 1.0, log=True)

    # Regularization
    p["lambda_smooth"]     = trial.suggest_float("lambda_smooth", 1e-8, 100.0, log=True)
    p["lambda_link"]       = trial.suggest_float("lambda_link", 1e-8, 100.0, log=True)
    p["lambda_mlp_l2"]     = trial.suggest_float("lambda_mlp_l2", 1e-10, 100.0, log=True)
    p["lambda_latent_var"] = trial.suggest_float("lambda_latent_var", 1e-8, 100.0, log=True)
    p["lambda_latent_rough"] = trial.suggest_float("lambda_latent_rough", 1e-9, 100.0, log=True)
    p["lambda_latent_rough_S"] = trial.suggest_float("lambda_latent_rough_S", 1e-9, 1.0, log=True)
    p["lambda_s_var_shortfall"] = trial.suggest_float("lambda_s_var_shortfall", 1e-8, 1.0, log=True)
    p["target_s_var"] = trial.suggest_float("target_s_var", 0.08, 0.20)
    p["target_latent_var"] = 0.05

    # Features
    # p["use_film"] = trial.suggest_categorical("use_film", [True, False])
    # p["use_film"] = True # Enforced by user requirement
    p["use_film"] = bool(use_film)  # set by optimize_pinn(use_film=...)

    use_clip = trial.suggest_categorical("use_temporal_clip", [False, True])
    if use_clip:
        p["temporal_grad_clip"]  = trial.suggest_float("temporal_grad_clip", 0.05, 0.25)
        p["lambda_temporal_clip"]= trial.suggest_float("lambda_temporal_clip", 0.05, 0.8)
    else:
        p["temporal_grad_clip"]  = 0.0
        p["lambda_temporal_clip"]= 0.0

    p["use_fourier_head"] = True
    p["layers"]     = trial.suggest_int("layers", 3, 7)
    p["units"]      = trial.suggest_categorical("units", [32, 48, 64, 80])
    p["mlp_hidden"] = trial.suggest_categorical("mlp_hidden", [64, 96, 128])
    p["mlp_depth"]  = trial.suggest_int("mlp_depth", 2, 4)
    p["temporal_kernel_scale"] = trial.suggest_float("temporal_kernel_scale", 0.005, 0.04, log=True)
    p["oversample_factor"]     = trial.suggest_int("oversample_factor", 1, 5)
    p["num_domain"]            = 4000 # Enforced by user requirement (consistency)
    p["warmup_adam_iters"]     = trial.suggest_int("warmup_adam_iters", 1200, 2600, step=200)
    p["lr"]                    = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    p["ratio_min"] = 0.05
    p["ratio_max"] = 100.0
    return p

# ============== STREAM PREDICTION =================
def _pred_obs_from_df_stream(head_net: torch.nn.Module,
                             df: pd.DataFrame,
                             colmap: Dict[str, Any],
                             scales: Dict[str,float],
                             component: int,
                             batch_size: int,
                             device: torch.device,
                             use_amp: bool,
                             pinned: bool) -> Tuple[np.ndarray,np.ndarray]:
    if df is None or len(df) == 0:
        return np.array([]), np.array([])
    xcol,ycol,tcol,hcol = (colmap["gwt"]["x"], colmap["gwt"]["y"],
                           colmap["gwt"]["t"], colmap["gwt"]["h"])
    N = len(df)
    pred_parts=[]; obs_parts=[]
    H_mid, Lh = scales["H_mid"], scales["Lh"]
    head_net.eval()

    x_arr = df[xcol].to_numpy(np.float32)
    y_arr = df[ycol].to_numpy(np.float32)
    t_arr = df[tcol].to_numpy(np.float32)
    h_scaled_obs = df[hcol].to_numpy(np.float32)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start+batch_size, N)
            batch_np = np.stack([x_arr[start:end], y_arr[start:end], t_arr[start:end]], axis=1)
            if pinned and device.type == 'cuda':
                batch_t = torch.empty((end-start,3), dtype=torch.float32, pin_memory=True)
                batch_t.copy_(torch.from_numpy(batch_np))
                batch_t = batch_t.to(device, non_blocking=True)
            else:
                batch_t = torch.tensor(batch_np, dtype=torch.float32, device=device)
            use_amp_local = use_amp and (device.type == "cuda")
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp_local):
                out = head_net(batch_t).detach()
            out_cpu = out.float().cpu().numpy()
            if out_cpu.ndim == 1:
                out_cpu = out_cpu[:, None]
            h_scaled_pred = out_cpu[:, component]
            pred_phys = H_mid + Lh * h_scaled_pred
            obs_phys  = H_mid + Lh * h_scaled_obs[start:end]
            pred_parts.append(pred_phys); obs_parts.append(obs_phys)

    return np.concatenate(pred_parts, 0), np.concatenate(obs_parts, 0)


def predict_head_grid_chunked(head_net, scales, times_scaled,
                              nx, ny, device, use_amp, chunk_pixels=20000):
    head_net.eval()
    x_orig = np.linspace(scales["X_min"], scales["X_min"]+scales["Lx"], nx, dtype=np.float32)
    y_orig = np.linspace(scales["Y_min"], scales["Y_min"]+scales["Ly"], ny, dtype=np.float32)
    Xo, Yo = np.meshgrid(x_orig, y_orig)
    xs = (Xo - scales["X_min"]) / scales["Lx"]
    ys = (Yo - scales["Y_min"]) / scales["Ly"]
    flat_x, flat_y = xs.ravel(), ys.ravel()
    total = flat_x.size
    results = {}
    H_mid, Lh = scales["H_mid"], scales["Lh"]

    with torch.no_grad():
        for t in times_scaled:
            h_store = np.empty(total, dtype=np.float32)
            for s in range(0, total, chunk_pixels):
                e = min(s+chunk_pixels, total)
                pts = np.stack([flat_x[s:e], flat_y[s:e], np.full(e-s, t, np.float32)], axis=1)
                pts_t = torch.tensor(pts, dtype=torch.float32, device=device)
                use_amp_local = use_amp and (device.type=='cuda')
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp_local):
                    out = head_net(pts_t).detach()
                out_np = out.float().cpu().numpy()
                if out_np.ndim == 1:
                    out_np = out_np[:,None]
                h_store[s:e] = H_mid + Lh * out_np[:,0]
            results[float(t)] = {"h": h_store.reshape(ys.shape)}
    return results

# ============== PLOTTING HELPERS =================
def _plot_head_maps(head_grid, times_sorted, gwt_train, gwt_test, colmap, scales, save_path):
    if not head_grid: return
    nT = len(times_sorted)
    fig, axes = plt.subplots(1, nT, figsize=(4.6*nT, 4.8), constrained_layout=True)
    if nT == 1: axes=[axes]
    xcol,ycol = colmap["gwt"]["x"], colmap["gwt"]["y"]
    wells_frames=[]
    if gwt_train is not None and len(gwt_train): wells_frames.append(gwt_train[[xcol,ycol]])
    if gwt_test is not None and len(gwt_test): wells_frames.append(gwt_test[[xcol,ycol]])
    wells_union = (pd.concat(wells_frames, ignore_index=True).drop_duplicates()
                   if wells_frames else pd.DataFrame(columns=[xcol,ycol]))
    for ax, t in zip(axes, times_sorted):
        hmap = head_grid[t]["h"]
        im = ax.imshow(hmap, origin="lower", cmap="viridis",
                       vmin=float(hmap.min()), vmax=float(hmap.max()), extent=[0,1,0,1])
        days = scales["T_min"] + t * scales["Lt"]
        dts  = pd.to_datetime(days, unit="D", origin="unix")
        ax.set_title(f"h t'={t:.3f}\n{dts.strftime('%Y-%m-%d')}", fontsize=10)
        if not wells_union.empty:
            ax.scatter(wells_union[xcol], wells_union[ycol],
                       s=36, marker='o', facecolor='none', edgecolor='k', lw=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Head Predictions (fixed split / single-branch)", y=1.02)
    fig.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close(fig)

def _plot_K_fields(K_field, S_field, scales, save_path, nx=121, ny=101, device=None, use_amp=False):
    from math import log10
    K_field.eval(); S_field.eval()
    xs = np.linspace(0,1,nx,dtype=np.float32); ys = np.linspace(0,1,ny,dtype=np.float32)
    Xg,Yg = np.meshgrid(xs,ys)
    flat = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
    with torch.no_grad():
        xy = torch.tensor(flat, dtype=torch.float32, device=device)
        use_amp_local = use_amp and (device is not None and device.type=='cuda')
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp_local):
            Kx_u, Ky_u = K_field(xy)
            S_u        = S_field(xy)
        Kx = map_unit_to_range_exp(Kx_u, 1e-2, 5e2).float().cpu().numpy().reshape(ny,nx)
        Ky = map_unit_to_range_exp(Ky_u, 1e-2, 5e2).float().cpu().numpy().reshape(ny,nx)
        S  = map_unit_to_range_exp(S_u , 0.02, 0.35).float().cpu().numpy().reshape(ny,nx)
    panels=6
    fig, axes = plt.subplots(1, panels, figsize=(4*panels,4.2), constrained_layout=True)
    plots=[
        (Kx,"Kx","viridis"),
        (Ky,"Ky","viridis"),
        (S ,"S" ,"magma"),
        (np.log10(Kx+1e-30),"log10 Kx","plasma"),
        (np.log10(Ky+1e-30),"log10 Ky","plasma"),
        (Ky/(Kx+1e-30),"Ky/Kx","coolwarm")
    ]
    for ax,(arr,title,cmap) in zip(axes,plots):
        im=ax.imshow(arr,origin="lower",cmap=cmap)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im,ax=ax,fraction=0.045,pad=0.04)
    fig.savefig(save_path,dpi=160,bbox_inches='tight')
    plt.close(fig)

def _scatter_metrics_plot(pred, obs, title_prefix, save_path):
    m = compute_basic_metrics(pred, obs)
    fig = plt.figure(figsize=(5,5))
    if pred.size:
        plt.scatter(obs, pred, s=14, c="#1f77b4", alpha=0.65, edgecolors="none")
        mn = float(min(pred.min(), obs.min()))
        mx = float(max(pred.max(), obs.max()))
        plt.plot([mn,mx],[mn,mx],"k--",lw=1)
    plt.xlabel("Observed Head (phys)")
    plt.ylabel("Predicted Head (phys)")
    plt.title(f"{title_prefix}\nRMSE={m['rmse']:.3f} MAE={m['mae']:.3f} R2={m['r2']:.3f}")
    plt.tight_layout()
    fig.savefig(save_path,dpi=160,bbox_inches='tight')
    plt.close(fig)
    return m

def _time_series_plot(head_net, gwt_train, gwt_test, colmap, scales,
                      save_path, device, use_amp, eval_chunk_size, component=0):
    xcol,ycol,tcol,hcol = colmap["gwt"]["x"], colmap["gwt"]["y"], colmap["gwt"]["t"], colmap["gwt"]["h"]
    parts=[]
    if gwt_train is not None and len(gwt_train): parts.append(gwt_train[[xcol,ycol]])
    if gwt_test  is not None and len(gwt_test):  parts.append(gwt_test[[xcol,ycol]])
    if not parts: return
    wells = pd.concat(parts, ignore_index=True).drop_duplicates().to_numpy()
    cols = min(5, wells.shape[0]); rows = int(np.ceil(wells.shape[0]/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.6, rows*3.2), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    H_mid, Lh = scales["H_mid"], scales["Lh"]
    head_net.eval()

    with torch.no_grad():
        for i,(xs,ys) in enumerate(wells):
            ax=axes[i]
            tr = gwt_train[(gwt_train[xcol]==xs)&(gwt_train[ycol]==ys)] if gwt_train is not None else pd.DataFrame()
            te = gwt_test [(gwt_test [xcol]==xs)&(gwt_test [ycol]==ys)] if gwt_test  is not None else pd.DataFrame()
            t_collect=[]
            if len(tr): t_collect.append(tr[tcol].to_numpy(np.float32))
            if len(te): t_collect.append(te[tcol].to_numpy(np.float32))
            if not t_collect:
                ax.axis('off'); continue
            t_all=np.concatenate(t_collect)
            t_dense=np.linspace(t_all.min(), t_all.max(), 150, dtype=np.float32)
            preds=[]
            for s in range(0,len(t_dense),eval_chunk_size):
                e=min(s+eval_chunk_size,len(t_dense))
                batch=np.stack([np.full(e-s,xs,dtype=np.float32),
                                np.full(e-s,ys,dtype=np.float32),
                                t_dense[s:e]],axis=1)
                batch_t=torch.tensor(batch,dtype=torch.float32,device=device)
                use_amp_local = use_amp and (device.type=='cuda')
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp_local):
                    out=head_net(batch_t).detach()
                out_np=out.float().cpu().numpy()
                if out_np.ndim==1: out_np=out_np[:,None]
                preds.append(H_mid + Lh*out_np[:,component])
            h_pred=np.concatenate(preds)
            t_dense_days=scales["T_min"]+t_dense*scales["Lt"]
            t_dense_dt=pd.to_datetime(t_dense_days,unit="D",origin="unix")
            ax.plot(t_dense_dt,h_pred,color='dimgray',lw=1.2,label="Pred")
            if len(tr):
                t_tr_days=scales["T_min"]+tr[tcol].to_numpy(np.float32)*scales["Lt"]
                ax.scatter(pd.to_datetime(t_tr_days,unit="D",origin="unix"),
                           H_mid+Lh*tr[hcol].to_numpy(np.float32),
                           c='#1f77b4',s=16,edgecolors='k',linewidths=0.35,label="Train")
            if len(te):
                t_te_days=scales["T_min"]+te[tcol].to_numpy(np.float32)*scales["Lt"]
                ax.scatter(pd.to_datetime(t_te_days,unit="D",origin="unix"),
                           H_mid+Lh*te[hcol].to_numpy(np.float32),
                           c='#d62728',s=18,marker='^',edgecolors='k',linewidths=0.35,label="Test")
            ax.set_title(f"W{i+1} (x={xs:.2f}, y={ys:.2f})",fontsize=8)
            ax.tick_params(axis='x',rotation=45)
            ax.grid(alpha=0.3)
            if i%cols==0: ax.set_ylabel("Head")
            if i==0: ax.legend(fontsize=7,loc="best",framealpha=0.6)
        last=i if wells.shape[0]>0 else -1
        for j in range(last+1,len(axes)):
            axes[j].axis('off')
    fig.suptitle("Wells Time Series (fixed split)", y=1.02)
    fig.savefig(save_path,dpi=160,bbox_inches='tight')
    plt.close(fig)

def _plot_q_well_maps(q_fun, scales, save_path, nx=120, ny=100, device=None, use_amp=False,
                      times_scaled=(0.0,0.5,1.0)):
    xs=np.linspace(0,1,nx,dtype=np.float32)
    ys=np.linspace(0,1,ny,dtype=np.float32)
    Xg,Yg=np.meshgrid(xs,ys)
    flat_xy=np.stack([Xg.ravel(),Yg.ravel()],axis=1)
    maps=[]
    with torch.no_grad():
        for t in times_scaled:
            pts=np.concatenate([flat_xy,np.full((flat_xy.shape[0],1),t,dtype=np.float32)],axis=1)
            pts_t=torch.tensor(pts,dtype=torch.float32,device=device)
            use_amp_local=use_amp and (device is not None and device.type=='cuda')
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp_local):
                q=q_fun(pts_t)
            maps.append(q.float().cpu().numpy().reshape(ny,nx))
    allv=np.concatenate([m.ravel() for m in maps])
    vmin=float(np.nanmin(allv)); vmax=float(np.nanmax(allv))
    fig,axes=plt.subplots(1,len(times_scaled),figsize=(4.6*len(times_scaled),4.8),constrained_layout=True)
    if len(times_scaled)==1: axes=[axes]
    for ax,t,arr in zip(axes,times_scaled,maps):
        im=ax.imshow(arr,origin='lower',cmap='coolwarm',vmin=vmin,vmax=vmax,extent=[0,1,0,1])
        days=scales["T_min"]+t*scales["Lt"]
        dt=pd.to_datetime(days,unit="D",origin="unix")
        ax.set_title(f"q_well t'={t:.3f}\n{dt.strftime('%Y-%m-%d')}",fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    fig.suptitle("q_well Heat Maps (fixed split)", y=1.02)
    fig.savefig(save_path,dpi=160,bbox_inches='tight')
    plt.close(fig)

# ============== PLOT AGGREGATED TRIAL =============
def plot_trial(trial_dir, results, gwt_train, gwt_test, times_to_plot,
               device, use_amp, eval_chunk_size, disable_grid_maps):
    net=results["final_net"]
    if net is None: return
    scales=results["scales"]; colmap=results["colmap"]
    pred_tr, obs_tr = _pred_obs_from_df_stream(net, gwt_train, colmap, scales, 0,
                                               eval_chunk_size, device, use_amp, False)
    pred_te, obs_te = _pred_obs_from_df_stream(net, gwt_test,  colmap, scales, 0,
                                               eval_chunk_size, device, use_amp, False)
    _scatter_metrics_plot(pred_te, obs_te, "Test Head vs Prediction (fixed split)",
                          os.path.join(trial_dir,"fig_scatter_test.pdf"))
    _scatter_metrics_plot(pred_tr, obs_tr, "Train Head vs Prediction (fixed split)",
                          os.path.join(trial_dir,"fig_scatter_train.pdf"))
    if not disable_grid_maps and len(times_to_plot)>0:
        head_grid=predict_head_grid_chunked(net,scales,times_to_plot,
                                            nx=120,ny=100,device=device,use_amp=use_amp)
        _plot_head_maps(head_grid, sorted(head_grid.keys()),
                        gwt_train, gwt_test, colmap, scales,
                        os.path.join(trial_dir,"fig_head_maps.pdf"))
    _plot_K_fields(results["final_K_field"],results["final_S_field"],scales,
                   os.path.join(trial_dir,"fig_K_fields.pdf"),
                   device=device,use_amp=use_amp)
    _time_series_plot(net,gwt_train,gwt_test,colmap,scales,
                      os.path.join(trial_dir,"fig_timeseries.pdf"),
                      device=device,use_amp=use_amp,
                      eval_chunk_size=eval_chunk_size)
    # Optional q_well
    try:
        if "make_rbf_source_from_wells_torch" in globals():
            q_fun = make_rbf_source_from_wells_torch(
                well_s,
                colmap["well"]["x"], colmap["well"]["y"],
                colmap["well"]["t"], colmap["well"]["Q"],
                scales["Q_min"], scales["Q_range"], scales["Lh"],
                rbf_lengthscales=(0.08,0.08,0.08),
                temporal_kernel_scale=None,
                max_centers=800
            )
            _plot_q_well_maps(q_fun, scales,
                              os.path.join(trial_dir,"fig_q_well_heatmaps.pdf"),
                              nx=120, ny=100, device=device, use_amp=use_amp)
    except Exception as e:
        with open(os.path.join(trial_dir,"plot_error.txt"),"a") as f:
            f.write(f"q_well plotting error: {e}\n")

# ============== FIXED RANDOM SPLIT ===============
def fixed_random_split(df: pd.DataFrame, frac_train: float, seed: int):
    """
    TEMPORAL SPLIT (Fixed): Wrapper around the 3-way split.
    """
    return random_train_val_test_split(df, frac_train, 0.15, seed)

# ============== OBJECTIVE (KGE maximize) =========
def objective_builder(global_cfg, plot_each_trial, times_to_plot,
                      eval_chunk_size, use_amp, offload_after_trial,
                      disable_grid_maps, low_mem, force_cpu):
    def objective(trial: optuna.Trial) -> float:
        global GLOBAL_GWT_TRAIN, GLOBAL_GWT_VAL, GLOBAL_GWT_TEST, GLOBAL_TEST_INDICES
        p = build_search_space(trial, use_film=global_cfg.get("use_film", False))

        run_kwargs=dict(
            warmup_adam_iters=p["warmup_adam_iters"],
            adam_iters=FIXED_ADAM_ITERS,
            num_domain=p["num_domain"],
            lr=p["lr"],
            layers=p["layers"],
            units=p["units"],
            hybrid_n_anchor=FIXED_HYBRID_N_ANCHOR,
            mlp_hidden=p["mlp_hidden"],
            mlp_depth=p["mlp_depth"],
            lambda_smooth=p["lambda_smooth"],
            lambda_link=p["lambda_link"],
            phys_weight=p["phys_weight"],
            lambda_latent_var=p["lambda_latent_var"],
            lambda_latent_rough=p["lambda_latent_rough"],
            lambda_latent_rough_S=p["lambda_latent_rough_S"],
            lambda_mlp_l2=0.0,
            lambda_grad_raw=p["lambda_grad_raw"],
            lambda_grad_t_raw=p["lambda_grad_t_raw"],
            lambda_h_tt=p["lambda_h_tt"],
            w_bc=p.get("w_bc", 1.0),
            w_ic=p.get("w_ic", 1.0),
            temporal_grad_clip=p["temporal_grad_clip"],
            lambda_temporal_clip=p["lambda_temporal_clip"],
            target_latent_var=p.get("target_latent_var",0.05),
            target_s_var=p["target_s_var"],
            lambda_s_var_shortfall=p["lambda_s_var_shortfall"],
            temporal_kernel_scale=p["temporal_kernel_scale"],
            ratio_min=p["ratio_min"],
            ratio_max=p["ratio_max"],
            use_fourier_head=True,
            fourier_freqs=(1,2,4,8,16,32,64),
            gwt_train_frac=1.0,
            cluster_bins=4,
            oversample_factor=p["oversample_factor"],
            return_penalty_history=True,
            use_film=p["use_film"],
        )

        trial_dir=os.path.join(global_cfg["output_dir"], f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        device_model=global_cfg["device_model"]
        if force_cpu: device_model="cpu"

        common=dict(
            stage_s=stage_s,
            gwt_s=GLOBAL_GWT_TRAIN,
            well_s=well_s,
            stats_df=stats_df,
            et_rch_csv_path=global_cfg["et_rch_csv_path"],
            ztop_extn_csv_path=global_cfg["ztop_extn_csv_path"],
            river_csv_path=global_cfg["river_csv_path"],
            ic_tif_path=global_cfg.get("ic_tif_path"),
            augment_river_dirichlet=True,
            warmup_iters=1,
            num_iters=FIXED_MAIN_NUM_ITERS,
            include_forcings=True,
            device_model=device_model,
            skip_lbfgs=True,
            save_dir=trial_dir
        )

        try:
            results = run_pinn_iterations(**common, **run_kwargs)
        except RuntimeError as e:
            trial.set_user_attr("error", str(e))
            with open(os.path.join(trial_dir,"trial_error.txt"),"w") as f: f.write(str(e))
            if low_mem and torch.cuda.is_available(): torch.cuda.empty_cache()
            return -1e6

        pen_paths = _persist_penalties_from_results(results, trial_dir, trial.number)
        trial.set_user_attr("penalty_artifacts", pen_paths)

        net=results.get("final_net")
        if net is None:
            trial.set_user_attr("error", "no final_net")
            if low_mem and torch.cuda.is_available(): torch.cuda.empty_cache()
            return -1e6

        colmap=results["colmap"]; scales=results["scales"]
        device=next(net.parameters()).device

        pred_train, obs_train = _pred_obs_from_df_stream(
            net, GLOBAL_GWT_TRAIN, colmap, scales, component=0,
            batch_size=eval_chunk_size, device=device, use_amp=use_amp, pinned=False
        )
        pred_val, obs_val = _pred_obs_from_df_stream(
            net, GLOBAL_GWT_VAL, colmap, scales, component=0,
            batch_size=eval_chunk_size, device=device, use_amp=use_amp, pinned=False
        )

        train_metrics = compute_basic_metrics(pred_train, obs_train)
        val_metrics  = compute_basic_metrics(pred_val, obs_val)
        val_kge = val_metrics["kge"]

        trial.report(val_kge, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()

        trial.set_user_attr("train_metrics", train_metrics)
        trial.set_user_attr("val_metrics", val_metrics)
        trial.set_user_attr("params_used", p)

        # SAVE METRICS TO DISK (per user request)
        try:
            metrics_path = os.path.join(trial_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump({"train": train_metrics, "val": val_metrics, "params": p}, f, indent=2)
            print(f"[INFO] Saved metrics to {metrics_path}")
        except Exception as e:
            print(f"[WARN] Failed to save metrics.json: {e}")

        # SAVE PREDICTIONS CSV (Train/Validation)
        try:
            # Helper to build dataframe
            def make_pred_df(pred, obs, src_df):
                # Ensure length matches
                n = len(pred)
                if len(src_df) != n:
                    # Should match if batch stream used full df
                    return pd.DataFrame()
                df_out = src_df.copy().reset_index(drop=True)
                df_out["predicted_head"] = pred.flatten()
                df_out["observed_head"] = obs.flatten() # Double check if obs_phys matches
                return df_out

            train_df_out = make_pred_df(pred_train, obs_train, GLOBAL_GWT_TRAIN)
            val_df_out = make_pred_df(pred_val, obs_val, GLOBAL_GWT_VAL)
            
            if not train_df_out.empty:
                train_df_out.to_csv(os.path.join(trial_dir, "predictions_train.csv"), index=False)
            if not val_df_out.empty:
                val_df_out.to_csv(os.path.join(trial_dir, "predictions_val.csv"), index=False)
            print(f"[INFO] Saved predictions CSVs to {trial_dir}")
        except Exception as e:
            print(f"[WARN] Failed to save prediction CSVs: {e}")

        # SAVE WEIGHTS JSON
        try:
            weights_path = os.path.join(trial_dir, "weights.json")
            # Save params 'p'.
            with open(weights_path, "w") as f:
                json.dump(p, f, indent=2)
            print(f"[INFO] Saved weights.json to {weights_path}")
        except Exception as e:
            print(f"[WARN] Failed to save weights.json: {e}")

        if plot_each_trial:
            try:
                plot_trial(trial_dir, results, GLOBAL_GWT_TRAIN, GLOBAL_GWT_TEST,
                           times_to_plot, device=device, use_amp=use_amp,
                           eval_chunk_size=eval_chunk_size,
                           disable_grid_maps=disable_grid_maps)
            except Exception as pe:
                with open(os.path.join(trial_dir,"plot_error.txt"),"w") as f:
                    f.write(str(pe))

        if offload_after_trial:
            net.cpu()
            results["final_K_field"].cpu()
            results["final_S_field"].cpu()

        del pred_train, obs_train, pred_val, obs_val
        if low_mem and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return val_kge
    return objective

def temporal_train_val_test_split(df: pd.DataFrame, frac_train: float = 0.7, frac_val: float = 0.15, seed: int = 1234):
    """
    3-Way TEMPORAL SPLIT (INTERPOLATION): Sort by Date_scaled.
    """
    if len(df) == 0: return df.copy(), df.copy(), df.copy()
    t_col = "Date_scaled" if "Date_scaled" in df.columns else "t"
    if t_col in df.columns:
        df = df.sort_values(t_col).reset_index(drop=True)
    else:
        df = df.sort_index()

    N = len(df)
    n_val = int(round(frac_val * N))
    n_te = N - int(round(frac_train * N)) - n_val
    
    start_val = int(round(N * (frac_train) / 2))
    start_te = start_val + n_val
    end_te = start_te + n_te
    
    val = df.iloc[start_val:start_te].copy()
    test = df.iloc[start_te:end_te].copy()
    train = pd.concat([df.iloc[:start_val], df.iloc[end_te:]]).copy()
    
    global GLOBAL_TEST_INDICES
    # Hash BOTH validation and test for robust leakage guard in downstream functions
    GLOBAL_TEST_INDICES = set(val.index.tolist() + test.index.tolist())
    
    print(f"[INFO] Temporal Split ({frac_train*100:.0f}/{frac_val*100:.0f}/{(1-frac_train-frac_val)*100:.0f}): Train={len(train)} Val={len(val)} Test={len(test)}")
    return train, val, test

def retrain_best(best_params: Dict[str,Any],
                 long_cfg: Dict[str,Any],
                 output_dir: str,
                 times_to_plot: Iterable[float],
                 eval_chunk_size: int,
                 use_amp: bool,
                 offload_after_trial: bool,
                 disable_grid_maps: bool,
                 force_cpu: bool):
    global GLOBAL_GWT_TRAIN, GLOBAL_GWT_VAL, GLOBAL_GWT_TEST, GLOBAL_TEST_INDICES
    os.makedirs(output_dir,exist_ok=True)
    final_params = best_params.copy()
    final_params.update(dict(
        hybrid_n_anchor=FIXED_HYBRID_N_ANCHOR,
        adam_iters=FIXED_ADAM_ITERS,
        gwt_train_frac=1.0,
        use_fourier_head=True,
        fourier_freqs=(1,2,4,8,16,32,64),
        return_penalty_history=True,
    ))
    final_params.update(long_cfg)
    if "w_bc" not in final_params: final_params["w_bc"]=1.0
    if "w_ic" not in final_params: final_params["w_ic"]=1.0
    device_model = long_cfg.get("device_model","cuda")
    if force_cpu: device_model="cpu"
    

    print("[INFO] Retraining best config on FIXED train split.")
    
    for k in ["save_dir", "et_rch_csv_path", "ztop_extn_csv_path", "river_csv_path", "device_model"]:
        if k in final_params: del final_params[k]
        
    results = run_pinn_iterations(
        stage_s=stage_s,
        gwt_s=GLOBAL_GWT_TRAIN,
        well_s=well_s,
        stats_df=stats_df,
        et_rch_csv_path=long_cfg.get("et_rch_csv_path","ET_RCH_VARUNA_ML.csv"),
        ztop_extn_csv_path=long_cfg.get("ztop_extn_csv_path","Z_top_extn_depth.csv"),
        river_csv_path=long_cfg.get("river_csv_path","varuna_points_latlong.csv"),
        augment_river_dirichlet=True,
        include_forcings=True,
        device_model=device_model,
        skip_lbfgs=True,
        save_dir=output_dir,
        **final_params
    )

    _persist_penalties_from_results(results, output_dir, trial_num=-1)

    net=results["final_net"]; colmap=results["colmap"]; scales=results["scales"]
    device=next(net.parameters()).device
    
    # Generate Predictions for Train and Test
    pred_train, obs_train = _pred_obs_from_df_stream(
        net, GLOBAL_GWT_TRAIN, colmap, scales, 0, eval_chunk_size, device, use_amp, False)
    pred_test,  obs_test  = _pred_obs_from_df_stream(
        net, GLOBAL_GWT_TEST,  colmap, scales, 0, eval_chunk_size, device, use_amp, False)
    
    # Save predictions
    tr_df = GLOBAL_GWT_TRAIN.copy(); tr_df["pred_h"] = pred_train; tr_df["obs_h"] = obs_train
    te_df = GLOBAL_GWT_TEST.copy();  te_df["pred_h"] = pred_test;  te_df["obs_h"] = obs_test
    tr_df.to_csv(os.path.join(output_dir, "predictions_train.csv"), index=False)
    te_df.to_csv(os.path.join(output_dir, "predictions_test.csv"), index=False)
        
    train_metrics=compute_basic_metrics(pred_train, obs_train)
    test_metrics =compute_basic_metrics(pred_test,  obs_test)
    with open(os.path.join(output_dir,"retrain_train_metrics.json"),"w") as f:
        json.dump(train_metrics,f,indent=2)
    with open(os.path.join(output_dir,"retrain_test_metrics.json"),"w") as f:
        json.dump(test_metrics,f,indent=2)
        
    print(f"[FINAL RETRAIN] KGE={test_metrics['kge']:.4f} RMSE={test_metrics['rmse']:.4f} MAE={test_metrics['mae']:.4f} R2={test_metrics['r2']:.4f}")
    
    plot_trial(output_dir, results, GLOBAL_GWT_TRAIN, GLOBAL_GWT_TEST,
               times_to_plot, device, use_amp, eval_chunk_size, disable_grid_maps)
               
    # Ensure Aquifer Properties are extracted (K and S)
    try:
        K_field, S_field = results["final_K_field"], results["final_S_field"]
        nx_ks, ny_ks = 100, 100
        Kx_map, Ky_map, S_map = sample_K_S_maps(K_field, S_field, nx=nx_ks, ny=ny_ks)
        xg, yg = np.meshgrid(np.linspace(0, 1, nx_ks, dtype=np.float32),
                             np.linspace(0, 1, ny_ks, dtype=np.float32))
        ks_df = pd.DataFrame({
            "x_scaled": xg.ravel(), "y_scaled": yg.ravel(),
            "x": scales["X_min"] + xg.ravel() * scales["Lx"],
            "y": scales["Y_min"] + yg.ravel() * scales["Ly"],
            "Kx_m_per_day": Kx_map.ravel(), "Ky_m_per_day": Ky_map.ravel(), "S": S_map.ravel(),
        })
        ks_df.to_csv(os.path.join(output_dir, "aquifer_properties_K_S.csv"), index=False)
        print("[INFO] Extracted Aquifer properties K and S to csv.")
    except Exception as e:
        print(f"[WARN] Failed to extract Aquifer properties: {e}")
               
    if offload_after_trial:
        net.cpu(); results["final_K_field"].cpu(); results["final_S_field"].cpu()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
    return dict(results=results, test_metrics=test_metrics, train_metrics=train_metrics)

def optimize_pinn(n_trials: int = 30,
                  retrain: bool = False,
                  sampler: str = "tpe",
                  study_name: str = "pinn_hpo_single_branch_fixedsplit",
                  storage: Optional[str] = None,
                  split_type: str = "random",
                  use_film: bool = False,
                  disable_progress_in_notebook: bool = True,
                  fixed_split_seed: int = 1234,
                  train_fraction: float = 0.7,
                  plot_per_trial: bool = True,
                  times_to_plot: Iterable[float] = (0.0,0.5,1.0),
                  eval_chunk_size: int = 4096,
                  use_amp: bool = False,
                  offload_after_trial: bool = False,
                  disable_grid_maps: bool = False,
                  low_mem: bool = False,
                  force_cpu: bool = False,
                  global_seed: Optional[int] = 42,
                  output_dir: Optional[str] = None
                  ) -> Tuple[optuna.study.Study, Optional[Dict[str,Any]]]:
    global GLOBAL_GWT_TRAIN, GLOBAL_GWT_VAL, GLOBAL_GWT_TEST, GLOBAL_TEST_INDICES, DEFAULT_OUTPUT_DIR
    if output_dir:
        DEFAULT_OUTPUT_DIR = output_dir
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    if "gwt_s" not in globals():
        raise RuntimeError("No data loaded. Call load_and_preprocess_data() before optimize_pinn().")

    # _require_globals()
    if global_seed is not None:
        set_global_seed(global_seed)
        print(f"[INFO] Global seed set to {global_seed}")

    print(f"[INFO] Creating fixed 3-way partition: {split_type}")
    if split_type == "random":
        GLOBAL_GWT_TRAIN, GLOBAL_GWT_VAL, GLOBAL_GWT_TEST = random_train_val_test_split(gwt_s, 0.7, 0.15, fixed_split_seed)
    elif split_type == "temporal":
        GLOBAL_GWT_TRAIN, GLOBAL_GWT_VAL, GLOBAL_GWT_TEST = temporal_train_val_test_split(gwt_s, 0.7, 0.15, fixed_split_seed)
    elif split_type == "temporal_lowdata":
        GLOBAL_GWT_TRAIN, GLOBAL_GWT_VAL, GLOBAL_GWT_TEST = temporal_train_val_test_split(gwt_s, 0.3, 0.15, fixed_split_seed)
    wells_union = (pd.concat([
        GLOBAL_GWT_TRAIN[['x_scaled','y_scaled']],
        GLOBAL_GWT_TEST[['x_scaled','y_scaled']]
    ], ignore_index=True).drop_duplicates().shape[0])
    print(f"[INFO] Train rows={len(GLOBAL_GWT_TRAIN)} Test rows={len(GLOBAL_GWT_TEST)} Unique wells={wells_union}")

    device_model = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    print(f"[INFO] Using device_model={device_model} (force_cpu={force_cpu})")
    if use_amp and device_model != "cuda":
        print("[WARN] AMP requested but CUDA not available; disabling AMP.")
        use_amp = False

    if sampler == "botorch":
        try:
            from optuna.integration import BoTorchSampler
            sampler_obj = BoTorchSampler()
        except Exception:
            print("[WARN] BoTorch not available; falling back to TPE.")
            sampler_obj = TPESampler(multivariate=True)
    else:
        sampler_obj = TPESampler(multivariate=True)
    pruner = MedianPruner(n_warmup_steps=1)
    show_bar = not disable_progress_in_notebook

    global_cfg = dict(
        output_dir=DEFAULT_OUTPUT_DIR,
        et_rch_csv_path=data_path("ET_RCH_VARUNA_ML.csv"),
        ztop_extn_csv_path=data_path("Z_top_extn_depth.csv"),
        river_csv_path=data_path("varuna_points_latlong.csv"),
        ic_tif_path=data_path(os.path.join("kriging_results", "kriging_2022-05-25.tif")),
        device_model=device_model,
        use_film=use_film
    )

    objective = objective_builder(global_cfg,
                                  plot_each_trial=plot_per_trial,
                                  times_to_plot=times_to_plot,
                                  eval_chunk_size=eval_chunk_size,
                                  use_amp=use_amp,
                                  offload_after_trial=offload_after_trial,
                                  disable_grid_maps=disable_grid_maps,
                                  low_mem=low_mem,
                                  force_cpu=force_cpu)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler_obj,
        pruner=pruner,
        storage=storage,
        load_if_exists=bool(storage)
    )

    print(f"[INFO] Starting HPO: trials={n_trials} main_iters={FIXED_MAIN_NUM_ITERS} "
          f"adam_iters={FIXED_ADAM_ITERS} anchor={FIXED_HYBRID_N_ANCHOR}")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_bar)

    best = study.best_trial
    print("\n[BEST TRIAL]")
    print(f"  Trial #{best.number} KGE={best.value:.6f}")
    for k,v in best.params.items():
        print(f"    {k}: {v}")

    summary = dict(
        best_value_kge=best.value,
        best_params=best.params,
        trial_number=best.number,
        datetime=datetime.utcnow().isoformat(),
        fixed=dict(
            adam_iters=FIXED_ADAM_ITERS,
            hybrid_n_anchor=FIXED_HYBRID_N_ANCHOR,
            num_iters=FIXED_MAIN_NUM_ITERS,
            train_fraction=train_fraction,
            split_seed=fixed_split_seed,
            global_seed=global_seed
        ),
        settings=dict(
            eval_chunk_size=eval_chunk_size,
            use_amp=use_amp,
            offload_after_trial=offload_after_trial,
            disable_grid_maps=disable_grid_maps,
            low_mem=low_mem
        )
    )
    with open(os.path.join(DEFAULT_OUTPUT_DIR,"best_trial_summary.json"),"w") as f:
        json.dump(summary,f,indent=2)

    def _save_df(df: pd.DataFrame, path: str):
        try:
            df.to_parquet(path + ".parquet", index=False)
        except Exception:
            df.to_csv(path + ".csv", index=False)

    _save_df(GLOBAL_GWT_TRAIN, os.path.join(DEFAULT_OUTPUT_DIR,"GLOBAL_gwt_train"))
    _save_df(GLOBAL_GWT_TEST , os.path.join(DEFAULT_OUTPUT_DIR,"GLOBAL_gwt_test"))
    _save_df(stage_s         , os.path.join(DEFAULT_OUTPUT_DIR,"GLOBAL_stage_s"))
    _save_df(well_s          , os.path.join(DEFAULT_OUTPUT_DIR,"GLOBAL_well_s"))

    records=[]
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            tr = t.user_attrs.get("train_metrics", {})
            va = t.user_attrs.get("val_metrics", {})
            rec = dict(
                trial=t.number,
                state=str(t.state),
                kge_val=va.get("kge"), rmse_val=va.get("rmse"),
                mae_val=va.get("mae"), r2_val=va.get("r2"),
                kge_train=tr.get("kge"), rmse_train=tr.get("rmse"),
                mae_train=tr.get("mae"), r2_train=tr.get("r2")
            )
            rec.update(t.params)
            records.append(rec)
        else:
            rec=dict(trial=t.number,state=str(t.state))
            rec.update(t.params)
            records.append(rec)
    if records:
        pd.DataFrame(records).to_csv(
            os.path.join(DEFAULT_OUTPUT_DIR,"all_trials_metrics.csv"), index=False
        )

    retrain_results=None
    if retrain:
        bp=best.params
        best_train_params=dict(
            mlp_hidden=bp["mlp_hidden"],
            mlp_depth= bp["mlp_depth"],
            lambda_smooth=bp["lambda_smooth"],
            lambda_link=  bp["lambda_link"],
            phys_weight=  bp["phys_weight"],
            lambda_latent_var= bp["lambda_latent_var"],
            lambda_latent_rough= bp["lambda_latent_rough"],
            lambda_latent_rough_S=bp["lambda_latent_rough_S"],
            lambda_mlp_l2=0.0,
            lambda_grad_raw=   bp["lambda_grad_raw"],
            lambda_grad_t_raw= bp["lambda_grad_t_raw"],
            lambda_h_tt=       bp["lambda_h_tt"],
            temporal_grad_clip=  bp.get("temporal_grad_clip",0.0),
            lambda_temporal_clip=bp.get("lambda_temporal_clip",0.0),
            target_latent_var=bp.get("target_latent_var",0.05),
            target_s_var=bp["target_s_var"],
            lambda_s_var_shortfall=bp["lambda_s_var_shortfall"],
            temporal_kernel_scale=bp["temporal_kernel_scale"],
            ratio_min=0.05,
            ratio_max=20.0,
            layers=bp["layers"],
            units= bp["units"],
            use_fourier_head=True,
            lr=bp["lr"],
            gwt_train_frac=1.0,
            cluster_bins=4,
            oversample_factor=bp["oversample_factor"],
            use_film=use_film,
        )
        long_cfg=dict(
            warmup_iters=1,
            num_iters=FIXED_MAIN_NUM_ITERS,
            warmup_adam_iters=min(2000, int(bp.get("warmup_adam_iters",1600)*1.1)),
            num_domain=min(20000, int(bp.get("num_domain",12000)*1.3)),
            save_dir=os.path.join(DEFAULT_OUTPUT_DIR,"best_retrain"),
            et_rch_csv_path=data_path("ET_RCH_VARUNA_ML.csv"),
            ztop_extn_csv_path=data_path("Z_top_extn_depth.csv"),
            river_csv_path=data_path("varuna_points_latlong.csv"),
            device_model=device_model
        )
        retrain_results = retrain_best(best_train_params, long_cfg,
                                       long_cfg["save_dir"],
                                       times_to_plot=times_to_plot,
                                       eval_chunk_size=eval_chunk_size,
                                       use_amp=use_amp,
                                       offload_after_trial=offload_after_trial,
                                       disable_grid_maps=disable_grid_maps,
                                       force_cpu=force_cpu)

    return study, retrain_results

# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================
def load_and_preprocess_data(data_dir: Optional[str] = None,
                             stage_csv: str = "Observed_stage.csv",
                             gwt_csv: str = "Observed_GWT.csv",
                             well_csv: str = "Well_extraction_child.csv",
                             well_q_threshold: float = -150.0):
    """
    Load and scale the observation datasets, and make them available to the
    training / optimisation functions of this module.

    Parameters
    ----------
    data_dir : directory containing the input files (default: DATA_DIR, i.e.
               $PINN_DATA_DIR or ./data). Also used for the auxiliary files
               (ET/RCH, aquifer geometry, river polyline, IC raster).
    stage_csv, gwt_csv, well_csv : file names inside data_dir.
    well_q_threshold : abstraction records with Q below this value are dropped.

    Returns
    -------
    stage_s, gwt_s, well_s, stats_df
    """
    global DATA_DIR, stage, gwt, well_demand, stage_s, gwt_s, well_s, bundle, stats_df, DERIV_SCALES
    if data_dir is not None:
        DATA_DIR = data_dir

    print(f"[INFO] Loading data from '{DATA_DIR}' ...")
    stage = pd.read_csv(data_path(stage_csv))
    gwt = pd.read_csv(data_path(gwt_csv))
    well_demand = pd.read_csv(data_path(well_csv))

    # --- Clean and Parse ---
    stage = drop_unnamed_and_empty(stage)
    gwt = drop_unnamed_and_empty(gwt)
    well_demand = drop_unnamed_and_empty(well_demand)

    stage = ensure_datetime(stage, "Date")
    gwt = ensure_datetime(gwt, "Date")
    well_demand = ensure_datetime(well_demand, "Date")

    # Drop empty
    stage = drop_unnamed_and_empty(stage)
    gwt = drop_unnamed_and_empty(gwt)
    well_demand = drop_unnamed_and_empty(well_demand)

    # --- Scaling (Head Mid-Range) ---
    print("[INFO] Running preprocessing/scaling...")
    stage_s, gwt_s, well_s, bundle, stats_df = fit_transform_with_midrange_head(
        stage, gwt, well_demand,
        t_col="Date",
        well_q_threshold=well_q_threshold
    )
    print("[INFO] Preprocessing done.")
    print("Stats Head:\n", stats_df.head())

    # --- Load Derivative Scales ---
    DERIV_SCALES = {}
    if os.path.exists(data_path("derivative_scaling_factors.csv")):
        try:
            ds_df = pd.read_csv(data_path("derivative_scaling_factors.csv"))
            # Handle both naming conventions
            if "parameter" in ds_df.columns and "scale_factor" in ds_df.columns:
                DERIV_SCALES = dict(zip(ds_df["parameter"], ds_df["scale_factor"]))
                print(f"[INFO] Loaded {len(DERIV_SCALES)} derivative scales.")
            elif "derivative" in ds_df.columns and "w_max" in ds_df.columns:
                DERIV_SCALES = dict(zip(ds_df["derivative"], ds_df["w_max"]))
                print(f"[INFO] Loaded {len(DERIV_SCALES)} derivative scales.")
            else:
                print("[WARN] derivative_scaling_factors.csv missing required columns.")
        except Exception as e:
            print(f"[WARN] Failed to load derivative scales: {e}")
    else:
        print("[WARN] derivative_scaling_factors.csv not found. Using defaults.")

    return stage_s, gwt_s, well_s, stats_df

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--study-name", type=str, default="pinn_hpo_unified")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_study.db")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--split-type", type=str, default="random", choices=["random", "temporal", "temporal_lowdata"])
    parser.add_argument("--use-film", action="store_true")
    parser.add_argument("--sampler", choices=["tpe","botorch"], default="tpe")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--fixed-split-seed", type=int, default=1234)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--times", type=str, default="0.0,0.5,1.0")
    parser.add_argument("--eval-chunk-size", type=int, default=4096)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--offload-after-trial", action="store_true")
    parser.add_argument("--no-grid-plots", action="store_true")
    parser.add_argument("--low-mem", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--no-plot-per-trial", action="store_true")
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Directory with the input files (see data/README.md)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for study outputs (default: DEFAULT_OUTPUT_DIR)")
    args,_ = parser.parse_known_args()

    load_and_preprocess_data(data_dir=args.data_dir)

    times = [float(x) for x in args.times.split(",") if x.strip()]

    study, _ = optimize_pinn(
        n_trials=args.n_trials,
        retrain=args.retrain,
        sampler=args.sampler,
        study_name=args.study_name,
        storage=args.storage,
        split_type=args.split_type,
        use_film=args.use_film,
        disable_progress_in_notebook=not args.progress,
        fixed_split_seed=args.fixed_split_seed,
        train_fraction=args.train_frac,
        plot_per_trial=not args.no_plot_per_trial,
        times_to_plot=times,
        eval_chunk_size=args.eval_chunk_size,
        use_amp=args.amp,
        offload_after_trial=args.offload_after_trial,
        disable_grid_maps=args.no_grid_plots,
        low_mem=args.low_mem,
        force_cpu=args.force_cpu,
        global_seed=args.global_seed,
        output_dir=args.output_dir
    )
    print(f"\n[SUMMARY] Best validation KGE: {study.best_trial.value:.6f}")

if __name__ == "__main__":
    import sys
    try:
        pass # _require_globals()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    main_cli()
