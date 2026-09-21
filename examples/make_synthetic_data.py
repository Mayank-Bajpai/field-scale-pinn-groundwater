#!/usr/bin/env python
"""
Generate a small SYNTHETIC dataset with the same file names and column schema
as the (restricted) field data used by pinn_unified.py.

The values are artificial: a smooth regional head gradient with a seasonal
cycle, a straight "river", random abstraction wells and gridded recharge/ET.
They are NOT physically meaningful and do not represent any real site. Their
only purpose is to let users install, run and test the workflow end to end.

Usage
-----
    python examples/make_synthetic_data.py --out data
"""
import argparse
import os

import numpy as np
import pandas as pd

LX, LY = 20_000.0, 10_000.0          # synthetic domain size (m), local coordinates
X0, Y0 = 0.0, 0.0


def head_field(x, y, t_days):
    """Smooth synthetic head (m): regional gradient toward the river + seasonal cycle."""
    dist_to_river = np.abs(y - (Y0 + 0.5 * LY))
    base = 60.0 + 4.0e-4 * dist_to_river + 1.5e-4 * (x - X0)
    season = 1.5 * np.sin(2 * np.pi * t_days / 365.25)
    return base + season


def main(out, seed=0, n_wells=20, n_dates=36):
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    dates = pd.date_range("2022-05-25", periods=n_dates, freq="MS")
    t = (dates - dates[0]).days.to_numpy(float)

    # Observed_GWT.csv : ObsWell, X, Y, Date, GWT
    wx = rng.uniform(X0 + 500, X0 + LX - 500, n_wells)
    wy = rng.uniform(Y0 + 500, Y0 + LY - 500, n_wells)
    rows = []
    for i in range(n_wells):
        keep = rng.random(n_dates) > 0.2                      # irregular records
        for d, td in zip(dates[keep], t[keep]):
            h = head_field(wx[i], wy[i], td) + rng.normal(0, 0.2)
            rows.append((f"SW{i+1:02d}", wx[i], wy[i], d.strftime("%Y-%m-%d"), round(h, 3)))
    pd.DataFrame(rows, columns=["ObsWell", "X", "Y", "Date", "GWT"]).to_csv(
        os.path.join(out, "Observed_GWT.csv"), index=False)

    # Observed_stage.csv : ObsPoint, Date, X, Y, Stage_masl  (points along the river)
    sx = np.linspace(X0 + 1000, X0 + LX - 1000, 8)
    sy = np.full_like(sx, Y0 + 0.5 * LY)
    rows = []
    for j in range(len(sx)):
        for d, td in zip(dates[::2], t[::2]):
            stage = head_field(sx[j], sy[j], td) - 0.5 + rng.normal(0, 0.05)
            rows.append((f"SP{j+1}", d.strftime("%Y-%m-%d"), sx[j], sy[j], round(stage, 3)))
    pd.DataFrame(rows, columns=["ObsPoint", "Date", "X", "Y", "Stage_masl"]).to_csv(
        os.path.join(out, "Observed_stage.csv"), index=False)

    # Well_extraction_child.csv : <index>, Wellid, X, Y, Date, Q_md  (negative = abstraction)
    n_pump = 40
    px = rng.uniform(X0, X0 + LX, n_pump)
    py = rng.uniform(Y0, Y0 + LY, n_pump)
    rows = []
    for k in range(n_pump):
        for d in dates:
            rows.append((k + 1, px[k], py[k], d.strftime("%Y-%m-%d"), round(-rng.uniform(5, 120), 2)))
    pd.DataFrame(rows, columns=["Wellid", "X", "Y", "Date", "Q_md"]).to_csv(
        os.path.join(out, "Well_extraction_child.csv"), index=True)

    # ET_RCH_VARUNA_ML.csv : sno, X, Y, Date, RCH, ET  (gridded, monthly)
    gx, gy = np.meshgrid(np.linspace(X0, X0 + LX, 10), np.linspace(Y0, Y0 + LY, 5))
    rows, sno = [], 1
    for d, td in zip(dates, t):
        wet = max(0.0, np.sin(2 * np.pi * td / 365.25))
        for x, y in zip(gx.ravel(), gy.ravel()):
            rows.append((sno, x, y, d.strftime("%Y-%m-%d"),
                         round(2.0 * wet + rng.uniform(0, 0.2), 4), round(1.0 + rng.uniform(0, 0.5), 4)))
            sno += 1
    pd.DataFrame(rows, columns=["sno", "X", "Y", "Date", "RCH", "ET"]).to_csv(
        os.path.join(out, "ET_RCH_VARUNA_ML.csv"), index=False)

    # Z_top_extn_depth.csv : OBJECTID, ID, I, J, K, X, Y, Ztop, Dz
    zx, zy = np.meshgrid(np.linspace(X0, X0 + LX, 20), np.linspace(Y0, Y0 + LY, 10))
    rows = []
    for n, (x, y) in enumerate(zip(zx.ravel(), zy.ravel()), start=1):
        i, j = divmod(n - 1, 20)
        rows.append((n, n, i + 1, j + 1, 1, x, y, round(75 + rng.normal(0, 1), 2), round(60 + rng.normal(0, 3), 2)))
    pd.DataFrame(rows, columns=["OBJECTID", "ID", "I", "J", "K", "X", "Y", "Ztop", "Dz"]).to_csv(
        os.path.join(out, "Z_top_extn_depth.csv"), index=False)

    # varuna_points_latlong.csv : FID, Longitude, Latitude
    # NOTE: despite the column names, the code expects PROJECTED coordinates in the
    # same system as X/Y above.
    rx = np.linspace(X0 + 100, X0 + LX - 100, 100)
    ry = Y0 + 0.5 * LY + 200 * np.sin(rx / 3000.0)
    pd.DataFrame({"FID": np.arange(len(rx)), "Longitude": rx, "Latitude": ry}).to_csv(
        os.path.join(out, "varuna_points_latlong.csv"), index=False)

    print(f"Synthetic dataset written to '{out}/'")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", help="output directory (default: data)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.out, seed=a.seed)
