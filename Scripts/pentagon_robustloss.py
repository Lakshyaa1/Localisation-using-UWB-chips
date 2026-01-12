#!/usr/bin/env python3

import json
import math
import numpy as np
from scipy.optimize import least_squares
import os
from statistics import mean, median

# ============================================================
# CONFIG
# ============================================================
INPUT_JSON = "(100,100)_C_08_01_2026.json"
OUTPUT_JSON = "uwb_positions_nlls_robust.json"

MIN_ANCHORS = 3
MAX_RMSE = 50.0

# Robust loss parameters (cm)
ROBUST_LOSS = "soft_l1"
F_SCALE = 25.0

# ============================================================
# SAFETY CHECKS
# ============================================================
if not os.path.exists(INPUT_JSON):
    raise FileNotFoundError(f"{INPUT_JSON} not found")

if os.path.getsize(INPUT_JSON) == 0:
    raise RuntimeError(f"{INPUT_JSON} is empty")

# ============================================================
# LINEAR LEAST SQUARES (INITIAL GUESS)
# ============================================================
def trilaterate_linear_ls(anchor_pos, distances):
    macs = list(anchor_pos.keys())
    ref = macs[0]

    x1, y1 = anchor_pos[ref]
    d1 = distances[ref]

    A, b = [], []

    for mac in macs[1:]:
        xi, yi = anchor_pos[mac]
        di = distances[mac]

        A.append([2 * (xi - x1), 2 * (yi - y1)])
        b.append(d1**2 - di**2 + xi**2 + yi**2 - x1**2 - y1**2)

    A = np.array(A)
    b = np.array(b)

    pos, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return float(pos[0]), float(pos[1])

# ============================================================
# NLLS RESIDUALS (NO WEIGHTS)
# ============================================================
def residuals_nlls(pos, anchors, distances):
    x, y = pos
    return [
        math.hypot(x - ax, y - ay) - distances[mac]
        for mac, (ax, ay) in anchors.items()
    ]

# ============================================================
# RMSE (for reporting only)
# ============================================================
def compute_rmse(x, y, anchor_pos, distances):
    err = 0.0
    for mac, (ax, ay) in anchor_pos.items():
        d_est = math.hypot(x - ax, y - ay)
        err += (d_est - distances[mac]) ** 2
    return math.sqrt(err / len(anchor_pos))

# ============================================================
# MAIN
# ============================================================
def main():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    anchors = {
        int(mac, 16): tuple(pos)
        for mac, pos in data["anchor_positions"].items()
    }

    measurements = data["measurements"]

    last_x, last_y = None, None
    xs, ys = [], []
    results = []

    print("\nUWB POSITIONING (ROBUST NLLS – NO WEIGHTS)")
    print("------------------------------------------------\n")

    for frame in measurements:
        distances = {
            int(mac, 16): float(d)
            for mac, d in frame["distances"].items()
            if int(mac, 16) in anchors
        }

        if len(distances) < MIN_ANCHORS:
            continue

        sub_anchors = {m: anchors[m] for m in distances}

        # ---- Initial Guess ----
        if last_x is None:
            x0, y0 = trilaterate_linear_ls(sub_anchors, distances)
        else:
            x0, y0 = last_x, last_y

        # ---- ROBUST NLLS ----
        result = least_squares(
            residuals_nlls,
            x0=[x0, y0],
            args=(sub_anchors, distances),
            loss=ROBUST_LOSS,
            f_scale=F_SCALE
        )

        x, y = result.x
        rmse = compute_rmse(x, y, sub_anchors, distances)

        if rmse > MAX_RMSE:
            last_x, last_y = None, None
            continue

        last_x, last_y = x, y
        xs.append(x)
        ys.append(y)

        print(
            f"{frame['sample']:05d} | {frame['timestamp']} | "
            f"X={x:7.2f} cm | Y={y:7.2f} cm | "
            f"Anchors={len(sub_anchors)} | RMSE={rmse:5.2f}"
        )

        results.append({
            "sample": frame["sample"],
            "timestamp": frame["timestamp"],
            "x_cm": round(x, 2),
            "y_cm": round(y, 2),
            "rmse": round(rmse, 2)
        })

    # ========================================================
    # FINAL STATISTICS
    # ========================================================
    print("\n------------------------------------------------")
    print("FINAL STATISTICS")
    print("------------------------------------------------")

    if xs:
        print(f"Samples used : {len(xs)}")
        print(f"Mean position   : X = {mean(xs):.2f} cm , Y = {mean(ys):.2f} cm")
        print(f"Median position : X = {median(xs):.2f} cm , Y = {median(ys):.2f} cm")
    else:
        print("No valid samples available.")

    print("------------------------------------------------")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved output to {OUTPUT_JSON}")

# ============================================================
if __name__ == "__main__":
    main()
