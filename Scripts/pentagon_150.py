#!/usr/bin/env python3

import json
import math
import numpy as np
from scipy.optimize import least_squares
from datetime import datetime
import os
from statistics import mean, median

# ============================================================
# CONFIG
# ============================================================
INPUT_JSON = "(100,100)_C_08_01_2026.json"
OUTPUT_JSON = "uwb_reprocessed_nlls.json"

MIN_ANCHORS = 3
MAX_RMSE = 50.0

# ============================================================
# NLLS RESIDUAL FUNCTION (WITH WEIGHTS)
# ============================================================
def residuals_nlls(pos, anchors, distances, weights):
    x, y = pos
    res = []

    for mac, (ax, ay) in anchors.items():
        d_est = math.hypot(x - ax, y - ay)
        r = d_est - distances[mac]
        res.append(math.sqrt(weights[mac]) * r)

    return np.array(res)

# ============================================================
# WEIGHTED NLLS (IRLS)
# ============================================================
def trilaterate_nlls_weighted(anchor_pos, distances, x0, iters=3):
    weights = {mac: 1.0 for mac in anchor_pos}
    x_est, y_est = x0

    for _ in range(iters):
        result = least_squares(
            residuals_nlls,
            x0=[x_est, y_est],
            args=(anchor_pos, distances, weights),
            loss="soft_l1",
            f_scale=10.0
        )

        x_est, y_est = result.x

        # Update weights (same logic as your live code)
        for mac, (ax, ay) in anchor_pos.items():
            d_est = math.hypot(x_est - ax, y_est - ay)
            r = d_est - distances[mac]
            weights[mac] = 1.0 / (1e-3 + abs(r))

    return x_est, y_est, weights

# ============================================================
# RMSE
# ============================================================
def compute_rmse(x, y, anchors, distances):
    err = 0.0
    for mac, (ax, ay) in anchors.items():
        d_est = math.hypot(x - ax, y - ay)
        err += (d_est - distances[mac]) ** 2
    return math.sqrt(err / len(anchors))

# ============================================================
# MAIN (JSON REPLAY)
# ============================================================
def main():
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(INPUT_JSON)

    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    # Parse anchors
    ANCHORS = {
        int(mac, 16): tuple(pos)
        for mac, pos in data["anchor_positions"].items()
    }

    measurements = data["measurements"]

    last_x, last_y = None, None
    sample_count = 0
    output = []

    xs, ys = [], []

    print("\nUWB POSITIONING (JSON → NLLS + ADAPTIVE WEIGHTS)")
    print("------------------------------------------------\n")

    for frame in measurements:
        distances = {
            int(mac, 16): float(d)
            for mac, d in frame["distances"].items()
            if int(mac, 16) in ANCHORS
        }

        if len(distances) < MIN_ANCHORS:
            continue

        sub_anchors = {m: ANCHORS[m] for m in distances}

        # -------- Initial guess --------
        if last_x is None:
            x0, y0 = (150.0, 150.0)   # cold start
        else:
            x0, y0 = last_x, last_y   # warm start

        x, y, weights = trilaterate_nlls_weighted(
            sub_anchors, distances, x0=(x0, y0), iters=3
        )

        rmse = compute_rmse(x, y, sub_anchors, distances)
        if rmse > MAX_RMSE:
            last_x, last_y = None, None
            continue

        last_x, last_y = x, y
        sample_count += 1

        xs.append(x)
        ys.append(y)

        print(
            f"{sample_count:05d} | {frame['timestamp']} | "
            f"X={x:7.2f} cm | Y={y:7.2f} cm | "
            f"Anchors={len(sub_anchors)} | RMSE={rmse:5.2f}"
        )

        output.append({
            "sample": sample_count,
            "timestamp": frame["timestamp"],
            "x_cm": round(x, 2),
            "y_cm": round(y, 2),
            "rmse": round(rmse, 2),
            "anchors": len(sub_anchors),
            "distances": {hex(k): v for k, v in distances.items()},
            "weights": {hex(k): round(v, 3) for k, v in weights.items()}
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
        json.dump(output, f, indent=2)

    print(f"\nSaved reprocessed data to {OUTPUT_JSON}")

# ============================================================
if __name__ == "__main__":
    main()
