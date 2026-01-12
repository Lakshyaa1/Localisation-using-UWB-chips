#!/usr/bin/env python3

import json
import math
import numpy as np
import os
from itertools import combinations
from statistics import mean, median

# ============================================================
# CONFIG
# ============================================================
INPUT_JSON = "(100,100)_C_08_01_2026.json"
OUTPUT_JSON = "uwb_combination_trilateration.json"

MIN_ANCHORS = 3
MAX_RESIDUAL = 30.0   # cm, per-anchor residual check
MAX_SPREAD = 50.0     # cm, reject wildly spread solutions

# ============================================================
# SAFETY
# ============================================================
if not os.path.exists(INPUT_JSON):
    raise FileNotFoundError(INPUT_JSON)

# ============================================================
# PURE TRILATERATION (3 ANCHORS)
# ============================================================
def trilaterate_3(a1, a2, a3, d1, d2, d3):
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = a3

    A = np.array([
        [2*(x2 - x1), 2*(y2 - y1)],
        [2*(x3 - x1), 2*(y3 - y1)]
    ])

    b = np.array([
        d1**2 - d2**2 + x2**2 + y2**2 - x1**2 - y1**2,
        d1**2 - d3**2 + x3**2 + y3**2 - x1**2 - y1**2
    ])

    try:
        sol = np.linalg.solve(A, b)
        return float(sol[0]), float(sol[1])
    except np.linalg.LinAlgError:
        return None

# ============================================================
# RESIDUAL CHECK
# ============================================================
def residual_ok(x, y, anchors, distances):
    for mac, (ax, ay) in anchors.items():
        d_est = math.hypot(x - ax, y - ay)
        if abs(d_est - distances[mac]) > MAX_RESIDUAL:
            return False
    return True

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

    xs, ys = [], []
    output = []

    print("\nUWB POSITIONING (COMBINATION TRILATERATION)")
    print("------------------------------------------------\n")

    for frame in measurements:
        distances = {
            int(mac, 16): float(d)
            for mac, d in frame["distances"].items()
            if int(mac, 16) in anchors
        }

        if len(distances) < MIN_ANCHORS:
            continue

        solutions = []

        # ---- all 3-anchor combinations ----
        for combo in combinations(distances.keys(), 3):
            m1, m2, m3 = combo

            sol = trilaterate_3(
                anchors[m1], anchors[m2], anchors[m3],
                distances[m1], distances[m2], distances[m3]
            )

            if sol is None:
                continue

            x, y = sol

            # residual validation only on this triplet
            triplet_anchors = {
                m1: anchors[m1],
                m2: anchors[m2],
                m3: anchors[m3]
            }
            triplet_distances = {
                m1: distances[m1],
                m2: distances[m2],
                m3: distances[m3]
            }

            if residual_ok(x, y, triplet_anchors, triplet_distances):
                solutions.append((x, y))

        if len(solutions) == 0:
            continue

        xs_frame = [p[0] for p in solutions]
        ys_frame = [p[1] for p in solutions]

        # ---- reject wildly inconsistent clusters ----
        if (max(xs_frame) - min(xs_frame)) > MAX_SPREAD:
            continue
        if (max(ys_frame) - min(ys_frame)) > MAX_SPREAD:
            continue

        # ---- final fused position ----
        x_final = median(xs_frame)
        y_final = median(ys_frame)

        xs.append(x_final)
        ys.append(y_final)

        print(
            f"{frame['sample']:05d} | {frame['timestamp']} | "
            f"X={x_final:7.2f} cm | Y={y_final:7.2f} cm | "
            f"Triplets={len(solutions)}"
        )

        output.append({
            "sample": frame["sample"],
            "timestamp": frame["timestamp"],
            "x_cm": round(x_final, 2),
            "y_cm": round(y_final, 2),
            "triplets_used": len(solutions)
        })

    # ========================================================
    # FINAL STATS
    # ========================================================
    print("\n------------------------------------------------")
    print("FINAL STATISTICS")
    print("------------------------------------------------")

    if xs:
        print(f"Samples used : {len(xs)}")
        print(f"Mean position   : X = {mean(xs):.2f} cm , Y = {mean(ys):.2f} cm")
        print(f"Median position : X = {median(xs):.2f} cm , Y = {median(ys):.2f} cm")
    else:
        print("No valid samples.")

    print("------------------------------------------------")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved output to {OUTPUT_JSON}")

# ============================================================
if __name__ == "__main__":
    main()
