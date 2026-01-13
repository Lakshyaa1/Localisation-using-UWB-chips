#!/usr/bin/env python3

import serial
import re
import time
import math
import json
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from itertools import combinations
from statistics import median
import os

# ============================================================
# DATA MODE: "fake" | "serial" | "json"
# ============================================================
DATA_MODE = "json"

JSON_FILE = "/home/lakshya/Localisation-using-UWB-chips/Localisation_Data/(100,200)_C_08_01_2026.json"

# ============================================================
# SERIAL CONFIG (used only in serial mode)
# ============================================================
PORT = "/dev/ttyACM0"
BAUD = 115200
SERIAL_TIMEOUT = 0.2

# ============================================================
# PARAMETERS
# ============================================================
MIN_ANCHORS = 3
MIN_DISTANCE = 10
MAX_DISTANCE = 500

MAX_RESIDUAL = 10.0   # cm
MAX_SPREAD = 15.0     # cm

PATTERN = re.compile(
    r"\[mac_address=0x([0-9a-fA-F]+),.*?distance\[cm\]=(-?\d+)\]"
)

# ============================================================
# LOAD JSON DATA IF NEEDED
# ============================================================
json_frames = []
json_index = 0

if DATA_MODE == "json":
    if not os.path.exists(JSON_FILE):
        raise FileNotFoundError(JSON_FILE)

    with open(JSON_FILE, "r") as f:
        data = json.load(f)

    ANCHORS = {
        int(mac, 16): tuple(pos)
        for mac, pos in data["anchor_positions"].items()
    }
    if "measurements" in data:
        json_frames = data["measurements"]
    elif "position_data" in data:
        json_frames = data["position_data"]
    else:
        raise KeyError("No measurements or position_data found in JSON")



else:
    # Hardcoded anchors for fake / serial
    ANCHORS = {
        0x0000: (0.0,   0.0),
        0x0002: (0.0, 150.0),
        0x0003: (0.0, 300.0),
        0x0004: (150.0, 300.0),
        0x0006: (300.0, 300.0),
        0x0007: (300.0, 150.0),
        0x0009: (300.0,   0.0),
        0x000A: (150.0,   0.0),
    }

# ============================================================
# TRILATERATION (3 ANCHORS)
# ============================================================
def trilaterate_3(a1, a2, a3, d1, d2, d3):
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = a3

    A = np.array([
        [2*(x2-x1), 2*(y2-y1)],
        [2*(x3-x1), 2*(y3-y1)]
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

def residual_ok(x, y, anchors, distances):
    for mac, (ax, ay) in anchors.items():
        d_est = math.hypot(x-ax, y-ay)
        if abs(d_est - distances[mac]) > MAX_RESIDUAL:
            return False
    return True

# ============================================================
# FAKE DISTANCE GENERATOR (DISTANCES ONLY)
# ============================================================
def generate_fake_distances(t, anchors):
    true_x = 150 + 80 * math.cos(t)
    true_y = 150 + 80 * math.sin(t)

    distances = {}
    for mac, (ax, ay) in anchors.items():
        d = math.hypot(true_x-ax, true_y-ay)
        d += np.random.normal(0, 3.0)
        if MIN_DISTANCE <= d <= MAX_DISTANCE:
            distances[mac] = d

    return distances

# ============================================================
# GUI SETUP
# ============================================================
plt.ion()
fig, ax = plt.subplots(figsize=(7,7))
plt.show(block=False)

ax.set_aspect("equal")
ax.grid(True)
ax.set_title("UWB Robust Trilateration (Static Tag)")
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")

xs = [p[0] for p in ANCHORS.values()]
ys = [p[1] for p in ANCHORS.values()]
ax.set_xlim(min(xs)-50, max(xs)+50)
ax.set_ylim(min(ys)-50, max(ys)+50)

for mac, (x, y) in ANCHORS.items():
    ax.plot(x, y, "bo")
    ax.text(x+5, y+5, hex(mac), fontsize=9)

circle_artists = {}
for mac in ANCHORS:
    c = plt.Circle((0,0), 0, fill=False, alpha=0.3)
    ax.add_patch(c)
    circle_artists[mac] = c

tag_point, = ax.plot([], [], "ro", markersize=8)
text_info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

# ============================================================
# MAIN LOOP
# ============================================================
def main():
    global json_index
    ser = None

    if DATA_MODE == "serial":
        ser = serial.Serial(PORT, BAUD, timeout=SERIAL_TIMEOUT)
        time.sleep(1)
        ser.reset_input_buffer()
        ser.write(b"stop\r\n")
        time.sleep(0.2)
        ser.write(b"initf -multi -addr=1 -paddr=[0,2,3,4,6,7,9,10]\r\n")
        time.sleep(0.2)

    t = 0.0
    distances = {}

    print("UWB GUI running")
    print("DATA MODE:", DATA_MODE.upper())

    try:
        while True:

            # ------------------------------------------------
            # INPUT SOURCE
            # ------------------------------------------------
            if DATA_MODE == "fake":
                distances = generate_fake_distances(t, ANCHORS)
                t += 0.05

            elif DATA_MODE == "json":
                frame = json_frames[json_index]
                json_index = (json_index + 1) % len(json_frames)

                distances = {
                    int(mac, 16): float(d)
                    for mac, d in frame["distances"].items()
                    if int(mac, 16) in ANCHORS
                }

                time.sleep(0.05)

            else:  # serial
                line = ser.readline().decode(errors="ignore").strip()
                if not line:
                    plt.pause(0.001)
                    continue

                m = PATTERN.search(line)
                if not m:
                    continue

                mac = int(m.group(1), 16)
                dist = float(m.group(2))

                if mac not in ANCHORS:
                    continue
                if dist < MIN_DISTANCE or dist > MAX_DISTANCE:
                    continue

                distances[mac] = dist

            # ------------------------------------------------
            # DRAW CIRCLES
            # ------------------------------------------------
            for mac, d in distances.items():
                cx, cy = ANCHORS[mac]
                circle_artists[mac].center = (cx, cy)
                circle_artists[mac].radius = d
                circle_artists[mac].set_visible(True)

            if len(distances) < MIN_ANCHORS:
                tag_point.set_data([], [])
                plt.pause(0.01)
                continue

            # ------------------------------------------------
            # COMBINATION TRILATERATION
            # ------------------------------------------------
            solutions = []

            for combo in combinations(distances.keys(), 3):
                m1, m2, m3 = combo
                sol = trilaterate_3(
                    ANCHORS[m1], ANCHORS[m2], ANCHORS[m3],
                    distances[m1], distances[m2], distances[m3]
                )

                if sol is None:
                    continue

                x, y = sol

                if residual_ok(
                    x, y,
                    {m1: ANCHORS[m1], m2: ANCHORS[m2], m3: ANCHORS[m3]},
                    {m1: distances[m1], m2: distances[m2], m3: distances[m3]}
                ):
                    solutions.append((x, y))

            if not solutions:
                tag_point.set_data([], [])
                plt.pause(0.01)
                continue

            xs_sol = [p[0] for p in solutions]
            ys_sol = [p[1] for p in solutions]

            if (max(xs_sol)-min(xs_sol)) > MAX_SPREAD or (max(ys_sol)-min(ys_sol)) > MAX_SPREAD:
                tag_point.set_data([], [])
                plt.pause(0.01)
                continue

            x_final = median(xs_sol)
            y_final = median(ys_sol)

            tag_point.set_data([x_final], [y_final])

            text_info.set_text(
                f"MODE: {DATA_MODE.upper()}\n"
                f"Anchors: {len(distances)}\n"
                f"Triplets: {len(solutions)}\n"
                f"X={x_final:.1f}, Y={y_final:.1f}"
            )

            plt.pause(0.01)

    except KeyboardInterrupt:
        print("\nStopped")

    finally:
        if ser:
            ser.write(b"stop\r\n")
            ser.close()
        plt.ioff()
        plt.show()

# ============================================================
if __name__ == "__main__":
    main()
