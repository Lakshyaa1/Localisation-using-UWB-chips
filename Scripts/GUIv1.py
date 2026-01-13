#!/usr/bin/env python3

import serial
import re
import time
import math
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ============================================================
# MODE SWITCH
# ============================================================
FAKE_DISTANCES = False   # <-- SET False WHEN USING REAL UWB

# ============================================================
# SERIAL CONFIG (USED ONLY IF FAKE_DISTANCES = False)
# ============================================================
PORT = "/dev/ttyACM0"
BAUD = 115200
SERIAL_TIMEOUT = 0.2

# ============================================================
# ANCHOR POSITIONS (cm)
# ============================================================
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

MIN_ANCHORS = 3
MIN_DISTANCE = 10
MAX_DISTANCE = 500

PATTERN = re.compile(
    r"\[mac_address=0x([0-9a-fA-F]+),.*?distance\[cm\]=(-?\d+)\]"
)

# ============================================================
# TRILATERATION (UNCHANGED – SAME AS REAL)
# ============================================================
def trilaterate_least_squares(anchor_pos, distances):
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

def compute_rmse(x, y, anchor_pos, distances):
    err = 0.0
    for mac, (ax, ay) in anchor_pos.items():
        d_est = math.hypot(x - ax, y - ay)
        err += (d_est - distances[mac]) ** 2
    return math.sqrt(err / len(anchor_pos))

# ============================================================
# FAKE UWB DISTANCE GENERATOR (MEASUREMENTS ONLY)
# ============================================================
def generate_fake_uwb_measurements(t, anchors):
    """
    Generates UWB-like distance measurements.
    The solver NEVER sees the true position.
    """

    # Hidden true position (only for distance generation)
    true_x = 150 + 80 * math.cos(t)
    true_y = 150 + 80 * math.sin(t)

    distances = {}
    for mac, (ax, ay) in anchors.items():
        d = math.hypot(true_x - ax, true_y - ay)
        d += np.random.normal(0, 3.0)  # ~3 cm noise
        if MIN_DISTANCE <= d <= MAX_DISTANCE:
            distances[mac] = d

    return distances

# ============================================================
# GUI SETUP
# ============================================================
plt.ion()
fig, ax = plt.subplots(figsize=(7, 7))
plt.show(block=False)

ax.set_xlim(-50, 350)
ax.set_ylim(-50, 350)
ax.set_aspect("equal")
ax.set_title("UWB Real-Time Trilateration")
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.grid(True)

# Plot anchors
for mac, (x, y) in ANCHORS.items():
    ax.plot(x, y, "bo")
    ax.text(x + 5, y + 5, hex(mac), fontsize=9)

circle_artists = {}
for mac in ANCHORS:
    c = plt.Circle((0, 0), 0, fill=False, alpha=0.3)
    ax.add_patch(c)
    circle_artists[mac] = c

tag_point, = ax.plot([], [], "ro", markersize=8)
text_info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

# ============================================================
# MAIN LOOP
# ============================================================
def main():

    ser = None
    if not FAKE_DISTANCES:
        ser = serial.Serial(PORT, BAUD, timeout=SERIAL_TIMEOUT)
        time.sleep(1)
        ser.reset_input_buffer()
        ser.write(b"stop\r\n")
        time.sleep(0.2)
        ser.write(b"initf -multi -addr=1 -paddr=[0,2,3,4,6,7,9,10]\r\n")
        time.sleep(0.2)

    distances = {}
    t = 0.0

    print("UWB GUI running")
    print("MODE:", "FAKE DISTANCES" if FAKE_DISTANCES else "REAL UWB")

    try:
        while True:

            # ------------------------------------------------
            # DISTANCE INPUT
            # ------------------------------------------------
            if FAKE_DISTANCES:
                distances = generate_fake_uwb_measurements(t, ANCHORS)
                t += 0.05

            else:
                line = ser.readline().decode(errors="ignore").strip()
                if not line:
                    plt.pause(0.001)
                    continue

                m = PATTERN.search(line)
                if not m:
                    plt.pause(0.001)
                    continue

                mac = int(m.group(1), 16)
                dist = float(m.group(2))

                if mac not in ANCHORS:
                    continue
                if dist < MIN_DISTANCE or dist > MAX_DISTANCE:
                    continue

                distances[mac] = dist

            # ------------------------------------------------
            # GUI UPDATE
            # ------------------------------------------------
            for mac, d in distances.items():
                cx, cy = ANCHORS[mac]
                circle_artists[mac].center = (cx, cy)
                circle_artists[mac].radius = d
                circle_artists[mac].set_visible(True)

            if len(distances) < MIN_ANCHORS:
                plt.pause(0.01)
                continue

            sub_anchors = {m: ANCHORS[m] for m in distances}
            x, y = trilaterate_least_squares(sub_anchors, distances)
            rmse = compute_rmse(x, y, sub_anchors, distances)

            tag_point.set_data([x], [y])

            text_info.set_text(
                f"MODE: {'FAKE' if FAKE_DISTANCES else 'REAL'}\n"
                f"Anchors: {len(distances)}\n"
                f"RMSE: {rmse:.2f} cm\n"
                f"X={x:.1f}, Y={y:.1f}"
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
