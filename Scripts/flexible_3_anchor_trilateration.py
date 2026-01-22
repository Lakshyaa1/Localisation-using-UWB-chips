#!/usr/bin/env python3

import serial
import re
import time
import math
import numpy as np
from itertools import combinations
from statistics import median
from datetime import datetime

# ============================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================

# Serial port settings
PORT = "/dev/ttyACM0"
BAUD = 115200
SERIAL_TIMEOUT = 0.2

# ============================================================
# ANCHOR CONFIGURATION - MODIFY AS NEEDED
# ============================================================
# Format: MAC_ADDRESS: (X_position_cm, Y_position_cm)
# You can add/remove anchors as needed

# ANCHORS = {
#     # 4-anchor configuration (square)
#     0x0004: (0.0,0.0),    # Bottom-left
#     0x0009: (0.0,480.0),    # Bottom-right
#     0x000a: (300.0,420.0),  # Top-right
#     0x0006: (300.0,0.0),  # Top-left
# }

# Alternative 8-anchor configuration (uncomment to use)
ANCHORS = {
    0x000a: (0.0,0.0),    
    0x0004: (1400.0,0.0),    
    0x0009: (750.0,0.0), 
    0x0002: (1400.0,750.0), 
    0x0006: (750.0, 1500.0),
    0x0000: (1400.0, 1500.0),
    0x0007: (0.0,1500.0),
    0x0003: (0.0,750.0),
}

# Alternative 6-anchor configuration (uncomment to use)
# ANCHORS = {
#     0x0004: (0.0,0.0),    
#     0x0009: (0.0,480.0),    
#     0x000a: (300.0,420.0), 
#     0x0006: (300.0,0.0), 
#     0x0002: (0.0, 240.0),
#     0x0003: (300.0, 240.0),
# }

# ============================================================
# ALGORITHM PARAMETERS
# ============================================================
MIN_ANCHORS = 3          # Minimum anchors needed for trilateration
TRIPLET_SIZE = 3         # Always 3 for 2D trilateration

MIN_DISTANCE = 10        # cm - reject distances below this
MAX_DISTANCE = 1200      # cm - reject distances above this

MAX_RESIDUAL = 50.0      # cm - per-anchor error tolerance
MAX_SPREAD = 50.0        # cm - maximum cluster spread allowed

# ============================================================
# AUTO-GENERATE INITIALIZATION COMMAND
# ============================================================
def generate_init_command():
    """Automatically generate the UWB init command based on ANCHORS"""
    anchor_ids = sorted([mac & 0xFF for mac in ANCHORS.keys()])
    addr_list = ",".join(map(str, anchor_ids))
    return f"initf -multi -addr=1 -paddr=[{addr_list}]\r\n"

# ============================================================
# SERIAL PARSER
# ============================================================
PATTERN = re.compile(
    r"\[mac_address=0x([0-9a-fA-F]+),.*?distance\[cm\]=(-?\d+)\]"
)

# ============================================================
# PURE 3-ANCHOR TRILATERATION
# ============================================================
def trilaterate_3(a1, a2, a3, d1, d2, d3):
    """
    Solve for (x,y) position given 3 anchor positions and distances.
    
    Args:
        a1, a2, a3: Anchor positions as (x, y) tuples
        d1, d2, d3: Distances to each anchor in cm
    
    Returns:
        (x, y) tuple or None if unsolvable
    """
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
# RESIDUAL CHECK (PER TRIPLET)
# ============================================================
def residual_ok(x, y, anchors, distances):
    """
    Check if position (x,y) has acceptable error for all anchors.
    
    Args:
        x, y: Estimated position
        anchors: Dictionary of anchor positions for this triplet
        distances: Dictionary of measured distances for this triplet
    
    Returns:
        True if all residuals are within MAX_RESIDUAL
    """
    for mac, (ax, ay) in anchors.items():
        d_est = math.hypot(x - ax, y - ay)
        if abs(d_est - distances[mac]) > MAX_RESIDUAL:
            return False
    return True

# ============================================================
# MAIN
# ============================================================
def main():
    # Display configuration
    print("=" * 60)
    print("UWB POSITIONING - FLEXIBLE N-ANCHOR CONFIGURATION")
    print("=" * 60)
    print(f"\nConfigured Anchors: {len(ANCHORS)}")
    for mac, (x, y) in sorted(ANCHORS.items()):
        print(f"  0x{mac:04X}: ({x:6.1f}, {y:6.1f}) cm")
    
    num_combinations = math.comb(len(ANCHORS), TRIPLET_SIZE)
    print(f"\nPossible {TRIPLET_SIZE}-anchor combinations: {num_combinations}")
    print(f"Min anchors required: {MIN_ANCHORS}")
    print(f"Max residual: {MAX_RESIDUAL} cm")
    print(f"Max cluster spread: {MAX_SPREAD} cm")
    print("=" * 60)
    
    # Initialize serial connection
    print(f"\nOpening serial port: {PORT} @ {BAUD} baud")
    ser = serial.Serial(PORT, BAUD, timeout=SERIAL_TIMEOUT)
    time.sleep(1)
    ser.reset_input_buffer()

    # Auto-generate and send initialization command
    init_cmd = generate_init_command()
    print(f"\nInitializing UWB...")
    print(f"Command: {init_cmd.strip()}")
    
    ser.write(b"stop\r\n")
    time.sleep(0.3)
    ser.write(init_cmd.encode())
    time.sleep(0.5)

    print("\n" + "=" * 60)
    print("LIVE POSITIONING STARTED")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()

    distances = {}
    sample = 0

    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            m = PATTERN.search(line)
            if not m:
                continue

            mac = int(m.group(1), 16)
            dist = int(m.group(2))

            # Filter: only known anchors
            if mac not in ANCHORS:
                continue
            
            # Filter: distance range
            if dist < MIN_DISTANCE or dist > MAX_DISTANCE:
                continue

            distances[mac] = float(dist)

            # Need at least MIN_ANCHORS to proceed
            if len(distances) < MIN_ANCHORS:
                continue

            solutions = []

            # Generate all 3-anchor combinations
            for combo in combinations(distances.keys(), TRIPLET_SIZE):
                m1, m2, m3 = combo

                # Solve trilateration
                sol = trilaterate_3(
                    ANCHORS[m1], ANCHORS[m2], ANCHORS[m3],
                    distances[m1], distances[m2], distances[m3]
                )

                if sol is None:
                    continue

                x, y = sol

                # Build triplet-specific anchor and distance dicts
                triplet_anchors = {
                    m1: ANCHORS[m1],
                    m2: ANCHORS[m2],
                    m3: ANCHORS[m3]
                }

                triplet_distances = {
                    m1: distances[m1],
                    m2: distances[m2],
                    m3: distances[m3]
                }

                # Validate with residual check
                if residual_ok(x, y, triplet_anchors, triplet_distances):
                    solutions.append((x, y))

            # Need at least one valid solution
            if not solutions:
                continue

            # Extract X and Y coordinates
            xs = [p[0] for p in solutions]
            ys = [p[1] for p in solutions]

            # Check cluster spread
            if (max(xs) - min(xs)) > MAX_SPREAD:
                continue
            if (max(ys) - min(ys)) > MAX_SPREAD:
                continue

            # Fuse solutions using median
            x_final = median(xs)
            y_final = median(ys)

            sample += 1
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            print(
                f"{sample:05d} | {ts} | "
                f"X={x_final:7.2f} cm | Y={y_final:7.2f} cm | "
                f"Anchors={len(distances)} | Triplets={len(solutions)}"
            )

    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("STOPPED BY USER")
        print("=" * 60)

    finally:
        ser.write(b"stop\r\n")
        ser.close()
        print("Serial port closed.")

# ============================================================
if __name__ == "__main__":
    main()