#!/usr/bin/env python3

import serial
import re
import time
import math
import numpy as np
import json
import argparse
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
MAX_SPREAD = 150.0        # cm - maximum cluster spread allowed

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
# PROCESS SINGLE MEASUREMENT
# ============================================================
def process_measurement(distances, anchors):
    """
    Process a single measurement and return the estimated position.
    
    Args:
        distances: Dictionary of {mac: distance} for this measurement
        anchors: Dictionary of anchor positions
    
    Returns:
        Tuple of (x, y, num_triplets) or None if insufficient data
    """
    # Need at least MIN_ANCHORS to proceed
    if len(distances) < MIN_ANCHORS:
        return None

    solutions = []

    # Generate all 3-anchor combinations
    for combo in combinations(distances.keys(), TRIPLET_SIZE):
        m1, m2, m3 = combo

        # Solve trilateration
        sol = trilaterate_3(
            anchors[m1], anchors[m2], anchors[m3],
            distances[m1], distances[m2], distances[m3]
        )

        if sol is None:
            continue

        x, y = sol

        # Build triplet-specific anchor and distance dicts
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

        # Validate with residual check
        if residual_ok(x, y, triplet_anchors, triplet_distances):
            solutions.append((x, y))

    # Need at least one valid solution
    if not solutions:
        return None

    # Extract X and Y coordinates
    xs = [p[0] for p in solutions]
    ys = [p[1] for p in solutions]

    # Check cluster spread
    if (max(xs) - min(xs)) > MAX_SPREAD:
        return None
    if (max(ys) - min(ys)) > MAX_SPREAD:
        return None

    # Fuse solutions using median
    x_final = median(xs)
    y_final = median(ys)

    return (x_final, y_final, len(solutions))

# ============================================================
# LOAD ANCHORS FROM JSON
# ============================================================
def load_anchors_from_json(json_data):
    """
    Load anchor positions from JSON file.
    
    Args:
        json_data: Parsed JSON data dictionary
    
    Returns:
        Dictionary of {mac: (x, y)} anchor positions
    """
    anchors = {}
    anchor_positions = json_data.get("anchor_positions", {})
    
    for mac_str, coords in anchor_positions.items():
        # Convert hex string to integer (e.g., "0xa" -> 10)
        mac = int(mac_str, 16)
        anchors[mac] = tuple(coords)
    
    return anchors

# ============================================================
# PROCESS JSON FILE
# ============================================================
def process_json_file(filename, verbose=False):
    """Process measurements from a JSON file."""
    
    # Load JSON data
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Load anchors from JSON
    anchors = load_anchors_from_json(data)
    
    # Display configuration
    print("=" * 60)
    print("UWB POSITIONING - JSON FILE MODE")
    print("=" * 60)
    print(f"\nJSON File: {filename}")
    
    if "session_info" in data:
        info = data["session_info"]
        print(f"Session Start: {info.get('start_time', 'N/A')}")
        print(f"Session End: {info.get('end_time', 'N/A')}")
        print(f"Duration: {info.get('duration_seconds', 'N/A')} seconds")
        print(f"Total Samples: {info.get('total_samples', 'N/A')}")
    
    print(f"\nConfigured Anchors: {len(anchors)}")
    for mac, (x, y) in sorted(anchors.items()):
        print(f"  0x{mac:04X}: ({x:6.1f}, {y:6.1f}) cm")
    
    num_combinations = math.comb(len(anchors), TRIPLET_SIZE)
    print(f"\nPossible {TRIPLET_SIZE}-anchor combinations: {num_combinations}")
    print(f"Min anchors required: {MIN_ANCHORS}")
    print(f"Max residual: {MAX_RESIDUAL} cm")
    print(f"Max cluster spread: {MAX_SPREAD} cm")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("PROCESSING MEASUREMENTS")
    print("=" * 60)
    print()
    
    measurements = data.get("measurements", [])
    processed_count = 0
    
    # Statistics tracking
    stats = {
        'total': 0,
        'insufficient_anchors': 0,
        'no_valid_solutions': 0,
        'excessive_spread': 0,
        'distance_filtered': 0,
        'success': 0
    }
    
    for measurement in measurements:
        stats['total'] += 1
        
        # Extract distance data
        distances = {}
        raw_distances = measurement.get("distances", {})
        distances_before_filter = len(raw_distances)
        
        for mac_str, dist in raw_distances.items():
            mac = int(mac_str, 16)
            
            # Filter: only known anchors
            if mac not in anchors:
                continue
            
            # Filter: distance range
            if dist < MIN_DISTANCE or dist > MAX_DISTANCE:
                continue
            
            distances[mac] = float(dist)
        
        if len(distances) < distances_before_filter:
            stats['distance_filtered'] += 1
        
        # Check if we have enough anchors
        if len(distances) < MIN_ANCHORS:
            stats['insufficient_anchors'] += 1
            if verbose:
                print(f"Sample {measurement.get('sample', '?')}: Insufficient anchors ({len(distances)} < {MIN_ANCHORS})")
            continue
        
        # Process the measurement
        result = process_measurement(distances, anchors)
        
        if result is not None:
            x_final, y_final, num_triplets = result
            processed_count += 1
            stats['success'] += 1
            
            sample = measurement.get("sample", processed_count)
            timestamp = measurement.get("timestamp", "N/A")
            
            print(
                f"{sample:05d} | {timestamp} | "
                f"X={x_final:7.2f} cm | Y={y_final:7.2f} cm | "
                f"Anchors={len(distances)} | Triplets={num_triplets}"
            )
        else:
            # Determine why it failed
            if len(distances) >= MIN_ANCHORS:
                # Try to determine if it's spread or no solutions
                solutions = []
                for combo in combinations(distances.keys(), TRIPLET_SIZE):
                    m1, m2, m3 = combo
                    sol = trilaterate_3(
                        anchors[m1], anchors[m2], anchors[m3],
                        distances[m1], distances[m2], distances[m3]
                    )
                    if sol is None:
                        continue
                    x, y = sol
                    triplet_anchors = {m1: anchors[m1], m2: anchors[m2], m3: anchors[m3]}
                    triplet_distances = {m1: distances[m1], m2: distances[m2], m3: distances[m3]}
                    if residual_ok(x, y, triplet_anchors, triplet_distances):
                        solutions.append((x, y))
                
                if not solutions:
                    stats['no_valid_solutions'] += 1
                    if verbose:
                        print(f"Sample {measurement.get('sample', '?')}: No valid solutions (residual too high)")
                else:
                    stats['excessive_spread'] += 1
                    xs = [p[0] for p in solutions]
                    ys = [p[1] for p in solutions]
                    if verbose:
                        print(f"Sample {measurement.get('sample', '?')}: Excessive spread (X: {max(xs)-min(xs):.1f}, Y: {max(ys)-min(ys):.1f})")
    
    print("\n" + "=" * 60)
    print(f"PROCESSING COMPLETE")
    print(f"Processed {processed_count} / {len(measurements)} measurements ({100*processed_count/len(measurements):.1f}%)")
    print("=" * 60)
    print("\nFailure Analysis:")
    print(f"  Total measurements:        {stats['total']}")
    print(f"  ✓ Successfully processed:  {stats['success']}")
    print(f"  ✗ Insufficient anchors:    {stats['insufficient_anchors']}")
    print(f"  ✗ No valid solutions:      {stats['no_valid_solutions']}")
    print(f"  ✗ Excessive spread:        {stats['excessive_spread']}")
    print(f"  ⚠ Distance filtered:       {stats['distance_filtered']}")
    print("\nTip: Use --verbose flag to see detailed rejection reasons")
    print("=" * 60)

# ============================================================
# LIVE SERIAL MODE
# ============================================================
def process_live_serial():
    """Process measurements from live serial connection."""
    
    # Display configuration
    print("=" * 60)
    print("UWB POSITIONING - LIVE SERIAL MODE")
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

            # Process the measurement
            result = process_measurement(distances, ANCHORS)
            
            if result is not None:
                x_final, y_final, num_triplets = result
                sample += 1
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                print(
                    f"{sample:05d} | {ts} | "
                    f"X={x_final:7.2f} cm | Y={y_final:7.2f} cm | "
                    f"Anchors={len(distances)} | Triplets={num_triplets}"
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
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='UWB Positioning System - Live or JSON file mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live serial mode
  python3 uwb_positioning.py
  
  # JSON file mode
  python3 uwb_positioning.py --json measurements.json
  python3 uwb_positioning.py -j data/session_001.json
  
  # JSON file mode with verbose output
  python3 uwb_positioning.py -j measurements.json --verbose
        """
    )
    
    parser.add_argument(
        '-j', '--json',
        type=str,
        metavar='FILE',
        help='Process measurements from a JSON file instead of live serial'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed rejection reasons for each failed measurement'
    )
    
    args = parser.parse_args()
    
    if args.json:
        # JSON file mode
        process_json_file(args.json, verbose=args.verbose)
    else:
        # Live serial mode
        process_live_serial()

# ============================================================
if __name__ == "__main__":
    main()