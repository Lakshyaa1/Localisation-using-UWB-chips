#!/usr/bin/env python3

import serial
import re
import time
import math
import json
import numpy as np
from datetime import datetime

PORT = "/dev/ttyACM0"
BAUD = 115200
SERIAL_TIMEOUT = 0.2

ANCHORS = {
    0x0004: (0.0,   0.0),
    0x0006: (300.0, 300.0),
    0x0007: (450.0, 150.0),
    0x0009: (0.0, 300.0),
    0x000A: (300.0, 0.0),
}

ANCHOR_IDS = list(ANCHORS.keys())
MIN_ANCHORS = 3

PATTERN = re.compile(
    r"\[mac_address=0x([0-9a-fA-F]+),.*?distance\[cm\]=(-?\d+)\]"
)

MIN_DISTANCE = 10
MAX_DISTANCE = 500
MAX_RMSE = 50.0

def trilaterate_least_squares(anchor_pos, distances):
    macs = list(anchor_pos.keys())
    ref = macs[0]

    x1, y1 = anchor_pos[ref]
    d1 = distances[ref]

    A = []
    b = []

    for mac in macs[1:]:
        xi, yi = anchor_pos[mac]
        di = distances[mac]

        A.append([2 * (xi - x1), 2 * (yi - y1)])
        b.append(d1**2 - di**2 + xi**2 + yi**2 - x1**2 - y1**2)

    A = np.array(A)
    b = np.array(b)

    try:
        pos, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return float(pos[0]), float(pos[1])
    except np.linalg.LinAlgError:
        return None, None

def compute_rmse(x, y, anchor_pos, distances):
    err = 0.0
    for mac, (ax, ay) in anchor_pos.items():
        d_est = math.sqrt((x - ax)**2 + (y - ay)**2)
        err += (d_est - distances[mac])**2
    return math.sqrt(err / len(anchor_pos))

def main():
    ser = serial.Serial(PORT, BAUD, timeout=SERIAL_TIMEOUT)
    time.sleep(1)
    ser.reset_input_buffer()

    print("Initializing UWB...")
    ser.write(b"stop\r\n")
    time.sleep(0.3)
    ser.write(b"initf -multi -addr=0 -paddr=[4,6,7,9,10]\r\n")
    time.sleep(0.3)

    print("\nUWB POSITIONING (STATIONARY MODE – ALL ANCHORS)")
    print("Ctrl+C to stop\n")

    distances = {}
    sample_count = 0
    
    position_data = []
    session_start = datetime.now()

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

            if mac not in ANCHORS:
                continue
            if dist < MIN_DISTANCE or dist > MAX_DISTANCE:
                continue

            distances[mac] = float(dist)

            if len(distances) < MIN_ANCHORS:
                continue

            sub_anchors = {m: ANCHORS[m] for m in distances}
            sub_distances = {m: distances[m] for m in distances}

            x, y = trilaterate_least_squares(sub_anchors, sub_distances)
            if x is None:
                continue

            rmse = compute_rmse(x, y, sub_anchors, sub_distances)

            sample_count += 1
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            data_point = {
                "sample": sample_count,
                "timestamp": ts,
                "x_cm": round(x, 2),
                "y_cm": round(y, 2),
                "num_anchors": len(sub_anchors),
                "rmse": round(rmse, 2),
                "distances": {hex(k): v for k, v in sub_distances.items()}
            }
            position_data.append(data_point)

            print(
                f"{sample_count:05d} | {ts} | "
                f"X={x:7.2f} cm | Y={y:7.2f} cm | "
                f"Anchors={len(sub_anchors)} | RMSE={rmse:5.2f}"
            )

    except KeyboardInterrupt:
        print("\nStopped.")
    
    finally:
        ser.write(b"stop\r\n")
        ser.close()
        
        if position_data:
            session_end = datetime.now()
            output_data = {
                "session_info": {
                    "start_time": session_start.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": session_end.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_seconds": (session_end - session_start).total_seconds(),
                    "total_samples": sample_count
                },
                "anchor_positions": {hex(k): v for k, v in ANCHORS.items()},
                "position_data": position_data
            }
            
            filename = f"uwb_log_{session_start.strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\nData saved to: {filename}")
            print(f"Total samples: {sample_count}")

if __name__ == "__main__":
    main()

