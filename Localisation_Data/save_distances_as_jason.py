#!/usr/bin/env python3

import serial
import re
import time
import json
from datetime import datetime

PORT = "/dev/ttyACM0"
BAUD = 115200
SERIAL_TIMEOUT = 0.2

ANCHORS = {
    0x0004: [0.0, 0.0],
    0x0006: [300.0, 300.0],
    0x0007: [450.0, 150.0],
    0x0009: [0.0, 300.0],
    0x000A: [300.0, 0.0],
}

PATTERN = re.compile(
    r"\[mac_address=0x([0-9a-fA-F]+),.*?distance\[cm\]=(-?\d+)\]"
)

def main():
    print("Connecting to serial port...")
    ser = serial.Serial(PORT, BAUD, timeout=SERIAL_TIMEOUT)
    time.sleep(1)
    ser.reset_input_buffer()

    print("Initializing UWB...")
    ser.write(b"stop\r\n")
    time.sleep(0.3)
    ser.write(b"initf -multi -addr=0 -paddr=[4,6,7,9,10]\r\n")
    time.sleep(0.3)

    print("\n=== UWB DATA LOGGING ===")
    print("Recording data... Press Ctrl+C to stop\n")

    measurements = []
    session_start = datetime.now()
    sample_count = 0
    current_frame = {}
    frame_number = 0

    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            if "SESSION_INFO_NTF" in line:
                if current_frame and len(current_frame) >= 3:
                    frame_number += 1
                    sample_count += 1
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    
                    measurement = {
                        "sample": sample_count,
                        "timestamp": ts,
                        "frame_number": frame_number,
                        "num_anchors": len(current_frame),
                        "distances": {k: v for k, v in current_frame.items()}
                    }
                    measurements.append(measurement)
                    
                    dist_str = ", ".join([f"{k}={v}cm" for k, v in current_frame.items()])
                    print(f"Sample {sample_count:04d} | {ts} | Anchors={len(current_frame)} | {dist_str}")
                
                current_frame = {}
                continue

            m = PATTERN.search(line)
            if m:
                mac = int(m.group(1), 16)
                dist = int(m.group(2))
                
                if mac in ANCHORS:
                    current_frame[hex(mac)] = dist

    except KeyboardInterrupt:
        print("\n\nStopping...")

    finally:
        ser.write(b"stop\r\n")
        ser.close()
        
        if current_frame and len(current_frame) >= 3:
            frame_number += 1
            sample_count += 1
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            measurement = {
                "sample": sample_count,
                "timestamp": ts,
                "frame_number": frame_number,
                "num_anchors": len(current_frame),
                "distances": current_frame
            }
            measurements.append(measurement)
        
        session_end = datetime.now()
        output_data = {
            "session_info": {
                "start_time": session_start.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": session_end.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": round((session_end - session_start).total_seconds(), 2),
                "total_samples": sample_count
            },
            "anchor_positions": {hex(k): v for k, v in ANCHORS.items()},
            "measurements": measurements
        }
        
        filename = f"uwb_data_{session_start.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n Data saved to: {filename}")
        print(f" Total samples: {sample_count}")
        print(f" Duration: {output_data['session_info']['duration_seconds']:.2f} seconds")

if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as e:
        print(f"\nERROR: Could not access serial port")
        print(f"Details: {e}")
        print("\nTroubleshooting:")
        print("1. Check device connection: ls -l /dev/ttyACM*")
        print("2. Check permissions: sudo chmod 666 /dev/ttyACM0")
        print("3. Close other programs using the port")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

