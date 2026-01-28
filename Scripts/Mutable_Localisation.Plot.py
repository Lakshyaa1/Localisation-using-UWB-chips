#!/usr/bin/env python3

import serial
import re
import time
import math
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import combinations
from statistics import median
from datetime import datetime
from collections import deque
import threading
import queue

# ============================================================
# CONFIGURATION
# ============================================================

PORT = "/dev/ttyACM0"
BAUD = 115200
SERIAL_TIMEOUT = 0.2

# Default anchor configuration
ANCHORS = {
    0x0000: (0.0, 0.0),
    0x0002: (0.0, 750.0),
    0x0003: (0.0, 1500.0),
    0x0004: (1400.0, 1500.0),
    0x0006: (2800, 1500.0),
    0x0007: (2800.0, 750.0),
    0x0009: (2800.0, 0.0),
    0x000a: (1400.0, 0.0),
}

# Algorithm parameters
MIN_ANCHORS = 3
TRIPLET_SIZE = 3
MIN_DISTANCE = 10
MAX_DISTANCE = 3000
MAX_RESIDUAL = 100.0
MAX_SPREAD = 100.0
TRAIL_LENGTH = 100

# Serial pattern
PATTERN = re.compile(r"\[mac_address=0x([0-9a-fA-F]+),.*?distance\[cm\]=(-?\d+)\]")

# ============================================================
# TRILATERATION
# ============================================================

def trilaterate_3(a1, a2, a3, d1, d2, d3):
    """Solve for (x,y) position given 3 anchor positions and distances."""
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = a3

    A = np.array([
        [2 * (x2 - x1), 2 * (y2 - y1)],
        [2 * (x3 - x1), 2 * (y3 - y1)]
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
    """Check if position has acceptable error."""
    for mac, (ax, ay) in anchors.items():
        d_est = math.hypot(x - ax, y - ay)
        if abs(d_est - distances[mac]) > MAX_RESIDUAL:
            return False
    return True

def process_measurement(distances, anchors):
    """Process distances and return position."""
    if len(distances) < MIN_ANCHORS:
        return None

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
        return None

    xs = [p[0] for p in solutions]
    ys = [p[1] for p in solutions]

    if (max(xs) - min(xs)) > MAX_SPREAD or (max(ys) - min(ys)) > MAX_SPREAD:
        return None

    return (median(xs), median(ys), len(solutions))

# ============================================================
# LIVE VISUALIZER CLASS
# ============================================================

class LiveVisualizer:
    def __init__(self, anchors, trail_length=TRAIL_LENGTH):
        self.anchors = anchors
        self.trail_length = trail_length
        
        # Data storage
        self.x_history = deque(maxlen=trail_length)
        self.y_history = deque(maxlen=trail_length)
        self.position_count = 0
        
        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.setup_plot()
        
        # Plot elements
        self.trail_line, = self.ax.plot([], [], 'b-', alpha=0.4, linewidth=2, label='Trail')
        self.position_scatter = self.ax.scatter([], [], c='blue', s=150, zorder=5, 
                                               marker='o', edgecolors='darkblue', 
                                               linewidths=2, label='Current Position')
        
        # Draw anchors
        anchor_x = [pos[0] for pos in anchors.values()]
        anchor_y = [pos[1] for pos in anchors.values()]
        self.ax.scatter(anchor_x, anchor_y, c='red', s=300, marker='^', 
                       zorder=10, label='Anchors', edgecolors='darkred', linewidths=2)
        
        # Anchor labels
        for mac, (x, y) in anchors.items():
            self.ax.annotate(
                f'0x{mac:X}',
                (x, y),
                xytext=(12, 12),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.9)
            )
        
        # Info text
        self.info_text = self.ax.text(
            0.02, 0.98, '',
            transform=self.ax.transAxes,
            verticalalignment='top',
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9)
        )
        
        self.ax.legend(loc='upper right', fontsize=10)
        
    def setup_plot(self):
        """Setup plot bounds and styling."""
        xs = [pos[0] for pos in self.anchors.values()]
        ys = [pos[1] for pos in self.anchors.values()]
        
        margin = 200
        x_min, x_max = min(xs) - margin, max(xs) + margin
        y_min, y_max = min(ys) - margin, max(ys) + margin
        
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlabel('X Position (cm)', fontsize=13, fontweight='bold')
        self.ax.set_ylabel('Y Position (cm)', fontsize=13, fontweight='bold')
        self.ax.set_title('UWB Real-Time Positioning', fontsize=15, fontweight='bold')
        self.ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        self.ax.set_aspect('equal')
        
    def update(self, x, y, num_anchors=0, num_triplets=0):
        """Update visualization with new position."""
        self.x_history.append(x)
        self.y_history.append(y)
        self.position_count += 1
        
        # Update trail
        self.trail_line.set_data(list(self.x_history), list(self.y_history))
        
        # Update current position
        self.position_scatter.set_offsets([[x, y]])
        
        # Update info
        info = f'Position: ({x:.1f}, {y:.1f}) cm\n'
        info += f'Sample: {self.position_count}\n'
        info += f'Anchors: {num_anchors}\n'
        info += f'Triplets: {num_triplets}\n'
        info += f'Trail points: {len(self.x_history)}'
        self.info_text.set_text(info)
        
        return self.trail_line, self.position_scatter, self.info_text

# ============================================================
# LIVE SERIAL MODE
# ============================================================

def run_live_serial(port=PORT, baud=BAUD):
    """Run live visualization from serial port."""
    print("=" * 70)
    print("UWB LIVE VISUALIZATION - SERIAL MODE")
    print("=" * 70)
    print(f"\nConfigured Anchors: {len(ANCHORS)}")
    for mac, (x, y) in sorted(ANCHORS.items()):
        print(f"  0x{mac:04X}: ({x:7.1f}, {y:7.1f}) cm")
    print("=" * 70)
    
    # Initialize serial
    print(f"\nOpening serial port: {port} @ {baud} baud")
    ser = serial.Serial(port, baud, timeout=SERIAL_TIMEOUT)
    time.sleep(1)
    ser.reset_input_buffer()
    
    # Send init command
    anchor_ids = sorted([mac & 0xFF for mac in ANCHORS.keys()])
    addr_list = ",".join(map(str, anchor_ids))
    init_cmd = f"initf -multi -addr=1 -paddr=[{addr_list}]\r\n"
    
    print("Initializing UWB...")
    ser.write(b"stop\r\n")
    time.sleep(0.3)
    ser.write(init_cmd.encode())
    time.sleep(0.5)
    
    print("\nStarting live visualization...")
    print("Close the plot window to stop")
    print("=" * 70)
    print()
    
    # Create visualizer
    viz = LiveVisualizer(ANCHORS)
    
    distances = {}
    running = True
    
    def update_plot(frame):
        if not running:
            return viz.trail_line, viz.position_scatter, viz.info_text
            
        # Read serial data
        try:
            for _ in range(10):  # Process multiple lines per frame
                line = ser.readline().decode(errors="ignore").strip()
                if not line:
                    continue
                
                m = PATTERN.search(line)
                if not m:
                    continue
                
                mac = int(m.group(1), 16)
                dist = int(m.group(2))
                
                if mac not in ANCHORS or not (MIN_DISTANCE <= dist <= MAX_DISTANCE):
                    continue
                
                distances[mac] = float(dist)
                
                result = process_measurement(distances, ANCHORS)
                
                if result is not None:
                    x, y, num_triplets = result
                    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    print(f"{viz.position_count + 1:05d} | {ts} | "
                          f"X={x:7.2f} cm | Y={y:7.2f} cm | "
                          f"Anchors={len(distances)} | Triplets={num_triplets}")
                    return viz.update(x, y, len(distances), num_triplets)
        except Exception as e:
            print(f"Error: {e}")
        
        return viz.trail_line, viz.position_scatter, viz.info_text
    
    try:
        ani = FuncAnimation(viz.fig, update_plot, interval=50, blit=True, cache_frame_data=False)
        plt.show()
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        running = False
        ser.write(b"stop\r\n")
        ser.close()
        print("Serial port closed.")

# ============================================================
# JSON PLAYBACK MODE
# ============================================================

def run_json_playback(filename, speed=1.0):
    """Playback visualization from JSON file."""
    print("=" * 70)
    print("UWB PLAYBACK VISUALIZATION - JSON MODE")
    print("=" * 70)
    
    # Load JSON
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Load anchors from JSON
    anchors = {}
    for mac_str, coords in data.get("anchor_positions", {}).items():
        mac = int(mac_str, 16)
        anchors[mac] = tuple(coords)
    
    print(f"\nJSON File: {filename}")
    if "session_info" in data:
        info = data["session_info"]
        print(f"Session: {info.get('start_time', 'N/A')}")
        print(f"Duration: {info.get('duration_seconds', 'N/A')} seconds")
        print(f"Samples: {info.get('total_samples', 'N/A')}")
    
    print(f"\nAnchors: {len(anchors)}")
    for mac, (x, y) in sorted(anchors.items()):
        print(f"  0x{mac:04X}: ({x:7.1f}, {y:7.1f}) cm")
    print(f"\nPlayback speed: {speed}x")
    print("=" * 70)
    print()
    
    # Create visualizer
    viz = LiveVisualizer(anchors)
    
    # Process measurements
    measurements = data.get("measurements", [])
    position_data = []
    
    for measurement in measurements:
        distances = {}
        for mac_str, dist in measurement.get("distances", {}).items():
            mac = int(mac_str, 16)
            if mac in anchors and MIN_DISTANCE <= dist <= MAX_DISTANCE:
                distances[mac] = float(dist)
        
        result = process_measurement(distances, anchors)
        if result is not None:
            x, y, num_triplets = result
            position_data.append({
                'x': x, 'y': y,
                'anchors': len(distances),
                'triplets': num_triplets,
                'timestamp': measurement.get('timestamp', ''),
                'sample': measurement.get('sample', 0)
            })
    
    print(f"Valid positions: {len(position_data)}/{len(measurements)}")
    print("\nStarting playback...")
    print("Close window to exit")
    print("=" * 70)
    print()
    
    # Animation
    current_idx = [0]
    
    def update_plot(frame):
        if current_idx[0] >= len(position_data):
            current_idx[0] = 0  # Loop
        
        pos = position_data[current_idx[0]]
        print(f"{current_idx[0] + 1:05d} | {pos['timestamp']} | "
              f"X={pos['x']:7.2f} cm | Y={pos['y']:7.2f} cm | "
              f"Anchors={pos['anchors']} | Triplets={pos['triplets']}")
        
        result = viz.update(pos['x'], pos['y'], pos['anchors'], pos['triplets'])
        current_idx[0] += 1
        return result
    
    interval = int(100 / speed)  # Adjust speed
    ani = FuncAnimation(viz.fig, update_plot, interval=interval, blit=True, cache_frame_data=False)
    plt.show()

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='UWB Live Position Visualizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live visualization from serial port
  python3 visualize_uwb.py --live
  
  # Playback from JSON file
  python3 visualize_uwb.py --json data.json
  
  # Playback at 2x speed
  python3 visualize_uwb.py --json data.json --speed 2.0
  
  # Custom serial port
  python3 visualize_uwb.py --live --port /dev/ttyUSB0
        """
    )
    
    parser.add_argument('--live', action='store_true', help='Live visualization from serial port')
    parser.add_argument('--json', type=str, metavar='FILE', help='Playback from JSON file')
    parser.add_argument('--port', type=str, default=PORT, help=f'Serial port (default: {PORT})')
    parser.add_argument('--baud', type=int, default=BAUD, help=f'Baud rate (default: {BAUD})')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier (default: 1.0)')
    
    args = parser.parse_args()
    
    if args.live:
        run_live_serial(args.port, args.baud)
    elif args.json:
        run_json_playback(args.json, args.speed)
    else:
        parser.print_help()
        print("\nError: Please specify --live or --json")

if __name__ == "__main__":
    main()