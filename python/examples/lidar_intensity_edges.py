import os
import ydlidar
import time
import sys
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.patches import Arc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import numpy as np

# Apply the dark theme globally
plt.style.use('dark_background')

RMAX = 4.0
RMIN = 0.02

# Global variable to track frame timing for frequency calculation
last_time = time.time()

fig = plt.figure(figsize=(16, 14))
fig.canvas.manager.set_window_title('YDLidar LIDAR Monitor')
lidar_polar = plt.subplot(polar=True)
lidar_polar.autoscale_view(True,True,True)
lidar_polar.set_rmax(RMAX)
lidar_polar.grid(True, color='white', alpha=0.3)
ports = ydlidar.lidarPortList()
port = "/dev/ydlidar"
for key, value in ports.items():
    port = value

laser = ydlidar.CYdLidar()
laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 230400)
laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
laser.setlidaropt(ydlidar.LidarPropScanFrequency, 5.0)
laser.setlidaropt(ydlidar.LidarPropSampleRate, 4)
laser.setlidaropt(ydlidar.LidarPropSingleChannel, False)
laser.setlidaropt(ydlidar.LidarPropMaxAngle, 180.0)
laser.setlidaropt(ydlidar.LidarPropMinAngle, -180.0)
laser.setlidaropt(ydlidar.LidarPropMaxRange, RMAX)
laser.setlidaropt(ydlidar.LidarPropMinRange, RMIN)
laser.setlidaropt(ydlidar.LidarPropIntenstiy, True)
scan = ydlidar.LaserScan()

def animate(num):
    global last_time
    r = laser.doProcessSimple(scan)
    if r:
        # Calculate real-time scanner frequency
        current_time = time.time()
        time_diff = current_time - last_time
        scan_freq = 1.0 / time_diff if time_diff > 0 else 0.0
        last_time = current_time
        
        point_count = len(scan.points)
        
        # 1. Extract data into NumPy arrays (keeping the x-axis flip)
        angles = np.array([-point.angle for point in scan.points])
        ranges = np.array([point.range for point in scan.points])
        intensities = np.array([point.intensity for point in scan.points])
        
        # 2. Sort points by angle for a continuous perimeter
        sort_indices = np.argsort(angles)
        sorted_angles = angles[sort_indices]
        sorted_ranges = ranges[sort_indices]
        sorted_intensities = intensities[sort_indices]
        
        # 3. Create line segments from the points
        # Reshape points into (N, 1, 2) and concatenate to get segments of shape (N-1, 2, 2)
        points = np.column_stack([sorted_angles, sorted_ranges]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # 4. Mask out segments that bridge large depth gaps
        depth_threshold = 0.2 
        range_diffs = np.abs(np.diff(sorted_ranges))
        
        # Convert any NaNs (from our origin fix) to a large number so they exceed the threshold
        range_diffs[np.isnan(range_diffs)] = 999.0 
        valid_mask = range_diffs <= depth_threshold
        
        valid_segments = segments[valid_mask]
        segment_intensities = sorted_intensities[:-1][valid_mask]
        
        lidar_polar.clear()
        lidar_polar.set_rmax(RMAX)
        lidar_polar.grid(True, color='white', alpha=0.3)
        
        # 5. Add Live Text Overlay
        # transAxes anchors the text relative to the plot axes (0,0 is bottom-left, 1,1 is top-right)
        overlay_text = f"Points: {point_count}\nFreq: {scan_freq:.1f} Hz"
        lidar_polar.text(-0.05, 1.05, overlay_text, transform=lidar_polar.transAxes, 
                         color='cyan', fontsize=12, fontweight='bold', 
                         verticalalignment='bottom', horizontalalignment='left')
        
        # 6. Create the LineCollection and map the intensity
        lc = LineCollection(valid_segments, cmap='hsv', linewidth=5)
        lc.set_array(segment_intensities)
        
        # Add the collection to the polar plot
        lidar_polar.add_collection(lc)
        
ret = laser.initialize()
if ret:
    ret = laser.turnOn()
    if ret:
        # Give the sensor a tiny moment to spin up before animating
        time.sleep(1) 
        last_time = time.time()
        ani = animation.FuncAnimation(fig, animate, interval=50)
        plt.show()
    laser.turnOff()
laser.disconnecting()
plt.close()