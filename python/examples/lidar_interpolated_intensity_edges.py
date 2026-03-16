import os
import ydlidar
import time
import sys
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.interpolate import make_interp_spline # Added for interpolation

# Apply the dark theme globally
plt.style.use('dark_background')

RMAX = 4.0
RMIN = 0.02
INTENSITY_MIN = 0
INTENSITY_MAX = 255
INTERP_MULTIPLIER = 3 # Generates 3x as many points for smooth curves

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
        current_time = time.time()
        time_diff = current_time - last_time
        scan_freq = 1.0 / time_diff if time_diff > 0 else 0.0
        last_time = current_time
        
        point_count = len(scan.points)

        # 1. Extract data into NumPy arrays
        angles = np.array([-point.angle for point in scan.points])
        ranges = np.array([point.range for point in scan.points])
        intensities = np.array([point.intensity for point in scan.points])
        
        # Neutralize invalid/zero ranges
        ranges[ranges <= 0.02] = np.nan 
        
        # 2. Sort points by angle
        sort_indices = np.argsort(angles)
        sorted_angles = angles[sort_indices]
        sorted_ranges = ranges[sort_indices]
        sorted_intensities = intensities[sort_indices]
        
        # 3. Find structural gaps to split the room into individual objects
        depth_threshold = 0.2 
        range_diffs = np.abs(np.diff(sorted_ranges))
        range_diffs[np.isnan(range_diffs)] = 999.0 
        
        # Get indices where the gap exceeds the threshold (used to slice the arrays)
        gap_indices = np.where(range_diffs > depth_threshold)[0] + 1
        
        angle_chunks = np.split(sorted_angles, gap_indices)
        range_chunks = np.split(sorted_ranges, gap_indices)
        intensity_chunks = np.split(sorted_intensities, gap_indices)
        
        smooth_angles, smooth_ranges, smooth_intensities = [], [], []
        
        # 4. Interpolate strictly WITHIN each physical object chunk
        for a_chunk, r_chunk, i_chunk in zip(angle_chunks, range_chunks, intensity_chunks):
            # Splines require strictly increasing X-values (angles). Remove duplicates:
            if len(a_chunk) > 0:
                valid_idx = np.insert(np.diff(a_chunk) > 0, 0, True)
                a_chunk = a_chunk[valid_idx]
                r_chunk = r_chunk[valid_idx]
                i_chunk = i_chunk[valid_idx]
            
            # A cubic spline requires at least 4 points to calculate the curve
            if len(a_chunk) > 3:
                # Generate a denser set of X-coordinates (angles)
                dense_a = np.linspace(a_chunk[0], a_chunk[-1], len(a_chunk) * INTERP_MULTIPLIER)
                
                # Calculate the smooth curve for the ranges
                spline_r = make_interp_spline(a_chunk, r_chunk, k=3)
                dense_r = spline_r(dense_a)
                
                # Linearly interpolate the color intensities
                dense_i = np.interp(dense_a, a_chunk, i_chunk)
                
                smooth_angles.extend(dense_a)
                smooth_ranges.extend(dense_r)
                smooth_intensities.extend(dense_i)
            elif len(a_chunk) > 0:
                # If chunk is too small to smooth, just use the raw points
                smooth_angles.extend(a_chunk)
                smooth_ranges.extend(r_chunk)
                smooth_intensities.extend(i_chunk)
                
        smooth_angles = np.array(smooth_angles)
        smooth_ranges = np.array(smooth_ranges)
        smooth_intensities = np.array(smooth_intensities)
        
        lidar_polar.clear()
        lidar_polar.set_rmax(RMAX)
        lidar_polar.grid(True, color='white', alpha=0.3)
        
        # 5. Build the final line segments from the smoothed data
        if len(smooth_angles) > 1:
            points = np.column_stack([smooth_angles, smooth_ranges]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Re-apply the depth gap mask to prevent drawing lines between the newly smoothed chunks
            s_range_diffs = np.abs(np.diff(smooth_ranges))
            valid_mask = s_range_diffs <= depth_threshold
            
            valid_segments = segments[valid_mask]
            segment_intensities = smooth_intensities[:-1][valid_mask]
            
            lc = LineCollection(valid_segments, cmap='hsv', linewidth=2)
            lc.set_array(segment_intensities)
            lc.set_clim(INTENSITY_MIN, INTENSITY_MAX)
            lidar_polar.add_collection(lc)

        # 6. Add Live Text Overlay
        overlay_text = f"Raw Points: {point_count}\nSmoothed: {len(smooth_angles)}\nFreq: {scan_freq:.1f} Hz"
        lidar_polar.text(-0.05, 1.05, overlay_text, transform=lidar_polar.transAxes, 
                         color='cyan', fontsize=12, fontweight='bold', 
                         verticalalignment='bottom', horizontalalignment='left')

ret = laser.initialize()
if ret:
    ret = laser.turnOn()
    if ret:
        time.sleep(1) 
        last_time = time.time()
        ani = animation.FuncAnimation(fig, animate, interval=50)
        plt.show()
    laser.turnOff()
laser.disconnecting()
plt.close()