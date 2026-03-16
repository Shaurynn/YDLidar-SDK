import os
import ydlidar
import time
import sys
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Apply the dark theme globally
plt.style.use('dark_background')

SCAN_ANGLE = 90.0
RMAX = 4.0
RMIN = 0.02
INTENSITY_MIN = 0
INTENSITY_MAX = 255
DECAY_TIME = 2.0 # Points live for exactly 2 seconds

# Time-based point cloud buffers
pc_x = np.array([])
pc_y = np.array([])
pc_z = np.array([])
pc_i = np.array([])
pc_t = np.array([]) # Array to store the timestamp of each point

fig = plt.figure(figsize=(16, 14))
fig.canvas.manager.set_window_title('YDLidar 3D Scanner')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')

ports = ydlidar.lidarPortList()
port = "/dev/ydlidar"
for key, value in ports.items():
    port = value

laser = ydlidar.CYdLidar()
laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 230400)
laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
laser.setlidaropt(ydlidar.LidarPropScanFrequency, 10.0)
laser.setlidaropt(ydlidar.LidarPropSampleRate, 4)
laser.setlidaropt(ydlidar.LidarPropSingleChannel, False)
laser.setlidaropt(ydlidar.LidarPropMaxAngle, SCAN_ANGLE)
laser.setlidaropt(ydlidar.LidarPropMinAngle, -SCAN_ANGLE)
laser.setlidaropt(ydlidar.LidarPropMaxRange, RMAX)
laser.setlidaropt(ydlidar.LidarPropMinRange, RMIN)
laser.setlidaropt(ydlidar.LidarPropIntenstiy, True)
scan = ydlidar.LaserScan()

def animate(num):
    global pc_x, pc_y, pc_z, pc_i, pc_t
    r = laser.doProcessSimple(scan)
    if r:
        current_time = time.time()
        
        # 1. Faster Oscillation: 2 seconds per complete cycle
        # Using (current_time * math.pi) creates a full sine wave every 2 seconds
        pitch_deg = 15.0 * math.sin(current_time * math.pi)
        pitch_rad = np.radians(pitch_deg)
        
        yaw_rad = np.array([-point.angle for point in scan.points])
        ranges = np.array([point.range for point in scan.points])
        intensities = np.array([point.intensity for point in scan.points])
        
        valid_mask = ranges > 0.02
        yaw_rad = yaw_rad[valid_mask]
        ranges = ranges[valid_mask]
        intensities = intensities[valid_mask]
        
        # 2. Spherical to Cartesian Conversion
        x = ranges * np.cos(pitch_rad) * np.cos(yaw_rad)
        y = ranges * np.cos(pitch_rad) * np.sin(yaw_rad)
        z = ranges * np.sin(pitch_rad)
        
        # Create an array of the current timestamp for this batch of points
        t_batch = np.full(len(x), current_time)
        
        # Append new points to the global buffers
        pc_x = np.append(pc_x, x)
        pc_y = np.append(pc_y, y)
        pc_z = np.append(pc_z, z)
        pc_i = np.append(pc_i, intensities)
        pc_t = np.append(pc_t, t_batch)
        
        # 3. Time-based Decay Logic
        ages = current_time - pc_t
        alive_mask = ages <= DECAY_TIME
        
        # Purge dead points from the arrays
        pc_x = pc_x[alive_mask]
        pc_y = pc_y[alive_mask]
        pc_z = pc_z[alive_mask]
        pc_i = pc_i[alive_mask]
        pc_t = pc_t[alive_mask]
        ages = ages[alive_mask] # Update ages array to match the remaining points
        
        # 4. Calculate RGBA Colors for the Fade Effect
        # Normalize the intensity to a 0.0 - 1.0 scale
        norm_i = np.clip((pc_i - INTENSITY_MIN) / (INTENSITY_MAX - INTENSITY_MIN), 0, 1)
        cmap = plt.colormaps['hsv']
        
        # Get the base RGB colors from the colormap
        colors = cmap(norm_i) 
        
        # Modify the Alpha channel (column index 3) based on age
        # A point at age 0.0 has alpha 1.0. A point at age 2.0 has alpha 0.0.
        colors[:, 3] = np.clip(1.0 - (ages / DECAY_TIME), 0, 1)
        
        ax.clear()
        
        # Format axes and grid
        ax.set_xlim([-RMAX, RMAX])
        ax.set_ylim([-RMAX, RMAX])
        ax.set_zlim([-RMAX*0.5, RMAX*0.5])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(color='white', alpha=0.2)
        
        # 5. Draw Point Cloud
        # Note: depthshade=False stops Matplotlib from overriding our custom alpha fading
        ax.scatter(pc_x, pc_y, pc_z, c=colors, s=3, depthshade=False)
        
        # Live text readout
        overlay_text = f"Active Points: {len(pc_x)}\nSim Pitch: {pitch_deg:+.1f}°"
        ax.text2D(0.02, 0.95, overlay_text, transform=ax.transAxes, 
                  color='cyan', fontsize=12, fontweight='bold')

ret = laser.initialize()
if ret:
    ret = laser.turnOn()
    if ret:
        time.sleep(1) 
        ani = animation.FuncAnimation(fig, animate, interval=50)
        plt.show()
    laser.turnOff()
laser.disconnecting()
plt.close()