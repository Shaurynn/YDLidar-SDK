import os
import ydlidar
import time
import sys
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.patches import Arc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

RMAX = 16.0

fig = plt.figure(figsize=(16, 14))
fig.canvas.manager.set_window_title('YDLidar LIDAR Monitor')
lidar_polar = plt.subplot(polar=True)
lidar_polar.autoscale_view(True,True,True)
lidar_polar.set_rmax(RMAX)
lidar_polar.grid(True)
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
laser.setlidaropt(ydlidar.LidarPropSampleRate, 20)
laser.setlidaropt(ydlidar.LidarPropSingleChannel, False)
laser.setlidaropt(ydlidar.LidarPropMaxAngle, 180.0)
laser.setlidaropt(ydlidar.LidarPropMinAngle, -180.0)
laser.setlidaropt(ydlidar.LidarPropMaxRange, RMAX)
laser.setlidaropt(ydlidar.LidarPropMinRange, 0.02)
laser.setlidaropt(ydlidar.LidarPropIntenstiy, True)
scan = ydlidar.LaserScan()

def animate(num):
    r = laser.doProcessSimple(scan)
    if r:
        # 1. Extract and format data as NumPy arrays
        # Negate the angle to keep your x-axis flip
        angles = np.array([-point.angle for point in scan.points])
        ranges = np.array([point.range for point in scan.points])
        
        # 2. Sort points by angle for a continuous perimeter
        sort_indices = np.argsort(angles)
        sorted_angles = angles[sort_indices]
        sorted_ranges = ranges[sort_indices]
        
        # 3. Prevent lines from bridging disconnected objects
        # Calculate the difference in range between adjacent points
        # If the jump is larger than the threshold (e.g., 0.5 meters), break the line
        depth_threshold = 0.5 
        range_diffs = np.abs(np.diff(sorted_ranges, prepend=sorted_ranges[-1]))
        
        # Insert NaN where the jump is too large. Matplotlib stops drawing the line at NaN.
        sorted_ranges[range_diffs > depth_threshold] = np.nan
        
        lidar_polar.clear()
        lidar_polar.set_rmax(RMAX)
        lidar_polar.grid(True)
        
        # 4. Draw the edges using plot()
        lidar_polar.plot(sorted_angles, sorted_ranges, color='cyan', linewidth=2, linestyle='-')
        
        # Optional: You can keep a scatter plot underneath if you still want to see the distinct vertices
        # lidar_polar.scatter(sorted_angles, sorted_ranges, color='blue', s=5)

ret = laser.initialize()
if ret:
    ret = laser.turnOn()
    if ret:
        ani = animation.FuncAnimation(fig, animate, interval=50)
        plt.show()
    laser.turnOff()
laser.disconnecting()
plt.close()