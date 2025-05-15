#!/usr/bin/env python3

import threading
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, Quaternion

class OccupancyPublisher(Node):
    def __init__(self):
        super().__init__('occupancy_publisher')

        # ---- parameters ----
        self.ORIGIN_RADIUS = 0.5        # meters to ignore around robot
        self.GRID_BINS    = 40          # number of cells per axis
        self.resolution   = 4.0 / self.GRID_BINS  # meters per cell

        # Define local grid extents centered on robot
        half_cells = self.GRID_BINS // 2
        self.X_RANGE = (-half_cells * self.resolution,
                         half_cells * self.resolution)
        self.Y_RANGE = (-half_cells * self.resolution,
                         half_cells * self.resolution)

        # Bin edges for histogram2d
        self.x_edges = np.linspace(self.X_RANGE[0], self.X_RANGE[1], self.GRID_BINS + 1)
        self.y_edges = np.linspace(self.Y_RANGE[0], self.Y_RANGE[1], self.GRID_BINS + 1)

        # ---- state ----
        self.latest_points = None
        self.points_lock   = threading.Lock()

        # ---- publisher ----
        self.map_pub = self.create_publisher(OccupancyGrid, 'occupancy_grid', 10)

        # ---- QoS ----
        pc_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5
        )

        # Subscribe to your stereo depth pointcloud
        self.create_subscription(
            PointCloud2,
            '/stereo_front_pc',
            self.pc_callback,
            qos_profile=pc_qos
        )

        # Spin in background to handle callbacks
        threading.Thread(target=rclpy.spin, args=(self,), daemon=True).start()

    def pc_callback(self, msg: PointCloud2):
        # Convert PointCloud2 to Nx3 numpy
        pts = []
        for x, y, z in pc2.read_points(msg, field_names=('x','y','z'), skip_nans=True):
            pts.append((x, y, z))
        if pts:
            with self.points_lock:
                self.latest_points = np.array(pts, dtype=np.float32)

    def run(self):
        rate_hz = 10.0
        dt = 1.0 / rate_hz
        half_width  = (self.GRID_BINS * self.resolution) / 2.0
        half_height = (self.GRID_BINS * self.resolution) / 2.0

        while rclpy.ok():
            with self.points_lock:
                pts = None if self.latest_points is None else self.latest_points.copy()

            if pts is not None and pts.size:
                # Down-sample if too dense
                if pts.shape[0] > 50000:
                    idx = np.random.choice(pts.shape[0], 50000, replace=False)
                    pts = pts[idx]

                # Drop Z, work in robot's local XY frame
                xy = pts[:, :2]

                # Mask out points too close to robot
                mask = np.hypot(xy[:,0], xy[:,1]) > self.ORIGIN_RADIUS
                fx, fy = xy[mask,0], xy[mask,1]

                # Build occupancy histogram
                hist = np.histogram2d(
                    fy, fx,
                    bins=[self.y_edges, self.x_edges]
                )[0]

                # Threshold at mean
                occ = (hist > hist.mean()).astype(np.int8) * 100

                # Publish OccupancyGrid in robot frame
                grid = OccupancyGrid()
                grid.header.stamp = self.get_clock().now().to_msg()
                grid.header.frame_id = 'base_link'
                grid.info.resolution = self.resolution
                grid.info.width      = self.GRID_BINS
                grid.info.height     = self.GRID_BINS

                # Center origin at robot
                grid.info.origin.position = Point(
                    x=-half_width,
                    y=-half_height,
                    z=0.0
                )
                grid.info.origin.orientation = Quaternion(
                    x=0.0, y=0.0, z=0.0, w=1.0
                )

                # Flatten occupancy and publish
                grid.data = occ.flatten(order='C').tolist()
                self.map_pub.publish(grid)

            time.sleep(dt)


def main():
    rclpy.init()
    node = OccupancyPublisher()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
