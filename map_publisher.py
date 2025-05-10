#!/usr/bin/env python3

import threading, time
import numpy as np
from scipy.ndimage import median_filter
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion

class OccupancyPublisher(Node):
    def __init__(self):
        super().__init__('occupancy_publisher')
        # Parameters
        self.ORIGIN_RADIUS = 0.5      # ignore points within this radius (m)
        self.GRID_BINS = 40           # cells per axis
        self.scale = 10               # pixels per cell for display
        self.resolution = 4.0 / self.GRID_BINS  # meters per cell
        # X: 0→4m ahead
        self.X_RANGE = (0.0, 4.0)
        # Y: -2m→+2m
        half = self.GRID_BINS // 2
        self.Y_RANGE = (-half * self.resolution, half * self.resolution)

        # Bin edges
        self.x_edges = np.linspace(self.X_RANGE[0], self.X_RANGE[1], self.GRID_BINS + 1)
        self.y_edges = np.linspace(self.Y_RANGE[0], self.Y_RANGE[1], self.GRID_BINS + 1)

        # State
        self.latest_points = None
        self.points_lock = threading.Lock()

        # Publisher for occupancy grid
        self.map_pub = self.create_publisher(OccupancyGrid, 'occupancy_grid', 10)
        self.frame_id = 'map'

        # QoS
        pc_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=5)
        px4_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST, depth=1,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # Subscriptions
        self.create_subscription(PointCloud2, '/voa_pc_out', self.pc_callback, qos_profile=pc_qos)
        self.create_subscription(VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.position_callback, qos_profile=px4_qos)
        self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.attitude_callback, qos_profile=px4_qos)

        # OpenCV window
        cv2.namedWindow('Occupancy Grid', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Occupancy Grid', self.GRID_BINS * self.scale, self.GRID_BINS * self.scale)

        threading.Thread(target=rclpy.spin, args=(self,), daemon=True).start()

    def pc_callback(self, msg: PointCloud2):
        pts = [(x, y, z) for x, y, z in pc2.read_points(msg, field_names=('x','y','z'), skip_nans=True)]
        if pts:
            with self.points_lock:
                self.latest_points = np.array(pts, dtype=np.float32)

    def position_callback(self, msg: VehicleLocalPosition):
        # unused for map origin in this example
        pass

    def attitude_callback(self, msg: VehicleAttitude):
        # unused for map orientation in this example
        pass

    def run(self):
        rate_hz = 30.0
        dt = 1.0 / rate_hz
        while rclpy.ok():
            with self.points_lock:
                pts = None if self.latest_points is None else self.latest_points.copy()

            if pts is not None and pts.size:
                if pts.shape[0] > 50000:
                    idx = np.random.choice(pts.shape[0], 50000, replace=False)
                    pts = pts[idx]

                xs, ys = pts[:,0], pts[:,1]
                mask = np.hypot(xs, ys) > self.ORIGIN_RADIUS
                fx, fy = xs[mask], ys[mask]

                hist, _, _ = np.histogram2d(fy, fx, bins=[self.y_edges, self.x_edges])
                hist_filt = median_filter(hist, size=7)
                thresh = hist_filt.mean()

                # Binary occupancy: 0 free, 100 occupied
                occ = (hist_filt > thresh).astype(np.int8) * 100

                # Publish OccupancyGrid
                grid = OccupancyGrid()
                grid.header.stamp = self.get_clock().now().to_msg()
                grid.header.frame_id = self.frame_id
                grid.info.resolution = self.resolution
                grid.info.width = self.GRID_BINS
                grid.info.height = self.GRID_BINS
                grid.info.origin.position = Point(x=self.X_RANGE[0], y=self.Y_RANGE[0], z=0.0)
                grid.info.origin.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                grid.data = occ.flatten(order='C').tolist()
                self.map_pub.publish(grid)

                # Display
                occ_img = (occ > 50).astype(np.uint8) * 255
                disp = cv2.resize(occ_img, (self.GRID_BINS*self.scale, self.GRID_BINS*self.scale), interpolation=cv2.INTER_NEAREST)
                cv2.imshow('Occupancy Grid', disp)
                cv2.waitKey(1)

            time.sleep(dt)
        cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = OccupancyPublisher()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()