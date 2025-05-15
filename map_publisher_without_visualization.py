#!/usr/bin/env python3

import threading
import time
import numpy as np
import math
from scipy.ndimage import median_filter, uniform_filter

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Point, Quaternion

class OccupancyPublisher(Node):
    def __init__(self):
        super().__init__('occupancy_publisher')

        # ---- parameters ----
        self.ORIGIN_RADIUS = 0.5       # ignore points within this radius (m)
        self.GRID_BINS    = 40         # cells per axis
        self.scale        = 10         # pixels per cell for display (unused now)
        self.resolution   = 4.0 / self.GRID_BINS  # meters per cell
        self.X_RANGE      = (0.0, 4.0)             # forward 4 m
        half_cells = self.GRID_BINS // 2
        self.Y_RANGE      = (-half_cells * self.resolution,
                              half_cells * self.resolution)

        # bin edges for histogram2d
        self.x_edges = np.linspace(self.X_RANGE[0], self.X_RANGE[1],
                                   self.GRID_BINS + 1)
        self.y_edges = np.linspace(self.Y_RANGE[0], self.Y_RANGE[1],
                                   self.GRID_BINS + 1)

        # ---- state ----
        self.latest_points = None
        self.points_lock   = threading.Lock()
        self.drone_pos     = np.zeros(3, dtype=np.float32)
        self.att_q         = (1.0, 0.0, 0.0, 0.0)  # (w,x,y,z)

        # ---- publisher ----
        self.map_pub = self.create_publisher(OccupancyGrid,
                                             'occupancy_grid', 10)
        self.frame_id = 'map'

        # ---- QoS ----
        pc_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5
        )
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # ---- subscriptions ----
        # self.create_subscription(PointCloud2,
        #     '/voa_pc_out', self.pc_callback, qos_profile=pc_qos)
        
        self.create_subscription(PointCloud2,
            '/stereo_front_pc', self.pc_callback, qos_profile=pc_qos)
        # self.create_subscription(Odometry,
        #     '/local_position_odom',
        #     self.position_callback, 10)
        self.create_subscription(VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.position_callback, qos_profile=px4_qos)
        self.create_subscription(VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitude_callback, qos_profile=px4_qos)
        
        
        β = math.radians(90)
        R_rot_y = np.array([
            [ math.cos(β), 0, math.sin(β)],
            [ 0,           1, 0        ],
            [-math.sin(β), 0, math.cos(β)]
        ])
        alpha = math.radians(90)
        R_rot_z = np.array([
            [ math.cos(alpha), -math.sin(alpha), 0],
            [ math.sin(alpha),  math.cos(alpha), 0],
            [ 0,            0,           1]
        ])
        self.R_cam_to_ned = R_rot_y @ R_rot_z

        # spin callbacks in background
        threading.Thread(target=rclpy.spin, args=(self,), daemon=True).start()

    def pc_callback(self, msg: PointCloud2):
        pts = [self.R_cam_to_ned @ np.array([x,y,z]) for x,y,z in pc2.read_points(
            msg, field_names=('x','y','z'), skip_nans=True)]
        if pts:
            with self.points_lock:
                self.latest_points = np.array(pts, dtype=np.float32)

    def position_callback(self, msg: VehicleLocalPosition):
        # NED → just take x,y
        self.drone_pos = np.array([msg.x, msg.y, msg.z],
                                  dtype=np.float32)

    # def position_callback(self, msg: Odometry):
    #     # Update the drone's current position
    #     self.drone_pos = (
    #         msg.pose.pose.position.x,
    #         msg.pose.pose.position.y,
    #         msg.pose.pose.position.z
    #     )

    def attitude_callback(self, msg: VehicleAttitude):
        # store quaternion (w,x,y,z)
        self.att_q = (msg.q[0], msg.q[1],
                      msg.q[2], msg.q[3])

    def run(self):
        rate_hz = 30.0
        dt = 1.0 / rate_hz

        while rclpy.ok():
            with self.points_lock:
                pts = (None if self.latest_points is None
                       else self.latest_points.copy())

            if pts is not None and pts.size:
                # down‑sample
                if pts.shape[0] > 50000:
                    idx = np.random.choice(pts.shape[0],
                                           50000, replace=False)
                    pts = pts[idx]

                # subtract drone XY position and drop Z
                body_xy = pts[:, :2] - self.drone_pos[:2]

                # compute yaw from quaternion
                w, x, y, z = self.att_q
                yaw = np.arctan2(2*(w*z + x*y),
                                 1 - 2*(y*y + z*z))

                # 2×2 rotation matrix
                c, s = np.cos(yaw), np.sin(yaw)
                R2 = np.array([[ c, -s],
                               [ s,  c]], dtype=np.float32)

                # rotate into world frame
                world_xy = (R2 @ body_xy.T).T
                xs, ys = world_xy[:,0], world_xy[:,1]

                # mask out inner radius
                mask = np.hypot(xs, ys) > self.ORIGIN_RADIUS
                fx, fy = xs[mask], ys[mask]

                # histogram & filter
                hist = np.histogram2d(
                    fy, fx,
                    bins=[self.y_edges, self.x_edges]
                )[0]
                filt = uniform_filter(hist, size=9)
                filt = median_filter(filt, size=3)
                thresh = filt.mean()

                # occupancy: free=0, occ=100
                occ = (filt > thresh).astype(np.int8) * 100

                # publish OccupancyGrid
                grid = OccupancyGrid()
                grid.header.stamp = self.get_clock().now().to_msg()
                grid.header.frame_id = self.frame_id
                grid.info.resolution = self.resolution
                grid.info.width      = self.GRID_BINS
                grid.info.height     = self.GRID_BINS
                grid.info.origin.position = Point(
                    x=self.X_RANGE[0],
                    y=self.Y_RANGE[0],
                    z=0.0
                )
                grid.info.origin.orientation = Quaternion(
                    x=0.0, y=0.0, z=0.0, w=1.0
                )
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
