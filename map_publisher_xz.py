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
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Point, Quaternion

class OccupancyPublisherXZ(Node):
    def __init__(self):
        super().__init__('occupancy_publisher_xz')

        # ---- parameters ----
        self.ORIGIN_RADIUS = 0.3        # ignore points within this radius (m)
        self.GRID_BINS    = 40          # cells per axis
        self.scale        = 10          # pixels per cell for display
        self.resolution   = 4.0 / self.GRID_BINS  # meters per cell

        # X covers [0 … 4] m, Z covers [-2 … +2] m
        self.X_RANGE = (0.0, 2.0)
        self.Z_RANGE = (-2.0, 2.0)

        # bin edges for histogram2d: first dim=Z, second dim=X
        self.x_edges = np.linspace(self.X_RANGE[0], self.X_RANGE[1],
                                   self.GRID_BINS + 1)
        self.z_edges = np.linspace(self.Z_RANGE[0], self.Z_RANGE[1],
                                   self.GRID_BINS + 1)

        # ---- state ----
        self.latest_points = None
        self.points_lock   = threading.Lock()
        # drone_pos = (x, y, z)
        self.drone_pos     = np.zeros(3, dtype=np.float32)
        self.att_q         = (1.0, 0.0, 0.0, 0.0)  # (w,x,y,z)

        # ---- publisher ----
        self.map_pub = self.create_publisher(OccupancyGrid,
                                             'occupancy_grid_xz', 10)
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
        self.create_subscription(PointCloud2,
            '/voa_pc_out', self.pc_callback, qos_profile=pc_qos)
        # self.create_subscription(Odometry,
        #     '/local_position_odom',
        #     self.position_callback, 10)
        self.create_subscription(VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.position_callback, qos_profile=px4_qos)
        self.create_subscription(VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitude_callback, qos_profile=px4_qos)

        # OpenCV window
        cv2.namedWindow('Occupancy Grid XZ', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Occupancy Grid XZ',
                        self.GRID_BINS * self.scale,
                        self.GRID_BINS * self.scale)

        threading.Thread(target=rclpy.spin, args=(self,), daemon=True).start()

    def pc_callback(self, msg: PointCloud2):
        print(f'message data: {type(msg.data), len(msg.data), msg.data[:20]}')

        pts = [(x,y,z) for x,y,z in pc2.read_points(
            msg, field_names=('x','y','z'), skip_nans=True)]
        print(f'points: {len(pts)}')
        self.cloud_width  = msg.width
        self.cloud_height = msg.height
        self.latest_cloud = np.array(pts).reshape(
            (msg.height, msg.width, 3)
        )
        print(f'cloud width: {self.cloud_width}')
        print(f'cloud height: {self.cloud_height}')
        print(f'cloud shape: {self.latest_cloud.shape}')
        if pts:
            with self.points_lock:
                self.latest_points = np.array(pts, dtype=np.float32)

    def position_callback(self, msg: VehicleLocalPosition):
        # NED → just take x,y
        self.drone_pos = np.array([msg.x, msg.y, msg.z],
                                  dtype=np.float32)

    # def position_callback(self, msg: Odometry):
    #     self.drone_pos = (
    #         msg.pose.pose.position.x,
    #         msg.pose.pose.position.y,
    #         msg.pose.pose.position.z
    #     )

    def attitude_callback(self, msg: VehicleAttitude):
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

                # --- transform into world X,Z ---
                # subtract drone XY for X-axis rotation
                body_xy = pts[:, :2] - self.drone_pos[:2]

                # compute yaw from quaternion
                w, x, y, z = self.att_q
                yaw = np.arctan2(2*(w*z + x*y),
                                 1 - 2*(y*y + z*z))

                # rotate into world XY
                c, s = np.cos(yaw), np.sin(yaw)
                R2 = np.array([[ c, -s],
                               [ s,  c]], dtype=np.float32)
                world_xy = (R2 @ body_xy.T).T

                # world X and world Z
                xs = world_xy[:,0]
                zs = pts[:,2] - self.drone_pos[2]

                # mask out inner radius (in X–Z plane)
                mask = np.hypot(xs, zs) > self.ORIGIN_RADIUS
                fx, fz = xs[mask], zs[mask]

                # histogram over Z (rows) × X (cols)
                hist = np.histogram2d(
                    fz, fx,
                    bins=[self.z_edges, self.x_edges]
                )[0]
                filt   = median_filter(hist, size=3)
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
                    y=self.Z_RANGE[0],  # Z→grid‑Y axis
                    z=0.0
                )
                grid.info.origin.orientation = Quaternion(
                    x=0.0, y=0.0, z=0.0, w=1.0
                )
                grid.data = occ.flatten(order='C').tolist()
                self.map_pub.publish(grid)

                # display
                img  = (occ > 50).astype(np.uint8) * 255
                disp = cv2.resize(
                    img,
                    (self.GRID_BINS*self.scale,
                     self.GRID_BINS*self.scale),
                    interpolation=cv2.INTER_NEAREST
                )
                cv2.imshow('Occupancy Grid XZ', disp)
                cv2.waitKey(1)

            time.sleep(dt)

        cv2.destroyAllWindows()

def main():
    rclpy.init()
    node = OccupancyPublisherXZ()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
