#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import math
from px4_msgs.msg import (
    VehicleLocalPosition,
    VehicleControlMode,
    VehicleCommand,
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleAttitude,
)
import heapq

class DijkstraPlanner(Node):
    def __init__(self):
        super().__init__('dijkstra_planner')

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # state
        self.offboard_enabled = False
        self.drone_position  = None
        self.att_q           = (1.0, 0.0, 0.0, 0.0)  # quaternion (w,x,y,z)

        # subscribe to drone position and attitude
        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.position_callback,
            qos_profile=px4_qos
        )
        self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitude_callback,
            qos_profile=px4_qos
        )

        # subscribe to OFFBOARD and commands
        self.create_subscription(
            VehicleControlMode,
            '/fmu/out/vehicle_control_mode',
            self.control_mode_callback,
            qos_profile=px4_qos
        )
        self.create_subscription(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            self.command_callback,
            qos_profile=px4_qos
        )
        self.create_subscription(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            self.offboard_heartbeat_callback,
            qos_profile=px4_qos
        )

        # subscribe to occupancy grid
        self.create_subscription(
            OccupancyGrid,
            'occupancy_grid',
            self.grid_callback,
            10
        )

        # publishers
        self.offb_ctrl_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            px4_qos
        )
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)
        self.sp_pub   = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            px4_qos
        )

        # heartbeat timer
        self.create_timer(0.05, self.publish_offboard_heartbeat)

    def publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.position     = True
        msg.velocity     = False
        msg.acceleration = False
        msg.attitude     = False
        msg.body_rate    = False
        msg.timestamp    = int(self.get_clock().now().nanoseconds / 1000)
        self.offb_ctrl_pub.publish(msg)

    def position_callback(self, msg: VehicleLocalPosition):
        self.drone_position = np.array([msg.x, msg.y, msg.z], dtype=np.float32)

    def attitude_callback(self, msg: VehicleAttitude):
        self.att_q = (msg.q[0], msg.q[1], msg.q[2], msg.q[3])

    def control_mode_callback(self, msg: VehicleControlMode):
        self.offboard_enabled = bool(msg.flag_control_offboard_enabled)

    def command_callback(self, msg: VehicleCommand):
        # track last command
        pass

    def offboard_heartbeat_callback(self, msg: OffboardControlMode):
        # no-op
        pass

    def grid_callback(self, msg: OccupancyGrid):
        # extract grid
        h, w    = msg.info.height, msg.info.width
        res     = msg.info.resolution
        data    = np.array(msg.data, dtype=np.int8).reshape((h, w))
        cost    = np.where(data > 50, np.inf, 1.0)

        # start: vehicle position in grid frame
        if self.drone_position is None:
            start = (h // 2, 0)
        else:
            ox = msg.info.origin.position.x
            oy = msg.info.origin.position.y
            x, y = self.drone_position[0], self.drone_position[1]
            col = int((x - ox) / res)
            row = int((y - oy) / res)
            col = np.clip(col, 0, w-1)
            row = np.clip(row, 0, h-1)
            start = (row, col)

        # goal: same row, right edge
        goal = (start[0], w-1)

        path = self.compute_dijkstra(cost, start, goal)
        if path is None:
            self.get_logger().warn('No path found')
            return

        # get grid origin & orientation
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y
        qx = msg.info.origin.orientation.x
        qy = msg.info.origin.orientation.y
        qz = msg.info.origin.orientation.z
        qw = msg.info.origin.orientation.w
        # yaw from grid quaternion
        yaw0 = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        R0 = np.array([[math.cos(yaw0), -math.sin(yaw0)],
                       [math.sin(yaw0),  math.cos(yaw0)]])

        # publish full Path rotated into world
        path_msg = Path()
        path_msg.header = msg.header
        for (r, c) in path:
            local = np.array([(c+0.5)*res, (r+0.5)*res])
            wx, wy = (R0.dot(local) + np.array([ox, oy]))
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(wx)
            pose.pose.position.y = float(wy)
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

        # first waypoint setpoint
        if not self.offboard_enabled:
            self.get_logger().warn('OFFBOARD inactive, skip setpoint')
            return

        # compute first world waypoint
        (r0, c0) = path[0]
        local0 = np.array([(c0+0.5)*res, (r0+0.5)*res])
        wx0, wy0 = (R0.dot(local0) + np.array([ox, oy]))
        z_sp = float(self.drone_position[2]) if self.drone_position is not None else 0.0

        sp = TrajectorySetpoint()
        sp.timestamp    = int(self.get_clock().now().nanoseconds / 1000)
        sp.position     = [wx0, wy0, z_sp]
        sp.velocity     = [0.0, 0.0, 0.0]
        sp.acceleration = [0.0, 0.0, 0.0]
        sp.jerk         = [0.0, 0.0, 0.0]
        sp.yaw          = 0.0
        sp.yawspeed     = 0.0
        self.sp_pub.publish(sp)
        self.get_logger().info(f"SP→ X:{wx0:.2f}, Y:{wy0:.2f}, Z:{z_sp:.2f}")

    def compute_dijkstra(self, cost, start, goal):
        h, w = cost.shape
        dist = np.full((h, w), np.inf)
        prev = {}
        dist[start] = 0.0
        pq = [(0.0, start)]
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        while pq:
            d, (r, c) = heapq.heappop(pq)
            if (r, c) == goal:
                break
            if d > dist[r, c]:
                continue
            for dr, dc in dirs:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and cost[nr,nc] < np.inf:
                    nd = d + cost[nr,nc]
                    if nd < dist[nr,nc]:
                        dist[nr,nc] = nd
                        prev[(nr,nc)] = (r, c)
                        heapq.heappush(pq, (nd, (nr,nc)))
        if dist[goal] == np.inf:
            return None
        path = [goal]
        cur = goal
        while cur != start:
            cur = prev[cur]
            path.append(cur)
        return list(reversed(path))


def main():
    rclpy.init()
    node = DijkstraPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
