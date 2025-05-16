#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
from px4_msgs.msg import (
    VehicleLocalPosition,
    VehicleAttitude,
    TrajectorySetpoint,
    VehicleCommand,           # <-- for subscribing to our own commands
    OffboardControlMode,      # <-- for subscribing to the heartbeat
    VehicleControlMode,       # <-- to know when OFFBOARD is active
)
import heapq

class DijkstraPlanner(Node):
    def __init__(self):
        super().__init__('dijkstra_planner')

        # --- QoS for PX4 topics ---
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # state flags
        self.offboard_enabled = False
        self.last_cmd        = None

        # subscribe to current vehicle position
        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.position_callback,
            qos_profile=px4_qos
        )

        # subscribe to PX4 control‑mode to check OFFBOARD status
        self.create_subscription(
            VehicleControlMode,
            '/fmu/out/vehicle_control_mode',
            self.control_mode_callback,
            qos_profile=px4_qos
        )

        # optionally, watch the commands you're sending
        self.create_subscription(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            self.command_callback,
            qos_profile=px4_qos
        )

        # and watch your own offboard‑heartbeat
        self.create_subscription(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            self.offboard_control_mode_callback,
            qos_profile=px4_qos
        )

        # subscribe to occupancy grid
        self.create_subscription(
            OccupancyGrid,
            'occupancy_grid',
            self.grid_callback,
            10
        )

        self.offb_ctrl_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            px4_qos
        )

        # publisher for the full planned Path
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)
        # publisher for the first waypoint as a TrajectorySetpoint
        self.sp_pub   = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            px4_qos
        )

        self.drone_position = None
        self.create_timer(0.05, self.publish_offboard_control_heartbeat_signal)

    def publish_offboard_control_heartbeat_signal(self):
        msg = OffboardControlMode()
        msg.position     = True   # enable position+yaw control
        msg.velocity     = False
        msg.acceleration = False
        msg.attitude     = False
        msg.body_rate    = False
        msg.timestamp    = int(self.get_clock().now().nanoseconds / 1000)
        self.offb_ctrl_pub.publish(msg)

    def position_callback(self, msg: VehicleLocalPosition):
        self.drone_position = np.array([msg.x, msg.y, msg.z], dtype=np.float32)

    def control_mode_callback(self, msg: VehicleControlMode):
        # flag_control_offboard_enabled tells us if PX4 is actually in OFFBOARD
        self.offboard_enabled = bool(msg.flag_control_offboard_enabled)
        self.get_logger().debug(f"OFFBOARD enabled: {self.offboard_enabled}")

    def command_callback(self, msg: VehicleCommand):
        # keep track of every command you issue
        self.last_cmd = msg
        self.get_logger().debug(f"Sent VEHICLE_COMMAND: {msg.command}")

    def offboard_control_mode_callback(self, msg: OffboardControlMode):
        # see what offboard modes (pos/vel/accel) you have enabled
        self.get_logger().debug(
            f"OffboardControlMode → pos:{msg.position} vel:{msg.velocity}"
        )

    def grid_callback(self, msg: OccupancyGrid):
        # build a simple cost map
        h, w   = msg.info.height, msg.info.width
        res    = msg.info.resolution
        data   = np.array(msg.data, dtype=np.int8).reshape((h, w))
        cost   = np.where(data > 50, np.inf, 1.0)

        start = (h//2, 0)
        goal  = (h//2, w-1)
        path  = self.compute_dijkstra(cost, start, goal)
        if path is None:
            self.get_logger().warn('No path found')
            return

        # publish the full Path
        path_msg = Path()
        path_msg.header = msg.header
        for (r, c) in path:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = msg.info.origin.position.x + (c + 0.5) * res
            pose.pose.position.y = msg.info.origin.position.y + (r + 0.5) * res
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

        # only send a setpoint if PX4 is actually in OFFBOARD
        if not self.offboard_enabled:
            self.get_logger().warn("OFFBOARD not active – skipping setpoint publish")
            return

        # get the first waypoint in world coords
        first_r, first_c = path[0]
        first_x = msg.info.origin.position.x + (first_c + 0.5) * res
        first_y = msg.info.origin.position.y + (first_r + 0.5) * res
        z_sp    = float(self.drone_position[2]) if self.drone_position is not None else 0.0

        sp = TrajectorySetpoint()
        sp.timestamp    = int(self.get_clock().now().nanoseconds / 1000)
        sp.position     = [first_x, first_y, z_sp]
        sp.velocity     = [0.0, 0.0, 0.0]
        sp.acceleration = [0.0, 0.0, 0.0]
        sp.jerk         = [0.0, 0.0, 0.0]
        sp.yaw          = 0.0
        sp.yawspeed     = 0.0

        self.sp_pub.publish(sp)
        self.get_logger().info(f"SP→ X:{first_x:.2f}, Y:{first_y:.2f}, Z:{z_sp:.2f}")

    def compute_dijkstra(self, cost, start, goal):
        h, w = cost.shape
        dist = np.full((h, w), np.inf, dtype=float)
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
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and cost[nr, nc] < np.inf:
                    nd = d + cost[nr, nc]
                    if nd < dist[nr, nc]:
                        dist[nr, nc] = nd
                        prev[(nr, nc)] = (r, c)
                        heapq.heappush(pq, (nd, (nr, nc)))

        if dist[goal] == np.inf:
            return None

        path = [goal]
        cur  = goal
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
