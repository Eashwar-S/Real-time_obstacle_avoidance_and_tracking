#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import heapq

class DijkstraPlanner(Node):
    def __init__(self):
        super().__init__('dijkstra_planner')

        self.scale = 10  # (display-related, left unused)
        # QoS for PX4 topics
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # subscriptions
        self.create_subscription(Odometry,
            '/local_position_odom',
            self.position_callback, 10)
        self.create_subscription(OccupancyGrid,
            'occupancy_grid', self.grid_callback, 10)

        self.path_pub = self.create_publisher(Path, 'planned_path', 10)
        self.drone_position = None

    def position_callback(self, msg: Odometry):
        self.drone_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )

    def grid_callback(self, msg: OccupancyGrid):
        h, w   = msg.info.height, msg.info.width
        res    = msg.info.resolution
        data   = np.array(msg.data, dtype=np.int8).reshape((h, w))

        cost   = np.where(data > 50, np.inf, 1.0)
        start  = (h//2, 0)
        goal   = (h//2, w-1)
        path   = self.compute_dijkstra(cost, start, goal)

        if path is None:
            self.get_logger().warn('No path found')
        else:
            # compute world setpoints
            world_pts = [
                (
                    msg.info.origin.position.x + (c + 0.5) * res,
                    msg.info.origin.position.y + (r + 0.5) * res
                )
                for (r, c) in path
            ]

            # publish Path message
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
                nr, nc = r+dr, c+dc
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
