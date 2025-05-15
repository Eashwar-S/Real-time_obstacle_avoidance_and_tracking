#!/usr/bin/env python3
"""
TFLiteFollower Node

Subscribes to:
  • /image_rect/compressed        (sensor_msgs/msg/CompressedImage)
  • /tflite_data                  (voxl_msgs/msg/Aidetection)
  • /fmu/out/vehicle_local_position (px4_msgs/msg/VehicleLocalPosition)
  • /fmu/out/vehicle_status        (px4_msgs/msg/VehicleStatus)
  • /fmu/out/vehicle_control_mode  (px4_msgs/msg/VehicleControlMode)

Publishes:
  • /image_rect/tflite_annotated  (sensor_msgs/msg/Image)
  • /fmu/in/trajectory_setpoint   (px4_msgs/msg/TrajectorySetpoint)
  • /fmu/in/offboard_control_mode (px4_msgs/msg/OffboardControlMode)
  • /fmu/in/vehicle_command       (px4_msgs/msg/VehicleCommand)
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from voxl_msgs.msg import Aidetection
from px4_msgs.msg import (
    VehicleLocalPosition,
    VehicleStatus,
    VehicleControlMode,
    TrajectorySetpoint,
    OffboardControlMode,
    VehicleCommand
)

class TfliteFollower(Node):
    def __init__(self):
        super().__init__('tflite_follower')

        # --- parameters ---
        self.declare_parameter('follow_distance', 0.5)
        self.declare_parameter('hover_height', -1.0)
        self.declare_parameter('yaw_gain', 1.0)

        self.follow_dist  = self.get_parameter('follow_distance').value
        self.hover_height = self.get_parameter('hover_height').value
        self.K_YAW        = self.get_parameter('yaw_gain').value

        # --- state ---
        self.current_local_pos = None
        self.offboard_enabled  = False
        self.target_index      = None
        self.desired_area      = None
        self.last_center       = None

        # buffer for latest detections
        self.detections = []  # each = {'box':(x1,y1,x2,y2), 'conf':float, 'class':str}

        # pixel→meter gains
        self.K_LAT  = 0.1
        self.K_VERT = 0.001
        self.K_FWD  = 0.1

        # PX4 QoS
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        # --- Subscribers ---
        self.create_subscription(
            CompressedImage,
            '/image_rect/compressed',
            self.image_callback,
            img_qos
        )
        self.create_subscription(
            Aidetection,
            '/tflite_data',
            self.detection_callback,
            img_qos
        )
        self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.local_position_callback,
            px4_qos
        )
        self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.status_callback,
            px4_qos
        )
        self.create_subscription(
            VehicleControlMode,
            '/fmu/out/vehicle_control_mode',
            self.control_mode_callback,
            px4_qos
        )

        # --- Publishers ---
        self.pub_annotated = self.create_publisher(
            Image,
            '/image_rect/tflite_annotated',
            QoSProfile(depth=1)
        )
        self.pub_sp = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            px4_qos
        )
        self.offb_ctrl_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            px4_qos
        )
        self.cmd_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            px4_qos
        )

        # heartbeat
        self.create_timer(0.05, self.publish_offboard_control_heartbeat_signal)

        cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
        self.bridge = CvBridge()
        self.get_logger().info('TFLite follower node ready.')

    def publish_offboard_control_heartbeat_signal(self):
        msg = OffboardControlMode()
        msg.position     = True
        msg.velocity     = False
        msg.acceleration = False
        msg.attitude     = False
        msg.body_rate    = False
        msg.timestamp    = int(self.get_clock().now().nanoseconds / 1000)
        self.offb_ctrl_pub.publish(msg)

    def status_callback(self, msg: VehicleStatus):
        armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.get_logger().info(f"Armed={armed}, nav_state={msg.nav_state}")

    def local_position_callback(self, msg: VehicleLocalPosition):
        self.current_local_pos = msg

    def control_mode_callback(self, msg: VehicleControlMode):
        self.offboard_enabled = bool(msg.flag_control_offboard_enabled)

    def detection_callback(self, msg: Aidetection):
        # buffer every detection; we'll clear after drawing each image
        box = (msg.x_min, msg.y_min, msg.x_max, msg.y_max)
        self.detections.append({
            'box': box,
            'conf': msg.detection_confidence,
            'class': msg.class_name
        })

    def image_callback(self, img_msg: CompressedImage):
        # decode image
        arr = np.frombuffer(img_msg.data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return
        h, w = img.shape[:2]

        # snapshot & clear detections buffer
        dets = self.detections
        self.detections = []

        # draw all detections
        boxes  = [d['box'] for d in dets]
        confs  = [d['conf'] for d in dets]
        clss   = [d['class'] for d in dets]
        for i, (box, conf, cls) in enumerate(zip(boxes, confs, clss)):
            x1,y1,x2,y2 = map(int, box)
            label = f'{i}: {cls} {conf:.2f}'
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img, label, (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

        cv2.imshow('Detections', img)
        key = cv2.waitKey(1) & 0xFF

        # select target if not yet set
        if self.target_index is None and boxes:
            if 48 <= key <= 57:
                idx = key - 48
                if idx < len(boxes):
                    self.target_index   = idx
                    x1,y1,x2,y2         = map(int, boxes[idx])
                    self.last_center    = ((x1+x2)/2, (y1+y2)/2)
                    self.desired_area   = (x2-x1)*(y2-y1)
                    self.get_logger().info(f'Selected detection {idx} area={self.desired_area:.0f}')

        # tracking & set‑point
        if self.target_index is not None and self.current_local_pos is not None:
            if not boxes:
                self.get_logger().warn('No detections: clearing target')
                self.target_index = None
            else:
                # find the box closest to previous center
                centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in boxes]
                dists   = [math.hypot(cx-self.last_center[0], cy-self.last_center[1])
                           for cx,cy in centers]
                best    = int(np.argmin(dists))
                box     = boxes[best]
                cx,cy   = centers[best]
                area    = (box[2]-box[0])*(box[3]-box[1])
                self.last_center = (cx,cy)

                # pixel errors
                ex = (cx - w/2) / w
                ey = (cy - h/2) / h

                pos = self.current_local_pos
                north_sp = pos.x + self.K_FWD * ((self.desired_area-area)/self.desired_area)
                east_sp  = pos.y + self.K_LAT * ex
                down_sp  = pos.z + self.K_VERT * ey
                yaw_sp   = pos.heading + self.K_YAW * ex

                sp = TrajectorySetpoint()
                sp.timestamp    = int(self.get_clock().now().nanoseconds / 1000)
                sp.position     = [north_sp, east_sp, down_sp]
                sp.velocity     = [0.0, 0.0, 0.0]
                sp.acceleration = [0.0, 0.0, 0.0]
                sp.jerk         = [0.0, 0.0, 0.0]
                sp.yaw          = yaw_sp
                sp.yawspeed     = 0.0

                if self.offboard_enabled:
                    self.pub_sp.publish(sp)
                else:
                    self.get_logger().info('OFFBOARD not active; skipping setpoint')

        # publish annotated image
        out = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        out.header = img_msg.header
        self.pub_annotated.publish(out)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TfliteFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
