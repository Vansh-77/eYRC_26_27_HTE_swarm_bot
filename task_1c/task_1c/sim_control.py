#!/usr/bin/env python3
import argparse
import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from shape_interface.srv import GetShape

# Wheel <-> body-velocity mapping, columns are [left, right, back] wheel
# speed (rad/s); rows are body frame [vx, vy, wz] per unit wheel speed.
_WHEEL_TO_BODY = np.array([
    [0.0147224, -0.0147224,  0.0     ],  # vx per unit [left, right, back] wheel speed
    [0.0085,    0.0085,     -0.017   ],  # vy per unit [left, right, back] wheel speed
    [0.132564,  0.132564,  0.132564]   # wz per unit [left, right, back] wheel speed
]) 

_BODY_TO_WHEEL = np.linalg.inv(_WHEEL_TO_BODY)
_CTRL_LIMIT = 3.14      # rad/s, matches lekiwi.xml actuator ctrlrange

WAYPOINT_TOLERANCE = 0.01   # metres
CIRCLE_SEGMENTS     = 36
POSITION_KP         = 1.2
YAW_HOLD_KP         = 2
CONTROL_PERIOD      = 0.02

def body_to_wheels(vx, vy, wz):
    """Body-frame (vx, vy, wz) -> wheel angular velocities [left, right, back]."""
    # convert body velocity to wheel speeds using _BODY_TO_WHEEL,
    # uniform scaling instead of clipping as it preserves the direction of the bot
    body_velocities = np.array([vx, vy, wz])
    w = _BODY_TO_WHEEL @ body_velocities 
    peak = np.max(np.abs(w))
    if peak > _CTRL_LIMIT:
        w *= _CTRL_LIMIT / peak
    wheel_angular_velocities = w
    return wheel_angular_velocities.tolist()



def yaw_from_quat(w, x, y, z):
    # convert quaternion to yaw (radians).
     return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z)
    )


def _regular_polygon(cx, cy, n_sides, side_length, start_angle=math.pi / 2):
    """Vertices of a regular polygon centred on (cx, cy), closed back to the
    first vertex so the last waypoint returns the robot to where it started
    drawing."""
    r = side_length / (2 * math.sin(math.pi / n_sides))
    pts = [
        (cx + r * math.cos(start_angle + 2 * math.pi * i / n_sides),
         cy + r * math.sin(start_angle + 2 * math.pi * i / n_sides))
        for i in range(n_sides)
    ]
    return pts + [pts[0]]


def build_waypoints(shape_name, data):
    """World-frame waypoints for `shape_name`, as returned by the get_shape
    service: data[0:2] is the shape's centre (x, y); the remaining entries
    are its size parameters (see shape_service.cpp's shape_map)."""
    cx, cy = data[0], data[1]

    if shape_name == "Circle":
        radius = data[2]
        return [
            (cx + radius * math.cos(2 * math.pi * i / CIRCLE_SEGMENTS),
             cy + radius * math.sin(2 * math.pi * i / CIRCLE_SEGMENTS))
            for i in range(1, CIRCLE_SEGMENTS + 1)
        ]

    if shape_name == "Square":
        return _regular_polygon(cx, cy, 4, data[2], start_angle=math.pi / 4)

    if shape_name == "Triangle":
        return _regular_polygon(cx, cy, 3, data[2])

    if shape_name == "Pentagon":
        return _regular_polygon(cx, cy, 5, data[2])

    if shape_name == "Rectangle":
        w, h = data[2], data[3]
        corners = [
            (cx - w / 2, cy - h / 2),
            (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2),
            (cx - w / 2, cy + h / 2),
        ]
        return corners + [corners[0]]

    raise ValueError(f"unknown shape '{shape_name}'")


class ShapeController(Node):
    def __init__(self, speed):
        super().__init__("shape_controller")
        self.speed = speed

        self.pose = None        # (x, y, yaw), latest ground truth
        self.start_pose = None  # (x, y, yaw), recorded on first odom message
        self.wp_index = 0
        self.done = False
#Add the publsiher and subscriber scripts
        self.cmd_pub = self.create_publisher(Float64MultiArray, "/wheel_commands", 10)
        self.odom_subscriber = self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.timer = self.create_timer(CONTROL_PERIOD,self._control_step)
        self.shape , self.waypoints = self._request_shape()
        self.get_logger().info(f"shape : {self.shape} , waypoints : {self.waypoints}")
        
    def _request_shape(self):
        client = self.create_client(GetShape, "get_shape")
        while not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for get_shape service...")

        future = client.call_async(GetShape.Request())
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(
                f"get_shape service call failed: {response and response.message}"
            )

        return response.shape_name, build_waypoints(response.shape_name, list(response.data))

    def _odom_cb(self, msg):
        # extract (x, y, yaw) from msg.pose.pose into self.pose,
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        w = yaw_from_quat(q.w, q.x, q.y, q.z)

        self.pose = (x, y, w) 
        # record self.start_pose on the first callback.
        if self.start_pose is None:
            self.start_pose = self.pose 
            self.get_logger().info(f"start pose : {self.start_pose}") 

    def _publish(self, wheels):
        self.cmd_pub.publish(Float64MultiArray(data=wheels))

    def _control_step(self):
        if self.done or self.pose is None:
            return        
        # return when self.waypoints is none
        if self.waypoints is None:
            return
        # run only when wp_index is less then lenght of self.waypoints
        if self.wp_index < len(self.waypoints):
            corner = self.waypoints[self.wp_index]
            error_x = corner[0] - self.pose[0]
            error_y = corner[1] - self.pose[1]
            error_theta = self.start_pose[2] - self.pose[2]
            vx = POSITION_KP * error_x
            vy = POSITION_KP * error_y
            wz = YAW_HOLD_KP * error_theta
            error = math.sqrt(error_x**2 + error_y**2)
            # move to next waypoint when error is less than waypoint tolerance
            if error<WAYPOINT_TOLERANCE:
                self.get_logger().info(f"wp {self.wp_index} reached, err={error:.4f}")
                self.wp_index += 1
            self._publish(body_to_wheels(vx, vy, wz))
        else:
            self.done = True
            self._publish(body_to_wheels(0,0,0))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.25,
                         help="max approach speed, m/s")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = ShapeController(args.speed)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish([0.0, 0.0, 0.0])
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
