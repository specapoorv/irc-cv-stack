#!/usr/bin/env python3
import rclpy as r
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8, Bool, Float32MultiArray


class GoToCone(Node):
    def __init__(self):
        super().__init__("go_to_cone_node")

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # Publishers
        self.twist_pub = self.create_publisher(Twist, "/motion", qos)
        self.rot_pub   = self.create_publisher(Int8, "/rot", qos)

        # Subscribers
        self.yolo_sub  = self.create_subscription(Float32MultiArray, "/cone_bbox", self.yolo_callback, qos)
        self.state_sub = self.create_subscription(Bool, "/state", self.state_callback, qos)

        # Autonomous state
        self.state = False        # Manual at start
        self.have_yolo = False      # YOLO not seen yet
        self.last_yolo_update = None
        self.last_yolo_data = None    

        # YOLO data
        self.cx = None
        self.cy = None
        self.w  = None
        self.h  = None
        self.depth = None

        # Rover control params
        self.Kp_ang = 0.005               # angular velocity gain (tune)
        self.forward_speed = 30           # linear speed
        self.center_align_threshold = 40  # pixels
        self.image_width = 1280
        self.image_height = 720           
        self.goal_depth = 0.50            #when to stop

        self.timer = self.create_timer(0.1, self.timer_callback)



    def state_callback(self, msg):
        self.state = msg.data



    def yolo_callback(self, msg: Float32MultiArray):

        

        cx, cy, w, h, depth = msg.data

        cx = float(cx)
        cy = float(cy)
        w  = float(w)
        h  = float(h)
        depth = float(depth)

        new_data = (cx, cy, w, h, depth)

        current_time = self.get_clock().now() #timestampt time.time() wont work

        # First time
        if self.last_yolo_update is None:
            self.last_yolo_update = current_time
            self.last_yolo_data = new_data
            self.have_yolo = True
        else:
            #check if values changed
            if new_data != self.last_yolo_data:
                self.last_yolo_update = current_time     #update timestamp
                self.last_yolo_data = new_data           
                self.have_yolo = True
            else:
                #if didnt change, do search
                time_diff = (current_time - self.last_yolo_update).nanoseconds / 1e9
                if time_diff > 2.0:
                    self.have_yolo = False

        self.cx, self.cy, self.w, self.h, self.depth = new_data


    def align_to_cone(self):
        """
        Align rover such that cone lies in center of camera.
        Angular velocity proportional to horizontal pixel offset.
        """
        center_x = self.image_width // 2
        error = center_x - self.cx

        if abs(error) < self.center_align_threshold:
            return True  # aligned

        twist = Twist()
        twist.angular.z = error * self.Kp_ang
        self.twist_pub.publish(twist)
        return False


    def timer_callback(self):

        if not self.state:
            self.twist_pub.publish(Twist())  # stop
            self.have_yolo = False
            return

        if not self.have_yolo:
            twist = Twist()
            twist.linear.x = self.forward_speed
            self.twist_pub.publish(twist)
            self.get_logger().warn("NO YOLO = driving straight")
            return
        
        aligned = self.align_to_cone()
        if not aligned:
            self.get_logger().info("ALIGNING TO CONE")
            return
        

        if self.depth is not None:
            if self.depth < self.goal_depth:
                self.get_logger().info("REACHED CONE = stopping")
                self.twist_pub.publish(Twist())
                return

        twist = Twist()
        twist.linear.x = self.forward_speed
        self.twist_pub.publish(twist)
        self.get_logger().info(f"DRIVING TOWARDS CONE, depth={self.depth:.2f}")


def main(args=None):
    r.init(args=args)
    node = GoToCone()
    try:
        r.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        r.shutdown()


if __name__ == "__main__":
    main()
