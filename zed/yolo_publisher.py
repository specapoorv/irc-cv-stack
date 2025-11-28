import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO


class YOLO(Node):
    def __init__(self):
        super().__init__("yolo_publisher")

        self.bridge = CvBridge()

        # YOLO model
        self.model = YOLO("./weights/model_old.pt")

        # Subscribers to ZED camera images
        self.sub_left = self.create_subscription(
            Image,
            "/zed/zed_node/left/image_rect_color",
            self.left_callback,
            10
        )

        self.sub_depth = self.create_subscription(
            Image,
            "/zed/zed_node/depth/depth_registered",
            self.depth_callback,
            10
        )

        # Output publisher (cx, cy, w, h, depth)
        self.pub = self.create_publisher(Float32MultiArray, "/cone_bbox", 10)

        # Frame buffers
        self.left_frame = None
        self.depth_frame = None

    def left_callback(self, msg):
        self.left_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.process()

    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, msg.encoding)  # float32 depth in meters

    def process(self):
        if self.left_frame is None or self.depth_frame is None:
            return

        frame = self.left_frame
        depth = self.depth_frame

        # YOLO detection
        results = self.model(frame)[0]

        if len(results.boxes) == 0:
            return

        box = results.boxes[0]  # pick the first detection
        xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()

        cx = int((xmin + xmax) / 2)
        cy = int((ymin + ymax) / 2)
        w = xmax - xmin
        h = ymax - ymin

        # Depth at center pixel
        z = float(depth[cy, cx]) if not np.isnan(depth[cy, cx]) else -1.0

        msg = Float32MultiArray()
        msg.data = [float(cx), float(cy), float(w), float(h), z]

        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = YOLO()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
