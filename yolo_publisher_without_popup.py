import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from sensor_msgs.msg import Image, NavSatFix
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import csv
from datetime import date
import os
import glob

class YoloPub(Node):
    def __init__(self):
        super().__init__("yolo_publisher")

        self.bridge = CvBridge()

        print(f"cuda is available : {torch.cuda.is_available()}")
        self.model = YOLO("./model_inside.pt")
        self.model.to("cuda")
        self.get_logger().warn(f"[DEPLOY] yolo deployed on device : {self.model.device} succesfully!")


        self.sub_left = self.create_subscription(Image, "/zed/zed_node/rgb/color/rect/image", self.left_callback, 10)
        self.sub_depth = self.create_subscription(Image, "/zed/zed_node/depth/depth_registered", self.depth_callback, 10)
        self.sub_state = self.create_subscription(Bool, "/state", self.state_callback, 10)
        self.sub_gps = self.create_subscription(NavSatFix, "/gps/fix", self.gps_callback, 10)

        self.log_service = self.create_service(Trigger, "log_service", self.log_callback)

        self.pub = self.create_publisher(Float32MultiArray, "/cone_bbox", 10)

        self.timer = self.create_timer(0.1, self.timer_callback) #10 hz

        self.left_frame = None 
        self.depth_frame = None
        self.object_number = 1
        self.gps = None
        self.state = False
        self.save_data = []

        today = date.today()
        log_dir = "/home/orin/logs"
        session = 1
        self.csv_path = f"{log_dir}/log_{today}_{session}.csv"
        while os.path.exists(self.csv_path):
            session += 1
            self.csv_path = f"{log_dir}/log_{today}_{session}.csv"

        




                  
                  

    def left_callback(self, msg):
        self.left_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, msg.encoding)  
    
    def gps_callback(self, msg : NavSatFix):
        self.gps = msg

    def log_callback(self, request, response):
        
        self.process()

        if len(self.save_data) == 0:
            response.success = False
            response.message = "CONE NOT DETECTED, move rover little away from cone"
            return response

        self.get_logger().info("Preparing to log color.....")

        min_z = None
        cone_to_log_idx: int = None
        for i, data in enumerate(self.save_data):
            z = data[2]
            if i==0:
                min_z = z
                cone_to_log_idx = 0
            if z > 0 and z < min_z:
                min_z = z
                cone_to_log_idx = i 

        H, S, V = self.save_data[cone_to_log_idx][3:]

        self.get_logger().info("Preparing to log GPS coords.....")
        if self.gps is None:
            self.get_logger().warn("Not getting GPS from topic, GPS message is still None!!!")
            latitude = 0.0
            longitude = 0.0
        else:
            latitude = self.gps.latitude
            longitude = self.gps.longitude

        self.get_logger().info(f"Logging object_number {self.object_number} at GPS ({latitude}, {longitude})!")
        self.get_logger().info(f"Logging color H={H}, S={S}, V={V}")

        
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["object_number", "latitude", "longitude", "H", "S", "V"])

            writer.writerow([self.object_number, latitude, longitude, H, S, V])


        self.object_number += 1

        self.get_logger().info("ONE STEP CLOSER TO WINNING IRC LETS GO!!!")


        response.success = True
        response.message = "logged sucessfully"
        return response
    
    def state_callback(self, msg):
        self.state = msg.data


    def cone_matching(self)


    def process(self):
        if self.left_frame is None or self.depth_frame is None:
            return

        frame = self.left_frame
        depth = self.depth_frame

        results = self.model(frame)[0]

        if len(results.boxes) == 0:
            self.get_logger().warn("[MODEL] 😢 i dont see any cone bro i am blind")
            return

        all_bbox_per_frame = []
        for box in results.boxes:
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            xmin, ymin, xmax, ymax = bbox
            cx = int((xmin + xmax) / 2)
            cy = int((ymin + ymax) / 2)
            w = xmax - xmin
            h = ymax - ymin
            z = float(depth[cy, cx]) if not np.isnan(depth[cy, cx]) else -1.0
            #NOTE i am giving depth as -1 if zed skill issues

            #if aspect ratio is fcked up, skip
            aspect_ratio = w / h
            #we have to test and see the conditions 

            # take a patch and give avg hsv
            patch_w = int(w * 0.1)
            patch_h = int(h * 0.1)
            x1 = max(cx - patch_w, 0)
            x2 = min(cx + patch_w, frame.shape[1] - 1)
            y1 = max(cy - patch_h, 0)
            y2 = min(cy + patch_h, frame.shape[0] - 1)

            roi = frame[y1:y2, x1:x2]

            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            H = float(np.mean(hsv_roi[:, :, 0]))
            S = float(np.mean(hsv_roi[:, :, 1]))
            V = float(np.mean(hsv_roi[:, :, 2]))

            all_bbox_per_frame.append([float(cx), float(cy), z, H, S, V])
            #NOT PUBLISHING W AND H ANYMORE


        msg = Float32MultiArray()
        msg.data = [item for cone in all_bbox_per_frame for item in cone]  # flatten
        self.save_data = all_bbox_per_frame.copy() #saving to write in csv if service requests

        num_cones = len(all_bbox_per_frame) // 6
        if num_cones > 10:
            self.get_logger().info("o shit it detected more than 10 cones 💀")
            return
        
        self.pub.publish(msg)
        self.get_logger().warn(f"[MODEl] 🥳 deteced {num_cones} cones")


    def run_model(self, iterations):
        '''
        runs self.process for iteration times and gives an average value (removes outliers as well) takes 1 iteration takes 50 ms so i guess we can run for 250 ms and give average values 
        '''


    
    def timer_callback(self):
        if self.left_frame is None:
            self.get_logger().info("[WAITING] for zed boi to give frames")
            return
        if self.depth_frame is None:
            self.get_logger().info("[WAITING] for zed depth image")
            return
        
        if self.state:
            self.get_logger().info("[AUTONOMOUS]")           
            self.process()
        
        if not self.state:
            self.get_logger().info("[MANUAL]")




def main(args=None):
    rclpy.init(args=args)
    node = YoloPub()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
