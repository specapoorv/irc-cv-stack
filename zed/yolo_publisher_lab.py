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
import ast

#color for coloured prints brooo!!!
RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"


class YoloPub(Node):
    def __init__(self):
        super().__init__("yolo_publisher")

        self.bridge = CvBridge()

        self.model = YOLO("./model_inside.pt")
        self.model.to("cuda")
        print(torch.cuda.is_available())
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
        self.last_state = None

        today = date.today()
        self.file_path = f"/home/orin/log_{today}.csv"
        self.hist_to_go_to = None

                  
    def left_callback(self, msg):
        self.left_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, msg.encoding)  
    
    def state_callback(self, msg):
        self.state = msg.data
    
    def gps_callback(self, msg : NavSatFix):
        self.gps = msg

    def log_callback(self, request, response):
        
        self.process()

        if len(self.save_data) == 0:
            #response.success = False
            #response.message = "CONE NOT DETECTED, move rover little away from cone"
            #return response
            pass

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

        cx, cy, z, hist = self.save_data[cone_to_log_idx]

        self.get_logger().info("Preparing to log GPS coords.....")
        if self.gps is None:
            self.get_logger().warn("Not getting GPS from topic, GPS message is still None!!!")
            latitude = 0.0
            longitude = 0.0
        else:
            latitude = self.gps.latitude
            longitude = self.gps.longitude

        self.get_logger().info(f"Logging object_number {self.object_number} at GPS ({latitude}, {longitude})!")
        self.get_logger().info(f"Logging LAB hist | mean={np.mean(hist):.4f}, max={np.max(hist):.4f}")


        file_exists = os.path.exists(self.file_path)
        with open(self.file_path, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["object_number", "latitude", "longitude", "hist"])

            writer.writerow([self.object_number, latitude, longitude, hist.tolist()])


        self.object_number += 1

        self.get_logger().info(f"{GREEN}{BOLD}ONE STEP CLOSER TO WINNING IRC LETS GO!!!{RESET}")
        #NOTE comment out for popup while loggin
       # self.show_popup(self.left_frame, cx, cy, H, S, V)
        
        response.success = True
        response.message = "logged sucessfully"


        return response

    #HELPERS
    
    def euclidean(self, p, q):
        return sum((a - b) ** 2 for a, b in zip(p, q))
    
    def chi_square(self, h1, h2, eps=1e-10):
        return 0.5 * np.sum((h1 - h2) ** 2 / (h1 + h2 + eps))

    def process(self):
        '''
        runs yolo basically and saves data in self.save_data
        '''
        if self.left_frame is None or self.depth_frame is None:
            return

        frame = self.left_frame
        depth = self.depth_frame

        results = self.model(frame)[0]

        if len(results.boxes) == 0:
            self.get_logger().warn(f"{YELLOW}[MODEL] 😢 i dont see any cone bro i am blind{RESET}")
            return

        all_bbox_per_frame = []
        for box in results.boxes:
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            xmin, ymin, xmax, ymax = bbox
            cx = int((xmin + xmax) / 2)
            cy = int((ymin + ymax) / 2)
            w = xmax - xmin
            h = ymax - ymin
            dpatch = depth[cy-2:cy+3, cx-2:cx+3]
            dpatch = dpatch[np.isfinite(dpatch)]
            if len(dpatch) == 0:
                continue
            z = float(np.median(dpatch))

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
            #earlier it was HSV now LAB (L doesnt matter A and B are important)
            lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            a = lab_roi[:, :, 1]
            b = lab_roi[:, :, 2]
            hist = cv2.calcHist(
                [a, b],
                [0, 1],
                None,
                [32, 32],
                [0, 256, 0, 256]
            )
            hist = cv2.normalize(hist, hist).flatten()

            all_bbox_per_frame.append([float(cx), float(cy), z, hist])


        self.save_data = all_bbox_per_frame.copy() #saving to write in csv if service requests and will be used in publishing

        num_cones = len(all_bbox_per_frame)
         
        # if num_cones > 10:
        #     self.get_logger().info(f"{RED}o shit it detected more than 10 cones 💀{RESET}")
        #     return
        
        self.get_logger().info(f"{GREEN}[MODEl] 🥳 deteced {num_cones} cones{RESET}")

    def get_hist_to_go_to(self):
        curr_lat = self.gps.latitude
        curr_lon = self.gps.longitude
        gps = [curr_lat, curr_lon]
        min_distance = float("inf")
        with open(self.file_path, newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                object_number = int(row["object_number"])
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                gps_log = [lat, lon]
                hist = np.array(ast.literal_eval(row["hist"]), dtype=np.float32) #idk bro wtf is this 

                dist = self.euclidean(gps, gps_log)
                if dist < min_distance:
                    min_distance = dist
                    self.hist_to_go_to = hist.copy()
                else:
                    self.get_logger.info("no cone less than min distance")


    def msg_publisher(self):
        #now we will go in all bbox per frame and take the cone with lowest z value and match with hsv as well
        #based on the log_file.csv closest gps coords will be selected and then its hsv will be noted now that hsv will be used to find for all the cones until you reach to the cone and deliver
        # hist_to_go_to = self.get_hist_to_go_to()
        # if len(hist_to_go_to) == 0:
        #     return

        #now we will filter all bbox per frame, such that only bbox whose hsvs lie within a threshold (cosine similarity) exist and then just pick the lowest z valu
        similarity_thres = 0.5
        filterd_bbox_per_frame = []
        for i, cone in enumerate(self.save_data):
            current_hist = cone[3]
            dist = self.chi_square(self.hist_to_go_to, current_hist)
            similarity = np.exp(-dist)
            if similarity > similarity_thres:
                filterd_bbox_per_frame.append([cone[0], cone[1], cone[2]])
            
        min_z = float("inf")
        chosen_cx = None
        chosen_cy = None
        chosen_depth = None
        for i, cone in enumerate(filterd_bbox_per_frame):
            current_z = cone[2]
            if current_z < min_z:
                min_z = current_z
                chosen_cx = cone[0]
                chosen_cy = cone[1]
                chosen_depth = cone[2]
    
        msg = Float32MultiArray()
        msg.data = [chosen_cx, chosen_cy, chosen_depth]
        self.pub.publish(msg)

    
    def timer_callback(self):
        if self.left_frame is None:
            self.get_logger().info(f"{CYAN}[WAITING] for zed boi to give frames, TURN ON ZEDWRAPPER{RESET}")

        if self.depth_frame is None:
            self.get_logger().info("[WAITING] for zed depth image")
           
        
        if self.state != self.last_state:
            if self.state:
                self.get_logger().info(f"{MAGENTA}{BOLD}[AUTONOMOUS MODE]{RESET}")
                self.get_hist_to_go_to()
            else:
                self.get_logger().info(f"{RED}{BOLD}[MANUAL MODE]{RESET}")

            self.last_state = self.state
        
        if self.state:
                self.process()
                self.msg_publisher()

    #Visuals
    def show_popup(self, frame, cx, cy, H, S, V):
        frame_disp = frame.copy()

        # Draw red dot at cone center
        cv2.circle(frame_disp, (int(cx), int(cy)), 8, (0, 0, 255), -1)

        # --- Create color patch ---
        patch_color = np.uint8([[[H, S, V]]])
        patch_bgr = cv2.cvtColor(patch_color, cv2.COLOR_HSV2BGR)[0][0].tolist()

        legend_x = 20
        legend_y = 20

        # Legend background
        cv2.rectangle(frame_disp,
                    (legend_x - 10, legend_y - 10),
                    (legend_x + 200, legend_y + 70),
                    (30, 30, 30),
                    -1)

        # Color patch square
        cv2.rectangle(frame_disp,
                    (legend_x, legend_y),
                    (legend_x + 50, legend_y + 50),
                    patch_bgr,
                    -1)

        # HSV text
        text = f"H:{H:.1f}  S:{S:.1f}  V:{V:.1f}"
        cv2.putText(frame_disp,
                    text,
                    (legend_x + 60, legend_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA)

        # LOGGED message
        cv2.putText(frame_disp,
                    "LOGGED!",
                    (legend_x, legend_y + 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)

        cv2.imshow("LOG CONFIRMATION", frame_disp)

        # wait 1 ms for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # press 'q' to close window
            cv2.destroyWindow("LOG CONFIRMATION")


def main(args=None):
    rclpy.init(args=args)
    node = YoloPub()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
