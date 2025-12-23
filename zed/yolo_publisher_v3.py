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
import torch.nn as nn
from torchvision import transforms
from PIL import Image as PILImage
import csv
from datetime import date
import os
import math

#color for coloured prints brooo!!!
RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"

class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)


class YoloPub(Node):
    def __init__(self):
        super().__init__("yolo_publisher")

        self.bridge = CvBridge()

        self.model = YOLO("./weights/model_inside.pt")
        self.model.to("cuda")
        print(torch.cuda.is_available())
        self.get_logger().warn(f"[DEPLOY] yolo deployed on device : {self.model.device} succesfully!")

        # Load encoder checkpoint
        self.encoder = CNN1().to("cuda")
        checkpoint_path = "./weights/simclr_checkpoint_20dec_730.pth"
        checkpoint = torch.load(checkpoint_path)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.encoder.eval()
        self.get_logger().warn(f"[DEPLOY] SimCLR encoder deployed on device : {self.model.device} succesfully!")

        # Transform for encoder
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])


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

        self.embedding_to_go_to = None
        self.hue_to_go_to = None
                  
    def left_callback(self, msg):
        self.left_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, msg.encoding)  
    
    def state_callback(self, msg):
        self.state = msg.data
    
    def gps_callback(self, msg : NavSatFix):
        self.gps = msg

    def get_embedding(self, cropped_image):
        """Get embedding for cropped cone image"""
        if cropped_image is None or cropped_image.size == 0:
            return None
        
        try:
            # Convert to PIL
            image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(image_rgb)
            
            # Transform and get embedding
            img_tensor = self.transform(pil_img).unsqueeze(0).to("cuda")
            
            with torch.no_grad():
                embedding = self.encoder(img_tensor).cpu().numpy()[0]
            
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding
        
        except Exception as e:
            self.get_logger().error(f"Error getting embedding: {e}")
            return None

    def compute_similarity(self, emb1, emb2, h1, h2):
        """Compute weighted similarity (cosine + hue distance)"""
        if emb1 is None or emb2 is None:
            return None
        
        # Cosine similarity
        cosine_similarity = np.dot(emb1, emb2)
        
        # Hue distance (normalized)
        dist = min(abs(h2 - h1), 360 - abs(h2 - h1))
        dist_normalized = dist / 180.0
        
        # Weighted combination
        similarity = (0.6 * cosine_similarity) + (0.4 * (1.0 - dist_normalized))
        
        return similarity

    def log_callback(self, request, response):
        
        self.process()

        if len(self.save_data) == 0:
            response.success = False
            response.message = "CONE NOT DETECTED, move rover little away from cone"
            self.get_logger().info(f"{RED}Service called but no cones detected{RESET")

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

        cx, cy, z, H, embedding = self.save_data[cone_to_log_idx]

        self.get_logger().info("Preparing to log GPS coords.....")
        if self.gps is None:
            self.get_logger().warn("Not getting GPS from topic, GPS message is still None!!!")
            latitude = 0.0
            longitude = 0.0
        else:
            latitude = self.gps.latitude
            longitude = self.gps.longitude

        self.get_logger().info(f"Logging object_number {self.object_number} at GPS ({latitude}, {longitude})!")
        self.get_logger().info(f"Logging color H={H}, and embedding")

        embedding_str = ','.join(map(str, embedding.tolist()))



        file_exists = os.path.exists(self.file_path)
        with open(self.file_path, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["object_number", "latitude", "longitude", "H", "embedding"])

            writer.writerow([self.object_number, latitude, longitude, H, embedding_str])


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
    
    def get_cosine_similarity(self, list1, list2):
        if len(list1) != len(list2):
            raise ValueError("Lists must have the same length")

        dot_product = sum(x * y for x, y in zip(list1, list2))
        magnitude_a = math.sqrt(sum(x**2 for x in list1))
        magnitude_b = math.sqrt(sum(y**2 for y in list2))

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)
    
    def reset_target(self):
        """Call this after successful delivery to find next target"""
        self.embedding_to_go_to = None
        self.hue_to_go_to = None
    

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
            self.get_logger().info(f"{YELLOW}[MODEL] 😢 i dont see any cone bro i am blind{RESET}")
            return

        all_bbox_per_frame = []
        for box in results.boxes:
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            xmin, ymin, xmax, ymax = bbox
            cx = int((xmin + xmax) / 2)
            cy = int((ymin + ymax) / 2)
            w = xmax - xmin
            h = ymax - ymin

            cropped = frame[ymin:ymax, xmin:xmax]

            embedding = self.get_embedding(cropped)

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

            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            H = float(np.mean(hsv_roi[:, :, 0]))
            S = float(np.mean(hsv_roi[:, :, 1]))
            V = float(np.mean(hsv_roi[:, :, 2]))

            all_bbox_per_frame.append([float(cx), float(cy), z, H, embedding])

        self.save_data = all_bbox_per_frame.copy() #saving to write in csv if service requests and will be used in publishing

        num_cones = len(all_bbox_per_frame)
         
        # if num_cones > 10:
        #     self.get_logger().info(f"{RED}o shit it detected more than 10 cones 💀{RESET}")
        #     return
        
        self.get_logger().info(f"{GREEN}[MODEl] 🥳 deteced {num_cones} cones{RESET}")

    def get_embedding_to_go_to(self):
        curr_lat = self.gps.latitude
        curr_lon = self.gps.longitude
        gps = [curr_lat, curr_lon]
        min_distance = float("inf")
        embedding_to_go_to = []
        hue_to_go_to = []
        with open(self.file_path, newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                object_number = int(row["object_number"])
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                gps_log = [lat, lon]
                H = float(row["H"])
                embedding = np.array([float(x) for x in row["embedding"].split(',')])
                dist = self.euclidean(gps, gps_log)
                if dist < min_distance:
                    min_distance = dist
                    embedding_to_go_to = embedding.copy()
                    hue_to_go_to = H
        
        return embedding_to_go_to, hue_to_go_to


    def get_hsv_to_go_to(self):
        curr_lat = self.gps.latitude
        curr_lon = self.gps.longitude
        gps = [curr_lat, curr_lon]
        min_distance = float("inf")
        hsv_to_go_to = []
        with open(self.file_path, newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                object_number = int(row["object_number"])
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                gps_log = [lat, lon]
                H = float(row["H"])
                S = float(row["S"])
                V = float(row["V"])
                hsv = [H, S, V]

                dist = self.euclidean(gps, gps_log)
                if dist < min_distance:
                    min_distance = dist
                    hsv_to_go_to = hsv.copy()

        return hsv_to_go_to


    def msg_publisher(self):
        #now we will go in all bbox per frame and take the cone with lowest z value and match with hsv as well
        #based on the log_file.csv closest gps coords will be selected and then its hsv will be noted now that hsv will be used to find for all the cones until you reach to the cone and deliver
        if self.embedding_to_go_to is None:
            self.embedding_to_go_to, self.hue_to_go_to = self.get_embedding_to_go_to() #we have to reset this as well which we did while turning into manual mode 

        #NOTE : if you press manual mode from some random place it will take embedding of the closest cone from that random GPS 

        #now we will filter all bbox per frame, such that only bbox whose embedding lie within a threshold (cosine similarity) exist and then just pick the lowest z valu
        similarity_thres = 0.5
        filterd_bbox_per_frame = []
        for i, cone in enumerate(self.save_data):
            cx, cy, z, embedding, H = cone
            current_similarity = self.compute_similarity(self.embedding_to_go_to, embedding, self.hue_to_go_to, H)
            if current_similarity > similarity_thres:
                filterd_bbox_per_frame.append([cx, cy, z, current_similarity])

        if len(filterd_bbox_per_frame) == 0:
            self.get_logger().info(f"{RED}No matching cones found after filtering{RESET}")
            return
            
        min_z = float("inf")
        min
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
                self.embedding_to_go_to, self.hue_to_go_to = self.get_embedding_to_go_to()

            else:
                self.get_logger().info(f"{RED}{BOLD}[MANUAL MODE]{RESET}")
                self.reset_target() #resets embedding_to_go_to

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
