import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from sensor_msgs.msg import Image, NavSatFix
from std_srvs.srv import Trigger
from rcl_interfaces.msg import SetParametersResult
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
from collections import deque

#colour for coloured prints brooo!!!
RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"


# preprocessing stuff / retinex utils
def single_scale_retinex(img, sigma):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    retinex = np.log10(img + 1.0) - np.log10(blur + 1.0)
    return retinex

def multi_scale_retinex(img, sigmas):
    retinex = np.zeros_like(img, dtype=np.float32)
    for sigma in sigmas:
        retinex += single_scale_retinex(img, sigma)
    return retinex / len(sigmas)

def colour_restoration(img, alpha=125, beta=46):
    img_sum = np.sum(img, axis=2, keepdims=True)
    return beta * (np.log10(alpha * img) - np.log10(img_sum + 1.0))

def msrcr(img, sigmas=(15, 80, 250), alpha=125, beta=46):
    img = img.astype(np.float32) + 1.0
    msr = multi_scale_retinex(img, sigmas)
    cr = colour_restoration(img, alpha, beta)
    msrcr_img = msr * cr

    # Normalize each channel independently
    for c in range(3):
        channel = msrcr_img[:, :, c]
        channel = (channel - channel.min()) / (channel.max() - channel.min())
        msrcr_img[:, :, c] = channel * 255

    return np.uint8(msrcr_img)


    
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Linear(128, 6)  # update if your dataset has a different number of colours

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class YoloPub(Node):
    def __init__(self):
        super().__init__("yolo_publisher")

        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = YOLO("./weights/model_inside.pt")
        self.model.to(self.device)
        print(torch.cuda.is_available())
        self.get_logger().warn(f"[DEPLOY] yolo deployed on device : {self.model.device} succesfully!")

        # Load encoder checkpoint
        self.cnn = SimpleCNN().to(self.device)
        cnn_checkpoint_path = "./weights/best_cnn_retinex_2.pth"
        self.cnn.load_state_dict(torch.load(cnn_checkpoint_path, map_location=self.device))
        self.cnn.eval()
        self.get_logger().warn(f"[DEPLOY] CNN colour classifier deployed successfully!")

        # Transform for CNN colour classification
        self.cnn_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        # Color class names
        self.class_names = ['blue', 'cyan', 'green', 'orange', 'red', 'yellow']


        self.sub_left = self.create_subscription(Image, "/zed/zed_node/rgb/color/rect/image", self.left_callback, 10)
        self.sub_depth = self.create_subscription(Image, "/zed/zed_node/depth/depth_registered", self.depth_callback, 10)
        self.sub_state = self.create_subscription(Bool, "/state", self.state_callback, 10)
        self.sub_gps = self.create_subscription(NavSatFix, "/gps/fix", self.gps_callback, 10)

        #Service and Param
        self.log_service = self.create_service(Trigger, "log_service", self.log_callback)
        self.declare_parameter("colour_override", "None")
        self.add_on_set_parameters_callback(self.colour_override_callback)

        self.pub = self.create_publisher(Float32MultiArray, "/cone_bbox", 10)

        self.timer = self.create_timer(0.1, self.timer_callback) #10 hz

        self.left_frame = None 
        self.depth_frame = None
        self.object_number = 1
        self.gps = None
        self.state = False
        self.save_data = []
        self.last_state = None
        self.depth_queue = deque(maxlen=10)

        today = date.today()
        self.file_path = f"/home/orin/log_{today}.csv"

        self.colour_to_go_to = None

                  
    def left_callback(self, msg):
        self.left_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, msg.encoding)  
    
    def state_callback(self, msg):
        self.state = msg.data
    
    def gps_callback(self, msg : NavSatFix):
        if msg is not None:
            self.gps = msg

    def classify_colour_with_retinex(self, cropped_image):
        """Classify colour of cropped cone image using Retinex preprocessing"""
        if cropped_image is None or cropped_image.size == 0:
            return None, None
        
        try:
            # Apply Retinex preprocessing
            retinex_img = msrcr(cropped_image)
            
            # Convert to PIL and transform
            image_rgb = cv2.cvtColor(retinex_img, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(image_rgb)
            img_tensor = self.cnn_transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.cnn(img_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_class = np.argmax(probs)
                confidence = probs[pred_class]
            
            colour_name = self.class_names[pred_class]
            return colour_name, confidence
        
        except Exception as e:
            self.get_logger().error(f"Error classifying colour: {e}")
            return None, None


    def log_callback(self, request, response):
        
        self.process()

        if len(self.save_data) == 0:
            response.success = False
            response.message = "CONE NOT DETECTED, move rover little away from cone"
            self.get_logger().info(f"{RED}Service called but no cones detected{RESET}")

            return response

        self.get_logger().info("Preparing to log colour.....")

        min_z = None
        cone_to_log_idx = None
        for i, data in enumerate(self.save_data):
            z = data[2]
            if i==0:
                min_z = z
                cone_to_log_idx = 0
            if z > 0 and z < min_z:
                min_z = z
                cone_to_log_idx = i 

        cx, cy, z, colour_name, colour_conf = self.save_data[cone_to_log_idx]

        self.get_logger().info("Preparing to log GPS coords.....")
        if self.gps is None:
            self.get_logger().warn("Not getting GPS from topic, GPS message is still None!!!")
            latitude = 0.0
            longitude = 0.0
        else:
            latitude = self.gps.latitude
            longitude = self.gps.longitude

        self.get_logger().info(f"Logging object_number {self.object_number} at GPS ({latitude}, {longitude})!")
        self.get_logger().info(f"Logging colour: {colour_name} (confidence: {colour_conf:.2%})")


        file_exists = os.path.exists(self.file_path)
        with open(self.file_path, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(["object_number", "latitude", "longitude", "colour_name", "colour_conf"])

            writer.writerow([self.object_number, latitude, longitude, colour_name, colour_conf])

        self.object_number += 1

        self.get_logger().info(f"{GREEN}{BOLD}ONE STEP CLOSER TO WINNING IRC LETS GO!!!{RESET}")
        #NOTE comment out for popup while loggin
       # self.show_popup(self.left_frame, cx, cy, H, S, V)
        
        response.success = True
        response.message = "logged sucessfully"
        return response
    
    def colour_override_callback(self, params):
        """
        Called whenever a parameter is changed.
        """
        for param in params:
            if param.name == "colour_override":
                if param.value.lower() == "none":
                    self.colour_to_go_to = None
                    self.get_logger().info(f"{YELLOW}Manual override cleared{RESET}")
                else:
                    self.colour_to_go_to = param.value.lower()
                    self.get_logger().info(f"{MAGENTA}Manual override set to: {self.colour_to_go_to}{RESET}")
        return SetParametersResult(successful=True)

    #HELPERS
    
    def euclidean(self, p, q):
        return sum((a - b) ** 2 for a, b in zip(p, q))
    
    
    def reset_target(self):
        """Call this after successful delivery to find next target"""
        self.colour_to_go_to = None
    

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

            colour_name, colour_conf = self.classify_colour_with_retinex(cropped)

            dpatch = depth[cy-2:cy+3, cx-2:cx+3]
            dpatch = dpatch[np.isfinite(dpatch)]
            if len(dpatch) == 0:
                continue
            z = float(np.median(dpatch))

            #if aspect ratio is fcked up, skip
            aspect_ratio = w / h
            #we have to test and see the conditions 

            all_bbox_per_frame.append([float(cx), float(cy), z, colour_name, colour_conf])

        self.save_data = all_bbox_per_frame.copy() #saving to write in csv if service requests and will be used in publishing

        num_cones = len(all_bbox_per_frame)
        
        self.get_logger().info(f"{GREEN}[MODEl] 🥳 deteced {num_cones} cones{RESET}")
    
    def get_colour_to_go_to(self):
        """Get target colour from nearest logged cone"""
        if self.gps is None:
            return None
        
        curr_lat = self.gps.latitude
        curr_lon = self.gps.longitude
        gps = [curr_lat, curr_lon]
        min_distance = float("inf")
        colour_to_go_to = None
        
        if not os.path.exists(self.file_path):
            return None
        
        with open(self.file_path, newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                gps_log = [lat, lon]
                colour_name = row["colour_name"]
                
                # dist = self.euclidean(gps, gps_log)
                dist = math.dist(gps, gps_log)
                if dist < min_distance:
                    min_distance = dist
                    colour_to_go_to = colour_name
        
        return colour_to_go_to


    def msg_publisher(self):
        if self.colour_to_go_to is None:
            self.colour_to_go_to = self.get_colour_to_go_to()
            if self.colour_to_go_to is None:
                self.get_logger().info(f"{YELLOW}give me colour to go to babygirl{RESET}")
                return
1
        # Filter cones by colour match
        filterd_bbox_per_frame = []
        for cone in self.save_data:
            cx, cy, z, colour_name, colour_conf = cone

            if colour_name is None:
                self.get_logger().info(f"{RED}colour name is none{RESET}")
                continue
            
            if colour_name == self.colour_to_go_to and colour_conf > 0.5:
                filterd_bbox_per_frame.append([cx, cy, z, colour_conf])

        #now we will filter all bbox per frame, such that only bbox whose embedding lie within a threshold (cosine similarity) exist and then just pick the lowest z valu


        if len(filterd_bbox_per_frame) == 0:
            self.get_logger().info(f"{RED}No matching cones found after filtering{RESET}")
            return
            
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

        self.depth_queue.append(chosen_depth)
        chosen_depth_rect = sum(self.depth_queue) / len(self.depth_queue)
    
        msg = Float32MultiArray()
        msg.data = [chosen_cx, chosen_cy, chosen_depth_rect]
        self.pub.publish(msg)

    
    def timer_callback(self):
        self.state = True
        if self.left_frame is None:
            self.get_logger().info(f"{CYAN}[WAITING] for zed boi to give frames, TURN ON ZEDWRAPPER{RESET}")

        if self.depth_frame is None:
            self.get_logger().info(f"{CYAN}[WAITING] for zed depth image")
        
        if self.state != self.last_state:
            if self.state:
                self.get_logger().info(f"{MAGENTA}{BOLD}[AUTONOMOUS MODE]{RESET}")
                self.get_logger().info(f"{BOLD}TARGET CONE : {self.colour_to_go_to}")


            else:
                self.get_logger().info(f"{RED}{BOLD}[MANUAL MODE]{RESET}")
                self.reset_target() #resets embedding_to_go_to
            self.last_state = self.state

        if self.state:
            self.process()
            self.msg_publisher()

def main(args=None):
    rclpy.init(args=args)
    node = YoloPub()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
