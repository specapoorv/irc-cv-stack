import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class TwoImageColorMatch(Node):
    def __init__(self):
        super().__init__("two_image_color_match")
        self.bridge = CvBridge()
        self.model = YOLO("./model_inside.pt")
        self.model.to("cuda")
        print(f"YOLO deployed on {self.model.device}")

        self.sub_left = self.create_subscription(
            Image, "/zed/zed_node/rgb/color/rect/image", self.image_callback, 10
        )

        self.frame = None
        self.captures = []  # stores histograms of detected objects
        self.saved_crops = []  # stores cropped images for visualization

        self.get_logger().info("Press 'c' to capture an image (only 2 images needed)")

    def image_callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        display = self.frame.copy()
        cv2.putText(display, f"Captures: {len(self.captures)}/2. Press 'c' to capture",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("YOLO Color Match Test", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            if len(self.captures) < 2:
                self.capture_and_process()
            else:
                self.get_logger().info("Already captured 2 images!")

    def capture_and_process(self):
        if self.frame is None:
            self.get_logger().warn("No frame received yet!")
            return

        results = self.model(self.frame)[0]
        if len(results.boxes) == 0:
            self.get_logger().warn("No object detected!")
            return

        hist_list = []
        crops = []

        for idx, box in enumerate(results.boxes):
            xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy().astype(int)
            crop = self.frame[ymin:ymax, xmin:xmax]

            # --- Compute convex hull of the cone ---
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            mask = np.zeros_like(gray)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(largest_contour)  # convex hull
                cv2.drawContours(mask, [hull], -1, 255, -1)  # fill convex hull

            # apply mask
            masked_crop = cv2.bitwise_and(crop, crop, mask=mask)
            crops.append(masked_crop)

            # --- LAB histogram only on convex hull ---
            lab_crop = cv2.cvtColor(masked_crop, cv2.COLOR_BGR2LAB)
            a = lab_crop[:, :, 1]
            b = lab_crop[:, :, 2]
            hist = cv2.calcHist([a,b], [0,1], mask, [32,32], [0,256,0,256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_list.append(hist)

            cv2.imshow(f"Cropped_{len(self.captures)}_{idx}", masked_crop)

        self.captures.append(hist_list)
        self.saved_crops.append(crops)
        self.get_logger().info(f"Captured image {len(self.captures)}/2 with {len(hist_list)} objects")

        # If we have 2 captures, compare histograms
        if len(self.captures) == 2:
            self.compare_two_captures()

    def chi_square(self, h1, h2, eps=1e-10):
        return 0.5 * np.sum((h1 - h2) ** 2 / (h1 + h2 + eps))

    def compare_two_captures(self):
        self.get_logger().info("Comparing objects between the two captures...")
        hist1_list = self.captures[0]
        hist2_list = self.captures[1]

        for i, h1 in enumerate(hist1_list):
            for j, h2 in enumerate(hist2_list):
                dist = self.chi_square(h1, h2)
                similarity = np.exp(-dist)
                print(f"Object {i+1} in first image vs Object {j+1} in second image -> Similarity: {similarity:.4f}")

def main(args=None):
    rclpy.init(args=args)
    node = TwoImageColorMatch()
    rclpy.spin(node)
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
