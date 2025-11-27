"""
this files takes 1 image every 10 seconds and stops once total 10 images has ben captured
"""

import pyzed.sl as sl
import time
import cv2
import os

def main():
    # Create and open the ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # 1280x720
    init_params.camera_fps = 30
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera.")
        exit(1)

    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()

    # Create output directory
    output_dir = "zed_captures"
    os.makedirs(output_dir, exist_ok=True)

    print("Capturing an image every 10 seconds for 100 seconds...")
    start_time = time.time()

    for i in range(0, 100, 10):
        # Grab frame
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            
            # Convert to numpy array
            img_np = image.get_data()
            
            # Save image
            filename = os.path.join(output_dir, f"zed_image_{i:03d}.png")
            cv2.imwrite(filename, img_np)
            print(f"Saved {filename}")
        else:
            print("Failed to grab frame")

        # Wait for next capture (10 seconds interval)
        time.sleep(10)

        # Stop after 100 seconds
        if time.time() - start_time >= 100:
            break

    print("Capture completed.")
    zed.close()

if __name__ == "__main__":
    main()
