import cv2
import time
from ultralytics import YOLO
import pyzed.sl as sl

# ---- Settings ----
MODEL_PATH = "irc-cv-stack/weights/model2.pt"      # path to YOLO model
INFERENCE_RATE = 5              # Hz (YOLO runs every N frames per second)
DISPLAY = True                  # show realtime visualization
SAVE_OUTPUT = False             # set True to save output video

# ---- Initialize YOLO ----
model = YOLO(MODEL_PATH)

# ---- Initialize ZED ----'
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED camera.")
    exit(1)

runtime_params = sl.RuntimeParameters()
image_zed = sl.Mat()

# ---- Optional video writer ----
out_writer = None

if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_writer = cv2.VideoWriter('output.avi', fourcc, 30, (1280, 720))

# ---- Inference loop ----
print("Running... Press 'q' to quit.")

last_infer_time = 0
while True:
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        frame = image_zed.get_data()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        # --- Run YOLO inference at a fixed rate ---
        current_time = time.time()
        if current_time - last_infer_time >= 1.0 / INFERENCE_RATE:
            results = model.predict(source=frame, verbose=False)
            annotated_frame = results[0].plot()
            last_infer_time = current_time
        else:
            annotated_frame = frame

        # --- Display / Save ---
        if DISPLAY:
            cv2.imshow("ZED + YOLOv8", annotated_frame)

        if SAVE_OUTPUT and out_writer is not None:
            out_writer.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

        # --- Break condition ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

zed.close()
if out_writer:
    out_writer.release()
cv2.destroyAllWindows()
