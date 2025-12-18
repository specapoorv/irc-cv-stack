import cv2
from ultralytics import YOLO

# ------------------------
# CONFIG
# ------------------------
MODEL_PATH = "/home/specapoorv/irc-cv-stack/weights/model_inside.pt"          # change to your trained model
IMAGE_PATH = "/home/specapoorv/irc-cv-stack/data/cone_inside.v2i.yolov8/test/images/1000263013_jpg.rf.6d50a32ca1204b25b0d0d2f3d9cef213.jpg"            # change to your test image

# ------------------------
# LOAD MODEL
# ------------------------
model = YOLO(MODEL_PATH)

# ------------------------
# READ IMAGE
# ------------------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

# ------------------------
# RUN INFERENCE
# ------------------------
results = model(img, conf=0.7)[0]  # first prediction

# ------------------------
# DRAW DETECTIONS
# ------------------------
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    cls  = int(box.cls[0])
    label = f"{model.names[cls]} {conf:.2f}"

    # draw box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    
    # label
    cv2.putText(img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,255,0), 2)

# ------------------------
# SHOW RESULT
# ------------------------
cv2.imshow("YOLO Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# OPTIONAL: save output
cv2.imwrite("output.jpg", img)
