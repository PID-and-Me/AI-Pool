import cv2
import random
from ultralytics import YOLO
import time
import numpy as np

# Load the pre-trained YOLO model
model = YOLO('/home/jrusso/yolo/models/240_yolov8n_full_integer_quant_edgetpu.tflite')

imgsz = 256

# Open video capture
cap = cv2.VideoCapture(0)

# Read class list
with open('/home/jrusso/yolo/classes/coco128.txt', "r") as file:
    class_list = file.read().strip().split("\n")

random.seed(42)
class_colors = {class_name: (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                 for class_name in class_list}

frame_count = 0
fps_values = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of Input")
        break

    frame = cv2.resize(frame, (1280, 720))

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    start_time = time.time()

    # YOLO Prediction
    results = model.predict(frame, imgsz=imgsz, conf=.25)
    boxes = results[0].boxes.data.cpu().numpy() if hasattr(results[0].boxes, 'data') else np.array([])

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        conf = float(box[4])  # Ensure confidence is displayed as float
        class_id = int(box[5])
        class_name = class_list[class_id] if class_id < len(class_list) else "Unknown"
        color = class_colors.get(class_name, (0, 255, 0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + 115, y1), color, -1)
        cv2.putText(frame, f'{class_name} {conf:.2f}', (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    # FPS Calculation
    fps = 1 / (time.time() - start_time)
    fps_values.append(fps)
    avg_fps = np.mean(fps_values[-30:])
    cv2.putText(frame, f'FPS: {avg_fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 1)

    cv2.imshow("Output", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
