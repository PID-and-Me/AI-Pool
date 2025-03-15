from flask import Flask, Response
import cv2
import random
from ultralytics import YOLO
import time
import numpy as np

app = Flask(__name__)

# Load the pre-trained YOLO model
model = YOLO('/home/jrusso/yolo/models/240_yolov8n_full_integer_quant_edgetpu.tflite')

imgsz = 256
frame_count = 0
fps_values = []

# Open video capture
cap = cv2.VideoCapture(0)

# Check if video capture is successfully opened
if not cap.isOpened():
    print("Error: Couldn't open video capture.")
    exit()

# Read class list
with open('/home/jrusso/yolo/classes/coco128.txt', "r") as file:
    class_list = file.read().strip().split("\n")

random.seed(42)
class_colors = {class_name: (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                 for class_name in class_list}

def generate_frames():
    global frame_count
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))  # Reduce resolution for faster processing

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        start_time = time.time()

        # YOLO Prediction
        results = model.predict(frame, imgsz=imgsz, conf=.25)
        boxes = results[0].boxes.data.cpu().numpy() if hasattr(results[0].boxes, 'data') else np.array([])

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            conf = float(box[4])
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

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Run the Flask app on 10.0.0.27:5000
    app.run(host='10.0.0.27', port=5000, debug=False, threaded=True)
