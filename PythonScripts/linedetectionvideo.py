import cv2
import random
from ultralytics import YOLO
import time

# Load the pre-trained YOLO model
model = YOLO('/home/jrusso/yolo/models/yolov5nu_full_integer_quant_edgetpu.tflite')

imgsz = 256

# Open video capture
cap = cv2.VideoCapture('/home/jrusso/yolo/inputs/people.mp4')

# Get input video's FPS
input_fps = cap.get(cv2.CAP_PROP_FPS)

# Read class list
with open('/home/jrusso/yolo/classes/coco128.txt', "r") as file:
    class_list = file.read().split("\n")

# Generate consistent bright colors for each class
random.seed(42)  # Ensures consistent colors every run
class_colors = {class_name: (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                 for class_name in class_list}

frame_count = 0
prev_time = time.time()

# Start video processing
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of Input")
        break

    # Preprocessing: Resize frame for faster processing
    frame = cv2.resize(frame, (1024, 600))

    frame_count += 1

    # Start timer for FPS calculation
    start_time = time.time()

    # Run the YOLO model
    results = model.predict(frame, imgsz=imgsz, verbose=True, conf=.05)
    boxes = results[0].boxes.data.numpy()  # Get prediction results as NumPy array

    class_positions = {}

    # Iterate through detections and draw boxes
    for box in boxes:
        x1, y1, x2, y2, conf, class_id = map(int, box)
        class_name = class_list[class_id]
        color = class_colors[class_name]

        # Track class positions for drawing connecting lines
        if class_name not in class_positions:
            class_positions[class_name] = []
        class_positions[class_name].append(((x1 + x2) // 2, (y1 + y2) // 2))

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_width, y1), color, -1)  # Background same as class color
        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # White text

    # Draw lines connecting instances of the same class
    for class_name, points in class_positions.items():
        color = class_colors[class_name]
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 2)

    # Calculate FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f'FPS:{fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 1)

    # Show the processed frame
    cv2.imshow("Output", frame)

    # Synchronize the frame rate with the input FPS
    elapsed_time = time.time() - prev_time
    wait_time = max(1 / input_fps - elapsed_time, 0)  # Ensure we don't wait negative time
    time.sleep(wait_time)  # Wait for the correct time between frames
    prev_time = time.time()  # Update the prev_time

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
