import cv2
import random
from ultralytics import YOLO
import time

# Load the pre-trained YOLO model
model = YOLO('/home/jrusso/yolo/models/yolov5nu_full_integer_quant_edgetpu.tflite')

imgsz = 256

# Open video capture
cap = cv2.VideoCapture('/home/jrusso/yolo/inputs/people.mp4')

# Read class list
with open('/home/jrusso/yolo/classes/coco128.txt', "r") as file:
    class_list = file.read().split("\n")

# Generate consistent bright colors for each class
random.seed(42)  # Ensures consistent colors every run
class_colors = {class_name: (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                 for class_name in class_list}

frame_count = 0

# Start video processing
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of Input")
        break

    # Preprocessing: Resize frame for faster processing
    frame = cv2.resize(frame, (800, 600))

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    # Start timer for FPS calculation
    start_time = time.time()

    # Run the YOLO model to get predictions
    results = model.predict(frame, imgsz=imgsz)
    boxes = results[0].boxes.data.numpy()  # Get prediction results as NumPy array

    # Iterate through detections and draw boxes
    for box in boxes:
        x1, y1, x2, y2, conf, class_id = map(int, box)
        class_name = class_list[class_id]
        color = class_colors[class_name]

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1-1, y1 - 20), (x1 + text_width, y1), color, -1)  # Background same as class color
        cv2.putText(frame, class_name, (x1 + 10, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # White text

    # Calculate FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f'FPS:{fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 1)

    # Show the processed frame
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
