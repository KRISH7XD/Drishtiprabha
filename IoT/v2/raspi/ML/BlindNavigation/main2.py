import cv2
import time
import threading
import queue
import os
import sys

# Path for custom utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

# Import the shared ultrasonic function
from utils.ultrasonic_sensor import get_distance
from ultralytics import YOLO

# Initialize TTS queue
tts_queue = queue.Queue()

# Load YOLOv8n model (make sure the file is in the correct path)
model = YOLO('yolov8n.pt')

# Function to speak text using eSpeak
def speak(text):
    os.system(f'espeak "{text}" --stdout | aplay')

# Thread function to process TTS queue
def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        speak(text)
        tts_queue.task_done()

# Start TTS thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# Initialize camera
cap = cv2.VideoCapture(0)  # Adjust index if needed

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Run object detection
        results = model(frame)
        detections = results[0].boxes
        labels = results[0].names

        # Get distance from ultrasonic sensor
        distance = get_distance()

        # Process detections
        for box in detections:
            cls_id = int(box.cls[0])
            label = labels[cls_id]
            x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2

            frame_width = frame.shape[1]
            if x_center < frame_width / 3:
                position = "left"
            elif x_center > 2 * frame_width / 3:
                position = "right"
            else:
                position = "center"

            # Generate audio message based on proximity
            if distance < 50:
                message = f"Warning: {label} ahead at {int(distance)} centimeters on your {position}."
            elif distance < 150:
                message = f"{label} detected {int(distance)} centimeters ahead on your {position}."
            else:
                message = f"Path is clear on your {position}."

            # Avoid message overlap
            if tts_queue.empty():
                tts_queue.put(message)

        time.sleep(0.1)  # Slight delay for responsiveness

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    cap.release()
    tts_queue.put(None)
    tts_thread.join()
