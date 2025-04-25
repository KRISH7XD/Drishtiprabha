import time
import threading
import queue
import os
from utils.ultrasonic_sensor import get_distance
from utils.camera import Camera  
from ultralytics import YOLO

# Initialize TTS queue
tts_queue = queue.Queue()

# Load YOLOv8n model
model = YOLO('yolov8n.pt')  # Ensure the path is correct

# Initialize camera using custom Camera class
camera = Camera(resolution=(320, 240))

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

try:
    while True:
        # Capture image from the camera
        frame = camera.capture_image()
        if frame is None:
            continue  # Skip if frame capture failed

        # Perform object detection
        results = model(frame)
        detections = results[0].boxes
        labels = results[0].names

        # Measure distance using the ultrasonic sensor
        distance = get_distance()  # Clean: only use this from utils

        # Process detections
        for box in detections:
            cls_id = int(box.cls[0])
            label = labels[cls_id]
            x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2

            # Determine position
            frame_width = frame.shape[1]
            if x_center < frame_width / 3:
                position = "left"
            elif x_center > 2 * frame_width / 3:
                position = "right"
            else:
                position = "center"

            # Generate message based on distance
            if distance < 50:
                message = f"Warning: {label} ahead at {distance:.1f} centimeters on your {position}."
            elif distance < 150:
                message = f"{label} detected {distance:.1f} centimeters ahead on your {position}."
            else:
                message = f"Path is clear on your {position}."

            # Add message to TTS queue
            if tts_queue.empty():
                tts_queue.put(message)

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    camera.release_camera()
    tts_queue.put(None)
    tts_thread.join()
