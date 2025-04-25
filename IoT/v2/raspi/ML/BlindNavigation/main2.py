import time
import threading
import queue
import os
import sys
import RPi.GPIO as GPIO
from utils.ultrasonic_sensor import get_distance
from utils.camera import Camera  # Import the Camera class
from ultralytics import YOLO

# GPIO setup for HC-SR04 (ultrasonic sensor)
TRIG = 23  # GPIO pin 23
ECHO = 24  # GPIO pin 24

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# Initialize TTS queue
tts_queue = queue.Queue()

# Load YOLOv8n model
model = YOLO('yolov8n.pt')  # Ensure this path is correct

# Initialize camera (use the new Camera class)
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

# Function to measure distance using the ultrasonic sensor
def measure_distance():
    return get_distance()  # Use the imported get_distance function

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

        # Measure distance using ultrasonic sensor
        distance = measure_distance()

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
                message = f"Warning: {label} ahead at {distance} centimeters on your {position}."
            elif distance < 150:
                message = f"{label} detected {distance} centimeters ahead on your {position}."
            else:
                message = f"Path is clear on your {position}."

            # Add message to TTS queue
            if tts_queue.empty():
                tts_queue.put(message)

        time.sleep(0.1)  # Adjust for desired responsiveness

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    camera.release_camera()  # Release camera when done
    GPIO.cleanup()
    tts_queue.put(None)
    tts_thread.join()
