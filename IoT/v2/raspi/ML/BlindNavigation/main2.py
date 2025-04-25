import cv2
import time
import threading
import queue
import os
import RPi.GPIO as GPIO
from ultralytics import YOLO

# GPIO setup for HC-SR04
TRIG = 23  # GPIO pin 23
ECHO = 24  # GPIO pin 24

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# Initialize TTS queue
tts_queue = queue.Queue()

# Load YOLOv8n model
model = YOLO('yolov8n.pt')  # Ensure this path is correct

# Function to measure distance using HC-SR04
def measure_distance():
    GPIO.output(TRIG, False)
    time.sleep(0.05)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    pulse_start = time.time()
    timeout = pulse_start + 0.04
    while GPIO.input(ECHO) == 0 and pulse_start < timeout:
        pulse_start = time.time()

    pulse_end = time.time()
    timeout = pulse_end + 0.04
    while GPIO.input(ECHO) == 1 and pulse_end < timeout:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150  # Speed of sound: 34300 cm/s
    distance = round(distance, 2)
    return distance

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
cap = cv2.VideoCapture(0)  # Adjust if necessary

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Perform object detection
        results = model(frame)
        detections = results[0].boxes
        labels = results[0].names

        # Measure distance
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
    cap.release()
    GPIO.cleanup()
    tts_queue.put(None)
    tts_thread.join()
