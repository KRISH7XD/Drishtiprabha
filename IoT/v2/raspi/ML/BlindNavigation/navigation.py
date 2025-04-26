import cv2
import numpy as np
from ultralytics import YOLO
import RPi.GPIO as GPIO
import time
import pyttsx3
from queue import Queue
import threading
import torch
from torchvision import transforms
import asyncio
from heapq import heappop, heappush

# Initialize YOLOv8 Nano model
model = YOLO("yolov8n.pt")  # Download from Ultralytics or use pretrained weights

# Initialize Pi Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize ultrasonic sensor
GPIO.setmode(GPIO.BCM)
TRIG = 23  # Ultrasonic trigger pin
ECHO = 24  # Ultrasonic echo pin
VIBRATION = 18  # Vibration motor pin
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.setup(VIBRATION, GPIO.OUT)

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speech rate
tts_queue = Queue()
tts_lock = threading.Lock()

# Initialize MiDaS for depth estimation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
device = torch.device("cpu")  # Use CPU on Raspberry Pi
midas.to(device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Occupancy grid parameters
GRID_SIZE = (10, 10)  # 10x10 meters, 1 cell = 1 meter
GRID_RESOLUTION = 1.0  # 1 meter per cell
user_pos = [0, 0]  # User starts at (0, 0)
target_pos = [5, 0]  # Target (e.g., room exit) at (5, 0) meters
occupancy_grid = np.zeros(GRID_SIZE, dtype=np.float32)

def get_ultrasonic_distance():
    """Measure distance using HC-SR04 ultrasonic sensor."""
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    pulse_start = time.time()
    pulse_end = time.time()

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150  # Speed of sound in cm
    distance = round(distance, 2)
    if 2 < distance < 400:  # Valid range for HC-SR04
        return distance
    return None

def get_depth_map(frame):
    """Estimate depth using MiDaS."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        depth = midas(img)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1), size=(480, 640), mode="bicubic", align_corners=False
        ).squeeze()
    
    depth = depth.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())  # Normalize
    return depth

def is_low_light(frame):
    """Detect low-light conditions based on frame brightness."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < 30  # Threshold for low light

def update_occupancy_grid(objects, depth_map, frame_shape):
    """Update 2D occupancy grid with detected objects."""
    global occupancy_grid
    occupancy_grid.fill(0)  # Reset grid

    for label, center_x, depth_meters in objects:
        # Map depth to grid coordinates (assume user at (0, 0))
        grid_x = int(depth_meters / GRID_RESOLUTION)
        grid_y = int((center_x - 0.5) * 4)  # Map x-position to ±2 meters
        grid_x = min(max(grid_x, 0), GRID_SIZE[0] - 1)
        grid_y = min(max(grid_y + GRID_SIZE[1] // 2, 0), GRID_SIZE[1] - 1)
        occupancy_grid[grid_x, grid_y] = 1.0  # Mark as occupied

def a_star(start, goal, grid):
    """A* path planning algorithm."""
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current_f, current = heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < GRID_SIZE[0] and 0 <= neighbor[1] < GRID_SIZE[1]:
                if grid[neighbor[0], neighbor[1]] > 0.5:  # Obstacle
                    continue
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

    return []  # No path found

def get_navigation_instruction(path, current_pos):
    """Generate navigation instruction based on A* path."""
    if not path:
        return "No clear path to target. Stop and wait."
    
    next_pos = path[0]
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]

    if dx > 0:
        return "Move forward 1 meter."
    elif dx < 0:
        return "Move backward 1 meter."
    elif dy > 0:
        return "Turn right 90 degrees and move forward 1 meter."
    elif dy < 0:
        return "Turn left 90 degrees and move forward 1 meter."
    return "Continue forward."

def tts_worker():
    """Process TTS queue to avoid overlapping audio."""
    while True:
        message = tts_queue.get()
        if message is None:
            break
        with tts_lock:
            engine.say(message)
            engine.runAndWait()
        tts_queue.task_done()

def add_to_tts_queue(message, priority=False):
    """Add message to TTS queue, prioritizing critical messages."""
    if priority:
        while not tts_queue.empty():
            try:
                tts_queue.get_nowait()
                tts_queue.task_done()
            except:
                pass
    tts_queue.put(message)

def activate_vibration(duration=0.5):
    """Activate vibration motor for haptic feedback."""
    GPIO.output(VIBRATION, True)
    time.sleep(duration)
    GPIO.output(VIBRATION, False)

async def main():
    # Start TTS worker thread
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()

    last_speech_time = time.time()
    min_speech_interval = 2.0  # Minimum time between speeches
    last_vibration_time = time.time()
    min_vibration_interval = 1.0  # Minimum time between vibrations

    global user_pos

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Check for low-light conditions
        if is_low_light(frame):
            ultrasonic_dist = get_ultrasonic_distance()
            if ultrasonic_dist and ultrasonic_dist < 100:  # <1 meter
                message = f"Obstacle {ultrasonic_dist / 100:.1f} meters ahead. Stop."
                add_to_tts_queue(message, priority=True)
                activate_vibration()
            continue

        # Object detection with YOLOv8
        results = model(frame, verbose=False)
        detected_objects = []
        
        for box in results[0].boxes:
            label = model.names[int(box.cls)]
            confidence = box.conf.item()
            if confidence > 0.5:
                x, y, w, h = box.xywh[0].tolist()
                center_x = x / frame.shape[1]  # Normalized x position
                detected_objects.append((label, center_x))

        # Get depth map (skip if no objects to save CPU)
        depth_map = get_depth_map(frame) if detected_objects else None

        # Process detected objects
        objects_with_depth = []
        closest_object = None
        closest_distance = float('inf')
        closest_position = None

        for label, center_x in detected_objects:
            region_y = int(frame.shape[0] * 0.3)
            region_x = int(center_x * frame.shape[1])
            depth_value = depth_map[region_y, region_x]
            depth_meters = 1 / (depth_value + 0.1) * 5  # Rough scaling
            
            if depth_meters < closest_distance:
                closest_distance = depth_meters
                closest_object = label
                closest_position = center_x
            
            objects_with_depth.append((label, center_x, depth_meters))

        # Validate with ultrasonic sensor
        ultrasonic_dist = get_ultrasonic_distance()
        if ultrasonic_dist and closest_distance > ultrasonic_dist / 100:
            closest_distance = ultrasonic_dist / 100

        # Update occupancy grid
        update_occupancy_grid(objects_with_depth, depth_map, frame.shape)

        # Plan path using A*
        path = a_star(tuple(user_pos), tuple(target_pos), occupancy_grid)

        # Generate navigation instructions
        current_time = time.time()
        if closest_object and closest_distance < 1.0 and (current_time - last_speech_time) > min_speech_interval:
            direction = "left" if closest_position > 0.5 else "right"
            message = f"{closest_object} {closest_distance:.1f} meters ahead. Turn {direction} to avoid."
            add_to_tts_queue(message, priority=True)
            last_speech_time = current_time
            if (current_time - last_vibration_time) > min_vibration_interval:
                activate_vibration()
                last_vibration_time = current_time
        elif (current_time - last_speech_time) > min_speech_interval:
            message = get_navigation_instruction(path, user_pos)
            add_to_tts_queue(message)
            last_speech_time = current_time

        # Update user position (simplified: assume user follows instructions)
        if path:
            user_pos = list(path[0])

        # Control frame rate
        await asyncio.sleep(0.1)  # ~10 FPS

        # Exit on key press (for testing)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    tts_queue.put(None)  # Signal TTS thread to exit

# Run async main
if __name__ == "__main__":
    asyncio.run(main())