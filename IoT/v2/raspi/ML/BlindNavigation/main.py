import asyncio
import cv2
import numpy as np
from object_detection import ObjectDetector
from sensor import SensorManager
from path_planning import PathPlanner
from feedback import FeedbackManager
from utils.camera import Camera   # <-- Import your new Camera class
import yaml

async def main():
    # Load configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize components
    detector = ObjectDetector(config["model"]["yolo_weights"])
    sensor = SensorManager(
        trig_pin=config["sensor"]["trig_pin"],
        echo_pin=config["sensor"]["echo_pin"],
        vibration_pin=config["sensor"]["vibration_pin"]
    )
    planner = PathPlanner(
        grid_size=config["path_planning"]["grid_size"],
        grid_resolution=config["path_planning"]["grid_resolution"],
        target_pos=config["path_planning"]["target_pos"]
    )
    feedback = FeedbackManager(
        speech_rate=config["feedback"]["speech_rate"],
        min_speech_interval=config["feedback"]["min_speech_interval"],
        min_vibration_interval=config["feedback"]["min_vibration_interval"]
    )
    camera = Camera(resolution=(640, 480))  # <-- Initialize your Camera object

    user_pos = [0, 0]  # Starting position
    last_speech_time = 0
    last_vibration_time = 0

    try:
        while True:
            frame = camera.capture_image()  # <-- Capture image using new Camera class
            if frame is None:
                print("Failed to capture frame")
                break

            # Check for low-light conditions
            if sensor.is_low_light(frame):
                ultrasonic_dist = sensor.get_ultrasonic_distance()
                if ultrasonic_dist and ultrasonic_dist < 100:  # <1 meter
                    feedback.add_to_tts_queue(
                        f"Obstacle {ultrasonic_dist / 100:.1f} meters ahead. Stop.",
                        priority=True
                    )
                    sensor.activate_vibration()
                await asyncio.sleep(0.1)
                continue

            # Object detection
            detected_objects = detector.detect(frame)

            # Depth estimation
            depth_map = sensor.get_depth_map(frame) if detected_objects else None

            # Process objects with depth
            objects_with_depth = []
            closest_object = None
            closest_distance = float('inf')
            closest_position = None

            for label, center_x in detected_objects:
                if depth_map is not None:
                    depth_meters = sensor.estimate_depth(depth_map, center_x, frame.shape)
                    if depth_meters < closest_distance:
                        closest_distance = depth_meters
                        closest_object = label
                        closest_position = center_x
                    objects_with_depth.append((label, center_x, depth_meters))

            # Validate with ultrasonic sensor
            ultrasonic_dist = sensor.get_ultrasonic_distance()
            if ultrasonic_dist and closest_distance > ultrasonic_dist / 100:
                closest_distance = ultrasonic_dist / 100

            # Update occupancy grid and plan path
            planner.update_occupancy_grid(objects_with_depth, depth_map, frame.shape)
            path = planner.plan_path(user_pos)
            instruction = planner.get_navigation_instruction(path, user_pos)

            # Generate feedback
            current_time = asyncio.get_event_loop().time()
            if (closest_object and 
                closest_distance < 1.0 and 
                (current_time - last_speech_time) > config["feedback"]["min_speech_interval"]):
                direction = "left" if closest_position > 0.5 else "right"
                message = f"{closest_object} {closest_distance:.1f} meters ahead. Turn {direction} to avoid."
                feedback.add_to_tts_queue(message, priority=True)
                last_speech_time = current_time
                if (current_time - last_vibration_time) > config["feedback"]["min_vibration_interval"]:
                    sensor.activate_vibration()
                    last_vibration_time = current_time
            elif (current_time - last_speech_time) > config["feedback"]["min_speech_interval"]:
                feedback.add_to_tts_queue(instruction)
                last_speech_time = current_time

            # Update user position (simplified)
            if path:
                user_pos = list(path[0])

            # Control frame rate
            await asyncio.sleep(0.1)  # ~10 FPS

    finally:
        # Cleanup
        camera.release_camera()    # <-- Release your Camera properly
        cv2.destroyAllWindows()
        sensor.cleanup()
        feedback.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
