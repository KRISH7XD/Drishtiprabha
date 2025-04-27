import asyncio
import cv2
import numpy as np
from object_detection import ObjectDetector
from sensor import SensorManager
from path_planning import PathPlanner
from feedback import FeedbackManager
from utils.camera import Camera
import yaml

async def main():
    print("Starting main loop...")
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Initializing components...")
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
    camera = Camera(resolution=(640, 480))

    user_pos = [0, 0]
    last_speech_time = 0
    last_vibration_time = 0

    try:
        while True:
            print("Capturing frame...")
            frame = camera.capture_image()
            if frame is None:
                print("Failed to capture frame")
                break

            print("Checking low light...")
            if sensor.is_low_light(frame):
                print("Low light detected")
                ultrasonic_dist = sensor.get_ultrasonic_distance()
                if ultrasonic_dist and ultrasonic_dist < 1.0:
                    message = f"Obstacle {ultrasonic_dist:.1f} meters ahead. Stop."
                    print(f"Adding to TTS: {message}")
                    feedback.add_to_tts_queue(message, priority=True)
                    sensor.activate_vibration()
                await asyncio.sleep(0.1)
                continue

            print("Detecting objects...")
            detected_objects = detector.detect(frame)

            print("Estimating depth...")
            depth_map = sensor.get_depth_map(frame) if detected_objects else None

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

            print("Validating with ultrasonic...")
            ultrasonic_dist = sensor.get_ultrasonic_distance()
            if ultrasonic_dist and closest_distance > ultrasonic_dist:
                closest_distance = ultrasonic_dist
                print(f"Using ultrasonic distance: {closest_distance:.2f} meters")

            print("Updating occupancy grid...")
            planner.update_occupancy_grid(objects_with_depth, depth_map, frame.shape)
            print("Planning path...")
            path = planner.plan_path(user_pos)
            print("Generating instruction...")
            instruction = planner.get_navigation_instruction(path, user_pos)
            print(f"Instruction: {instruction}")

            current_time = asyncio.get_event_loop().time()
            if (closest_object and 
                closest_distance < 1.0 and 
                (current_time - last_speech_time) > config["feedback"]["min_speech_interval"]):
                direction = "left" if closest_position > 0.5 else "right"
                message = f"{closest_object} {closest_distance:.1f} meters ahead. Turn {direction} to avoid."
                print(f"Adding to TTS: {message}")
                feedback.add_to_tts_queue(message, priority=True)
                last_speech_time = current_time
                if (current_time - last_vibration_time) > config["feedback"]["min_vibration_interval"]:
                    sensor.activate_vibration()
                    last_vibration_time = current_time
            elif (current_time - last_speech_time) > config["feedback"]["min_speech_interval"]:
                print(f"Adding to TTS: {instruction}")
                feedback.add_to_tts_queue(instruction)
                last_speech_time = current_time

            if path:
                user_pos = list(path[0])
                print(f"Updated user position: {user_pos}")

            await asyncio.sleep(0.1)

    except Exception as e:
        print(f"Main loop error: {e}")
    finally:
        print("Cleaning up...")
        camera.release_camera()
        cv2.destroyAllWindows()
        sensor.cleanup()
        feedback.cleanup()
        print("Cleanup complete")

if __name__ == "__main__":
    asyncio.run(main())