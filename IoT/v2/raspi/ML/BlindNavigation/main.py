import asyncio
import cv2
import numpy as np
from object_detection import ObjectDetector
from sensor import SensorManager
from path_planning import PathPlanner
from feedback import FeedbackManager
from voice_input import VoiceInput
import yaml
import math

async def main():
    # Load configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize components
    detector = ObjectDetector(config["model"]["yolo_weights"])
    sensor = SensorManager(
        trig_pin=config["sensor"]["trig_pin"],
        echo_pin=config["sensor"]["echo_pin"],
        vibration_pin=config["sensor"]["vibration_pin"],
        extra_sensors=config["sensor"].get("extra_sensors", [])
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
    voice = VoiceInput()

    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    user_pos = [0, 0]  # Starting position
    last_speech_time = 0
    last_vibration_time = 0
    last_voice_check = 0
    voice_check_interval = 10.0  # Check for voice commands every 10 seconds

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            # Check for voice commands
            current_time = asyncio.get_event_loop().time()
            if (current_time - last_voice_check) > voice_check_interval:
                command = voice.get_command()
                if command and "exit" in command:
                    planner.target_pos = [5, 0]  # Example: set target to room exit
                    feedback.add_to_tts_queue("Navigating to room exit.", priority=True)
                last_voice_check = current_time

            # Get user orientation from IMU
            yaw = sensor.get_imu_orientation()

            # Check for low-light conditions
            if sensor.is_low_light(frame):
                ultrasonic_dist = sensor.get_ultrasonic_distance()
                if ultrasonic_dist and ultrasonic_dist < 1.0:
                    feedback.add_to_tts_queue(
                        f"Obstacle {ultrasonic_dist:.1f} meters ahead. Stop.",
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
                    # Adjust object position based on yaw
                    relative_angle = (center_x - 0.5) * 60 + yaw  # Assume 60° FOV
                    if depth_meters < closest_distance:
                        closest_distance = depth_meters
                        closest_object = label
                        closest_position = center_x
                        closest_angle = relative_angle
                    objects_with_depth.append((label, center_x, depth_meters))

            # Validate with ultrasonic sensors
            ultrasonic_dist = sensor.get_ultrasonic_distance()
            if ultrasonic_dist and closest_distance > ultrasonic_dist:
                closest_distance = ultrasonic_dist
                closest_angle = yaw  # Assume forward-facing sensor

            # Check extra sensors
            extra_distances = sensor.get_extra_sensor_distances()
            for i, dist in enumerate(extra_distances):
                if dist and dist < 1.0:
                    sensor_name = ["left", "right", "rear"][i]
                    feedback.add_to_tts_queue(
                        f"Obstacle {dist:.1f} meters {sensor_name}. Stop.",
                        priority=True
                    )
                    sensor.activate_vibration()

            # Update occupancy grid and plan path
            planner.update_occupancy_grid(objects_with_depth, depth_map, frame.shape)
            path = planner.plan_path(user_pos)
            instruction = planner.get_navigation_instruction(path, user_pos, yaw)

            # Generate feedback
            if (closest_object and 
                closest_distance < 1.0 and 
                (current_time - last_speech_time) > config["feedback"]["min_speech_interval"]):
                angle_desc = "straight ahead" if abs(closest_angle) < 15 else \
                             f"at {int(abs(closest_angle))} degrees {'left' if closest_angle < 0 else 'right'}"
                message = f"{closest_object} {closest_distance:.1f} meters {angle_desc}. Turn {'left' if closest_angle > 0 else 'right'} to avoid."
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

            # Exit on key press (for testing)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        sensor.cleanup()
        feedback.cleanup()

if __name__ == "__main__":
    asyncio.run(main())