import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import torch
from torchvision import transforms

class SensorManager:
    def __init__(self, trig_pin, echo_pin, vibration_pin):
        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.vibration_pin = vibration_pin
        GPIO.setup(trig_pin, GPIO.OUT)
        GPIO.setup(echo_pin, GPIO.IN)
        GPIO.setup(vibration_pin, GPIO.OUT)

        # Initialize MiDaS
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas.eval()
        self.device = torch.device("cpu")
        self.midas.to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def get_ultrasonic_distance(self):
        """Measure distance using HC-SR04."""
        GPIO.output(self.trig_pin, True)
        time.sleep(0.00001)
        GPIO.output(self.trig_pin, False)

        pulse_start = time.time()
        pulse_end = time.time()

        while GPIO.input(self.echo_pin) == 0:
            pulse_start = time.time()
        while GPIO.input(self.echo_pin) == 1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  # Speed of sound in cm
        distance = round(distance, 2)
        if 2 < distance < 400:
            return distance
        return None

    def get_depth_map(self, frame):
        """Estimate depth using MiDaS."""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            depth = self.midas(img)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1), size=(480, 640), mode="bicubic", align_corners=False
            ).squeeze()
        
        depth = depth.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        return depth

    def estimate_depth(self, depth_map, center_x, frame_shape):
        """Estimate depth at object's position."""
        region_y = int(frame_shape[0] * 0.3)
        region_x = int(center_x * frame_shape[1])
        depth_value = depth_map[region_y, region_x]
        return 1 / (depth_value + 0.1) * 5  # Rough scaling

    def is_low_light(self, frame):
        """Detect low-light conditions."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        return avg_brightness < 30

    def activate_vibration(self, duration=0.5):
        """Activate vibration motor."""
        GPIO.output(self.vibration_pin, True)
        time.sleep(duration)
        GPIO.output(self.vibration_pin, False)

    def cleanup(self):
        """Cleanup GPIO."""
        GPIO.cleanup()