import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import torch
from torchvision import transforms
import smbus
import math

class SensorManager:
    def __init__(self, trig_pin, echo_pin, vibration_pin, extra_sensors=None):
        # Initialize GPIO
        GPIO.setmode(GPIO.BCM)
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.vibration_pin = vibration_pin
        GPIO.setup(trig_pin, GPIO.OUT)
        GPIO.setup(echo_pin, GPIO.IN)
        GPIO.setup(vibration_pin, GPIO.OUT)

        # Initialize extra ultrasonic sensors (if provided)
        self.extra_sensors = extra_sensors or []
        for trig, echo in self.extra_sensors:
            GPIO.setup(trig, GPIO.OUT)
            GPIO.setup(echo, GPIO.IN)

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

        # Initialize MPU-6050
        self.bus = smbus.SMBus(1)  # I2C bus 1
        self.mpu_address = 0x68  # MPU-6050 default address
        self._init_mpu6050()

    def _init_mpu6050(self):
        """Initialize MPU-6050 IMU."""
        try:
            # Wake up MPU-6050 (disable sleep mode)
            self.bus.write_byte_data(self.mpu_address, 0x6B, 0x00)
        except Exception as e:
            print(f"Failed to initialize MPU-6050: {e}")
            self.mpu_address = None

    def get_imu_orientation(self):
        """Get yaw angle from MPU-6050 (approximate)."""
        if not self.mpu_address:
            return 0.0  # Default to 0 if IMU not available

        try:
            # Read accelerometer and gyro data
            accel_x = self._read_word_2c(0x3B) / 16384.0
            accel_y = self._read_word_2c(0x3D) / 16384.0
            accel_z = self._read_word_2c(0x3F) / 16384.0

            # Calculate pitch and roll
            pitch = math.atan2(accel_y, math.sqrt(accel_x**2 + accel_z**2)) * 180 / math.pi
            roll = math.atan2(-accel_x, accel_z) * 180 / math.pi

            # Approximate yaw (simplified, requires magnetometer for accuracy)
            gyro_z = self._read_word_2c(0x47) / 131.0
            yaw = gyro_z  # Placeholder; integrate over time for better yaw

            return yaw  # In degrees
        except:
            return 0.0

    def _read_word_2c(self, reg):
        """Read 2 bytes from MPU-6050 and convert to signed integer."""
        high = self.bus.read_byte_data(self.mpu_address, reg)
        low = self.bus.read_byte_data(self.mpu_address, reg + 1)
        value = (high << 8) + low
        if value >= 0x8000:
            return -((65535 - value) + 1)
        return value

    def get_ultrasonic_distance(self):
        """Measure distance using HC-SR04 (returns meters)."""
        GPIO.output(self.trig_pin, True)
        time.sleep(0.00001)
        GPIO.output(self.trig_pin, False)
        start_time = time.time()
        stop_time = time.time()
        
        while GPIO.input(self.echo_pin) == 0:
            start_time = time.time()
        while GPIO.input(self.echo_pin) == 1:
            stop_time = time.time()
        
        elapsed_time = stop_time - start_time
        distance_inches = (elapsed_time * 34300) / 2 / 2.54  # Your code
        distance_meters = distance_inches * 0.0254  # Convert to meters
        if 0.05 < distance_meters < 4.0:
            return distance_meters
        return None

    def get_extra_sensor_distances(self):
        """Measure distances from extra ultrasonic sensors (returns list of meters)."""
        distances = []
        for trig, echo in self.extra_sensors:
            GPIO.output(trig, True)
            time.sleep(0.00001)
            GPIO.output(trig, False)
            start_time = time.time()
            stop_time = time.time()
            
            while GPIO.input(echo) == 0:
                start_time = time.time()
            while GPIO.input(echo) == 1:
                stop_time = time.time()
            
            elapsed_time = stop_time - start_time
            distance_inches = (elapsed_time * 34300) / 2 / 2.54
            distance_meters = distance_inches * 0.0254
            distances.append(distance_meters if 0.05 < distance_meters < 4.0 else None)
        return distances

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
        """Estimate depth at object's position (returns meters)."""
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