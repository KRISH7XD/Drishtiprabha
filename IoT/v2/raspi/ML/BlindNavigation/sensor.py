import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import torch
from torchvision import transforms
import smbus

class Ultrasonic:
    def __init__(self, trig_pin=15, echo_pin=16):
        self.TRIG = trig_pin
        self.ECHO = echo_pin

        # Board setup
        try:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(self.TRIG, GPIO.OUT)
            GPIO.setup(self.ECHO, GPIO.IN)
            print(f"Waiting for sensor (TRIG: {trig_pin}, ECHO: {echo_pin}) to settle")
            time.sleep(0.5)
        except Exception as e:
            print(f"GPIO setup failed: {e}")
            raise

    def measure_distance(self):
        try:
            GPIO.output(self.TRIG, False)
            time.sleep(0.00001)

            GPIO.output(self.TRIG, True)
            time.sleep(0.00001)
            GPIO.output(self.TRIG, False)

            start_time = time.time()
            timeout = start_time + 0.04  # 40ms timeout

            while GPIO.input(self.ECHO) == 0 and start_time < timeout:
                start_time = time.time()

            if start_time >= timeout:
                return 99999

            end_time = time.time()
            while GPIO.input(self.ECHO) == 1 and end_time < timeout:
                end_time = time.time()

            if end_time >= timeout:
                return 99999

            pulse_duration = end_time - start_time
            distance = (pulse_duration * 34300) / 2
            return round(distance, 2)
        except Exception as e:
            print(f"Distance measurement failed: {e}")
            return 99999

    def cleanup(self):
        pass  # Cleanup handled by SensorManager

class SensorManager:
    def __init__(self, trig_pin, echo_pin, vibration_pin, extra_sensors=None, i2c_bus=1):
        # Clean up any existing GPIO setup
        try:
            GPIO.cleanup()
        except:
            pass

        # Initialize primary ultrasonic sensor
        self.primary_sensor = Ultrasonic(trig_pin=trig_pin, echo_pin=echo_pin)
        
        # Initialize vibration pin
        GPIO.setmode(GPIO.BOARD)
        self.vibration_pin = vibration_pin
        try:
            GPIO.setup(vibration_pin, GPIO.OUT)
        except RuntimeError as e:
            print(f"Vibration pin setup failed: {e}")
            raise

        # Initialize extra ultrasonic sensors
        self.extra_sensors = []
        for trig, echo in (extra_sensors or []):
            try:
                sensor = Ultrasonic(trig_pin=trig, echo_pin=echo)
                self.extra_sensors.append(sensor)
            except RuntimeError as e:
                print(f"Extra sensor setup failed (TRIG: {trig}, ECHO: {echo}): {e}")
                raise

        # Initialize MiDaS
        try:
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.midas.eval()
            self.device = torch.device("cpu")
            self.midas.to(self.device)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((384, 384)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        except Exception as e:
            print(f"MiDaS initialization failed: {e}")
            self.midas = None

        # Initialize MPU-6050 (optional)
        self.mpu_address = None
        self.i2c_bus = i2c_bus
        try:
            self.bus = smbus.SMBus(self.i2c_bus)
            self.mpu_address = 0x68
            self._init_mpu6050()
        except Exception as e:
            print(f"IMU initialization failed on bus {self.i2c_bus}: {e}. Continuing without IMU.")

    def _init_mpu6050(self):
        try:
            self.bus.write_byte_data(self.mpu_address, 0x6B, 0x00)
        except Exception as e:
            print(f"Failed to initialize MPU-6050: {e}")
            self.mpu_address = None

    def get_imu_orientation(self):
        if not self.mpu_address:
            return 0.0
        try:
            accel_x = self._read_word_2c(0x3B) / 16384.0
            accel_y = self._read_word_2c(0x3D) / 16384.0
            accel_z = self._read_word_2c(0x3F) / 16384.0
            pitch = math.atan2(accel_y, math.sqrt(accel_x**2 + accel_z**2)) * 180 / math.pi
            roll = math.atan2(-accel_x, accel_z) * 180 / math.pi
            gyro_z = self._read_word_2c(0x47) / 131.0
            yaw = gyro_z
            return yaw
        except:
            return 0.0

    def _read_word_2c(self, reg):
        high = self.bus.read_byte_data(self.mpu_address, reg)
        low = self.bus.read_byte_data(self.mpu_address, reg + 1)
        value = (high << 8) + low
        if value >= 0x8000:
            return -((65535 - value) + 1)
        return value

    def get_ultrasonic_distance(self):
        distance_cm = self.primary_sensor.measure_distance()
        if distance_cm >= 99999:
            return None
        distance_meters = distance_cm / 100.0
        if 0.05 < distance_meters < 4.0:
            return distance_meters
        return None

    def get_extra_sensor_distances(self):
        distances = []
        for sensor in self.extra_sensors:
            distance_cm = sensor.measure_distance()
            if distance_cm >= 99999:
                distances.append(None)
            else:
                distance_meters = distance_cm / 100.0
                distances.append(distance_meters if 0.05 < distance_meters < 4.0 else None)
        return distances

    def get_depth_map(self, frame):
        if self.midas is None:
            return None
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
        if depth_map is None:
            return float('inf')
        region_y = int(frame_shape[0] * 0.3)
        region_x = int(center_x * frame_shape[1])
        depth_value = depth_map[region_y, region_x]
        return 1 / (depth_value + 0.1) * 5

    def is_low_light(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        return avg_brightness < 30

    def activate_vibration(self, duration=0.5):
        try:
            GPIO.output(self.vibration_pin, True)
            time.sleep(duration)
            GPIO.output(self.vibration_pin, False)
        except Exception as e:
            print(f"Vibration activation failed: {e}")

    def cleanup(self):
        try:
            GPIO.cleanup()
        except:
            pass