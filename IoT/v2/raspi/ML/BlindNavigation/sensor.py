import cv2
import numpy as np
from gpiozero import DigitalOutputDevice, DigitalInputDevice
import time
import torch
from torchvision import transforms
import smbus
import math

class SensorManager:
    def __init__(self, trig_pin, echo_pin, vibration_pin, extra_sensors=None):
        # Initialize GPIO with gpiozero
        self.trig_pin = trig_pin
        self.echo_pin = echo_pin
        self.vibration_pin = vibration_pin
        self.trig = DigitalOutputDevice(trig_pin)
        self.echo = DigitalInputDevice(echo_pin)
        self.vibration = DigitalOutputDevice(vibration_pin)

        # Initialize extra ultrasonic sensors (if provided)
        self.extra_sensors = extra_sensors or []
        self.extra_trig_echo = [(DigitalOutputDevice(trig), DigitalInputDevice(echo)) 
                               for trig, echo in self.extra_sensors]

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
        self.bus = smbus.SMBus(1)
        self.mpu_address = 0x68
        self._init_mpu6050()

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
        """Measure distance using HC-SR04 (returns meters)."""
        timeout = 0.05
        self.trig.on()
        time.sleep(0.00001)
        self.trig.off()
        start_time = time.time()
        stop_time = time.time()
        
        while not self.echo.is_active:
            start_time = time.time()
            if start_time - stop_time > timeout:
                return None
        while self.echo.is_active:
            stop_time = time.time()
            if stop_time - start_time > timeout:
                return None
        
        elapsed_time = stop_time - start_time
        distance_inches = (elapsed_time * 34300) / 2 / 2.54
        distance_meters = distance_inches * 0.0254
        if 0.05 < distance_meters < 4.0:
            return distance_meters
        return None

    def get_extra_sensor_distances(self):
        distances = []
        timeout = 0.05
        for trig, echo in self.extra_trig_echo:
            trig.on()
            time.sleep(0.00001)
            trig.off()
            start_time = time.time()
            stop_time = time.time()
            
            while not echo.is_active:
                start_time = time.time()
                if start_time - stop_time > timeout:
                    distances.append(None)
                    break
            else:
                while echo.is_active:
                    stop_time = time.time()
                    if stop_time - start_time > timeout:
                        distances.append(None)
                        break
                else:
                    elapsed_time = stop_time - start_time
                    distance_inches = (elapsed_time * 34300) / 2 / 2.54
                    distance_meters = distance_inches * 0.0254
                    distances.append(distance_meters if 0.05 < distance_meters < 4.0 else None)
        return distances

    def get_depth_map(self, frame):
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
        region_y = int(frame_shape[0] * 0.3)
        region_x = int(center_x * frame_shape[1])
        depth_value = depth_map[region_y, region_x]
        return 1 / (depth_value + 0.1) * 5

    def is_low_light(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        return avg_brightness < 30

    def activate_vibration(self, duration=0.5):
        self.vibration.on()
        time.sleep(duration)
        self.vibration.off()

    def cleanup(self):
        self.trig.close()
        self.echo.close()
        self.vibration.close()
        for trig, echo in self.extra_trig_echo:
            trig.close()
            echo.close()