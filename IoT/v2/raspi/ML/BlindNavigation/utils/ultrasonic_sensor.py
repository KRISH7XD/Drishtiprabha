import lgpio
import time

h = lgpio.gpiochip_open(0)
TRIG = 23
ECHO = 24

lgpio.gpio_claim_output(h, TRIG)
lgpio.gpio_claim_input(h, ECHO)

def get_distance():
    lgpio.gpio_write(h, TRIG, 1)
    time.sleep(0.00001)
    lgpio.gpio_write(h, TRIG, 0)

    start_time = time.time()
    stop_time = time.time()

    while lgpio.gpio_read(h, ECHO) == 0:
        start_time = time.time()

    while lgpio.gpio_read(h, ECHO) == 1:
        stop_time = time.time()

    time_elapsed = stop_time - start_time
    distance = (time_elapsed * 34300) / 2
    return distance
