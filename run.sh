#!/bin/bash
source ~/mlVenv/bin/activate
git pull origin main
python3 IoT/v2/raspi/ML/BlindNavigation/main.py