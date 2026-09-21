## Wrist camera access. The Robotiq wrist camera URCap serves single JPEG frames over HTTP on the
# robot's own IP (port 4242); there is no local /dev/video device to open.
import os

import cv2
import numpy as np
import requests


class Camera:
    def __init__(self, ur_ip=None, timeout=2):
        ur_ip = ur_ip or os.environ.get("UR_IP", "192.168.1.20")
        self.url = f"http://{ur_ip}:4242/current.jpg?type=color"
        self.timeout = timeout

    def get_frame(self):
        resp = requests.get(self.url, timeout=self.timeout)
        resp.raise_for_status()
        return cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
