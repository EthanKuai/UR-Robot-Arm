## Wrist camera access. The Robotiq wrist camera URCap serves single JPEG frames over HTTP on the
# robot's own IP (port 4242); there is no local /dev/video device to open.
#
# The camera autofocuses, so a frame grabbed right after the arm moves can come back blurred. Frames
# are focus-checked and re-read a few times before being handed to the caller.
import os
import time

import cv2
import numpy as np
import requests

FOCUS_THRESH = 30.0  # variance-of-Laplacian below this counts as out of focus
FOCUS_RETRIES = 6    # frames read before giving up and returning a blurred one
FOCUS_WAIT_S = 0.5   # wait between re-reads, giving autofocus time to settle


def focus_score(frame):
    # Variance of the Laplacian: a sharp image has strong high-frequency content, a blurred one has
    # little. Scene-dependent, so FOCUS_THRESH is only as good as the frames it was set from — the
    # sharp frame.png scores ~72 and the same frame mildly blurred ~18, hence 30. Re-check the
    # threshold with this function if the scene or working distance changes much.
    return cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


class Camera:
    def __init__(self, ur_ip=None, timeout=2):
        ur_ip = ur_ip or os.environ.get("UR_IP", "192.168.1.20")
        self.url = f"http://{ur_ip}:4242/current.jpg?type=color"
        self.timeout = timeout

    def _read(self):
        resp = requests.get(self.url, timeout=self.timeout)
        resp.raise_for_status()
        return cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)

    def get_frame(self, check_focus=True):
        # Pass check_focus=False to read continuously while the arm moves: waiting cannot clear
        # motion blur, and the retry would stall a loop that has to keep commanding or polling.
        if not check_focus:
            return self._read()
        for attempt in range(FOCUS_RETRIES):
            frame = self._read()
            score = focus_score(frame)
            if score >= FOCUS_THRESH:
                return frame
            if attempt < FOCUS_RETRIES - 1:
                time.sleep(FOCUS_WAIT_S)
        # Returning the blurred frame beats raising: the caller's detector fails on it and stops,
        # whereas an exception would tear down a running servo.
        print(f"Camera: out of focus after {FOCUS_RETRIES} reads (score {score:.0f} < {FOCUS_THRESH}); "
              f"using the blurred frame")
        return frame
