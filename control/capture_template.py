## Grab one wrist camera frame, drag a box around the tool, save the crop as visual_servo.py's template.
# Capture at the height calibration will start from; the full frame is also saved for offline testing.
import os

import cv2
import numpy as np
import requests

UR_IP = os.environ.get("UR_IP", "192.168.1.20")
TEMPLATE_PATH = os.environ.get("TEMPLATE_PATH", "tool.png")

resp = requests.get(f"http://{UR_IP}:4242/current.jpg?type=color", timeout=2)
resp.raise_for_status()
frame = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
cv2.imwrite("frame.png", frame)

x, y, w, h = cv2.selectROI("Drag a tight box around the tool, then Enter", frame)
cv2.destroyAllWindows()
if w and h:
    cv2.imwrite(TEMPLATE_PATH, frame[y:y + h, x:x + w])
    print(f"Saved {w}x{h} template to {TEMPLATE_PATH} (full frame in frame.png)")
else:
    print("No box selected; template not saved.")
