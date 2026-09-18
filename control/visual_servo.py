## Visual servoing: track a target with the Robotiq wrist camera, servo the TCP toward it.
# Camera feed comes from the wrist camera URCap's HTTP endpoint on the robot's own IP (port 4242),
# not a local /dev/video device.
#
# DETECTOR selects the targeting strategy (env var, default "aruco"):
#   aruco        - continuous closed-loop servo (speedL) on an ArUco marker (DICT_4X4_50).
#   vlm-only     - discrete jut moves: stop, ask ChatGPT vision to locate TARGET_DESC, moveL, repeat.
#   vlm-assisted - same VLM jut phase, then hands off to the aruco continuous corrector to finish.
#
# Camera->robot mapping is probe-calibrated at startup: the TCP jogs PROBE_STEP along tool X and Y,
# and the resulting image shifts form a 2x2 image Jacobian J (error per meter, tool frame). Moves
# are then d = -J^-1 * error in the tool frame, so sign, axis swap and scale (at the calibrated
# depth) come from measurement. Assumes the target stays still and depth stays roughly constant.
import base64
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
import requests
import rtde_control
import rtde_receive

DETECTOR = os.environ.get("DETECTOR", "aruco")  # aruco | vlm-only | vlm-assisted
TARGET_DESC = os.environ.get("TARGET_DESC", "the ArUco marker")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-terra")  # verify against current OpenAI vision models

MAX_SPEED = 0.05       # m/s cap for continuous servo
SERVO_GAIN = 1.0        # 1/s, proportional gain on the estimated tool-frame offset
ACCEL = 0.3
JUT_STEP = 0.02         # max meters per discrete VLM jut
CENTER_THRESH = 0.05    # normalized error below which a jut phase is considered centered
MAX_JUT_ITERS = 20
PROBE_STEP = 0.02       # meters jogged along each tool axis during calibration
MIN_PROBE_SHIFT = 0.01  # normalized error; weaker/degenerate probe response rejects calibration
SETTLE_S = 0.5          # wait after a probe move so the camera serves a fresh frame


def clip_norm(v, max_norm):
    n = np.linalg.norm(v)
    return v * (max_norm / n) if n > max_norm else v


def detect_aruco(frame, detector):
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return None
    cx, cy = corners[0][0].mean(axis=0)
    h, w = frame.shape[:2]
    return (w / 2 - cx) / (w / 2), (h / 2 - cy) / (h / 2)


def detect_vlm(frame, client, target_desc):
    ok, jpg = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(jpg.tobytes()).decode()
    prompt = (
        f"Locate {target_desc} in this image. Respond with ONLY a JSON object, no markdown: "
        '{"found": bool, "x": float, "y": float, "confidence": float} '
        "where x,y are the target center as a fraction of image width/height "
        "(0.0=left/top, 1.0=right/bottom). Set found=false if not visible."
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
    )
    text = response.choices[0].message.content.strip().strip("`").removeprefix("json").strip()
    data = json.loads(text)
    if not data.get("found"):
        return None
    x, y, conf = data["x"], data["y"], data.get("confidence", 0.0)
    return (0.5 - x) * 2, (0.5 - y) * 2, conf


def draw_direction_hud(frame, dx, dy, max_val):
    # Screen-space plot of the last correction signal; not aligned to real-world directions.
    radius, margin = 40, 15
    center = (frame.shape[1] - margin - radius, margin + radius)
    cv2.circle(frame, center, radius, (200, 200, 200), 1)
    tip = (
        int(center[0] + (dx / max_val) * radius),
        int(center[1] + (dy / max_val) * radius),
    )
    if tip != center:
        cv2.arrowedLine(frame, center, tip, (0, 0, 255), 2, tipLength=0.3)


def main():
    print("=== Visual servo ===")
    print(f"DETECTOR={DETECTOR}")
    if DETECTOR == "aruco":
        print("Target: an ArUco marker (DICT_4X4_50) — a generated black/white bit-grid square, not a plain cross.")
        print("  Generate one to print, e.g.:")
        print("    python3 -c \"import cv2; cv2.imwrite('marker.png', "
              "cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 400))\"")
    else:
        print(f"Target description sent to {OPENAI_MODEL}: \"{TARGET_DESC}\" (set via TARGET_DESC env var).")
        print("Requires OPENAI_API_KEY in the environment.")
        if DETECTOR == "vlm-assisted":
            print("After the VLM roughly centers the target, an ArUco corrector takes over — target must be a marker.")
    print(f"Calibration first: TCP jogs {PROBE_STEP * 100:.0f} cm along tool X and Y — keep the target visible and still.")
    print("Top-right circle/arrow = direction & magnitude of the current correction.")
    print("Press 'q' in the video window to stop.")

    UR_IP = os.environ.get("UR_IP", "192.168.1.20")
    CAM_URL = f"http://{UR_IP}:4242/current.jpg?type=color"

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters(),
    )

    rtde_c = rtde_control.RTDEControlInterface(UR_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(UR_IP)

    log_name = os.path.splitext(os.path.basename(__file__))[0]
    log_path = f"{log_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
    log_file = open(log_path, "w")
    print(f"Logging corrections to {log_path}")

    def get_frame():
        resp = requests.get(CAM_URL, timeout=2)
        resp.raise_for_status()
        return cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)

    def log(dx, dy):
        log_file.write(f"{datetime.now().isoformat()},{dx:.6f},{dy:.6f}\n")

    def show(frame, quit_key="q"):
        cv2.imshow("wrist camera", frame)
        return cv2.waitKey(1) & 0xFF == ord(quit_key)

    def tool_to_base(d_tool_xy):
        R = cv2.Rodrigues(np.array(rtde_r.getActualTCPPose()[3:]))[0]
        return R[:, :2] @ d_tool_xy

    def calibrate(detect):
        # Returns J^-1, mapping normalized image error to the tool-frame XY offset (m) that produced it.
        start = rtde_r.getActualTCPPose()

        def error_at(pose):
            rtde_c.moveL(pose, speed=0.1, acceleration=ACCEL)
            time.sleep(SETTLE_S)
            error = detect(get_frame())
            if error is None:
                raise RuntimeError("Calibration: target not visible")
            return np.array(error[:2])

        e0 = error_at(start)
        cols = []
        for axis in range(2):
            probe = list(start)
            probe[:3] = (np.array(start[:3]) + tool_to_base(np.eye(2)[axis] * PROBE_STEP)).tolist()
            cols.append((error_at(probe) - e0) / PROBE_STEP)
        rtde_c.moveL(start, speed=0.1, acceleration=ACCEL)

        J = np.column_stack(cols)
        print(f"Calibration: J (normalized error per m, cols = tool X, tool Y) =\n{J}")
        if np.linalg.svd(J * PROBE_STEP, compute_uv=False)[-1] < MIN_PROBE_SHIFT:
            raise RuntimeError("Calibration: probe barely/degenerately moved the target; raise PROBE_STEP or move closer")
        return np.linalg.inv(J)

    def run_continuous_aruco_servo(J_inv):
        while True:
            frame = get_frame()
            error = detect_aruco(frame, detector)

            if error is not None:
                v_tool = clip_norm(-SERVO_GAIN * J_inv @ np.array(error), MAX_SPEED)
                rtde_c.speedL([*tool_to_base(v_tool).tolist(), 0, 0, 0], ACCEL)
            else:
                v_tool = np.zeros(2)
                rtde_c.speedStop()

            log(*v_tool)
            draw_direction_hud(frame, *v_tool, MAX_SPEED)
            if show(frame):
                return False
        return True

    def run_vlm_jut_phase(client, J_inv):
        for _ in range(MAX_JUT_ITERS):
            frame = get_frame()
            result = detect_vlm(frame, client, TARGET_DESC)

            if result is None:
                print("VLM: target not found")
                log(0.0, 0.0)
                draw_direction_hud(frame, 0.0, 0.0, 1.0)
                if show(frame):
                    return False
                continue

            ex, ey, conf = result
            print(f"VLM: error=({ex:.2f}, {ey:.2f}) confidence={conf:.2f}")
            log(ex, ey)
            draw_direction_hud(frame, ex, ey, 1.0)
            if show(frame):
                return False

            if abs(ex) < CENTER_THRESH and abs(ey) < CENTER_THRESH:
                print("VLM: target centered")
                return True

            d_tool = clip_norm(-J_inv @ np.array([ex, ey]), JUT_STEP)
            pose = rtde_r.getActualTCPPose()
            pose[:3] = (np.array(pose[:3]) + tool_to_base(d_tool)).tolist()
            rtde_c.moveL(pose, speed=0.1, acceleration=0.3)

        print("VLM: max jut iterations reached without centering")
        return True

    try:
        if DETECTOR == "aruco":
            J_inv = calibrate(lambda f: detect_aruco(f, detector))
            run_continuous_aruco_servo(J_inv)
        elif DETECTOR in ("vlm-only", "vlm-assisted"):
            from openai import OpenAI

            client = OpenAI()
            # Jut moves stay in the tool XY plane, so depth (and J) still holds for the aruco phase.
            J_inv = calibrate(lambda f: detect_vlm(f, client, TARGET_DESC))
            if run_vlm_jut_phase(client, J_inv) and DETECTOR == "vlm-assisted":
                run_continuous_aruco_servo(J_inv)
        else:
            raise ValueError(f"Unknown DETECTOR: {DETECTOR}")
    finally:
        rtde_c.speedStop()
        rtde_c.stopScript()
        cv2.destroyAllWindows()
        log_file.close()


if __name__ == "__main__":
    main()
