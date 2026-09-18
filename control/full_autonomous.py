## Fully autonomous visual servo: the VLM sees the wrist camera and drives the TCP itself.
# Same camera/robot setup as visual_servo.py, but no detector, probe calibration or Jacobian: the VLM
# must self-calibrate (probe, watch how the image shifts), search for the target and center it.
#
# Each turn is one stateless request: current frame + a state JSON holding the VLM's own notes from
# last turn, the move actually executed (after clamping) and the TCP offset from start. The VLM
# replies with a JSON tool-frame move and new notes. The camera closes the loop; notes carry memory.
#
# Safety: moves are clamped to MAX_STEP per turn, run at MAX_SPEED, stay within +-MAX_OFFSET of the
# start pose per tool axis, and never rotate. 'q' stops a move mid-way; during a VLM call it's read after.
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

from visual_servo import ACCEL, OPENAI_MODEL, SETTLE_S, clip_norm, draw_direction_hud

TARGET_DESC = os.environ.get("TARGET_DESC", "a flat metal circle with 9 drill holes")
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", (
    "You control a UR robot arm with a camera mounted on its wrist, looking roughly down. "
    "Your goal is to move the arm so the target is centered in the camera image. "
    "You do not know how tool-frame moves map to image motion: self-calibrate first by making small "
    "probe moves along each axis and noting how the target shifts in the image, then use that mapping "
    "to center it. If the target is not visible, search for it systematically. Avoid moving in z "
    "unless you are sure which way is away from the table."
))

MAX_SPEED = 0.01  # m/s, deliberately very slow
MAX_STEP = 0.04    # max meters per turn
MAX_OFFSET = 0.15  # max meters from the start pose, per tool axis

PROTOCOL = f"""

Each turn you get the current camera image and a state JSON with: "notes" (what you wrote last turn), \
"last_executed_move" (the move actually made last turn after clamping, null if none) and \
"offset_from_start" (current TCP offset from the start pose). All distances are meters in the tool frame. \
Nothing else is remembered between turns, so keep everything you need (calibration results, where the \
target was, your plan) in notes.
Reply with ONLY a JSON object:
{{"dx": float, "dy": float, "dz": float, "notes": str, "done": bool}}
dx, dy, dz is the next move. Moves are clamped to {MAX_STEP} m per turn and to +-{MAX_OFFSET} m per axis \
from the start pose. Set done=true once the goal is reached; no move is made then."""


def ask_vlm(client, frame, state):
    ok, jpg = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(jpg.tobytes()).decode()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + PROTOCOL},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Target: {TARGET_DESC}\nState: {json.dumps(state)}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


def main():
    print("=== Fully autonomous visual servo ===")
    print(f"Target description sent to {OPENAI_MODEL}: \"{TARGET_DESC}\" (set via TARGET_DESC env var).")
    print("System prompt overridable via SYSTEM_PROMPT env var. Requires OPENAI_API_KEY in the environment.")
    print(f"Moves: <= {MAX_STEP * 100:.0f} cm/turn at {MAX_SPEED * 1000:.0f} mm/s, "
          f"within +-{MAX_OFFSET * 100:.0f} cm of the start pose, no rotation.")
    print("Top-right circle/arrow = last executed tool-frame XY move.")
    print("Press 'q' in the video window to stop (halts a move mid-way; during a VLM call, takes effect after it).")

    UR_IP = os.environ.get("UR_IP", "192.168.1.20")
    CAM_URL = f"http://{UR_IP}:4242/current.jpg?type=color"

    from openai import OpenAI

    client = OpenAI()
    rtde_c = rtde_control.RTDEControlInterface(UR_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(UR_IP)

    log_name = os.path.splitext(os.path.basename(__file__))[0]
    log_path = f"{log_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
    log_file = open(log_path, "w")
    print(f"Logging moves to {log_path}")

    def get_frame():
        resp = requests.get(CAM_URL, timeout=2)
        resp.raise_for_status()
        return cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)

    def show(frame, quit_key="q"):
        cv2.imshow("wrist camera", frame)
        return cv2.waitKey(1) & 0xFF == ord(quit_key)

    start = rtde_r.getActualTCPPose()
    p0 = np.array(start[:3])
    R0 = cv2.Rodrigues(np.array(start[3:]))[0]
    offset = np.zeros(3)
    state = {"notes": "", "last_executed_move": None, "offset_from_start": offset.tolist()}

    try:
        while True:
            frame = get_frame()
            try:
                cmd = ask_vlm(client, frame, state)
                d = clip_norm(np.array([float(cmd.get(k, 0.0)) for k in ("dx", "dy", "dz")]), MAX_STEP)
            except (ValueError, TypeError, AttributeError) as e:
                print(f"VLM: invalid reply ({e}), not moving")
                state["error"] = f"Your last reply was invalid: {e}"
                state["last_executed_move"] = None
                if show(frame):
                    break
                continue

            state.pop("error", None)
            state["notes"] = str(cmd.get("notes", ""))
            print(f"VLM: move={d.round(4).tolist()} done={bool(cmd.get('done'))} notes={state['notes']}")

            draw_direction_hud(frame, d[0], d[1], MAX_STEP)
            if show(frame):
                break
            if cmd.get("done"):
                print("VLM: reports goal reached")
                break

            new_offset = np.clip(offset + d, -MAX_OFFSET, MAX_OFFSET)
            d, offset = new_offset - offset, new_offset
            target = p0 + R0 @ offset
            rtde_c.moveL([*target.tolist(), *start[3:]], speed=MAX_SPEED, acceleration=ACCEL, asynchronous=True)
            # Async so 'q' is polled mid-move; the move counts as done once the TCP is within 1 mm of target.
            while np.linalg.norm((actual := np.array(rtde_r.getActualTCPPose()[:3])) - target) > 1e-3:
                live = get_frame()
                draw_direction_hud(live, d[0], d[1], MAX_STEP)
                if show(live):
                    rtde_c.stopL()
                    print(f"q: stopped mid-move at offset {(R0.T @ (actual - p0)).round(4).tolist()} "
                          f"(target {offset.round(4).tolist()})")
                    return
            time.sleep(SETTLE_S)

            state["last_executed_move"] = d.round(4).tolist()
            state["offset_from_start"] = offset.round(4).tolist()
            log_file.write(f"{datetime.now().isoformat()},{d[0]:.6f},{d[1]:.6f},{d[2]:.6f},{json.dumps(cmd)}\n")
    finally:
        rtde_c.speedStop()
        rtde_c.stopScript()
        cv2.destroyAllWindows()
        log_file.close()


if __name__ == "__main__":
    main()
