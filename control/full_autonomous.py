## Fully autonomous visual servo: the VLM sees the wrist camera and drives the TCP itself.
# Same camera/robot setup as visual_servo.py, but no detector, probe calibration or Jacobian: the VLM
# must self-calibrate (probe, watch how the image shifts), search for the target and center it.
#
# Memory lives in VLM-context-<datetime>.md, rewritten every turn and fed back in full as the next
# turn's prompt. Its facts sections are written by this script (ground truth: executed moves, offset,
# tilt, rotation); the notes section is written by the VLM and is the only thing it carries between
# turns. At startup the notes can be inherited from a previous run's context file, so the VLM does not
# recalibrate from zero. Each turn also sends two images, previous and current, so image shifts are
# read by comparing pictures rather than by recalling a number; the raw frame of every turn is saved to
# full_autonomous-<datetime>/<turn>-<datetime>.png.
#
# The VLM reports which stage it is on; the stage 3 rotation (ROTATE_RIGHT about tool Z) is performed by
# this script, once, when the VLM reports stage 3.
#
# Safety: moves are clamped to MAX_STEP / MAX_TILT_STEP per turn, run at MAX_SPEED, stay within
# +-MAX_OFFSET of the start pose in tool x/y, +-MAX_TILT of the start orientation about tool X/Y, and
# no lower than MAX_DESCENT below the start height (base Z). 'q' in the video window stops a move
# mid-way; during a VLM call it's read after. Ctrl+C also shuts down cleanly.
import base64
import glob
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
import rtde_control
import rtde_receive

from capture import Camera
from visual_servo import ACCEL, OPENAI_MODEL, SETTLE_S, clip_norm, draw_direction_hud

TARGET_DESC = os.environ.get("TARGET_DESC", "a flat metal circle with 9 drill holes")
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", """\
You control a UR robot arm. A camera on its wrist, mounted in front of the claw and looking roughly \
down, gives you two images per turn: the frame before your last move, and the frame now. Your goal is \
to land the claw on the target.

Take the following as a message from someone who has watched this task fail:

Your weakest skill here is estimating small image shifts. Your reading of the target's center is only \
good to roughly the reading error stated in the context document. So:
- Never conclude anything from a probe that moved the target less than the minimum trustworthy shift \
stated there. That is noise, and a mapping built on noise will be confidently wrong. If a probe was \
too small, repeat the same move next turn, and again, accumulating a longer baseline; compute the gain \
over the whole accumulated move.
- Change ONE thing per turn while calibrating. Never a probe together with a correction, never two \
axes at once, never translation together with tilt. Otherwise you cannot attribute what you see.
- Use the executed move from the move log, never what you asked for. Commands get clamped.
- Compare the two images directly against each other. They are better evidence than your recollection \
of a number.
- Write every reading into the notes as a row: turn, center x px, center y px, apparent diameter px, \
and which executed move produced it. A reading you do not write down is gone forever.

Calibration protocol:
1. At the reference pose, record the target's position.
2. Probe one axis in one direction, repeating the same step over turns until the target has moved at \
least the minimum trustworthy shift. Record the position.
3. Reverse past the reference to the other side by twice that accumulated distance, again over as many \
turns as needed, and record. Gain = (position B - position A) / (executed distance from A to B). \
A two-sided probe halves your error.
4. Return to the reference pose and check the target reads where it did in step 1, within your reading \
error. If it does not, your readings are not reliable: re-measure, do not proceed.
5. Repeat for the other axis.
6. Check that the two axes do not produce nearly parallel image shifts. If they do, you cannot invert \
the mapping; write that down and re-probe with longer baselines.
7. Verify before trusting it: state the position you predict for a specific move, make that move, and \
compare. If the prediction is off by more than the minimum trustworthy shift, the calibration is wrong. \
Redo it instead of pressing on.

Only once verified, work toward the goal:
- Apply about 70% of the correction you compute, and re-measure every turn. Overshoot costs more turns \
than undershoot.
- If an observed shift is less than half or more than twice what you predicted, stop and re-probe.
- Gains scale with distance: if the apparent diameter changes by more than about 20%, rescale the gains \
by (old diameter / new diameter) and verify again.
- If the correction you need is larger than the travel left within the limits, write that in the notes \
and say so, rather than grinding against the limit.

Stages, in order; report the stage you are working on in the "stage" field (0 while calibrating), \
record it in the notes, and finish one before starting the next:
1. Translate so the target sits at bottom center: horizontally centered, with its lower edge just \
touching the bottom edge of the image.
2. Tilt forward. You do not know which axis or sign is forward: probe one tilt step on one axis and see \
which direction brings the target back up toward the middle of the image. That direction is forward. \
Continue until the target is near the vertical middle again; the claw is then over the target.
3. Rotate right 90 degrees. Report stage=3 with all values zero: the robot performs the rotation itself, \
once, and logs it. Your calibrated gains then apply to an image rotated by 90 degrees.
4. Translate until the target sits at the center of the rotated frame.
5. Descend. Down is whichever z direction makes the target grow; confirm with one small step before \
committing. Descend in small steps, re-measuring each turn, and set done=0.5 when the claw is very near \
the tool: the tool should disappear entirely from the camera feed. The descent limit is a safety floor, \
not a goal.
6. Do nothing for one turn: all values zero.
7. Tilt back until the camera is perfectly flat, facing the ground again (tilt from start reads 0, 0), \
and set done=1.0.

Recovery: if the target leaves the frame, reverse your last executed move exactly to get back to \
something you recognise, before improvising a search.
Never raise done just because you are out of ideas. Write what is wrong in the notes and hold still \
with all values zero.""")

MAX_SPEED = 0.01  # m/s, deliberately very slow
MAX_STEP = 0.04    # max meters per turn
MAX_OFFSET = 0.15  # max meters from the start pose, in tool x/y (tool z gets MAX_DESCENT)
MAX_DESCENT = 0.40  # max meters below the start height (base Z)
MAX_TILT_STEP = np.radians(3)  # max radians per turn, per tilt axis
MAX_TILT = np.radians([45, 30])  # max radians from the start orientation, about tool X, Y
ROTATE_RIGHT = np.radians(-90)  # stage 3 rotation about tool Z; flip the sign if it turns the wrong way
READ_ERR_FRAC = 0.03  # assumed VLM center-reading error, as a fraction of image width
MIN_SHIFT_FRAC = 0.10  # probe shift below this fraction of image width is treated as noise
LOG_ROWS = 30  # move log rows kept in the context document

NOTES_TEMPLATE = """\
### Stage
calibration - not started

### Readings
| turn | center x px | center y px | diameter px | after executed move |
|---|---|---|---|---|

### Gains (image shift in px per meter of executed move)
tool x: unknown
tool y: unknown

### Plan
Probe tool x in one direction, repeating the step until the shift exceeds the minimum trustworthy
shift, then reverse past the reference by twice that distance. Return to the reference and confirm the
reading repeats. Then do the same for tool y, then run a prediction test before correcting anything.
"""

PROTOCOL = f"""

Every turn you get the whole context document below, then the frame from before your last move, then \
the current frame. The document is all that survives between turns: its facts sections are written by \
the robot and you cannot change them, and its notes section is written only by you.
Reply with ONLY a JSON object:
{{"dx": float, "dy": float, "dz": float, "drx": float, "dry": float, "stage": int, "done": float, \
"notes": str}}
dx, dy, dz is the next translation in meters; drx, dry the next tilt in radians about tool X and Y \
(rotating about the TCP). Both are in the start tool frame. stage is the stage you are working on, 0 \
while calibrating. done is the fraction of the task complete: 0, then 0.5 as stage 5 says, then 1.0 as \
stage 7 says; the run ends at 1.0 and no move is made then.
Translations are clamped to {MAX_STEP} m per turn, +-{MAX_OFFSET} m in tool x and y from the start \
pose, +-{MAX_DESCENT} m in tool z and never below {MAX_DESCENT} m under the start height; tilts to \
{MAX_TILT_STEP:.3f} rad per turn and +-{MAX_TILT[0]:.3f} rad about tool X, +-{MAX_TILT[1]:.3f} rad about \
tool Y from the start.
"notes" replaces the notes section wholesale: write it out in full every turn, keeping its headings. \
Anything you leave out is lost."""

NOTES_HEADING = "## Notes (written by you, the only thing you carry between turns)\n"


def inherit_notes():
    # Offer the latest context files; the chosen one's notes section seeds this run's notes.
    paths = sorted(glob.glob("VLM-context-*.md"))[-9:][::-1]
    if not paths:
        return None, NOTES_TEMPLATE
    for i, path in enumerate(paths, 1):
        print(f"  {i}) {path}")
    choice = input(f"Continue from last session? [1-{len(paths)}/n] ").strip()
    if not (choice.isdigit() and 1 <= int(choice) <= len(paths)):
        return None, NOTES_TEMPLATE
    path = paths[int(choice) - 1]
    with open(path) as f:
        return path, f.read().split(NOTES_HEADING, 1)[1]


def encode(frame):
    ok, jpg = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(jpg.tobytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def ask_vlm(client, context, prev_frame, frame):
    content = [{"type": "text", "text": context}]
    if prev_frame is not None:
        content += [{"type": "text", "text": "PREVIOUS frame, before your last move:"}, encode(prev_frame)]
    content += [{"type": "text", "text": "CURRENT frame, now:"}, encode(frame)]
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + PROTOCOL},
            {"role": "user", "content": content},
        ],
    )
    return json.loads(response.choices[0].message.content)


def main():
    print("=== Fully autonomous visual servo ===")
    print(f"Target description sent to {OPENAI_MODEL}: \"{TARGET_DESC}\" (set via TARGET_DESC env var).")
    print("System prompt overridable via SYSTEM_PROMPT env var. Requires OPENAI_API_KEY in the environment.")
    print(f"Moves: <= {MAX_STEP * 100:.0f} cm/turn at {MAX_SPEED * 1000:.0f} mm/s, "
          f"within +-{MAX_OFFSET * 100:.0f} cm of the start pose in x/y and <= {MAX_DESCENT * 100:.0f} cm below it, "
          f"tilt <= {np.degrees(MAX_TILT_STEP):.0f} deg/turn and +-{np.degrees(MAX_TILT).round().tolist()} deg about x/y, "
          f"stage 3 rotation {np.degrees(ROTATE_RIGHT):.0f} deg about z.")
    print("Top-right circle/arrow = last executed tool-frame XY move, inverted; the one left of it = tilt (rx, ry).")
    print("Press 'q' in the video window (it must have focus) to stop; halts a move mid-way, during a VLM call "
          "takes effect after it. Ctrl+C also shuts down cleanly.")

    inherited, notes = inherit_notes()

    UR_IP = os.environ.get("UR_IP", "192.168.1.20")
    camera = Camera(UR_IP)

    from openai import OpenAI

    client = OpenAI()
    rtde_c = rtde_control.RTDEControlInterface(UR_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(UR_IP)

    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_name = os.path.splitext(os.path.basename(__file__))[0]
    log_path, context_path = f"{log_name}-{stamp}.log", f"VLM-context-{stamp}.md"
    frames_dir = f"{log_name}-{stamp}"
    os.makedirs(frames_dir)
    log_file = open(log_path, "w")
    print(f"Logging moves to {log_path}, VLM context to {context_path}, frames to {frames_dir}/")

    def show(frame, quit_key="q"):
        cv2.imshow("wrist camera", frame)
        return cv2.waitKey(1) & 0xFF == ord(quit_key)

    start = rtde_r.getActualTCPPose()
    p0 = np.array(start[:3])
    R0 = cv2.Rodrigues(np.array(start[3:]))[0]
    offset, tilt, rz = np.zeros(3), np.zeros(2), 0.0
    rows, turn, prev_frame, error, stage = [], 0, None, None, None

    h, w = camera.get_frame().shape[:2]
    header = f"""# Autonomous visual servo context, started {stamp}

## Setup (facts, written by the robot)
- Target: {TARGET_DESC}
- Image: {w} x {h} px; center ({w / 2:.0f}, {h / 2:.0f}). x px counts right from the left edge, \
y px counts down from the top edge.
- Assume your reading of the target's center is only accurate to about +-{READ_ERR_FRAC * w:.0f} px.
- Minimum trustworthy probe shift: {MIN_SHIFT_FRAC * w:.0f} px. Below that, treat a probe as noise.
- Distances are meters and angles radians in the start tool frame, whose origin is the pose at startup."""
    if inherited:
        header += (f"\n- This is a NEW RUN that inherited its notes from {inherited}. Everything else was reset: "
                   "the start pose (so offset, tilt and rotation read zero again), the move log and the turn "
                   "count. The movement limits and instructions may also have changed since that run, even if "
                   "your calibration has not.")

    def write_context():
        current = (f"- Turn: {turn}\n"
                   f"- Offset from start (tool x, y, z, m): {offset.round(4).tolist()}\n"
                   f"- Tilt from start (about tool x, y, rad): {tilt.round(4).tolist()}\n"
                   f"- Rotation from start (about tool z, rad): {rz:.4f}\n"
                   f"- Travel left before the limits: "
                   f"x/y +-{MAX_OFFSET} m, {MAX_DESCENT} m below the start height, "
                   f"tilt +-{MAX_TILT[0]:.3f}/{MAX_TILT[1]:.3f} rad about tool x/y")
        if error:
            current += f"\n- PROBLEM WITH YOUR LAST REPLY: {error}"
        table = ["| turn | requested dx,dy,dz,drx,dry | executed dx,dy,dz,drx,dry,drz | offset after | tilt after |",
                 "|---|---|---|---|---|", *rows[-LOG_ROWS:]] if rows else ["(no moves yet)"]
        context = "\n\n".join([
            header,
            "## Current state (facts, written by the robot)\n" + current,
            "## Move log (facts, written by the robot)\n" + "\n".join(table),
            NOTES_HEADING + notes,
        ])
        with open(context_path, "w") as f:
            f.write(context)
        return context

    def hud(img):
        # Move arrow is inverted (points where the image content goes); tilt arrow is raw (rx, ry).
        draw_direction_hud(img, -d[0], -d[1], MAX_STEP)
        draw_direction_hud(img, dr[0], dr[1], MAX_TILT_STEP, slot=1)

    try:
        while True:
            turn += 1
            frame = camera.get_frame()
            cv2.imwrite(f"{frames_dir}/{turn}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png", frame)
            try:
                cmd = ask_vlm(client, write_context(), prev_frame, frame)
                requested = [float(cmd.get(k, 0.0)) for k in ("dx", "dy", "dz", "drx", "dry")]
                d = clip_norm(np.array(requested[:3]), MAX_STEP)
                dr = np.clip(requested[3:], -MAX_TILT_STEP, MAX_TILT_STEP)
                new_stage, done = int(cmd.get("stage", 0)), float(cmd.get("done", 0.0))
            except (ValueError, TypeError, AttributeError) as e:
                print(f"VLM: invalid reply ({e}), not moving")
                error = f"it was not valid JSON in the required format ({e}); no move was made"
                if show(frame):
                    break
                continue

            error = None
            notes = str(cmd.get("notes", notes))
            print(f"VLM: move={d.round(4).tolist()} tilt={dr.round(4).tolist()} stage={new_stage} done={done:g}")
            if new_stage != stage:
                print(f"=== Stage {new_stage} ===" if new_stage else "=== Calibration ===")
                stage = new_stage

            hud(view := frame.copy())  # a copy, so the frame sent to the VLM as PREVIOUS stays clean
            if show(view):
                break
            if done >= 1:
                print("VLM: reports goal reached")
                break

            lim = np.array([MAX_OFFSET, MAX_OFFSET, MAX_DESCENT])  # tool z gets the descent range; base Z floored below
            target = p0 + R0 @ np.clip(offset + d, -lim, lim)
            target[2] = max(target[2], p0[2] - MAX_DESCENT)
            new_offset = R0.T @ (target - p0)
            d, offset = new_offset - offset, new_offset
            new_tilt = np.clip(tilt + dr, -MAX_TILT, MAX_TILT)
            dr, tilt = new_tilt - tilt, new_tilt
            drz = ROTATE_RIGHT if stage == 3 and rz == 0.0 else 0.0  # stage 3 rotation, once
            rz += drz
            # Tilt about the start tool X/Y, then roll about the tilted tool Z (the camera's view axis).
            R_target = R0 @ cv2.Rodrigues(np.array([*tilt, 0.0]))[0] @ cv2.Rodrigues(np.array([0.0, 0.0, rz]))[0]
            rtde_c.moveL([*target.tolist(), *cv2.Rodrigues(R_target)[0].ravel().tolist()],
                         speed=MAX_SPEED, acceleration=ACCEL, asynchronous=True)

            def remaining(pose):
                R_err = cv2.Rodrigues(np.array(pose[3:]))[0].T @ R_target
                return np.linalg.norm(np.array(pose[:3]) - target), np.linalg.norm(cv2.Rodrigues(R_err)[0])

            # Async so 'q' is polled mid-move; done once the TCP is within 1 mm and 0.5 deg of target.
            while (err := remaining(pose := rtde_r.getActualTCPPose()))[0] > 1e-3 or err[1] > np.radians(0.5):
                actual = np.array(pose[:3])
                hud(live := camera.get_frame())
                if show(live):
                    rtde_c.stopL()
                    print(f"q: stopped mid-move at offset {(R0.T @ (actual - p0)).round(4).tolist()} "
                          f"(target {offset.round(4).tolist()})")
                    return
            time.sleep(SETTLE_S)

            executed = [*d.round(4).tolist(), *dr.round(4).tolist(), round(drz, 4)]
            rows.append(f"| {turn} | {','.join(f'{v:+.4f}' for v in requested)} | "
                        f"{','.join(f'{v:+.4f}' for v in executed)} | "
                        f"{offset.round(4).tolist()} | {tilt.round(4).tolist()} |")
            prev_frame = frame
            log_file.write(f"{datetime.now().isoformat()},{','.join(f'{v:.6f}' for v in executed)},{json.dumps(cmd)}\n")
    except KeyboardInterrupt:
        print("\nCtrl+C: shutting down")
    finally:
        write_context()
        rtde_c.stopL()  # halts a move in flight (Ctrl+C mid-move); harmless when idle
        rtde_c.speedStop()
        rtde_c.stopScript()
        cv2.destroyAllWindows()
        log_file.close()


if __name__ == "__main__":
    main()
