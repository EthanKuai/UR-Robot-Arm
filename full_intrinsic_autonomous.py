## Fully autonomous visual servo with measured feedback: the VLM sees the wrist camera and, every turn,
# determine_transform's measurement of the target against its template (center, scale, tilt, roll), and
# drives the TCP itself. A copy of full_autonomous.py with the estimator piped into the context document
# and the protocol rewritten around it: one probe per axis instead of long eye-read baselines, and stage
# 2 tilts until the measured tilt reads TARGET_TILT_DEG rather than until the target looks centred.
#
# The estimator reports everything relative to the template's capture pose (see determine_transform.py),
# so tool.png must be captured with the camera square to the face for "tilt" to mean degrees off it.
#
# Memory lives in VLM-intrinsic-context-<datetime>.md, rewritten every turn and fed back in full as the
# next turn's prompt. Its facts sections are written by this script (ground truth: executed moves,
# offset, tilt, rotation, and the measured pose per turn); the notes section is written by the VLM and
# is the only thing it carries between turns. At startup a previous run's context file can be continued
# (arm has not moved: state, logs and notes carry on) or learned from (new arm position: only its gains
# and forward-tilt direction are kept), so the VLM does not recalibrate from zero. Each turn also sends
# two images, previous and current; the raw frame of every turn is saved to
# full_intrinsic_autonomous-<datetime>/<turn>-<datetime>.png.
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
import re
import time
from datetime import datetime

import cv2
import numpy as np
import rtde_control
import rtde_receive

from capture import Camera
from determine_transform import TEMPLATE_PATH, draw_overlay, estimate_transform
from visual_servo import ACCEL, OPENAI_MODEL, SETTLE_S, clip_norm, draw_direction_hud

TARGET_DESC = os.environ.get("TARGET_DESC", "a flat metal circle with 9 drill holes")
TARGET_TILT_DEG = 45  # stage 2 ends when the measured tilt of the target's face reads this
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", f"""\
You control a UR robot arm. A camera on its wrist, mounted in front of the claw and looking roughly \
down, gives you two images per turn: the frame before your last move, and the frame now. Your goal is \
to land the claw on the target.

A pose estimator also measures the target in every frame by matching a template of its face, and the \
context document's Measured table records the result for every turn: the target's center in pixels, \
its scale (apparent size relative to the template; 1.0 at the distance the template was taken from, \
larger = closer), the tilt of the target's face relative to the camera in degrees, the image direction \
the face recedes in, its roll, and the fit's ambiguity. These numbers are far better than your reading \
of the images: the center to about a pixel, the tilt to a degree or two. Trust them over your eyes, and \
over any rule of thumb about where the target should appear in the image. "Not found" means the target \
is out of view, hidden, or the frame was bad: unless you are descending in stage 5, reverse your last \
executed move exactly, and only then fall back to your eyes.

Rules:
- Use the executed move from the move log, never what you asked for. Commands get clamped.
- Change ONE thing per turn while calibrating: one axis, and either a translation or a tilt, so that \
what you measure can be attributed.
- A gain needs a measured center shift above the minimum trustworthy shift in the document. If a probe \
fell short, repeat the same move and use the accumulated baseline.
- If a turn's ambiguity is below 2.0, its tilt direction may be flipped by 180 degrees: trust the tilt \
magnitude that turn, not the direction.
- The notes are for conclusions: the stage, the gains and the scale they were measured at, which tilt \
is forward, what to do next. Measurements are already in the document; do not copy them.

Calibration (gains: measured center shift in px per meter of executed move):
1. Probe tool x by one step. Gain x = center shift / executed dx.
2. Return to the reference and check the measured center comes back within a few pixels.
3. Probe tool y the same way, and return.
4. Verify: predict the measured center for one move, make it, compare. If the prediction is off by \
more than the minimum trustworthy shift, redo the calibration instead of pressing on.
If the context document says this run continues or inherits a previous run, calibration is already \
done: do not repeat it. Continue at the stage in your notes, or, with inherited gains, begin at stage 1.

Working toward the goal:
- Apply about 70% of the correction you compute, and re-measure every turn. Overshoot costs more turns \
than undershoot.
- If a measured shift is less than half or more than twice what you predicted, stop and re-probe.
- Gains scale with distance: when the measured scale has changed by more than about 20% since the gains \
were measured, multiply them by (new scale / old scale).
- If the correction you need is larger than the travel left within the limits, write that in the notes \
and say so, rather than grinding against the limit.

Stages, in order; report the stage you are working on in the "stage" field (0 while calibrating), \
record it in the notes, and finish one before starting the next:
1. Translate so the measured center is horizontally at the image center and the target's lower edge \
just touches the bottom edge of the image.
2. Tilt forward until the measured tilt reads {TARGET_TILT_DEG} degrees. Forward is the tilt direction \
that moves the target up the image: probe one tilt step on one axis to find it, and record it. Judge \
this stage by the measured tilt alone, not by where the target sits in the image.
3. Rotate right 90 degrees. Report stage=3 with all values zero: the robot performs the rotation itself, \
once, and logs it. Your gains then apply to an image rotated by 90 degrees; the measured roll shows it.
4. Translate until the measured center sits at the image center.
5. Descend. Down is whichever z direction makes the measured scale grow; confirm with one small step \
before committing. Descend in small steps, re-measuring each turn, and set done=0.5 when the claw is \
very near the tool: the tool leaves the camera's view entirely and the estimator reports not found. The \
descent limit is a safety floor, not a goal.
6. Do nothing for one turn: all values zero.
7. Tilt back until the camera is perfectly flat, facing the ground again (tilt from start reads 0, 0), \
and set done=1.0.

Never raise done just because you are out of ideas. Write what is wrong in the notes and hold still \
with all values zero.""")

MAX_SPEED = 0.01  # m/s, deliberately very slow
MAX_STEP = 0.04    # max meters per turn
MAX_OFFSET = 0.15  # max meters from the start pose, in tool x/y (tool z gets MAX_DESCENT)
MAX_DESCENT = 0.40  # max meters below the start height (base Z)
MAX_TILT_STEP = np.radians(3)  # max radians per turn, per tilt axis
MAX_TILT = np.radians([45, 30])  # max radians from the start orientation, about tool X, Y
ROTATE_RIGHT = np.radians(-90)  # stage 3 rotation about tool Z; flip the sign if it turns the wrong way
READ_ERR_FRAC = 0.03  # assumed VLM center-reading error by eye, as a fraction of image width
MEASURED_ERR_PX = 2  # assumed error of the estimator's center on real frames (offline it is sub-pixel)
MEASURED_MIN_SHIFT_PX = 10  # a probe that moves the measured center less than this is treated as noise
LOG_ROWS = 30  # move log / measured rows kept in the context document
CONTEXT_PREFIX = "VLM-intrinsic-context-"  # distinct from full_autonomous's, whose files lack the Measured table

NOTES_TEMPLATE = """\
### Stage
{stage}

### Gains (measured center shift in px per meter of executed move, and the scale they were measured at)
{gains}

### Forward tilt (which tilt axis and sign moves the target up the image)
{forward}

### Plan
{plan}
"""
FRESH_NOTES = NOTES_TEMPLATE.format(
    stage="calibration - not started",
    gains="tool x: unknown\ntool y: unknown",
    forward="unknown",
    plan="Probe tool x by one step and read the measured center shift; return to the reference; the same for\n"
         "tool y; then one prediction test before correcting anything.")

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


def measure_row(turn, pose):
    if pose is None:
        return f"| {turn} | not found | - | - | - | - | - | - |"
    return (f"| {turn} | {pose.center[0]:.1f} | {pose.center[1]:.1f} | {pose.scale:.3f} | {pose.tilt_mag:.1f} | "
            f"{pose.tilt_dir:+.0f} | {pose.roll:+.1f} | {pose.ambiguity:.1f} |")


def choose_session():
    # Offer the latest context files. Returns (mode, path): mode "continue" (arm has not moved since
    # that run) or "learn" (new arm position, keep only its gains); (None, None) starts fresh.
    paths = sorted(glob.glob(f"{CONTEXT_PREFIX}*.md"))[-9:][::-1]
    if not paths:
        return None, None
    for i, path in enumerate(paths, 1):
        print(f"  {i}) {path}")
    n = len(paths)
    choice = input(f"Continue a session (c<n>: arm has not moved) or learn its gains (l<n>: new arm position)? "
                   f"[c1-{n}/l1-{n}/n] ").strip().lower()
    mode, idx = {"c": "continue", "l": "learn"}.get(choice[:1]), choice[1:]
    if mode is None or not (idx.isdigit() and 1 <= int(idx) <= n):
        return None, None
    return mode, paths[int(idx) - 1]


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
    print("=== Fully autonomous visual servo with measured pose feedback ===")
    template = cv2.imread(TEMPLATE_PATH)
    if template is None:
        raise SystemExit(f"Can't read template {TEMPLATE_PATH}; crop one with capture_template.py first.")
    print(f"Pose estimator template: {TEMPLATE_PATH} (set via TEMPLATE_PATH env var); stage 2 tilts until the "
          f"measured tilt reads {TARGET_TILT_DEG} deg.")
    print(f"Target description sent to {OPENAI_MODEL}: \"{TARGET_DESC}\" (set via TARGET_DESC env var).")
    print("System prompt overridable via SYSTEM_PROMPT env var. Requires OPENAI_API_KEY in the environment.")
    print(f"Moves: <= {MAX_STEP * 100:.0f} cm/turn at {MAX_SPEED * 1000:.0f} mm/s, "
          f"within +-{MAX_OFFSET * 100:.0f} cm of the start pose in x/y and <= {MAX_DESCENT * 100:.0f} cm below it, "
          f"tilt <= {np.degrees(MAX_TILT_STEP):.0f} deg/turn and +-{np.degrees(MAX_TILT).round().tolist()} deg about x/y, "
          f"stage 3 rotation {np.degrees(ROTATE_RIGHT):.0f} deg about z.")
    print("Top-right circle/arrow = last executed tool-frame XY move, inverted; the one left of it = tilt (rx, ry).")
    print("Press 'q' in the video window (it must have focus) to stop; halts a move mid-way, during a VLM call "
          "takes effect after it. Ctrl+C also shuts down cleanly.")

    mode, inherited = choose_session()

    UR_IP = os.environ.get("UR_IP", "192.168.1.20")
    camera = Camera(UR_IP)

    from openai import OpenAI

    client = OpenAI()
    rtde_c = rtde_control.RTDEControlInterface(UR_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(UR_IP)

    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_name = os.path.splitext(os.path.basename(__file__))[0]
    log_path, context_path = f"{log_name}-{stamp}.log", f"{CONTEXT_PREFIX}{stamp}.md"
    frames_dir = f"{log_name}-{stamp}"
    os.makedirs(frames_dir)
    log_file = open(log_path, "w")
    print(f"Logging moves to {log_path}, VLM context to {context_path}, frames to {frames_dir}/")

    def show(frame, quit_key="q"):
        cv2.imshow("wrist camera", frame)
        return cv2.waitKey(1) & 0xFF == ord(quit_key)

    offset, tilt, rz = np.zeros(3), np.zeros(2), 0.0
    rows, mrows, turn, prev_frame, error, stage, notes = [], [], 0, None, None, None, FRESH_NOTES
    if inherited:
        with open(inherited) as f:
            old = f.read()
    if mode == "learn":
        def section(name):
            return old.split(f"### {name}")[1].split("\n", 1)[1].split("\n### ")[0].strip()
        notes = NOTES_TEMPLATE.format(
            stage=f"calibration - gains inherited from {inherited}; start at stage 1",
            gains=section("Gains"), forward=section("Forward tilt"),
            plan="Calibration is inherited: do not re-probe. Begin stage 1 from the measured center.")
    elif mode == "continue":
        def fact(label, default=None):
            m = re.search(rf"^- {label}[^:]*: (.*)$", old, re.M)
            return m.group(1) if m else default
        turn, rz = int(fact("Turn")), float(fact("Rotation from start", 0.0))
        offset, tilt = np.array(json.loads(fact("Offset from start"))), np.array(json.loads(fact("Tilt from start")))
        rows = re.findall(r"^\| \d.*$", old.split("## Move log")[1].split("## Measured")[0], re.M)
        mrows = re.findall(r"^\| \d.*$", old.split("## Measured")[1].split("## Notes")[0], re.M)
        notes = old.split(NOTES_HEADING, 1)[1]
        old_frames = glob.glob(f"{log_name}-{inherited.removeprefix(CONTEXT_PREFIX).removesuffix('.md')}/*.png")
        if old_frames:
            prev_frame = cv2.imread(max(old_frames, key=lambda p: int(os.path.basename(p).split("-")[0])))

    start = rtde_r.getActualTCPPose()
    p0 = np.array(start[:3])
    R0 = cv2.Rodrigues(np.array(start[3:]))[0]
    if mode == "continue":
        # The arm still sits at that run's last pose, so its start frame is recovered from where we are
        # now: R_now = R0 @ R_tilt @ R_z and p_now = p0 + R0 @ offset.
        R0 = R0 @ cv2.Rodrigues(np.array([0.0, 0.0, rz]))[0].T @ cv2.Rodrigues(np.array([*tilt, 0.0]))[0].T
        p0 = p0 - R0 @ offset

    h, w = camera.get_frame().shape[:2]
    header = f"""# Autonomous visual servo context, started {stamp}

## Setup (facts, written by the robot)
- Target: {TARGET_DESC}
- Image: {w} x {h} px; center ({w / 2:.0f}, {h / 2:.0f}). x px counts right from the left edge, \
y px counts down from the top edge.
- A pose estimator measures the target every turn from the template {TEMPLATE_PATH}; see Measured. Its \
center is accurate to about +-{MEASURED_ERR_PX} px, and the minimum trustworthy probe shift of the measured \
center is {MEASURED_MIN_SHIFT_PX} px. Tilt is 0 when the camera faces the target's face as it did when the \
template was taken. Tilt dir is degrees in the image, 0 = right (+x), 90 = down (+y). Roll is degrees \
clockwise in the image. Ambiguity below 2.0 means tilt dir may be flipped that turn.
- By eye, your reading of the target's center is only accurate to about +-{READ_ERR_FRAC * w:.0f} px.
- Distances are meters and angles radians in the start tool frame, whose origin is the pose at startup."""
    if mode == "continue":
        header += (f"\n- This run CONTINUES {inherited}: the arm has not moved since, so the offset, tilt, "
                   "rotation, move log, turn count and your notes carry on from it unchanged. Only the movement "
                   "limits and instructions may have changed since that run.")
    elif mode == "learn":
        header += (f"\n- This is a NEW RUN from a new arm position. The gains in your notes were measured in "
                   f"{inherited} and are trusted: do not recalibrate. Everything else was reset: the start pose "
                   "(so offset, tilt and rotation read zero), the move log and the turn count. The movement "
                   "limits and instructions may also have changed since that run.")

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
        measured = ["| turn | center x px | center y px | scale | tilt deg | tilt dir deg | roll deg | ambiguity |",
                    "|---|---|---|---|---|---|---|---|", *mrows[-LOG_ROWS:]] if mrows else ["(nothing measured yet)"]
        context = "\n\n".join([
            header,
            "## Current state (facts, written by the robot)\n" + current,
            "## Move log (facts, written by the robot)\n" + "\n".join(table),
            "## Measured (facts, written by the robot's pose estimator on the frame at the start of each turn, "
            "i.e. after the previous turn's move)\n" + "\n".join(measured),
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
            pose = estimate_transform(frame, template)
            mrows.append(measure_row(turn, pose))
            pose_json = None if pose is None else {k: v for k, v in pose._asdict().items() if k not in ("quad", "rvec", "tvec")}
            print("measured: " + (f"center ({pose.center[0]:.1f}, {pose.center[1]:.1f}) scale {pose.scale:.3f} "
                                  f"tilt {pose.tilt_mag:.1f} deg toward {pose.tilt_dir:+.0f} roll {pose.roll:+.1f} "
                                  f"ambiguity {pose.ambiguity:.1f}" if pose else "target not found"))
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

            view = draw_overlay(frame, pose) if pose else frame.copy()  # copies: the VLM's PREVIOUS stays clean
            hud(view)
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
            log_file.write(f"{datetime.now().isoformat()},{','.join(f'{v:.6f}' for v in executed)},{json.dumps(cmd)},"
                           f"{json.dumps(pose_json)}\n")
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
