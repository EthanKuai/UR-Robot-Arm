## Locate the tool's flat face in a wrist-camera frame and measure how its pose differs from the template's.
# The tool is the flat end of a metal cylinder carrying 9 drill holes. The holes are the only thing on it
# worth tracking: the machined face is textureless and specular, so descriptor matching (ORB/SIFT) has
# nothing to hold on to, while the holes are dark, round and in an asymmetric pattern that pins down
# rotation. They are found as blobs, matched against the same pattern in the template, and the
# correspondence is solved for a pose.
#
# Why not a learned detector: YOLO returns an axis-aligned box, i.e. position only. Pose would need a
# 6-DoF head trained on a labelled set of this exact tool. There is one known rigid object here and an
# analytic model of it, which is precisely where geometry beats learning - no training data, sub-pixel
# output, and a residual you can threshold instead of a confidence you cannot interpret.
#
# What the pose is measured against: the template's hole pattern is used as the 3D model, in template
# pixel units, assumed flat and square to the camera. It is neither - tool.png is a crop of an ordinary
# frame, so it carries whatever tilt the camera had at capture. Everything reported is therefore
# RELATIVE TO THE CAPTURE POSE: tilt_mag reads 0 when the camera is back where the template was taken,
# not when the face is truly perpendicular. For servoing back onto the tool that is the useful zero.
#
# Because the model is literally the template's pixel coordinates, the capture distance in model units
# works out to the focal length in pixels, which is what makes `scale` (= f / t_z) a clean relative
# depth: 1.0 at capture height, >1 closer. That matches visual_servo.detect_tool's scale, so it can be
# fed to the same Jacobian rescaling.
#
# Accuracy, measured offline (see verify_transform.py): over synthetic tilts of 0-40 deg about four axes,
# 28 of 32 cases were recovered to within 2.6 deg (usually <1 deg). The 4 failures were gross - wrong
# correspondence sets - and every one of them showed a reprojection error above 10 px while every good
# fit stayed below 0.2 px. MAX_REPROJ_PX sits in that gap, so a bad fit is rejected rather than reported.
#
# Two caveats that offline tests cannot cover:
#   - The synthetic tilts are warps of the one frame available, so they exercise the geometry and the
#     matcher, not appearance changes: real re-captures differ in lighting, focus and specular highlights.
#   - intrinsics.py notes the lens autofocuses, so f drifts by a few percent with working distance.
#     That lands almost entirely on `scale`; tilt is barely affected.
import os
import sys
from typing import NamedTuple

import cv2
import numpy as np

from intrinsics import DIST, FRAME_SIZE, K

TEMPLATE_PATH = os.environ.get("TEMPLATE_PATH", "tool.png")
FRAME_PATH = os.environ.get("FRAME_PATH", "frame.png")
OVERLAY_PATH = os.environ.get("OVERLAY_PATH", "transform.png")

# Blob filter for the drill holes: dark, round, and roughly 10 px across in the template. The area band
# is wide because the tool grows as the camera descends; circularity/inertia are loose enough to keep a
# hole that foreshortening has squashed into an ellipse.
HOLE_MIN_AREA = 5
HOLE_MAX_AREA = 4000
HOLE_MIN_CIRCULARITY = 0.45
HOLE_MIN_INERTIA = 0.25
HOLE_MIN_CONVEXITY = 0.75

SCALE_MIN, SCALE_MAX = 0.4, 3.5   # template-to-frame size range considered, as in visual_servo
SEED_TOL_FRAC = 0.20              # seed inlier radius, as a fraction of the scaled pattern spread
REFINE_TOL_FRAC = 0.08            # tighter radius once a homography is fitted
TOPK_SEEDS = 20                   # distinct seeds carried into refinement
MIN_MATCHES = 5                   # below this the pose is under-constrained; 4 is the bare PnP minimum
MAX_REPROJ_PX = 2.0               # good fits measured <0.2 px, gross failures >10 px


class Pose(NamedTuple):
    dx: float          # normalized image offset of the tool centre, visual_servo's sign convention
    dy: float          # (positive = tool is left of / above centre)
    center: tuple      # tool centre in pixels
    scale: float       # apparent size vs. the template; 1.0 = capture height, >1 = camera closer
    roll: float        # deg, in-plane rotation vs. the template, clockwise in the image
    tilt_mag: float    # deg the face normal has swung away from the template's
    tilt_dir: float    # deg, image-space direction the face recedes in (atan2(y, x), y down)
    n_holes: int       # holes matched
    reproj_px: float   # RMS reprojection error of the accepted solution
    ambiguity: float   # reproj of the mirror solution / reproj of this one; near 1.0 = tilt_dir may flip
    quad: np.ndarray   # the template's outline projected into the frame, 4x2, for drawing
    rvec: np.ndarray   # raw model->camera rotation (Rodrigues) and translation, in template pixel units
    tvec: np.ndarray


def find_holes(img):
    # SimpleBlobDetector thresholds at many levels and keeps blobs that stay stable across them, so it
    # copes with the metal face changing brightness without a tuned global threshold.
    p = cv2.SimpleBlobDetector.Params()
    p.filterByColor, p.blobColor = True, 0
    p.minThreshold, p.maxThreshold, p.thresholdStep = 20, 220, 10
    p.filterByArea, p.minArea, p.maxArea = True, HOLE_MIN_AREA, HOLE_MAX_AREA
    p.filterByCircularity, p.minCircularity = True, HOLE_MIN_CIRCULARITY
    p.filterByInertia, p.minInertiaRatio = True, HOLE_MIN_INERTIA
    p.filterByConvexity, p.minConvexity = True, HOLE_MIN_CONVEXITY
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints = cv2.SimpleBlobDetector.create(p).detect(gray)
    return np.array([k.pt for k in keypoints], np.float64).reshape(-1, 2)


def _assign(predicted, points, tol):
    # Nearest-neighbour correspondence, keeping only pairs closer than tol.
    d = np.linalg.norm(predicted[:, None, :] - points[None, :, :], axis=2)
    nearest = d.argmin(1)
    ok = d[np.arange(len(predicted)), nearest] < tol
    return np.where(ok)[0], nearest[ok], d[np.arange(len(predicted)), nearest][ok]


def _refine(model_pts, frame_pts, mi, qi, rounds=4):
    # Fit a homography to the current correspondences, re-assign every model point through it, repeat.
    # This recovers points the similarity seed missed under strong tilt, and pulls apart a real match
    # from a lucky one: a wrong correspondence set cannot be made to fit a single homography.
    spread = np.linalg.norm(model_pts - model_pts.mean(0), axis=1).mean()
    residual = np.inf
    for _ in range(rounds):
        H, _ = cv2.findHomography(model_pts[mi].reshape(-1, 1, 2), frame_pts[qi].reshape(-1, 1, 2), 0)
        if H is None:
            break
        projected = cv2.perspectiveTransform(model_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        scale = np.sqrt(abs(np.linalg.det(H[:2, :2])))
        mi2, qi2, r = _assign(projected, frame_pts, max(2.0, REFINE_TOL_FRAC * scale * spread))
        if len(mi2) < 4:
            break
        settled = len(mi2) == len(mi) and (mi2 == mi).all() and (qi2 == qi).all()
        mi, qi, residual = mi2, qi2, r.mean()
        if settled:
            break
    return mi, qi, residual


def match_pattern(model_pts, frame_pts):
    # Two point correspondences fix a similarity (scale, rotation, translation), which approximates the
    # true homography closely enough to seed from while tilt is moderate. Every ordered pair from each
    # set is tried and scored by how many of the remaining holes land near a detected blob. With 8-9
    # template holes this is a few tens of thousands of cheap hypotheses, so no sampling is needed.
    spread = np.linalg.norm(model_pts - model_pts.mean(0), axis=1).mean()
    seeds = []
    for i in range(len(model_pts)):
        for j in range(len(model_pts)):
            if i == j:
                continue
            dp = model_pts[j] - model_pts[i]
            lp = np.linalg.norm(dp)
            if lp < 1e-6:
                continue
            for a in range(len(frame_pts)):
                for b in range(len(frame_pts)):
                    if a == b:
                        continue
                    dq = frame_pts[b] - frame_pts[a]
                    lq = np.linalg.norm(dq)
                    scale = lq / lp
                    if not SCALE_MIN <= scale <= SCALE_MAX:
                        continue
                    cos = (dp @ dq) / (lp * lq)
                    sin = (dp[0] * dq[1] - dp[1] * dq[0]) / (lp * lq)
                    S = scale * np.array([[cos, -sin], [sin, cos]])
                    mi, qi, r = _assign((model_pts - model_pts[i]) @ S.T + frame_pts[a],
                                        frame_pts, SEED_TOL_FRAC * scale * spread)
                    if len(mi) >= 4:
                        seeds.append((-len(mi), r.mean(), mi, qi))
    if not seeds:
        return None
    seeds.sort(key=lambda s: (s[0], s[1]))
    # Refine several competing seeds and keep the one that fits best, ranked by inlier count first.
    # Ranking on residual alone is degenerate: a homography through exactly 4 points fits them
    # perfectly, so a 4-point match would always win.
    best, seen = (0, np.inf, None, None), set()
    for _, _, mi, qi in seeds:
        key = (tuple(mi), tuple(qi))
        if key in seen:
            continue
        seen.add(key)
        m, q, r = _refine(model_pts, frame_pts, mi, qi)
        if (len(m), -r) > (best[0], -best[1]):
            best = (len(m), r, m, q)
        if len(seen) >= TOPK_SEEDS:
            break
    return (best[2], best[3]) if best[2] is not None else None


def estimate_transform(frame, template):
    # Returns the frame's pose relative to the template's capture pose, or None if the tool was not
    # found well enough to trust. In a loop, hoist find_holes(template) - it is recomputed here only to
    # keep the call simple.
    if (frame.shape[1], frame.shape[0]) != FRAME_SIZE:
        print(f"determine_transform: frame is {frame.shape[1]}x{frame.shape[0]}, but intrinsics.py was "
              f"derived for {FRAME_SIZE[0]}x{FRAME_SIZE[1]}; rescale K before trusting this.")
    template_pts, frame_pts = find_holes(template), find_holes(frame)
    if len(template_pts) < 4 or len(frame_pts) < 4:
        return None
    matched = match_pattern(template_pts, frame_pts)
    if matched is None:
        return None
    mi, qi = matched
    if len(mi) < MIN_MATCHES:
        return None

    # The model is the template's hole pattern, recentred, taken as a plane at z = 0 in template pixel
    # units. IPPE is the planar solver and returns both members of the mirror pair a flat patch always
    # admits; the better-fitting one is used and their ratio reported as `ambiguity`.
    origin = template_pts.mean(0)
    model = np.hstack([template_pts[mi] - origin, np.zeros((len(mi), 1))])
    ok, rvecs, tvecs, errors = cv2.solvePnPGeneric(
        model, frame_pts[qi].reshape(-1, 1, 2), K, DIST, flags=cv2.SOLVEPNP_IPPE)
    if not ok or len(rvecs) == 0:
        return None
    errors = np.asarray(errors).ravel()
    order = np.argsort(errors)
    rvec, tvec, reproj = rvecs[order[0]], tvecs[order[0]], float(errors[order[0]])
    if reproj > MAX_REPROJ_PX:
        return None
    ambiguity = float(errors[order[1]] / max(reproj, 1e-9)) if len(order) > 1 else float("inf")

    R = cv2.Rodrigues(rvec)[0]
    normal = R[:, 2] * (-1 if R[2, 2] > 0 else 1)   # face normal, oriented back toward the camera
    h, w = frame.shape[:2]
    cx, cy = cv2.projectPoints(np.zeros((1, 3)), rvec, tvec, K, DIST)[0].ravel()
    th, tw = template.shape[:2]
    corners = np.array([[-tw / 2, -th / 2, 0], [tw / 2, -th / 2, 0],
                        [tw / 2, th / 2, 0], [-tw / 2, th / 2, 0]], np.float64)
    quad = cv2.projectPoints(corners, rvec, tvec, K, DIST)[0].reshape(-1, 2)

    return Pose(dx=(w / 2 - cx) / (w / 2), dy=(h / 2 - cy) / (h / 2), center=(float(cx), float(cy)),
                scale=float(K[0, 0] / np.asarray(tvec).ravel()[2]),
                roll=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
                tilt_mag=float(np.degrees(np.arccos(min(1.0, abs(normal[2]))))),
                tilt_dir=float(np.degrees(np.arctan2(normal[1], normal[0]))),
                n_holes=len(mi), reproj_px=reproj, ambiguity=ambiguity,
                quad=quad, rvec=rvec, tvec=tvec)


def draw_overlay(frame, pose):
    out = frame.copy()
    cv2.polylines(out, [np.int32(pose.quad)], True, (0, 255, 0), 2)
    cv2.drawMarker(out, np.int32(pose.center), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
    # Arrow along tilt_dir, its length proportional to tilt_mag: it points the way the face recedes.
    if pose.tilt_mag > 0.5:
        a = np.radians(pose.tilt_dir)
        tip = np.int32(np.array(pose.center) + np.array([np.cos(a), np.sin(a)]) * (4 * pose.tilt_mag + 15))
        cv2.arrowedLine(out, np.int32(pose.center), tip, (0, 0, 255), 2, tipLength=0.3)
    lines = [f"tilt {pose.tilt_mag:.1f} deg toward {pose.tilt_dir:+.0f} deg",
             f"roll {pose.roll:+.1f} deg   scale {pose.scale:.3f}",
             f"offset ({pose.dx:+.3f}, {pose.dy:+.3f})",
             f"{pose.n_holes} holes  reproj {pose.reproj_px:.2f}px  amb {pose.ambiguity:.1f}"]
    for i, text in enumerate(lines):
        cv2.putText(out, text, (8, 18 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(out, text, (8, 18 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


def main():
    live = "--live" in sys.argv
    template = cv2.imread(TEMPLATE_PATH)
    if template is None:
        raise SystemExit(f"Can't read template {TEMPLATE_PATH}; crop one with capture_template.py first.")
    if live:
        from capture import Camera
        frame = Camera().get_frame()
    else:
        frame = cv2.imread(FRAME_PATH)
        if frame is None:
            raise SystemExit(f"Can't read frame {FRAME_PATH}; pass --live to grab one from the camera.")

    print(f"=== determine_transform === template={TEMPLATE_PATH} "
          f"frame={'live camera' if live else FRAME_PATH}")
    pose = estimate_transform(frame, template)
    if pose is None:
        print("Tool not found (too few holes matched, or the fit was rejected as untrustworthy).")
        return
    print(f"  centre     {pose.center[0]:.1f}, {pose.center[1]:.1f} px  "
          f"(normalized offset {pose.dx:+.4f}, {pose.dy:+.4f})")
    print(f"  scale      {pose.scale:.4f}  (1.0 = template capture height, >1 = camera closer)")
    print(f"  roll       {pose.roll:+.2f} deg in-plane vs. template")
    print(f"  tilt       {pose.tilt_mag:.2f} deg, receding toward image direction {pose.tilt_dir:+.1f} deg")
    print(f"  quality    {pose.n_holes} holes matched, reproj {pose.reproj_px:.3f} px, "
          f"ambiguity {pose.ambiguity:.2f}" + ("  <-- mirror solution nearly as good; tilt_dir may be "
          "flipped" if pose.ambiguity < 2.0 else ""))
    cv2.imwrite(OVERLAY_PATH, draw_overlay(frame, pose))
    print(f"  overlay    {OVERLAY_PATH}")


if __name__ == "__main__":
    main()
