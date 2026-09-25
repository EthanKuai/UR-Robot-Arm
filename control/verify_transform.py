## Offline checks for determine_transform.py. No robot and no camera needed - everything runs against
# the committed tool.png / frame.png.
#
# The only real frame available is the one tool.png was cropped from, so ground truth is manufactured:
# a frame is warped by the homography a known camera rotation about the tool centre would induce
# (H = K (R + t n^T / d) K^-1, with the rotation taken about a point on the tool plane so the tool
# centre stays put), and the estimator has to recover that rotation angle.
#
# This exercises the geometry and the correspondence search. It cannot exercise appearance: a real
# re-capture also changes lighting, focus and specular highlights, none of which a warp reproduces.
#
# The bar is not "always right" but "never confidently wrong": a case may be recovered accurately or
# rejected (None), and only a wrong answer that passed the quality gate counts as a failure.
import cv2
import numpy as np

from determine_transform import estimate_transform
from intrinsics import K

TILT_TOL_DEG = 3.0      # recovered tilt must land this close to ground truth
AXES = {"X": (1.0, 0.0), "Y": (0.0, 1.0), "XY": (1.0, 1.0), "X-Y": (1.0, -1.0)}
ANGLES = (0, 5, 10, 15, 20, 25, 30, 40)
TOOL_CENTRE = (334.0, 350.0)   # roughly the disc centre in frame.png; the warp pivots here


def tilt_homography(angle_deg, axis, centre):
    # Homography induced on the tool plane by rotating the camera `angle_deg` about `axis`, pivoting on
    # the plane point under `centre`. The pivot makes the tool centre a fixed point of the warp, which
    # is also the cheapest way to check the homography is right.
    ray = np.linalg.inv(K) @ np.array([centre[0], centre[1], 1.0])
    R = cv2.Rodrigues(np.radians(angle_deg) * np.array([axis[0], axis[1], 0.0])
                      / np.linalg.norm(axis))[0]
    return K @ (R + np.outer((np.eye(3) - R) @ ray, [0, 0, 1])) @ np.linalg.inv(K)


def main():
    frame, template = cv2.imread("frame.png"), cv2.imread("tool.png")
    assert frame is not None and template is not None, "need frame.png and tool.png"
    failures, recovered, rejected = [], 0, 0

    # 1. Identity: tool.png is a crop of frame.png, so the frame must read as the capture pose itself.
    pose = estimate_transform(frame, template)
    assert pose is not None, "identity: tool not found in the frame its own template came from"
    print(f"identity      tilt={pose.tilt_mag:5.2f} roll={pose.roll:+6.2f} scale={pose.scale:.4f} "
          f"holes={pose.n_holes} reproj={pose.reproj_px:.3f}")
    for name, value, limit in (("tilt", pose.tilt_mag, 1.0), ("roll", abs(pose.roll), 1.0),
                               ("scale-1", abs(pose.scale - 1.0), 0.01)):
        if value > limit:
            failures.append(f"identity {name}={value:.3f} exceeds {limit}")

    # 2. The homography itself: the pivot must be a fixed point, or the ground truth is wrong.
    p = tilt_homography(20, (1.0, 0.0), TOOL_CENTRE) @ np.array([*TOOL_CENTRE, 1.0])
    if np.linalg.norm(p[:2] / p[2] - np.array(TOOL_CENTRE)) > 1e-6:
        failures.append("tilt_homography does not hold the pivot fixed")

    # 3. Known tilts about four axes.
    print("\n axis  gt |  tilt   err | holes reproj   amb  | verdict")
    for name, axis in AXES.items():
        for gt in ANGLES:
            H = tilt_homography(gt, axis, TOOL_CENTRE)
            warped = cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]))
            pose = estimate_transform(warped, template)
            if pose is None:
                rejected += 1
                print(f"{name:>5s} {gt:3d} |   -       -  |   -      -      -    | rejected")
                continue
            err = abs(pose.tilt_mag - gt)
            ok = err <= TILT_TOL_DEG
            recovered += ok
            if not ok:
                failures.append(f"{name} {gt} deg -> {pose.tilt_mag:.1f} deg "
                                f"(reproj {pose.reproj_px:.2f} px passed the gate)")
            print(f"{name:>5s} {gt:3d} | {pose.tilt_mag:5.1f} {err:5.1f} | {pose.n_holes:3d} "
                  f"{pose.reproj_px:7.3f} {pose.ambiguity:6.2f} | {'ok' if ok else 'WRONG'}")

    # 4. Pure translation and zoom, which must show up as offset and scale but no tilt.
    print()
    for label, M in (("shift +40,-25", np.array([[1.0, 0, 40], [0, 1.0, -25], [0, 0, 1.0]])),
                     ("zoom 1.4x", K @ np.diag([1.4, 1.4, 1.0]) @ np.linalg.inv(K))):
        warped = cv2.warpPerspective(frame, M, (frame.shape[1], frame.shape[0]))
        pose = estimate_transform(warped, template)
        if pose is None:
            failures.append(f"{label}: tool not found")
            continue
        print(f"{label:<14s} centre=({pose.center[0]:6.1f},{pose.center[1]:6.1f}) "
              f"scale={pose.scale:.3f} tilt={pose.tilt_mag:4.1f} roll={pose.roll:+5.1f}")
        if pose.tilt_mag > TILT_TOL_DEG:
            failures.append(f"{label}: reported {pose.tilt_mag:.1f} deg tilt where there is none")

    total = len(AXES) * len(ANGLES)
    print(f"\nrecovered {recovered}/{total} within {TILT_TOL_DEG} deg, {rejected} rejected, "
          f"{total - recovered - rejected} wrong-but-accepted")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
