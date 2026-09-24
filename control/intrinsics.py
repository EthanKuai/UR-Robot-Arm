## Camera intrinsics for the Robotiq Wrist Camera (RWC-CAM-001) as served by capture.Camera.
#
# Robotiq does not publish these. They were recovered from the factory calibration embedded in
# Robotiq_Wrist_Camera-1.3.2.urcap and re-solved from its 3520 raw grid correspondences with
# cv2.calibrateCamera (0.33 px RMS over 10 poses). Full derivation, evidence and caveats:
# ../robotiq-specs.md
#
# These apply to the 611x459 frames capture.Camera returns. Robotiq's docs quote 616x463 for the
# same endpoint, so check frame.shape before trusting them: the underlying calibration is
# fx = fy = 1085.2 on the full 1280x960 sensor frame with the principal point at its centre, and
# everything below is that scaled by 611/1280. For any other frame size, rescale by width/1280.
#
# Two things to know before using these for metric work:
#   - The lens is a liquid autofocus lens, so focal length varies with the focus command. A
#     calibration at 0.5 m is wrong at 0.1 m by several percent. Calibrate at your working
#     distance if you need real numbers; treat these as a prior.
#   - The calibration grid covered only the central ~half of the frame, so distortion outside
#     that is extrapolated. k1 is the only distortion term worth keeping.
import numpy as np

FRAME_SIZE = (611, 459)  # width, height these were derived for

# Principal point is the image centre. The free fit put it ~9 px off centre in the 1280-wide
# frame, but the grid never reached a corner so that offset is inside the uncertainty.
K = np.array([[518.0,   0.0, 305.5],
              [  0.0, 518.9, 229.5],
              [  0.0,   0.0,   1.0]])

# OpenCV plumb-bob, k1 only. The full fit's k2/k3 (0.71, -3.11) are overfit to the covered region.
DIST = np.array([-0.008, 0.0, 0.0, 0.0, 0.0])

# Implied field of view on the full 1280x960 frame: 61 deg horizontal, 48 deg vertical. Robotiq's
# datasheet FoV table implies 50 x 39 deg instead; that table describes a centred sub-rectangle,
# not the full frame. See ../robotiq-specs.md.
