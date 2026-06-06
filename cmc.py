"""
Camera Motion Compensation (CMC) -- the core adaptation for a *moving* drone.

Plain ByteTrack/SORT assumes a roughly static camera: the Kalman filter
predicts where a target will be next frame using only the target's own
velocity. On a drone the whole scene shifts because the *camera* moves
(pan / roll / altitude change). The predicted box then lands in the wrong
place, IoU with the real detection collapses, the track is dropped, and a
brand-new ID is created -> an "ID switch".

CMC fixes this by estimating the frame-to-frame *global* motion (a 2x3
affine: rotation + scale + translation) and warping every track's predicted
state into the new frame's coordinate system *before* association.

Pipeline per frame:
  1. Convert to greyscale, optionally downscale (speed).
  2. Detect sparse corners (goodFeaturesToTrack) on the previous frame.
  3. Track them into the current frame (pyramidal Lucas-Kanade optical flow).
  4. Robustly fit an affine transform with RANSAC (rejects moving vehicles /
     pedestrians as outliers, keeping only the static background motion).

This is the same idea ("GMC") used inside BoT-SORT; implemented here from
OpenCV primitives so it stays dependency-light and runs fine on CPU/Jetson.
"""

import cv2
import numpy as np


class CameraMotionCompensation:
    def __init__(self, downscale=2, max_corners=1000, quality=0.01, min_distance=1):
        self.downscale = max(1, int(downscale))
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality,
            minDistance=min_distance,
            blockSize=3,
        )
        self.prev_gray = None

    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            h, w = gray.shape
            gray = cv2.resize(gray, (w // self.downscale, h // self.downscale))
        return gray

    def apply(self, frame):
        """Return a 2x3 affine matrix mapping previous-frame -> current-frame.

        On the first call (no history) it returns identity.
        """
        H = np.eye(2, 3, dtype=np.float32)
        gray = self._preprocess(frame)

        if self.prev_gray is None:
            self.prev_gray = gray
            return H

        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
        if prev_pts is None or len(prev_pts) < 8:
            self.prev_gray = gray
            return H

        # Track corners forward with optical flow.
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None
        )
        if curr_pts is None:
            self.prev_gray = gray
            return H

        status = status.reshape(-1).astype(bool)
        good_prev = prev_pts.reshape(-1, 2)[status]
        good_curr = curr_pts.reshape(-1, 2)[status]

        if len(good_prev) >= 8:
            M, _ = cv2.estimateAffinePartial2D(
                good_prev, good_curr, method=cv2.RANSAC,
                ransacReprojThreshold=3,
            )
            if M is not None:
                H = M.astype(np.float32)
                # Translation was estimated on the downscaled image; rescale it
                # back to full resolution. Rotation/scale are scale-invariant.
                if self.downscale > 1:
                    H[0, 2] *= self.downscale
                    H[1, 2] *= self.downscale

        self.prev_gray = gray
        return H
