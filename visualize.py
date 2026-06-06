"""
Drawing utilities: bounding boxes, unique ID labels, fading trajectory tails,
and a live FPS / count overlay.
"""

from collections import defaultdict, deque

import cv2
import numpy as np

# A fixed, perceptually-spread palette so IDs keep stable, distinct colours.
_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]


def color_for(track_id):
    return _PALETTE[track_id % len(_PALETTE)]


class Visualizer:
    def __init__(self, tail_length=30, box_thickness=2):
        self.tail_length = tail_length
        self.box_thickness = box_thickness
        self.tails = defaultdict(lambda: deque(maxlen=tail_length))

    def draw(self, frame, tracks, fps=None):
        """
        frame  : BGR image (modified in place and returned)
        tracks : list of STrack (must expose .track_id, .tlbr, .score)
        fps    : optional float to overlay
        """
        active = set()
        for t in tracks:
            tid = t.track_id
            active.add(tid)
            x1, y1, x2, y2 = t.tlbr.astype(int)
            cx, cy = (x1 + x2) // 2, y2  # anchor tail at feet (bottom-centre)
            self.tails[tid].append((cx, cy))
            col = color_for(tid)

            # fading trajectory tail
            pts = list(self.tails[tid])
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                thick = max(1, int(2 * alpha))
                cv2.line(frame, pts[i - 1], pts[i], col, thick, cv2.LINE_AA)

            # bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, self.box_thickness)

            # id label with filled background for readability
            label = f"{tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), col, -1)
            cv2.putText(
                frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

        # drop tails for ids no longer present (keeps memory bounded)
        for tid in list(self.tails.keys()):
            if tid not in active:
                self.tails[tid].clear()

        # HUD
        hud = f"tracks: {len(tracks)}"
        if fps is not None:
            hud = f"FPS: {fps:5.1f}   " + hud
        cv2.rectangle(frame, (0, 0), (260, 28), (0, 0, 0), -1)
        cv2.putText(
            frame, hud, (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
        )
        return frame
