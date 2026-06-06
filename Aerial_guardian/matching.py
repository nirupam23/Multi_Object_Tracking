"""
Association helpers: IoU cost matrices + Hungarian linear assignment.

We use scipy.optimize.linear_sum_assignment instead of the `lap` package so
there is no compiled dependency to build on a Jetson / fresh machine.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def ious(atlbrs, btlbrs):
    """Pairwise IoU between two sets of boxes in [x1, y1, x2, y2] form."""
    A, B = len(atlbrs), len(btlbrs)
    if A == 0 or B == 0:
        return np.zeros((A, B), dtype=np.float32)

    a = np.ascontiguousarray(atlbrs, dtype=np.float32)
    b = np.ascontiguousarray(btlbrs, dtype=np.float32)

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[..., 0] * wh[..., 1]

    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def iou_distance(atracks, btracks):
    """1 - IoU cost matrix. Accepts STrack lists or raw [x1,y1,x2,y2] arrays."""
    if len(atracks) and not isinstance(atracks[0], np.ndarray):
        atlbrs = [t.tlbr for t in atracks]
    else:
        atlbrs = atracks
    if len(btracks) and not isinstance(btracks[0], np.ndarray):
        btlbrs = [t.tlbr for t in btracks]
    else:
        btlbrs = btracks

    return 1.0 - ious(np.asarray(atlbrs), np.asarray(btlbrs))


def linear_assignment(cost_matrix, thresh):
    """Solve the assignment problem; drop matches with cost above `thresh`.

    Returns (matches Nx2, unmatched_a tuple, unmatched_b tuple).
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = [
        [r, c] for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] <= thresh
    ]
    matches = np.asarray(matches) if matches else np.empty((0, 2), dtype=int)

    matched_rows = set(matches[:, 0].tolist()) if len(matches) else set()
    matched_cols = set(matches[:, 1].tolist()) if len(matches) else set()
    unmatched_a = tuple(r for r in range(cost_matrix.shape[0]) if r not in matched_rows)
    unmatched_b = tuple(c for c in range(cost_matrix.shape[1]) if c not in matched_cols)
    return matches, unmatched_a, unmatched_b
