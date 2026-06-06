"""
End-to-end pipeline: frames/video -> detect -> track -> annotate -> MP4.

Also writes MOT-format results (one row per box per frame) so the output can
be scored with the standard CLEAR-MOT / HOTA tooling, and measures throughput
separately for the detector and the tracker so you can see where time goes.
"""

import os
import glob
import time

import cv2
import numpy as np

from .detector import Detector
from .byte_tracker import BYTETracker
from .visualize import Visualizer


def _frame_iter(source):
    """Yield (frame_index, BGR image). Accepts a video file OR a folder of
    images (VisDrone sequences are folders of numbered .jpg)."""
    if os.path.isdir(source):
        files = sorted(
            glob.glob(os.path.join(source, "*.jpg"))
            + glob.glob(os.path.join(source, "*.png"))
        )
        for i, f in enumerate(files, start=1):
            img = cv2.imread(f)
            if img is not None:
                yield i, img
    else:
        cap = cv2.VideoCapture(source)
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            i += 1
            yield i, img
        cap.release()


def _source_fps_and_size(source):
    if os.path.isdir(source):
        files = sorted(
            glob.glob(os.path.join(source, "*.jpg"))
            + glob.glob(os.path.join(source, "*.png"))
        )
        if not files:
            raise FileNotFoundError(f"No images found in {source}")
        h, w = cv2.imread(files[0]).shape[:2]
        return 30.0, (w, h), len(files)
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, (w, h), n


def run(
    source,
    weights,
    output="output.mp4",
    mot_txt=None,
    conf=0.25,
    iou=0.6,
    imgsz=1280,
    device="cpu",
    classes=(0,),
    half=False,
    use_sahi=False,
    slice_size=640,
    overlap=0.2,
    track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    use_cmc=True,
    cmc_downscale=2,
    tail_length=30,
    max_frames=None,
    detect_every=1,
):
    src_fps, (w, h), n_total = _source_fps_and_size(source)

    detector = Detector(
        weights, conf=conf, iou=iou, imgsz=imgsz, device=device,
        classes=list(classes) if classes is not None else None,
        half=half, use_sahi=use_sahi,
        slice_height=slice_size, slice_width=slice_size, overlap_ratio=overlap,
    )
    tracker = BYTETracker(
        track_thresh=track_thresh, track_buffer=track_buffer,
        match_thresh=match_thresh, frame_rate=src_fps,
        use_cmc=use_cmc, cmc_downscale=cmc_downscale,
    )
    vis = Visualizer(tail_length=tail_length)

    writer = cv2.VideoWriter(
        output, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h)
    )
    mot_file = open(mot_txt, "w") if mot_txt else None

    det_time = trk_time = 0.0
    n_done = 0
    n_detect = 0
    detect_every = max(1, int(detect_every))
    t_wall0 = time.time()

    for idx, frame in _frame_iter(source):
        run_detector = (n_done % detect_every == 0)
        t0 = time.time()
        if run_detector:
            dets = detector.detect(frame)
            t1 = time.time()
            tracks = tracker.update(dets, frame=frame if use_cmc else None)
            t2 = time.time()
            det_time += (t1 - t0)
            trk_time += (t2 - t1)
            n_detect += 1
        else:
            # skipped frame: let the tracker coast (Kalman + CMC), no detection
            t1 = time.time()
            tracks = tracker.coast(frame=frame if use_cmc else None)
            t2 = time.time()
            trk_time += (t2 - t1)

        n_done += 1

        inst_fps = 1.0 / max(1e-6, (t2 - t0))
        vis.draw(frame, tracks, fps=inst_fps)
        writer.write(frame)

        if mot_file:
            for t in tracks:
                x1, y1, x2, y2 = t.tlbr
                mot_file.write(
                    f"{idx},{t.track_id},{x1:.2f},{y1:.2f},"
                    f"{x2 - x1:.2f},{y2 - y1:.2f},{t.score:.4f},-1,-1,-1\n"
                )

        if n_done % 50 == 0:
            print(f"  frame {n_done}/{n_total}  ~{n_done / (time.time() - t_wall0):.1f} FPS")
        if max_frames and n_done >= max_frames:
            break

    writer.release()
    if mot_file:
        mot_file.close()

    wall = time.time() - t_wall0
    stats = {
        "frames": n_done,
        "detections_run": n_detect,
        "detect_every": detect_every,
        "wall_seconds": round(wall, 2),
        "pipeline_fps": round(n_done / wall, 2) if wall else 0.0,
        "detector_fps": round(n_detect / det_time, 2) if det_time else 0.0,
        "tracker_fps": round(n_done / trk_time, 2) if trk_time else 0.0,
        "detector_ms_per_frame": round(1000 * det_time / n_detect, 2) if n_detect else 0.0,
        "tracker_ms_per_frame": round(1000 * trk_time / n_done, 2) if n_done else 0.0,
        "output": output,
    }
    return stats
