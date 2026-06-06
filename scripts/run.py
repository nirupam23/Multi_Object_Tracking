#!/usr/bin/env python3
"""
Run the Aerial Guardian pipeline on a VisDrone sequence (folder of images) or
any video, and print an FPS + hardware report.

Examples
--------
# Track vehicles in one VisDrone-MOT-val sequence, on GPU, with CMC:
python scripts/run.py \
    --source /data/VisDrone2019-MOT-val/sequences/uav0000086_00000_v \
    --weights runs/aerial_guardian/yolov8_visdrone_vehicle/weights/best.pt \
    --output out_uav0000086.mp4 --device 0 --imgsz 1280

# No fine-tuned weights yet? Use a stock COCO model and keep the COCO vehicle
# classes (car=2, motorcycle=3, bus=5, truck=7):
python scripts/run.py --source <seq> --weights yolov8s.pt \
    --classes 2 3 5 7 --output out.mp4 --device cpu --imgsz 1280

# CPU-only (laptop), smaller resolution for speed:
python scripts/run.py --source clip.mp4 --weights best.pt \
    --output out.mp4 --device cpu --imgsz 960

# Maximum tiny-object recall with SAHI tiling (slower):
python scripts/run.py --source <seq> --weights best.pt \
    --use-sahi --slice-size 640 --output out_sahi.mp4
"""

import argparse
import json
import os
import platform
import sys

# Make the repo root importable so `python scripts/run.py` works from anywhere
# (no need to set PYTHONPATH).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aerial_guardian.pipeline import run  # noqa: E402


def hardware_string(device):
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        if str(device) != "cpu" and torch.cuda.is_available():
            info["device"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
        else:
            info["device"] = f"CPU ({platform.processor() or 'unknown'})"
    except Exception:
        info["device"] = str(device)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="image folder OR video file")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--output", default="output.mp4")
    ap.add_argument("--mot-txt", default=None, help="also write MOT-format results")

    # detection
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--classes", type=int, nargs="+", default=[0])
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--use-sahi", action="store_true")
    ap.add_argument("--slice-size", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)

    # tracking
    ap.add_argument("--track-thresh", type=float, default=0.5)
    ap.add_argument("--track-buffer", type=int, default=30)
    ap.add_argument("--match-thresh", type=float, default=0.8)
    ap.add_argument("--no-cmc", action="store_true",
                    help="disable camera motion compensation (for ablation)")
    ap.add_argument("--cmc-downscale", type=int, default=2)
    ap.add_argument("--tail-length", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--detect-every", type=int, default=1,
                    help="run the detector every Nth frame and let the tracker "
                         "coast (Kalman+CMC) between; N=2-3 boosts FPS with "
                         "little quality loss on drone video")
    args = ap.parse_args()

    hw = hardware_string(args.device)
    print("Hardware:", json.dumps(hw))
    print("Running...")

    stats = run(
        source=args.source,
        weights=args.weights,
        output=args.output,
        mot_txt=args.mot_txt,
        conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device,
        classes=tuple(args.classes), half=args.half,
        use_sahi=args.use_sahi, slice_size=args.slice_size, overlap=args.overlap,
        track_thresh=args.track_thresh, track_buffer=args.track_buffer,
        match_thresh=args.match_thresh, use_cmc=not args.no_cmc,
        cmc_downscale=args.cmc_downscale, tail_length=args.tail_length,
        max_frames=args.max_frames,
        detect_every=args.detect_every,
    )

    print("\n================  REPORT  ================")
    print(json.dumps({"hardware": hw, **stats}, indent=2))
    print("==========================================")


if __name__ == "__main__":
    main()
