#!/usr/bin/env python3
"""
Fine-tune YOLOv8 on VisDrone vehicles, tuned for SMALL objects.

What we change vs. an off-the-shelf COCO model (this is the "added value"):

  * Domain fine-tuning. COCO vehicles are large and ground-level; VisDrone
    vehicles are small and seen top-down. We start from COCO weights and adapt.

  * High training resolution (--imgsz 1280, even 1536 if VRAM allows). Small
    objects need pixels; this is the single biggest lever.

  * Optional P2 head (--p2). Standard YOLOv8 detects at strides 8/16/32. The
    P2 variant adds a stride-4 head, roughly doubling spatial resolution at the
    detection level -- built specifically for very small targets. It costs a
    little speed but markedly improves tiny-object recall.

  * Augmentations that help small/aerial objects: mosaic (forces the model to
    see objects at many scales/positions), copy-paste, mild scale jitter, and
    we DON'T over-rotate (drone footage is roughly nadir but upright).

  * close_mosaic in the final epochs so the model finishes on clean images.

The result stays tiny: YOLOv8n ~6 MB, YOLOv8s ~22 MB, even the P2 variants are
well under the 300 MB budget.

Usage:
    python scripts/train.py --data /data/visdrone_yolo/visdrone_vehicle.yaml \
        --model yolov8s.pt --imgsz 1280 --epochs 60 --batch 8
    # smaller / faster:
    python scripts/train.py --data ... --model yolov8n.pt --imgsz 1024 --epochs 50
    # extra small-object head:
    python scripts/train.py --data ... --model yolov8s-p2.yaml --imgsz 1280
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to visdrone_vehicle.yaml")
    ap.add_argument("--model", default="yolov8s.pt",
                    help="yolov8n.pt | yolov8s.pt | yolov8s-p2.yaml | yolov8n-p2.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=0, help="GPU index, or 'cpu'")
    ap.add_argument("--project", default="runs/aerial_guardian")
    ap.add_argument("--name", default="yolov8_visdrone_vehicle")
    ap.add_argument("--p2", action="store_true",
                    help="shortcut: swap a .pt model for its -p2 .yaml variant")
    args = ap.parse_args()

    from ultralytics import YOLO

    model_arg = args.model
    if args.p2 and model_arg.endswith(".pt"):
        # e.g. yolov8s.pt -> yolov8s-p2.yaml (architecture only, trains from scratch
        # on the new head but you can pass pretrained weights via `model.load`)
        model_arg = model_arg.replace(".pt", "-p2.yaml")

    model = YOLO(model_arg)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,

        # --- small-object / aerial-friendly recipe ---
        mosaic=1.0,
        close_mosaic=10,      # disable mosaic for last 10 epochs
        copy_paste=0.3,
        scale=0.5,            # scale jitter +/- 50%
        degrees=0.0,          # don't rotate -- aerial frames stay upright
        translate=0.1,
        fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,

        # detection-friendly optimisation
        optimizer="auto",
        lr0=0.01,
        patience=20,          # early stop
        cos_lr=True,
        plots=True,
    )

    # Export a lightweight ONNX too (handy for later TensorRT / edge work).
    try:
        model.export(format="onnx", imgsz=args.imgsz, simplify=True)
    except Exception as e:
        print(f"(ONNX export skipped: {e})")


if __name__ == "__main__":
    main()
