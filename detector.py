"""
Detection front-end.

Base model: a fine-tuned Ultralytics YOLOv8 (n/s). YOLOv8 is an anchor-free,
single-stage detector. Its FPN/PAN neck fuses features at strides 8/16/32
(heads P3-P5). On 1080p drone frames a vehicle can be ~20-40 px (a person even
smaller, ~10-20 px), which at stride 32 (P5) is only a cell or two and at
stride 8 (P3) a handful -- so small targets live almost entirely on the
highest-resolution head. We push recall on them in two complementary ways:

  1. High input resolution (--imgsz 1280/1536): more pixels per target.
  2. (optional) SAHI sliced inference: cut the frame into overlapping tiles,
     detect on each tile at native scale, then merge with NMS. A small object
     in a 1080p frame becomes ~2x larger in a 540p tile -- a size the
     detector handles far more reliably. Costs FPS, so it's a flag.

At training time we additionally enable a P2 (stride-4) head -- see
scripts/train.py and configs/ -- which adds a detection level specifically for
very small objects.

This wrapper returns detections as a plain numpy array [N, 5] = x1,y1,x2,y2,score
in full-frame pixel coordinates, ready for the tracker.
"""

import numpy as np


class Detector:
    def __init__(
        self,
        weights,
        conf=0.25,
        iou=0.6,
        imgsz=1280,
        device="cpu",
        classes=None,          # e.g. [0] for a fine-tuned 'vehicle' model
        half=False,            # FP16 on supported GPUs
        use_sahi=False,
        slice_height=640,
        slice_width=640,
        overlap_ratio=0.2,
    ):
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.classes = classes
        self.half = half
        self.use_sahi = use_sahi
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_ratio = overlap_ratio

        from ultralytics import YOLO

        self.model = YOLO(weights)

        self.sahi_model = None
        if use_sahi:
            from sahi import AutoDetectionModel

            # SAHI renamed the ultralytics backend across versions; try both.
            for mtype in ("ultralytics", "yolov8"):
                try:
                    self.sahi_model = AutoDetectionModel.from_pretrained(
                        model_type=mtype,
                        model_path=weights,
                        confidence_threshold=conf,
                        device=device,
                    )
                    break
                except Exception:
                    continue
            if self.sahi_model is None:
                raise RuntimeError(
                    "Could not initialise a SAHI model. Check your `sahi` version."
                )

    # ----------------------------------------------------------------------
    def detect(self, frame):
        """frame: BGR numpy image -> detections np.ndarray [N, 5]."""
        if self.use_sahi:
            return self._detect_sahi(frame)
        return self._detect_plain(frame)

    def _detect_plain(self, frame):
        res = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            half=self.half,
            device=self.device,
            verbose=False,
        )[0]

        if res.boxes is None or len(res.boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)

        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy().reshape(-1, 1)
        return np.concatenate([xyxy, conf], axis=1).astype(np.float32)

    def _detect_sahi(self, frame):
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(
            frame,                       # SAHI accepts a numpy BGR image
            self.sahi_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            verbose=0,
        )

        dets = []
        for obj in result.object_prediction_list:
            if self.classes is not None and obj.category.id not in self.classes:
                continue
            x1, y1, x2, y2 = obj.bbox.to_xyxy()
            dets.append([x1, y1, x2, y2, obj.score.value])

        if not dets:
            return np.empty((0, 5), dtype=np.float32)
        return np.asarray(dets, dtype=np.float32)
