# Aerial Guardian: Lightweight Vehicle Detection and Tracking for Drone Video

A compact detect-and-track pipeline for the VisDrone-MOT dataset. It detects vehicles from a moving drone and tracks each one with a consistent identity across frames, drawing a bounding box, a unique ID label, and a short trajectory trail per vehicle. The system is deliberately lightweight (the detector is about 22 MB, far below the 300 MB budget) and runs on a CPU or a single free Colab GPU, so no edge hardware is needed to reproduce it.

The two contributions on top of off-the-shelf components are a YOLOv8 detector fine-tuned on VisDrone with a small-object training recipe, and a ByteTrack tracker extended with Camera Motion Compensation so that vehicle identities survive the drone's own camera motion.

## Highlights

A single-class vehicle detector fine-tuned on VisDrone, reaching mAP@0.5 of 0.965 on the validation set.

A camera-motion-compensated ByteTrack tracker that keeps identities stable while the drone pans, demonstrated on a real sequence where parked vehicles retain their IDs across the entire clip.

An appearance-free, dependency-light design (no re-identification network, Hungarian assignment via SciPy rather than a compiled package) that runs on CPU and exports to ONNX, with a documented TensorRT path for edge devices.

Explicit speed and precision controls (model size, input resolution, FP16, sliced inference, and detector frame-skip) with a built-in profiler that separates detector time from tracker time.

## Results at a glance

Detection, on the VisDrone-MOT validation set, single class vehicle, YOLOv8s fine-tuned at input size 1024, 20 epochs, on a Tesla T4.

| Metric | Value |
| --- | --- |
| Precision | 0.965 |
| Recall | 0.929 |
| mAP at IoU 0.5 | 0.965 |
| mAP at IoU 0.5 to 0.95 | 0.729 |
| Best F1 | 0.95 at confidence 0.427 |

On the confusion matrix at the operating point, 60,517 vehicle instances were detected correctly, 3,303 were missed, and 4,440 background regions were predicted as vehicles, which corresponds to roughly 95 percent of true vehicles recovered.

Tracking, on validation sequence uav0000339_00001_v (275 frames, up to about 29 vehicles tracked at once): the tracker assigned 44 unique IDs in total, a low number for a busy intersection, and roadside parked vehicles kept the same ID from the first frame to the last while the drone was continuously panning.

Speed, on a Tesla T4 at input size 1024 with FP16: about 9.6 frames per second end to end, with the detector at about 43 frames per second in isolation and the tracker at about 28. Running the detector on one of every three frames and coasting the tracker between them raised the end-to-end rate to about 10.5 frames per second. The processed output video, with boxes, unique IDs, and trajectory trails, is provided alongside this repository.

Model size and cost: YOLOv8s is about 22 MB (YOLOv8n is about 6 MB), and the tracker is pure NumPy and OpenCV with essentially no model size, so the total is far below the 300 MB budget. The full fine-tune took about 96 minutes for 20 epochs on a single free Tesla T4.

## Target class

The challenge text is internally inconsistent: the description and the output-video item say vehicles, while the Target Classes field says Persons. This project targets vehicles. All wheeled-vehicle categories in VisDrone (3 bicycle, 4 car, 5 van, 6 truck, 7 tricycle, 8 awning-tricycle, 9 bus, 10 motor) are merged into a single class, 0 equals vehicle, so two-wheelers are included. Switching the target to persons is a one-line change in the class map inside scripts/prepare_visdrone.py.

## Repository structure

```
aerial-guardian/
  aerial_guardian/
    kalman_filter.py     constant-velocity Kalman filter for boxes
    cmc.py               camera motion compensation (optical-flow based)
    matching.py          IoU cost matrix and Hungarian assignment
    byte_tracker.py      ByteTrack association plus CMC integration and coast
    detector.py          YOLOv8 wrapper, optional SAHI tiling
    visualize.py         boxes, ID labels, trajectory trails, FPS overlay
    pipeline.py          end-to-end loop, writes the MP4 and FPS report
  scripts/
    prepare_visdrone.py  converts VisDrone-MOT into YOLO format
    train.py             fine-tunes YOLOv8 with the small-object recipe
    run.py               command-line runner, prints the FPS and hardware report
  tests/
    test_tracker.py      synthetic tracker and CMC sanity tests
  configs/
    visdrone_vehicle.yaml
  requirements.txt
  README.md
```

## Installation

```
git clone <your-private-repo-url> aerial-guardian
cd aerial-guardian
pip install -r requirements.txt
```

PyTorch and Ultralytics are pulled in by the requirements. On a server without a display, install opencv-python-headless instead of opencv-python. A quick check that the tracker works, needing no data or weights:

```
python tests/test_tracker.py
```

This reports stable IDs for a static-camera scene and a moving-camera scene.

## Dataset

The model is built on the VisDrone2019-MOT dataset. Each sequence is a folder of numbered JPG frames, with a matching annotation text file whose columns are frame index, target id, box left, box top, box width, box height, score, object category, truncation, and occlusion. The converter keeps the wheeled-vehicle categories and maps them to a single class.

A note on scale: the label distribution confirms why this is a small-object problem. The validation set contains 63,820 vehicle instances, and the great majority of boxes are below five percent of the frame in both width and height, which at the native resolution is only a few tens of pixels per vehicle.

A note on the split: only the validation set was provided for this challenge, so the model was fine-tuned on it. In production the model would be trained on the separate training split and evaluated on validation. This is discussed under Limitations.

## Method

### Detection and small-object handling

The base detector is YOLOv8, an anchor-free single-stage model. A CSP backbone extracts features, a PAN and FPN neck fuses them at strides 8, 16, and 32, and a decoupled head predicts class and box directly at each location using distribution focal loss for the box regression. At drone altitude a vehicle can be only twenty to forty pixels across, which at stride 32 is barely one grid cell, so small targets are detected almost entirely on the high-resolution stride-8 head.

Four measures address the small-object problem and stack together. First, domain fine-tuning, because COCO vehicles are large and seen from the ground whereas VisDrone vehicles are small and seen from above, so the model is fine-tuned on VisDrone rather than used as is. Second, high input resolution, which preserves pixels on small targets and is the single biggest lever. Third, an optional stride-4 detection head (the P2 variant) that adds a level dedicated to very small objects. Fourth, an optional sliced-inference mode (SAHI) that splits the frame into overlapping tiles so a small object becomes larger relative to its tile, at the cost of running detection several times per frame.

The training recipe is chosen for aerial data: mosaic augmentation to force many scales and positions, copy-paste, mild scale jitter, horizontal flip, and no rotation because nadir drone frames stay upright, with mosaic disabled for the final epochs so training finishes on clean images.

### Tracking and consistent identities under camera motion

The tracker is ByteTrack. Its central idea is to keep low-confidence detections instead of discarding them. It associates in two rounds. The first matches confirmed tracks to high-score detections using IoU and a Kalman motion prediction. The second matches still-unmatched tracks to the low-score detections that other trackers discard, which recovers vehicles that briefly drop in confidence due to occlusion, motion blur, or small size, without letting weak boxes corrupt the strong matches. Tracks that stay unmatched are kept in a lost state for a buffer of frames so the same ID can be recovered after a short disappearance. The tracker is appearance-free, using only motion and IoU rather than a re-identification network, which keeps it fast and edge-friendly.

The drone-specific problem is camera motion. A plain motion tracker assumes a roughly static camera and predicts each box from the target's own velocity. On a drone the whole scene shifts because the camera pans, rolls, or changes altitude, so the predicted box lands in the wrong place, its overlap with the true detection collapses, the track dies, and a new ID is created. This is an ID switch.

The fix is Camera Motion Compensation. Every frame the global camera motion is estimated as a two-by-three affine transform: the image is converted to grayscale and optionally downscaled, sparse corners are detected, tracked into the next frame with pyramidal Lucas-Kanade optical flow, and a robust affine is fit with RANSAC, which rejects the independently moving vehicles as outliers and locks onto the static background. That transform is applied to every track's Kalman state before association, so predictions live in the new frame's coordinate system. This is the same global-motion-compensation idea used inside BoT-SORT, implemented from OpenCV primitives so it stays dependency-light and runs on a CPU or a Jetson. Occlusion is handled by the combination of the low-score recovery round, the lost-track buffer, and the Kalman filter coasting on velocity through the gap, now also corrected for camera motion.

### Optimization

The runner times the detector and the tracker separately and prints a report with the device, the number of frames, the end-to-end rate, and the detector and tracker rates in isolation. FP16 half precision roughly doubles throughput on a supported GPU. A frame-skip option runs the detector on one of every N frames and lets the tracker coast (Kalman prediction plus camera-motion compensation) on the others, which raises throughput when the detector is the bottleneck. Profiling on this workload showed the detector is not the bottleneck; the camera-motion step and the video input and output dominate, so the practical speed levers here are the camera-motion downscale factor and the input resolution.

## Training details

The detector was fine-tuned from YOLOv8s COCO weights for 20 epochs at input size 1024 with batch size 16 on a single Tesla T4. The optimizer was left on automatic selection, which chose AdamW at a learning rate of 0.002 (overriding the requested 0.01), with a cosine learning-rate schedule, a short warmup, and early-stopping patience of 20. Training ran under automatic mixed precision. Augmentations followed the aerial recipe above. Training took about 96 minutes.

The training curves show smooth convergence. Box, classification, and distribution-focal losses on both the training and validation streams decreased steadily and flattened toward the final epochs, while precision, recall, and both mAP measures rose quickly in the first ten epochs and plateaued by around epoch 15. The validation mAP at IoU 0.5 improved from 0.83 at the first epoch to 0.965 at the last.

## Detailed results

Detection metrics at the final epoch are listed in the table above. The precision-recall curve is close to the top-right corner across most of the recall range, with an area (mAP at IoU 0.5) of 0.965. The F1-confidence curve is broad and flat, peaking at 0.95 around a confidence of 0.43, which means the detector is not sensitive to the exact confidence threshold. The precision-confidence curve reaches near 1.0 at high confidence, and the recall-confidence curve stays high until confidence climbs past about 0.6, after which it falls, which is the expected trade-off.

The confusion matrix indicates that the main error mode is a small number of missed and spurious detections rather than class confusion, which is expected for a single-class problem. The remaining misses are dominated by the smallest and most distant vehicles, which sliced inference or higher resolution would help recover at a speed cost.

Tracking behavior on the submitted sequence is described above: 44 unique IDs over 275 frames with up to about 29 vehicles at once, and stable IDs for parked vehicles across the clip despite continuous panning, which is the direct effect of the camera-motion compensation.

### Measured run (verbatim)

The following is the exact console output of the run that produced the submitted video, on the uav0000339_00001_v sequence with the fine-tuned model at input size 1024 and FP16. It is included unedited for reproducibility. The per-frame readout settles around 9 to 10 frames per second, and the final report gives a pipeline rate of 9.64, a detector rate of 42.58 in isolation, and a tracker rate of 28.04, with the detector taking about 23 milliseconds per frame and the tracker about 36, which is the basis for the statement that the detector is not the bottleneck.

```
weights: runs/detect/runs/aerial_guardian/yolov8_visdrone_vehicle/weights/best.pt
sequence: /content/drive/MyDrive/VisDrone2019-MOT-val(1)/VisDrone2019-MOT-val/sequences/uav0000339_00001_v
Hardware: {"python": "3.12.13", "platform": "Linux-6.6.122+-x86_64-with-glibc2.35", "torch": "2.11.0+cu128", "device": "Tesla T4", "cuda": "12.8"}
Running...
  frame 50/275   ~7.0 FPS
  frame 100/275  ~8.7 FPS
  frame 150/275  ~8.9 FPS
  frame 200/275  ~9.1 FPS
  frame 250/275  ~9.5 FPS

================  REPORT  ================
{
  "hardware": {
    "python": "3.12.13",
    "platform": "Linux-6.6.122+-x86_64-with-glibc2.35",
    "torch": "2.11.0+cu128",
    "device": "Tesla T4",
    "cuda": "12.8"
  },
  "frames": 275,
  "detections_run": 275,
  "detect_every": 1,
  "wall_seconds": 28.53,
  "pipeline_fps": 9.64,
  "detector_fps": 42.58,
  "tracker_fps": 28.04,
  "detector_ms_per_frame": 23.48,
  "tracker_ms_per_frame": 35.66,
  "output": "out_finetuned.mp4"
}
==========================================
```

## Reproduce

There are two paths. The quick path produces a tracked video in minutes with no training. The full path fine-tunes the detector first and is the intended submission.

Quick path, no training (stock COCO YOLOv8, keeping the COCO vehicle classes car 2, motorcycle 3, bus 5, truck 7):

```
python scripts/run.py --source path/to/VisDrone2019-MOT-val/sequences/uav0000086_00000_v --weights yolov8s.pt --classes 2 3 5 7 --output out_demo.mp4 --device cpu --imgsz 960
```

Full path, step 1, convert VisDrone to YOLO format (use the copy option on Windows because symbolic links need administrator rights there):

```
python scripts/prepare_visdrone.py --val-root path/to/VisDrone2019-MOT-val --out path/to/visdrone_yolo --copy
```

Full path, step 2, fine-tune the detector (a free Colab T4 is sufficient):

```
python scripts/train.py --data path/to/visdrone_yolo/visdrone_vehicle.yaml --model yolov8s.pt --imgsz 1024 --epochs 20 --batch 16 --device 0
```

Full path, step 3, run detection and tracking and write the output video:

```
python scripts/run.py --source path/to/VisDrone2019-MOT-val/sequences/uav0000339_00001_v --weights best.pt --classes 0 --half --output out_finetuned.mp4 --mot-txt out_finetuned.txt --device 0 --imgsz 1024
```

The runner prints a JSON report with the measured frames per second and the hardware. The optional mot-txt output is in standard MOT format and can be scored with tools such as TrackEval if ground truth is available. Useful options include the model choice (yolov8n is lighter than yolov8s), the input size, the half flag for FP16, the use-sahi flag for tiled inference, the cmc-downscale factor, and the detect-every factor for frame-skip.

## Summary report

Architecture and small-object handling. The architecture is a fine-tuned YOLOv8 detector feeding a ByteTrack tracker. Small objects are handled by domain fine-tuning, high input resolution, an optional stride-4 head, and optional sliced inference, supported by an aerial augmentation recipe. The detector stays small and exports to ONNX, with a TensorRT export path for edge deployment.

Handling ID switches from drone ego-motion and occlusion. Stability comes from ByteTrack's two-round association, the lost-track buffer, and Camera Motion Compensation, which warps each track's Kalman prediction by the estimated camera motion before matching so the drone's panning does not cause the prediction to miss its detection. A synthetic moving-camera test shows the effect directly, and the submitted sequence shows the same: parked vehicles keep their IDs across the whole clip and the total number of IDs stays low relative to the number of vehicles present.

Adapting to edge hardware such as NVIDIA Jetson. The design is already edge-oriented: a small model, an appearance-free tracker, and camera-motion compensation in plain OpenCV. On a Jetson the model would be exported to a TensorRT engine, which fuses layers and selects optimal kernels and typically gives a large speedup over eager PyTorch. INT8 quantization, calibrated on a few hundred VisDrone frames, would roughly double speed again and shrink the model further, with the accuracy drop validated. The operating point (model size, input resolution, and whether sliced inference is on) would be chosen per platform using the same runner options, with no code changes. The video path would use the hardware decode and encode engines through DeepStream or GStreamer so that decoding and encoding stay off the CPU, leaving the CPU for the light tracker. The camera-motion downscale factor keeps optical flow cheap on the device, and the power and clock profile would be fixed so the measured rate is stable.

Frames per second and hardware. On a Tesla T4 at input size 1024 with FP16, end-to-end throughput is about 9.6 frames per second, with the detector at about 43 frames per second in isolation and the tracker at about 28. Profiling shows the detector is not the bottleneck; the camera-motion step and the video input and output dominate. The frame-skip option raised the end-to-end rate to about 10.5 frames per second.

## What was added on top of open source

The work added on top of open source, rather than simply downloading and running a model, consists of a VisDrone to YOLO converter, a small-object training recipe (high resolution, optional stride-4 head, aerial-friendly augmentations), a Camera Motion Compensation module integrated into the tracker's Kalman states to remove ego-motion ID switches, a tracker coast mode and frame-skip option for throughput, optional sliced inference for tiny objects, and the trajectory-trail and FPS-report tooling. The detector and the ByteTrack association logic build on well-understood open-source work that can be explained block by block.

## Limitations

Because only the validation set was provided, the detector was both fine-tuned and evaluated on it. The reported metrics therefore measure fit to that data rather than generalization to an unseen split, and a held-out test set would give a more conservative estimate. The provided code supports training on a separate training split through the train-root option. The remaining detection errors are concentrated on the smallest and most distant vehicles, which sliced inference or higher resolution can recover at a speed cost. Tracking quality was assessed qualitatively on the output video and through a synthetic camera-motion test; formal HOTA, MOTA, and IDF1 scoring against ground truth is possible using the MOT-format output but was not run here.

## Acknowledgements and references

This project builds on Ultralytics YOLOv8 for detection, the ByteTrack association method (Zhang and colleagues, 2022), the global-motion-compensation idea from BoT-SORT, and the SAHI sliced-inference method (Akyon and colleagues, 2022). The dataset is VisDrone2019-MOT.
