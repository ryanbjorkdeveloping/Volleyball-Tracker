# Volleyball Tracker — Claude Reference

> **MANDATORY:** Update this file every time any change is made to the program — new features, modified logic, renamed files, added scripts, changed constants, updated dependencies, anything. This file is the single source of truth for future Claude sessions.

---

## Project Overview

A computer vision pipeline that detects and tracks a volleyball in video footage, predicts ball trajectory/landing, and measures velocity over time. Designed for offline batch processing of `.mp4`/`.mov` game recordings.

**Key output:** CSV of per-frame positions (`frame, time_s, x_pixel, y_pixel, confidence, source`) and optional annotated video.

---

## Architecture

```
src/
  detection/
    tracker.py           — BallTracker: Roboflow hosted or local YOLOv8 (original/legacy)
    tracknet_tracker.py  — TrackNetTracker: standalone TrackNetV2-only tracker
    fusion_tracker.py    — FusionTracker + process_video(): TrackNetV2 + YOLOv8 (RECOMMENDED)
    cv_tracker.py        — CVBallTracker: frame-diff + Hough circles + Kalman (no ML)
    yolo_detector.py     — YOLOBallDetector: single-frame YOLOv8n (COCO sports-ball class 32)
    unet2d.py            — TrackNetV2 model definition (UNet-based)
    unet2d_parts.py      — UNet building blocks (DoubleConv, Down, Up, OutConv)
  court/
    homography.py        — CourtMapper: pixel → real-world coords via homography (9m×18m court)
  physics/
    trajectory.py        — BallPhysics: velocity calc + 3D parabolic landing prediction
scripts/
  track_video.py         — Main CLI entrypoint for video tracking
  run_inference.py       — Single-image Roboflow inference test
  download_dataset.py    — Dataset download helper
  setup_env.py           — Environment setup helper
tests/
  test_trajectory.py     — Pytest tests for BallPhysics
models/
  tracknetv2_volleyball_best.pth.tar  — TrackNetV2 weights (WASB-SBDT, primary weights)
  tracknet_best.pt                    — Alternative TrackNetV2 weights
data/
  raw/videos/            — Input videos + annotated output videos
```

---

## Detection Modes

All modes are selected via `--mode` in `scripts/track_video.py`.

| Mode | Class | Description | When to use |
|------|-------|-------------|-------------|
| `fusion` (default) | `FusionTracker` / `process_video()` | TrackNetV2 + YOLOv8, 4-pass offline pipeline | Best coverage; batch/offline |
| `tracknet` | `TrackNetTracker` | TrackNetV2 heatmap tracker only | No YOLO, still good |
| `cv` | `CVBallTracker` | Frame-diff + Hough + Kalman; no ML | No GPU/weights needed |
| `hosted` | `BallTracker(mode="hosted")` | Roboflow serverless API | Needs API key; no local GPU |
| `local` | `BallTracker(mode="local")` | Local YOLOv8 `.pt` weights | Custom YOLO weights |

---

## Detection Pipeline (Fusion — Recommended)

`process_video()` in `src/detection/fusion_tracker.py` runs 4 passes:

1. **Pass 1** — TrackNetV2 (all 3 heatmap slots per triplet) + YOLOv8 on every frame. Positions filtered to play area (`PLAY` dict: 2–98% x, 8–80% y of frame). Static-object guard (`window=2, tol=15px`) prevents YOLO from locking on logos/ad boards. After all detections are collected, a **continuity filter** segments them by displacement (`_MAX_DISP=300px`, `_MAX_PX_PER_FRAME=250`) and a static-object guard (near-identical positions >8 frames apart). The best-scoring segment is kept; **segment stitching** then reconnects any other segment whose endpoint is within `_INTERP_MAX_GAP` frames and `_MAX_DISP` displacement from the kept segment's endpoints (bypasses false positives sandwiched between real ball segments).
2. **Pass 2** — Guided zoom: for frames with no detection between two known detections (gap ≤ 80), interpolate a guide point, then run both models on a tight crop (`_ZOOM_HW=250`, `_ZOOM_HH=200`). Reject if detected position is > 65% of zoom half-width from guide.
3. **Pass 2b** — Trajectory extension: zoomed inference forward/backward from segment endpoints (up to 30 frames, window `_EXT_ZOOM_HW=180`, `_EXT_ZOOM_HH=140`).
4. **Pass 3** — Linear interpolation through remaining gaps ≤ `_INTERP_MAX_GAP=20` frames. These get `source="interp"`.
5. **Pass 4** — Render annotated video with fading trail (30 frames) and confidence labels.

### Fusion Logic (`_fuse()`)
- Both detections within `_AGREE_DIST=180px` → weighted-average by confidence, boosted by `_CONF_BOOST=1.25`
- Both fire, disagree → higher-confidence wins
- One fires → use it
- Neither fires + Kalman active → Kalman prediction (`source="kalman"`)

---

## TrackNetV2 Details

- Model: UNet-based (`src/detection/unet2d.py` + `unet2d_parts.py`)
- Input: 3 consecutive frames stacked → tensor shape `(1, 9, 288, 512)` (3 frames × RGB = 9 channels)
- Output: 3 heatmaps `(3, 288, 512)` — one per input frame, after sigmoid
- Resolution: `_INP_W=512`, `_INP_H=288`
- Weights: `models/tracknetv2_volleyball_best.pth.tar` (WASB-SBDT pre-trained on volleyball)
- Device priority: MPS (Apple Silicon) → CUDA → CPU
- Court crop: strips browser/YouTube chrome — top 8% (`COURT_Y1_FRAC=0.08`), bottom 18% (`COURT_Y2_FRAC=0.82`)
- Score threshold: `_SCORE_THRESHOLD=0.5` (global); `TrackNetTracker` init overrides to 0.15

---

## TrackNetTracker Specifics

Key features over the basic version:
- **court_crop**: crops frame before inference so ball is proportionally larger
- **all_heatmaps**: uses all 3 heatmap output slots per inference; each frame gets 3 detection chances
- **guided_zoom**: when Kalman has a predicted position, runs a second zoomed inference (`ZOOM_W=900`, `ZOOM_H=700`); zoom threshold is 0.05 (lower than normal)
- **smart Kalman**: velocity-aware with bounce detection (`BOUNCE_Y_FRAC=0.52`, `BOUNCE_DAMPING=0.75`)
- **track init guard**: requires 3 detections within 8 frames before Kalman starts (`_INIT_REQUIRED=3`, `_INIT_WINDOW=8`)
- **MAX_KALMAN_GAP=25**: stops predicting after 25 consecutive frames without detection

---

## YOLOv8 Detector (`yolo_detector.py`)

- Model: `yolov8n.pt` (nano, COCO pre-trained) — auto-downloaded by ultralytics
- Class: COCO class 32 (`sports ball`)
- Default conf: 0.15 (tracker), 0.10 (fusion YOLO pass)
- Methods: `detect(frame)` → `dict | None`, `detect_zoom(frame, cx, cy, ...)` → `dict | None`
- Optional `court_crop` trims frame before inference

---

## Kalman Filter (shared pattern)

All trackers use `cv2.KalmanFilter(4, 2)` — state `[x, y, vx, vy]`, measurement `[x, y]`.

```
transitionMatrix:    [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]  (dt=1.0)
measurementMatrix:   [[1,0,0,0],[0,1,0,0]]
processNoiseCov:     eye(4) * 5e-3
measurementNoiseCov: eye(2) * 5.0
errorCovPost:        eye(4) * 10.0
```

Bounce physics: when predicted `py > BOUNCE_Y`, reverses `vy` with damping (0.75) once per arc.

---

## Detection Output Format

All trackers return `list[dict]` per frame:

```python
{
    "x": float,           # pixel x center
    "y": float,           # pixel y center
    "width": float,       # bounding box width (px)
    "height": float,      # bounding box height (px)
    "confidence": float,  # 0.0–1.0 (0.0 = kalman/interp)
    "class": "volleyball",
    "source": str,        # "tracknet" | "yolo" | "fusion" | "kalman" | "interp" | "detected"
}
```

---

## CSV Output Format

```
frame, time_s, x_pixel, y_pixel, confidence, source
```

Existing output CSVs (root dir): `positions_tracknet_v5.csv` is the latest iteration.

---

## Annotation Color Legend

| Source | Color (BGR) | Meaning |
|--------|-------------|---------|
| `tracknet` | (0, 255, 80) green | TrackNetV2 heatmap detection |
| `yolo` | (255, 180, 0) blue-ish | YOLOv8-only detection |
| `fusion` | (0, 220, 255) yellow | Both models agreed |
| `kalman` | (80, 180, 255) light blue | Kalman prediction, no detection |
| `interp` | (160, 100, 255) purple | Linear interpolation gap fill |
| `detected` | (0, 255, 0) green | Generic detected (cv mode) |

---

## Court Mapping (`src/court/homography.py`)

`CourtMapper` converts pixel (x, y) to real-world meters via homography. Standard court: 9m × 18m. Call `calibrate(pixel_points)` with 4 corner pixels, then `to_court_coords(px, py)`.

---

## Physics (`src/physics/trajectory.py`)

`BallPhysics(fps)`:
- `velocity(pos_prev, pos_curr)` → `np.ndarray` — pixel velocity in px/s
- `predict_landing(pos_3d, vel_3d)` → `(x, y) | None` — solves parabolic arc with `g=9.81 m/s²`

Note: `predict_landing` requires 3D coordinates in meters — requires court mapping first.

---

## Main CLI Usage

```bash
# Recommended (fusion mode)
python scripts/track_video.py video.mp4 --output-video annotated.mp4

# TrackNetV2 only
python scripts/track_video.py video.mp4 --mode tracknet

# OpenCV only (no ML)
python scripts/track_video.py video.mp4 --mode cv

# With all options
python scripts/track_video.py video.mp4 \
  --mode fusion \
  --output-csv positions.csv \
  --output-video out.mp4 \
  --conf 0.15 \
  --yolo-conf 0.10 \
  --model-path models/tracknetv2_volleyball_best.pth.tar
```

---

## Environment & Dependencies

**`.env` vars** (copy `.env.example`):
```
ROBOFLOW_API_KEY=<key>
ROBOFLOW_WORKSPACE=ryan-n5sq8
ROBOFLOW_WORKFLOW_ID=general-segmentation-api
ROBOFLOW_PROJECT=<project_slug>
ROBOFLOW_VERSION=1
```
Only needed for `--mode hosted`.

**`requirements.txt`:**
```
ultralytics>=8.0.0
roboflow>=1.1.0
inference-sdk>=0.9.0
opencv-python>=4.8.0
python-dotenv>=1.0.0
numpy>=1.24.0
```
PyTorch must be installed separately (not in requirements.txt). Python 3.10+ required for `str | None` union syntax.

---

## Key Constants to Know

| Constant | File | Value | Purpose |
|----------|------|-------|---------|
| `_INP_W / _INP_H` | tracknet_tracker.py | 512 / 288 | TrackNetV2 input resolution |
| `_FRAMES_IN` | tracknet_tracker.py | 3 | Sliding window size |
| `_SCORE_THRESHOLD` | tracknet_tracker.py | 0.5 (global) | Heatmap peak threshold |
| `COURT_Y1_FRAC` | tracknet_tracker.py | 0.08 | Top crop fraction |
| `COURT_Y2_FRAC` | tracknet_tracker.py | 0.82 | Bottom crop fraction |
| `BOUNCE_Y_FRAC` | tracknet_tracker.py | 0.52 | Platform/floor contact y-fraction |
| `BOUNCE_DAMPING` | tracknet_tracker.py | 0.75 | Kalman vy after bounce |
| `MAX_KALMAN_GAP` | tracknet_tracker.py | 25 | Max frames before Kalman stops |
| `_AGREE_DIST` | fusion_tracker.py | 180 | px — fusion agreement threshold |
| `_CONF_BOOST` | fusion_tracker.py | 1.25 | Confidence multiplier on agreement |
| `_INTERP_MAX_GAP` | fusion_tracker.py | 20 | Max gap to interpolate |
| `_ZOOM_HW / _ZOOM_HH` | fusion_tracker.py | 450 / 350 | Guided zoom half-dimensions |
| `ZOOM_W / ZOOM_H` | tracknet_tracker.py | 900 / 700 | Guided zoom window in TrackNetTracker |
| `_SCALE` | cv_tracker.py | 0.33 | Downscale for CV processing |
| `_SPORTS_BALL_CLASS` | yolo_detector.py | 32 | COCO class index |

---

## Tests

```bash
pytest tests/
```

`tests/test_trajectory.py` covers: velocity calculation, landing prediction returning a point, and landing returning `None` when ball is already grounded.

---

## Iteration History (annotated video versions)

| Version | File | Notes |
|---------|------|-------|
| v1 | `annotated.mp4` | Roboflow hosted detection |
| v2 | `annotated_v2.mp4` | Local CV tracker |
| cv | `annotated_cv.mp4` | CVBallTracker |
| tracknet | `annotated_tracknet.mp4` | TrackNetV2 first run |
| tracknet_v2–v5 | ... | Iterative improvements |
| fusion | `annotated_fusion.mp4` | TrackNetV2 + YOLO fusion |
| fusion_v2–v9 | ... | Iterative improvements to fusion pipeline |
| fusion_v10 | `annotated_fusion_v10.mp4` | Removed Kalman smoother (was displacing positions), added segment stitching |

Latest working CSVs: `positions_tracknet_v5.csv`, `positions_fusion_v10.csv`

---

## Gotchas & Non-Obvious Decisions

- The Kalman post-smoothing pass was removed from `process_video()`. With `measurementNoiseCov=10.0` it trusted velocity prediction 1000× more than raw measurements, displacing accurate detections by hundreds of pixels (f137 moved from (1429,820) to (953,547) in v9). Raw detector positions are more accurate than the smoothed ones.
- `unet2d.py` `forward()` returns `{0: logits}` (dict), not a tensor — callers must do `out[0]` before sigmoid.
- `TrackNetTracker` and `FusionTracker` use `_init_buf` to require 3 detections within 8 frames before emitting results — this suppresses false positives at video start.
- YOLO uses COCO class 32 (`sports ball`) not a custom volleyball model. This works reasonably well but can false-positive on heads/round objects.
- Zoom inference uses threshold 0.05 (not 0.15) — ball is much larger in the zoomed crop.
- `_is_static()` in `process_video()` guards against YOLO locking on a stationary logo/brand watermark by checking if the same position appeared in the last 4 frames.
- Court crop fractions are tuned for YouTube/browser screen recordings. For other sources (direct broadcast, phone), adjust `COURT_Y1_FRAC` and `COURT_Y2_FRAC`.
- `physics/trajectory.py` works in meters (3D). Pixel coordinates need homography conversion before being useful for landing prediction.
- `models/tracknet_best.pt` exists but the primary weights used everywhere are `models/tracknetv2_volleyball_best.pth.tar`.
