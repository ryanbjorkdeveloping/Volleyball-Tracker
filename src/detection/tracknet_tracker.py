"""
Standalone TrackNetV2 volleyball tracker.

Uses pre-trained WASB-SBDT weights (tracknetv2_volleyball_best.pth.tar).
Processes frames in sliding windows of 3, outputs a heatmap per frame.
No CUDA required — runs on MPS (Apple Silicon) or CPU.
"""
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Make unet2d importable from this directory
sys.path.insert(0, str(Path(__file__).parent))
from unet2d import TrackNetV2

# Model input resolution (from WASB-SBDT tracknetv2.yaml)
_INP_W = 512
_INP_H = 288
_FRAMES_IN = 3          # number of input frames
_SCORE_THRESHOLD = 0.5  # heatmap confidence threshold


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tracknet(weights_path: str) -> tuple:
    """Load TrackNetV2 model and return (model, device)."""
    device = _get_device()
    model = TrackNetV2(
        n_channels=_FRAMES_IN * 3,  # 3 frames × RGB = 9 channels
        n_classes=_FRAMES_IN,       # one heatmap per input frame
        bilinear=True,
        halve_channel=False,
        mode="nearest",
    )
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, device


def _preprocess_frame(frame: np.ndarray, crop: tuple | None = None) -> np.ndarray:
    """
    Resize BGR frame to (INP_H, INP_W) and convert to float32 [0,1] RGB.
    crop: optional (y1, y2, x1, x2) in pixels to zoom in before resizing.
    """
    if crop is not None:
        y1, y2, x1, x2 = crop
        frame = frame[y1:y2, x1:x2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (_INP_W, _INP_H))
    return resized.astype(np.float32) / 255.0


def _frames_to_tensor(frames: list, device: torch.device) -> torch.Tensor:
    """
    frames: list of 3 numpy arrays (H, W, 3) in [0,1] float32 RGB
    Returns tensor of shape (1, 9, 288, 512)
    """
    stacked = np.concatenate([f.transpose(2, 0, 1) for f in frames], axis=0)
    return torch.from_numpy(stacked).unsqueeze(0).to(device)


def _heatmap_to_position(hm: np.ndarray, crop: tuple, orig_w: int, orig_h: int) -> tuple | None:
    """
    hm: (H, W) sigmoid heatmap in [0,1]
    crop: (y1, y2, x1, x2) region of original frame that was passed to the model.
    Returns (x, y, score) in original frame coordinates, or None.
    """
    if hm.max() <= _SCORE_THRESHOLD:
        return None

    peak_y, peak_x = np.unravel_index(np.argmax(hm), hm.shape)
    score = float(hm[peak_y, peak_x])

    y1, y2, x1, x2 = crop
    crop_w = x2 - x1
    crop_h = y2 - y1

    # Map from model coords → crop coords → original coords
    x = peak_x * crop_w / _INP_W + x1
    y = peak_y * crop_h / _INP_H + y1
    return float(x), float(y), score


def _run_inference_raw(model, device, frames: list) -> np.ndarray:
    """Run model on a triplet, return raw heatmaps (3, H, W) after sigmoid."""
    with torch.no_grad():
        inp = _frames_to_tensor(frames, device)
        out = model(inp)
        hms = out[0].sigmoid().squeeze(0).cpu().numpy()
    return hms


class TrackNetTracker:
    """
    Frame-by-frame volleyball tracker using TrackNetV2.

    Improvements over the basic version:
    - court_crop: crops out browser/scoreboard chrome so the ball is larger in model input
    - all_heatmaps: uses all 3 heatmap outputs per inference, giving each frame 3 detection
      chances (as heatmap[2] at t, heatmap[1] at t+1, heatmap[0] at t+2)
    - guided_zoom: when a Kalman position is available, runs a second zoomed inference
      centred on the prediction for higher-confidence detection during fast movement
    - smart Kalman: velocity-aware prediction with bounce detection
    """

    # Court crop fractions — strip YouTube/browser chrome.
    # Tune these if using a different video source.
    COURT_Y1_FRAC = 0.08   # skip top 8%  (browser tabs + YouTube header)
    COURT_Y2_FRAC = 0.82   # skip bottom 18% (YouTube controls + recommendations)
    COURT_X1_FRAC = 0.0
    COURT_X2_FRAC = 1.0

    # Guided zoom window (pixels in original frame)
    ZOOM_W = 900
    ZOOM_H = 700

    MAX_KALMAN_GAP = 25   # stop predicting if no detection for this many frames

    # Bounce physics: when Kalman predicts below this y-fraction, reverse vy.
    # Represents the platform/floor contact point (roughly player forearm height).
    BOUNCE_Y_FRAC = 0.52   # ~52% of frame height
    BOUNCE_DAMPING = 0.75  # fraction of vy to retain after bounce

    def __init__(self, weights_path: str, score_threshold: float = 0.15):
        global _SCORE_THRESHOLD
        _SCORE_THRESHOLD = score_threshold

        self._model, self._device = load_tracknet(weights_path)
        self._frame_buf: list = []      # preprocessed court-crop frames
        self._raw_buf: list = []        # original frames (for zoom pass)
        self._orig_w = None
        self._orig_h = None
        self._court_crop = None         # (y1, y2, x1, x2) in pixels
        self._bounce_y: float | None = None
        self._bounced = False

        # Per-frame best score cache: frame_idx → (x, y, score)
        self._best: dict = {}

        # Kalman filter
        self._kf = None
        self._kf_initialized = False
        self._last_det_frame = -999
        self._frame_idx = 0

        # Track initialization: require N detections within a window before Kalman starts
        self._init_buf: list = []          # recent (frame_idx, x, y, score) detections
        self._INIT_REQUIRED = 3            # detections needed
        self._INIT_WINDOW   = 8            # within this many frames
        self._track_confirmed = False


    # ── public API ────────────────────────────────────────────────────────────

    def detect_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Process one frame. Returns a list with at most one detection dict,
        or an empty list. Call in sequence for every video frame.
        """
        if self._orig_w is None:
            self._orig_h, self._orig_w = frame.shape[:2]
            self._court_crop = self._compute_court_crop()
            self._bounce_y = self._orig_h * self.BOUNCE_Y_FRAC
            self._init_kalman()

        self._raw_buf.append(frame)

        # Preprocess using court crop
        pre = _preprocess_frame(frame, self._court_crop)
        self._frame_buf.append(pre)

        fi = self._frame_idx
        self._frame_idx += 1

        # Need 3 frames before inference
        if len(self._frame_buf) < _FRAMES_IN:
            return []

        triplet = self._frame_buf[-_FRAMES_IN:]

        # ── Primary inference on court crop ───────────────────────────────────
        hms = _run_inference_raw(self._model, self._device, triplet)

        # Fix 3: score all 3 heatmap slots against their respective frames
        for offset in range(_FRAMES_IN):
            slot_fi = fi - (_FRAMES_IN - 1) + offset
            if slot_fi < 0:
                continue
            hm = hms[offset]
            pos = _heatmap_to_position(hm, self._court_crop, self._orig_w, self._orig_h)
            if pos is not None:
                x, y, score = pos
                if slot_fi not in self._best or score > self._best[slot_fi][2]:
                    self._best[slot_fi] = (x, y, score)

        # ── Kalman update / predict ───────────────────────────────────────────
        if fi in self._best:
            x, y, score = self._best[fi]

            # Track init: accumulate detections; confirm after N within window
            if not self._track_confirmed:
                self._init_buf.append((fi, x, y, score))
                # Drop entries outside the init window
                self._init_buf = [(f, *rest) for f, *rest in self._init_buf
                                  if fi - f <= self._INIT_WINDOW]
                if len(self._init_buf) >= self._INIT_REQUIRED:
                    self._track_confirmed = True
                    # Seed Kalman from the first buffered detection
                    for bf, bx, by, _ in self._init_buf:
                        self._kalman_update(bx, by)
                        self._last_det_frame = bf
                else:
                    return []   # not yet confirmed — withhold result

            self._kalman_update(x, y)
            self._last_det_frame = fi
            # If ball is now above bounce line, it has completed the bounce
            if self._bounce_y is not None and y < self._bounce_y:
                self._bounced = False

        elif self._track_confirmed and (fi - self._last_det_frame) <= self.MAX_KALMAN_GAP:
            # Fix 2: guided zoom around Kalman prediction
            kx, ky = self._kalman_predict_pos()
            zoom_det = self._guided_zoom(frame, fi, kx, ky)
            if zoom_det is not None:
                zx, zy, zs = zoom_det
                self._best[fi] = (zx, zy, zs)
                self._kalman_update(zx, zy)
                self._last_det_frame = fi
            else:
                # Pure Kalman prediction
                kx, ky = self._kalman_predict_pos()
                return [self._make_det(kx, ky, 0.0, "kalman")]
        else:
            return []

        if fi in self._best:
            x, y, score = self._best[fi]
            source = "tracknet" if score > 0 else "kalman"
            return [self._make_det(x, y, score, source)]

        return []

    # ── internal helpers ──────────────────────────────────────────────────────

    def _compute_court_crop(self) -> tuple:
        y1 = int(self._orig_h * self.COURT_Y1_FRAC)
        y2 = int(self._orig_h * self.COURT_Y2_FRAC)
        x1 = int(self._orig_w * self.COURT_X1_FRAC)
        x2 = int(self._orig_w * self.COURT_X2_FRAC)
        return (y1, y2, x1, x2)

    def _guided_zoom(self, frame: np.ndarray, fi: int, kx: float, ky: float) -> tuple | None:
        """Run a second inference on a zoomed crop centred on (kx, ky)."""
        hw, hh = self.ZOOM_W // 2, self.ZOOM_H // 2
        zx1 = max(0, int(kx) - hw)
        zy1 = max(0, int(ky) - hh)
        zx2 = min(self._orig_w, zx1 + self.ZOOM_W)
        zy2 = min(self._orig_h, zy1 + self.ZOOM_H)
        zoom_crop = (zy1, zy2, zx1, zx2)

        # Need 3 raw frames; use last 3 if available
        if len(self._raw_buf) < _FRAMES_IN:
            return None

        zoom_triplet = [
            _preprocess_frame(f, zoom_crop)
            for f in self._raw_buf[-_FRAMES_IN:]
        ]

        # Lower threshold for zoom pass — ball is much larger here
        hms = _run_inference_raw(self._model, self._device, zoom_triplet)

        # Temporarily use a lower threshold for the zoomed crop
        peak_y, peak_x = np.unravel_index(np.argmax(hms[2]), hms[2].shape)
        score = float(hms[2][peak_y, peak_x])
        if score <= 0.05:
            return None

        zy1, zy2, zx1, zx2 = zoom_crop
        crop_w = zx2 - zx1
        crop_h = zy2 - zy1
        x = peak_x * crop_w / _INP_W + zx1
        y = peak_y * crop_h / _INP_H + zy1
        return float(x), float(y), score

    def _init_kalman(self):
        kf = cv2.KalmanFilter(4, 2)
        dt = 1.0
        kf.transitionMatrix    = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], np.float32)
        kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 5e-3
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0
        kf.errorCovPost        = np.eye(4, dtype=np.float32) * 10.0
        self._kf = kf

    def _kalman_update(self, x: float, y: float):
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        if not self._kf_initialized:
            self._kf.statePre  = np.array([[x],[y],[0],[0]], np.float32)
            self._kf.statePost = np.array([[x],[y],[0],[0]], np.float32)
            self._kf_initialized = True
        else:
            self._kf.predict()
            self._kf.correct(meas)

    def _kalman_predict_pos(self) -> tuple:
        pred = self._kf.predict()
        px = float(pred[0, 0])
        py = float(pred[1, 0])

        # Bounce physics: when ball crosses platform height, reverse vy once
        if self._bounce_y is not None and not self._bounced and py > self._bounce_y:
            vy = float(self._kf.statePost[3, 0])
            if vy > 0:  # was moving downward
                py = 2.0 * self._bounce_y - py    # mirror across bounce line
                self._kf.statePost[1, 0] = np.float32(py)
                self._kf.statePre[1, 0]  = np.float32(py)
                new_vy = -vy * self.BOUNCE_DAMPING
                self._kf.statePost[3, 0] = np.float32(new_vy)
                self._kf.statePre[3, 0]  = np.float32(new_vy)
                self._bounced = True

        px = float(np.clip(px, 0, self._orig_w - 1))
        py = float(np.clip(py, 0, self._orig_h - 1))
        return px, py

    @staticmethod
    def _make_det(x, y, score, source):
        return {
            "x": float(x), "y": float(y),
            "width": 30.0, "height": 30.0,
            "confidence": score,
            "class": "volleyball",
            "source": source,
        }
