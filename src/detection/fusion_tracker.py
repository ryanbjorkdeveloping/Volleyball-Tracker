"""
FusionTracker: combines TrackNetV2 (temporal heatmap) + YOLOv8 (single-frame)
for maximum volleyball tracking coverage.

Fusion logic per frame:
  - Both fire, agree  (dist ≤ AGREE_DIST) → weighted-average position, boosted confidence
  - Both fire, disagree                    → trust higher confidence
  - Only one fires                         → use it
  - Neither fires + Kalman active          → Kalman prediction
  - Neither fires + no Kalman              → no detection

For offline/batch use, call process_video() which runs a two-pass pipeline:
  Pass 1 – both models on every frame, all TrackNetV2 heatmap slots
  Pass 2 – guided zoom on gap frames
  Pass 3 – smooth interpolation between confirmed endpoints
"""
import csv
import os
from pathlib import Path

import cv2
import numpy as np

from src.detection.tracknet_tracker import (
    load_tracknet,
    _preprocess_frame,
    _run_inference_raw,
    _INP_W, _INP_H, _FRAMES_IN,
)
from src.detection.yolo_detector import YOLOBallDetector

# ── Fusion constants ──────────────────────────────────────────────────────────
_AGREE_DIST      = 180   # px — models "agree" when detections are this close
_CONF_BOOST      = 1.25  # multiply fused confidence when both models agree
_INTERP_MAX_GAP  = 20    # only interpolate gaps ≤ this many frames
_ZOOM_HW         = 250   # guided zoom half-width (tighter = ball bigger in model input)
_ZOOM_HH         = 200   # guided zoom half-height


def _fuse(a: dict | None, b: dict | None) -> dict | None:
    """Merge two detections. Returns None if both are None."""
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a

    dist = ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5
    if dist <= _AGREE_DIST:
        ca, cb = a["confidence"], b["confidence"]
        total = ca + cb
        x = (a["x"] * ca + b["x"] * cb) / total
        y = (a["y"] * ca + b["y"] * cb) / total
        conf = min(1.0, (total / 2) * _CONF_BOOST)
        return {
            "x": x, "y": y,
            "width": (a["width"] + b["width"]) / 2,
            "height": (a["height"] + b["height"]) / 2,
            "confidence": conf,
            "class": "volleyball",
            "source": "fusion",
        }

    # Disagree — trust the higher-confidence model
    return a if a["confidence"] >= b["confidence"] else b


class FusionTracker:
    """
    Real-time fusion tracker. Call detect_frame() on each frame in order.

    For best offline results use process_video() instead, which runs
    a two-pass pipeline with backward-aware gap interpolation.
    """

    COURT_Y1_FRAC = 0.08
    COURT_Y2_FRAC = 0.82
    MAX_KALMAN_GAP = 25

    def __init__(
        self,
        tracknet_weights: str,
        conf: float = 0.15,
        yolo_conf: float | None = None,
    ):
        self._conf      = conf
        self._yolo_conf = yolo_conf if yolo_conf is not None else max(0.10, conf)

        self._tn_model, self._tn_device = load_tracknet(tracknet_weights)
        self._yolo = YOLOBallDetector(conf=self._yolo_conf)

        self._orig_w = self._orig_h = None
        self._court_crop = None

        # TrackNetV2 frame buffer
        self._frame_buf: list = []
        self._raw_buf:   list = []
        self._best:      dict = {}

        # Track init guard
        self._init_buf: list  = []
        self._INIT_N    = 3
        self._INIT_WIN  = 8
        self._confirmed = False

        # Kalman
        self._kf           = None
        self._kf_ready     = False
        self._last_det_fi  = -999
        self._bounced      = False
        self._bounce_y: float | None = None

        self._fi = 0

    # ── public API ────────────────────────────────────────────────────────────

    def detect_frame(self, frame: np.ndarray) -> list[dict]:
        if self._orig_w is None:
            self._orig_h, self._orig_w = frame.shape[:2]
            y1 = int(self._orig_h * self.COURT_Y1_FRAC)
            y2 = int(self._orig_h * self.COURT_Y2_FRAC)
            self._court_crop = (y1, y2, 0, self._orig_w)
            self._yolo._court_crop = self._court_crop
            self._bounce_y = self._orig_h * 0.52
            self._init_kalman()

        fi = self._fi
        self._fi += 1

        self._raw_buf.append(frame)
        self._frame_buf.append(_preprocess_frame(frame, self._court_crop))

        # ── TrackNetV2 (all 3 heatmap slots) ─────────────────────────────────
        if len(self._frame_buf) >= _FRAMES_IN:
            hms = _run_inference_raw(
                self._tn_model, self._tn_device, self._frame_buf[-_FRAMES_IN:]
            )
            for offset in range(_FRAMES_IN):
                sfi = fi - (_FRAMES_IN - 1) + offset
                if sfi < 0:
                    continue
                score = float(hms[offset].max())
                if score >= self._conf:
                    py, px = np.unravel_index(np.argmax(hms[offset]), hms[offset].shape)
                    y1, y2, x1, x2 = self._court_crop
                    x = px * (x2 - x1) / _INP_W + x1
                    y = py * (y2 - y1) / _INP_H + y1
                    if sfi not in self._best or score > self._best[sfi]["confidence"]:
                        self._best[sfi] = {
                            "x": float(x), "y": float(y),
                            "width": 30.0, "height": 30.0,
                            "confidence": score,
                            "class": "volleyball", "source": "tracknet",
                        }

        # ── YOLO (current frame) ──────────────────────────────────────────────
        yolo_det = self._yolo.detect(frame)
        tn_det   = self._best.get(fi)

        fused = _fuse(tn_det, yolo_det)
        if fused:
            if fi not in self._best or fused["confidence"] > self._best[fi]["confidence"]:
                self._best[fi] = fused

        # ── Kalman + track init ───────────────────────────────────────────────
        det = self._best.get(fi)
        if det:
            x, y = det["x"], det["y"]

            if not self._confirmed:
                self._init_buf.append((fi, x, y))
                self._init_buf = [(f, *r) for f, *r in self._init_buf
                                  if fi - f <= self._INIT_WIN]
                if len(self._init_buf) >= self._INIT_N:
                    self._confirmed = True
                    for bf, bx, by in self._init_buf:
                        self._kalman_update(bx, by)
                        self._last_det_fi = bf
                else:
                    return []

            self._kalman_update(x, y)
            self._last_det_fi = fi
            if self._bounce_y and y < self._bounce_y:
                self._bounced = False
            return [det]

        elif self._confirmed and (fi - self._last_det_fi) <= self.MAX_KALMAN_GAP:
            kx, ky = self._kalman_predict()
            return [{
                "x": kx, "y": ky,
                "width": 30.0, "height": 30.0,
                "confidence": 0.0,
                "class": "volleyball", "source": "kalman",
            }]

        return []

    # ── Kalman internals ──────────────────────────────────────────────────────

    def _init_kalman(self):
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 5e-3
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5.0
        kf.errorCovPost        = np.eye(4, dtype=np.float32) * 10.0
        self._kf = kf

    def _kalman_update(self, x: float, y: float):
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        if not self._kf_ready:
            self._kf.statePre  = np.array([[x],[y],[0],[0]], np.float32)
            self._kf.statePost = np.array([[x],[y],[0],[0]], np.float32)
            self._kf_ready = True
        else:
            self._kf.predict()
            self._kf.correct(meas)

    def _kalman_predict(self) -> tuple:
        pred = self._kf.predict()
        px, py = float(pred[0, 0]), float(pred[1, 0])

        if self._bounce_y and not self._bounced and py > self._bounce_y:
            vy = float(self._kf.statePost[3, 0])
            if vy > 0:
                py = 2.0 * self._bounce_y - py
                self._kf.statePost[1, 0] = np.float32(py)
                self._kf.statePre[1, 0]  = np.float32(py)
                self._kf.statePost[3, 0] = np.float32(-vy * 0.75)
                self._kf.statePre[3, 0]  = np.float32(-vy * 0.75)
                self._bounced = True

        px = float(np.clip(px, 0, self._orig_w - 1))
        py = float(np.clip(py, 0, self._orig_h - 1))
        return px, py


# ── Offline batch pipeline ────────────────────────────────────────────────────

def process_video(
    video_path:       str,
    tracknet_weights: str,
    output_video:     str = "",
    output_csv:       str = "",
    conf:             float = 0.15,
    yolo_conf:        float = 0.10,
) -> dict:
    """
    Full two-pass fusion pipeline for offline video files.

    Pass 1 – TrackNetV2 (all heatmap slots) + YOLO on every frame
    Pass 2 – guided zoom on gap frames (both models)
    Pass 3 – smooth linear interpolation through remaining gaps ≤ 20 frames
    Pass 4 – render annotated video

    Returns summary dict with coverage stats.
    """
    cap = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    COURT     = (int(H*0.08), int(H*0.82), 0, W)
    # Play-area spatial filter — court zone only, excludes ad boards and floor
    # y2=0.60 cuts off shoes/floor; ad boards are excluded by tight y1=0.18
    # Wider than original to capture serve (full width, top 8% and bottom 20% cut)
    PLAY      = dict(x1=int(W*0.02), x2=int(W*0.98),
                     y1=int(H*0.08), y2=int(H*0.80))
    BOUNCE_Y  = H * 0.52

    def in_play(x, y):
        return PLAY["x1"] <= x <= PLAY["x2"] and PLAY["y1"] <= y <= PLAY["y2"]

    def _is_static(pos_history: list, x: float, y: float,
                   window: int = 2, tol: float = 15.0) -> bool:
        """Return True if (x,y) appeared at basically the same spot in recent frames."""
        if len(pos_history) < window:
            return False
        recent = [p for p in pos_history[-window:] if p is not None]
        if not recent:
            return False
        matches = sum(1 for px, py in recent
                      if abs(px - x) < tol and abs(py - y) < tol)
        return matches >= len(recent)

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading models...")
    tn_model, tn_device = load_tracknet(tracknet_weights)
    yolo = YOLOBallDetector(conf=yolo_conf, court_crop=COURT)
    print("  TrackNetV2 ✓  YOLOv8 ✓")

    # ── Pass 1 ────────────────────────────────────────────────────────────────
    print("\nPass 1: TrackNetV2 + YOLO on every frame...")
    frames_raw = []
    frame_buf  = []
    best: dict = {}   # fi → {x, y, confidence, source}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fi = 0
    yolo_pos_history: list = []   # track recent YOLO positions to catch static logos
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_raw.append(frame.copy())
        frame_buf.append(_preprocess_frame(frame, COURT))

        # TrackNetV2 — all 3 heatmap slots
        if len(frame_buf) >= _FRAMES_IN:
            hms = _run_inference_raw(tn_model, tn_device, frame_buf[-_FRAMES_IN:])
            for offset in range(_FRAMES_IN):
                sfi = fi - (_FRAMES_IN - 1) + offset
                if sfi < 0:
                    continue
                score = float(hms[offset].max())
                if score >= conf:
                    py, px = np.unravel_index(np.argmax(hms[offset]), hms[offset].shape)
                    y1, y2, x1, x2 = COURT
                    x = float(px * (x2-x1) / _INP_W + x1)
                    y = float(py * (y2-y1) / _INP_H + y1)
                    if in_play(x, y):
                        tn_det = {"x": x, "y": y, "confidence": score,
                                  "width": 30.0, "height": 30.0,
                                  "class": "volleyball", "source": "tracknet"}
                        if sfi not in best or score > best[sfi]["confidence"]:
                            best[sfi] = tn_det

        # YOLO — current frame (with static-object guard)
        yolo_det = yolo.detect(frame)
        if yolo_det and in_play(yolo_det["x"], yolo_det["y"]):
            yx, yy = yolo_det["x"], yolo_det["y"]
            if not _is_static(yolo_pos_history, yx, yy):
                fused = _fuse(best.get(fi), yolo_det)
                if fused and (fi not in best or fused["confidence"] > best[fi]["confidence"]):
                    best[fi] = fused
            yolo_pos_history.append((yx, yy))
        else:
            yolo_pos_history.append(None)
        if len(yolo_pos_history) > 8:
            yolo_pos_history.pop(0)

        fi += 1
        print(f"\r  {fi}/{total}  detections={len(best)}", end="", flush=True)

    cap.release()
    print(f"\n  Pass 1 detections: {len(best)}")
    for f in sorted(best):
        d = best[f]
        print(f"    f{f:3d}: ({d['x']:.0f},{d['y']:.0f}) conf={d['confidence']:.3f} [{d['source']}]")

    # ── Continuity filter: drop detections that jump too far between frames ───
    # Cap absolute max displacement at 300 px regardless of gap size — prevents
    # false positives on opposite sides of the frame being grouped together.
    _MAX_DISP = 300.0        # absolute max pixel displacement between any two consecutive dets
    _MAX_PX_PER_FRAME = 250.0  # also scale with gap (whichever is smaller)

    def _build_segments(dets: dict) -> list[list[int]]:
        frames = sorted(dets.keys())
        if not frames:
            return []
        segs, cur = [], [frames[0]]
        for i in range(1, len(frames)):
            f0, f1 = frames[i-1], frames[i]
            d0, d1 = dets[f0], dets[f1]
            dist = ((d0["x"]-d1["x"])**2 + (d0["y"]-d1["y"])**2) ** 0.5
            gap = f1 - f0
            # Static-object guard: near-identical positions far apart in time
            # (real balls always move; logos/heads don't)
            if dist < 25.0 and gap > 8:
                segs.append(cur); cur = [f1]; continue
            max_allowed = min(_MAX_PX_PER_FRAME * gap, _MAX_DISP)
            if dist <= max_allowed:
                cur.append(f1)
            else:
                segs.append(cur); cur = [f1]
        segs.append(cur)
        return segs

    segments = _build_segments(best)
    if len(segments) > 1:
        # Score: total confidence + length bonus (no motion term — it rewards noise)
        def _score(seg):
            return sum(best[f]["confidence"] for f in seg) + len(seg) * 0.1
        best_seg = max(segments, key=_score)

        # Stitch segments that are directly reachable from the kept segment's
        # endpoints — bypasses false-positive detections sandwiched between them.
        kept = set(best_seg)
        lo, hi = min(kept), max(kept)
        changed = True
        while changed:
            changed = False
            for seg in segments:
                if all(f in kept for f in seg):
                    continue
                seg_lo, seg_hi = min(seg), max(seg)
                # Try connecting to high endpoint
                if seg_lo > hi:
                    gap = seg_lo - hi
                    d = ((best[hi]["x"] - best[seg_lo]["x"])**2 +
                         (best[hi]["y"] - best[seg_lo]["y"])**2) ** 0.5
                    if gap <= _INTERP_MAX_GAP and d <= min(_MAX_PX_PER_FRAME * gap, _MAX_DISP):
                        kept.update(seg); hi = max(hi, seg_hi); changed = True
                # Try connecting to low endpoint
                elif seg_hi < lo:
                    gap = lo - seg_hi
                    d = ((best[seg_hi]["x"] - best[lo]["x"])**2 +
                         (best[seg_hi]["y"] - best[lo]["y"])**2) ** 0.5
                    if gap <= _INTERP_MAX_GAP and d <= min(_MAX_PX_PER_FRAME * gap, _MAX_DISP):
                        kept.update(seg); lo = min(lo, seg_lo); changed = True

        dropped = len(best) - len(kept)
        print(f"  Continuity filter: kept {len(kept)} frames "
              f"({len(kept)-len(best_seg)} stitched), "
              f"dropped {dropped} frames from {len(segments)-1} other segment(s)")
        best = {f: best[f] for f in kept}

    # ── Pass 2: guided zoom — fill within AND between segments ────────────────
    print("\nPass 2: guided zoom on gaps...")
    det_frames = sorted(best.keys())
    new_found  = 0

    for i in range(len(det_frames) - 1):
        f0, f1 = det_frames[i], det_frames[i+1]
        gap = f1 - f0
        if gap <= 1 or gap > 80:   # allow bridging larger inter-segment gaps
            continue
        x0, y0 = best[f0]["x"], best[f0]["y"]
        x1_, y1_ = best[f1]["x"], best[f1]["y"]

        for gi in range(f0+1, f1):
            if gi in best or gi < _FRAMES_IN - 1:
                continue
            t   = (gi - f0) / gap
            kx  = x0 + t * (x1_ - x0)
            ky  = y0 + t * (y1_ - y0)

            # TrackNetV2 zoomed
            zx1 = max(0, int(kx) - _ZOOM_HW); zy1 = max(0, int(ky) - _ZOOM_HH)
            zx2 = min(W, zx1+_ZOOM_HW*2);     zy2 = min(H, zy1+_ZOOM_HH*2)
            zoom = (zy1, zy2, zx1, zx2)
            zt = [_preprocess_frame(frames_raw[gi-2], zoom),
                  _preprocess_frame(frames_raw[gi-1], zoom),
                  _preprocess_frame(frames_raw[gi],   zoom)]
            hms = _run_inference_raw(tn_model, tn_device, zt)
            s   = float(hms[2].max())
            tn_zoom = None
            if s >= 0.05:
                py, px = np.unravel_index(np.argmax(hms[2]), hms[2].shape)
                rx = float(px * (zx2-zx1) / _INP_W + zx1)
                ry = float(py * (zy2-zy1) / _INP_H + zy1)
                tn_zoom = {"x": rx, "y": ry, "confidence": s,
                           "width": 30.0, "height": 30.0,
                           "class": "volleyball", "source": "tracknet"}

            # YOLO zoomed
            yolo_zoom = yolo.detect_zoom(frames_raw[gi], kx, ky,
                                         zoom_w=_ZOOM_HW*2, zoom_h=_ZOOM_HH*2,
                                         conf_override=0.05)

            fused = _fuse(tn_zoom, yolo_zoom)
            if fused:
                # Reject if the detection is too far from the guide point
                dist = ((fused["x"] - kx)**2 + (fused["y"] - ky)**2) ** 0.5
                if dist <= _ZOOM_HW * 0.65:
                    best[gi] = fused
                    new_found += 1
                    print(f"  Zoom f{gi}: ({fused['x']:.0f},{fused['y']:.0f}) "
                          f"conf={fused['confidence']:.3f} [{fused['source']}]")

    print(f"  New detections from zoom: {new_found}")

    # ── Pass 2b: trajectory extension — zoom backward/forward from endpoints ──
    # Use velocity from the first/last 3 real detections in the kept segment.
    print("\nPass 2b: trajectory extension from segment endpoints...")
    _EXT_FRAMES  = 30   # how many frames to search beyond each endpoint
    _EXT_ZOOM_HW = 180  # tighter window so ball appears larger in model input
    _EXT_ZOOM_HH = 140
    ext_found = 0

    sorted_det = sorted(best.keys())

    def _zoom_at(gi, kx, ky):
        """Run zoomed TrackNetV2 at predicted position. Returns det or None."""
        if gi < _FRAMES_IN - 1 or gi >= total:
            return None
        zx1 = max(0, int(kx) - _EXT_ZOOM_HW)
        zy1 = max(0, int(ky) - _EXT_ZOOM_HH)
        zx2 = min(W, zx1 + _EXT_ZOOM_HW * 2)
        zy2 = min(H, zy1 + _EXT_ZOOM_HH * 2)
        zoom = (zy1, zy2, zx1, zx2)
        zt = [_preprocess_frame(frames_raw[gi-2], zoom),
              _preprocess_frame(frames_raw[gi-1], zoom),
              _preprocess_frame(frames_raw[gi],   zoom)]
        hms = _run_inference_raw(tn_model, tn_device, zt)
        s = float(hms[2].max())
        if s < 0.05:
            return None
        py, px = np.unravel_index(np.argmax(hms[2]), hms[2].shape)
        rx = float(px * (zx2-zx1) / _INP_W + zx1)
        ry = float(py * (zy2-zy1) / _INP_H + zy1)
        if ((rx-kx)**2 + (ry-ky)**2) ** 0.5 > _EXT_ZOOM_HW * 0.8:
            return None
        return {"x": rx, "y": ry, "confidence": s,
                "width": 30.0, "height": 30.0,
                "class": "volleyball", "source": "tracknet"}

    if len(sorted_det) >= 2:
        # ── Backward extension: velocity from first 2 anchor frames ───────
        f_a, f_b = sorted_det[0], sorted_det[min(2, len(sorted_det)-1)]
        vx = (best[f_b]["x"] - best[f_a]["x"]) / max(1, f_b - f_a)
        vy = (best[f_b]["y"] - best[f_a]["y"]) / max(1, f_b - f_a)
        anchor = f_a
        for step in range(1, _EXT_FRAMES + 1):
            gi = anchor - 1
            if gi < _FRAMES_IN - 1 or gi in best:
                break
            kx = float(np.clip(best[anchor]["x"] - vx, 0, W - 1))
            ky = float(np.clip(best[anchor]["y"] - vy, 0, H - 1))
            det = _zoom_at(gi, kx, ky)
            if det is None:
                break
            best[gi] = det
            ext_found += 1
            sorted_det = sorted(best.keys())
            anchor = gi
            # Update velocity from latest 2 anchors
            nxt = sorted_det[sorted_det.index(gi) + 1]
            vx = (best[nxt]["x"] - det["x"]) / max(1, nxt - gi)
            vy = (best[nxt]["y"] - det["y"]) / max(1, nxt - gi)

        # ── Forward extension: velocity from last 2 anchor frames ─────────
        sorted_det = sorted(best.keys())
        f_c = sorted_det[max(0, len(sorted_det)-3)]
        f_d = sorted_det[-1]
        vx = (best[f_d]["x"] - best[f_c]["x"]) / max(1, f_d - f_c)
        vy = (best[f_d]["y"] - best[f_c]["y"]) / max(1, f_d - f_c)
        anchor = f_d
        for step in range(1, _EXT_FRAMES + 1):
            gi = anchor + 1
            if gi >= total or gi in best:
                break
            kx = float(np.clip(best[anchor]["x"] + vx, 0, W - 1))
            ky = float(np.clip(best[anchor]["y"] + vy, 0, H - 1))
            det = _zoom_at(gi, kx, ky)
            if det is None:
                break
            best[gi] = det
            ext_found += 1
            anchor = gi
            prev = sorted(best.keys())[-2]
            vx = (det["x"] - best[prev]["x"]) / max(1, gi - prev)
            vy = (det["y"] - best[prev]["y"]) / max(1, gi - prev)

    print(f"  Extension found {ext_found} additional frames")

    # ── Pass 3: interpolate remaining gaps ────────────────────────────────────
    print("\nPass 3: interpolating remaining gaps...")
    final: dict = {f: {**d, "source": d["source"]} for f, d in best.items()}

    det_frames = sorted(best.keys())
    interp_count = 0
    for i in range(len(det_frames) - 1):
        f0, f1 = det_frames[i], det_frames[i+1]
        gap = f1 - f0
        if gap <= 1 or gap > _INTERP_MAX_GAP:
            continue
        x0, y0 = best[f0]["x"], best[f0]["y"]
        x1_, y1_ = best[f1]["x"], best[f1]["y"]
        for gi in range(f0+1, f1):
            if gi not in final:
                t = (gi - f0) / gap
                final[gi] = {
                    "x": x0 + t*(x1_-x0), "y": y0 + t*(y1_-y0),
                    "width": 30.0, "height": 30.0,
                    "confidence": 0.0,
                    "class": "volleyball", "source": "interp",
                }
                interp_count += 1

    real_n  = sum(1 for d in final.values() if d["source"] != "interp")
    print(f"  Coverage: {len(final)}/{total} ({100*len(final)/total:.1f}%)  "
          f"real={real_n}  interp={interp_count}")

    # ── Pass 4: render video ──────────────────────────────────────────────────
    if output_video or output_csv:
        print("\nPass 4: rendering...")
        _render(video_path, final, fps, W, H, total, output_video, output_csv)

    return {
        "total_frames": total,
        "covered": len(final),
        "real_detections": real_n,
        "interpolated": interp_count,
        "coverage_pct": 100 * len(final) / total,
        "positions": final,
    }


def _render(video_path, final, fps, W, H, total, output_video, output_csv):
    SOURCE_COLORS = {
        "tracknet": (0, 255, 80),    # green
        "yolo":     (255, 180, 0),   # blue-ish
        "fusion":   (0, 220, 255),   # yellow
        "kalman":   (80, 180, 255),  # light blue
        "interp":   (160, 100, 255), # purple
    }
    TRAIL_LEN = 30

    cap    = cv2.VideoCapture(video_path)
    writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, fps, (W, H))

    trail = []
    rows  = []

    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det = final.get(fi)
        if det:
            xi = int(round(det["x"]))
            yi = int(round(det["y"]))
            src = det["source"]
            col = SOURCE_COLORS.get(src, (200, 200, 200))

            trail.append((xi, yi, src))
            if len(trail) > TRAIL_LEN:
                trail.pop(0)

            # Fading trail
            for j, (tx, ty, ts) in enumerate(trail[:-1]):
                a   = (j + 1) / len(trail)
                tc  = SOURCE_COLORS.get(ts, (200, 200, 200))
                tc  = tuple(int(c * a) for c in tc)
                cv2.circle(frame, (tx, ty), max(2, int(14*a)), tc, -1)

            # Current position ring
            label = f"{det['confidence']:.0%}" if det["confidence"] > 0 else f"~{src}"
            cv2.circle(frame, (xi, yi), 32, col, 3)
            cv2.circle(frame, (xi, yi), 7,  col, -1)
            cv2.putText(frame, label, (xi+36, yi+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)

            rows.append([fi, f"{fi/fps:.4f}", xi, yi,
                         f"{det['confidence']:.4f}", src])

        if writer:
            writer.write(frame)
        fi += 1
        print(f"\r  {fi}/{total}", end="", flush=True)

    cap.release()
    if writer:
        writer.release()

    if output_csv:
        with open(output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time_s", "x_pixel", "y_pixel", "confidence", "source"])
            w.writerows(rows)

    print(f"\n  Video: {output_video}")
    print(f"  CSV:   {output_csv}")
    print(f"\n  Legend:")
    print(f"    green  = TrackNetV2 detection")
    print(f"    yellow = YOLO + TrackNetV2 fusion (both agreed)")
    print(f"    blue   = YOLO-only detection")
    print(f"    purple = interpolated gap fill")
