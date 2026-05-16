"""
YOLOv8 single-frame volleyball detector.

Uses the COCO pre-trained YOLOv8n model with sports-ball class (32).
Runs on MPS (Apple Silicon), CUDA, or CPU.
"""
import numpy as np
import cv2
import torch

_SPORTS_BALL_CLASS = 32   # COCO class index for "sports ball"


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class YOLOBallDetector:
    """
    Single-frame volleyball detector using YOLOv8n (COCO weights).

    Accepts an optional court_crop so it only searches the play area,
    making the ball proportionally larger in the model input.
    """

    # Auto-download model size: n=nano (fastest), s=small, m=medium
    MODEL_SIZE = "n"

    def __init__(self, conf: float = 0.15, court_crop: tuple | None = None):
        from ultralytics import YOLO
        self._device = _get_device()
        self._model  = YOLO(f"yolov8{self.MODEL_SIZE}.pt")
        self._conf   = conf
        self._court_crop = court_crop   # (y1, y2, x1, x2) in original pixels

    def detect(self, frame: np.ndarray) -> dict | None:
        """
        Run YOLO on frame. Returns a detection dict or None.
        Dict keys: x, y, width, height, confidence, class, source.
        """
        if self._court_crop is not None:
            y1, y2, x1, x2 = self._court_crop
            img = frame[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            img = frame
            offset_x, offset_y = 0, 0

        results = self._model(
            img,
            conf=self._conf,
            classes=[_SPORTS_BALL_CLASS],
            device=self._device,
            verbose=False,
        )

        best = None
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if best is None or conf > best["confidence"]:
                    bx, by, bw, bh = box.xywh[0].tolist()
                    best = {
                        "x":          bx + offset_x,
                        "y":          by + offset_y,
                        "width":      bw,
                        "height":     bh,
                        "confidence": conf,
                        "class":      "volleyball",
                        "source":     "yolo",
                    }
        return best

    def detect_zoom(self, frame: np.ndarray, cx: float, cy: float,
                    zoom_w: int = 900, zoom_h: int = 700,
                    conf_override: float | None = None) -> dict | None:
        """
        Run YOLO on a tight crop centred on (cx, cy).
        Useful for guided search when Kalman has a predicted position.
        """
        H, W = frame.shape[:2]
        x1 = max(0, int(cx) - zoom_w // 2)
        y1 = max(0, int(cy) - zoom_h // 2)
        x2 = min(W, x1 + zoom_w)
        y2 = min(H, y1 + zoom_h)
        img = frame[y1:y2, x1:x2]

        old_conf = self._conf
        if conf_override is not None:
            self._conf = conf_override

        results = self._model(
            img,
            conf=self._conf,
            classes=[_SPORTS_BALL_CLASS],
            device=self._device,
            verbose=False,
        )
        self._conf = old_conf

        best = None
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if best is None or conf > best["confidence"]:
                    bx, by, bw, bh = box.xywh[0].tolist()
                    best = {
                        "x":          bx + x1,
                        "y":          by + y1,
                        "width":      bw,
                        "height":     bh,
                        "confidence": conf,
                        "class":      "volleyball",
                        "source":     "yolo",
                    }
        return best
