import base64
import os

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

_ROBOFLOW_WORKFLOW_URL = "https://serverless.roboflow.com/infer/workflows/{workspace}/{workflow_id}"


class BallTracker:
    """
    Detects/tracks a volleyball via either:
      - 'hosted': Roboflow serverless workflow (no local GPU needed)
      - 'local':  local YOLOv8 .pt weights with ByteTrack
    """

    def __init__(self, mode: str = "hosted", model_path: str | None = None, conf: float = 0.15):
        self.mode = mode
        self.conf = conf

        if mode == "hosted":
            self._api_key = os.environ["ROBOFLOW_API_KEY"]
            self._workspace = os.environ["ROBOFLOW_WORKSPACE"]
            self._workflow_id = os.environ["ROBOFLOW_WORKFLOW_ID"]
            self._url = _ROBOFLOW_WORKFLOW_URL.format(
                workspace=self._workspace,
                workflow_id=self._workflow_id,
            )

        elif mode == "local":
            if model_path is None:
                raise ValueError("model_path is required for local mode")
            from ultralytics import YOLO
            self._model = YOLO(model_path)

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'hosted' or 'local'.")

    def detect_image(self, image_path: str) -> list[dict]:
        """Run detection on a single image file."""
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        if self.mode == "hosted":
            return self._call_workflow(b64)
        results = self._model.predict(source=image_path, conf=self.conf, verbose=False)
        return self._parse_yolo_detections(results)

    def detect_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection on a single BGR numpy frame (e.g. from cv2.VideoCapture).
        Returns a list of dicts: [{x, y, width, height, confidence, class}, ...]
        """
        if self.mode == "hosted":
            _, buf = cv2.imencode(".jpg", frame)
            b64 = base64.b64encode(buf).decode("utf-8")
            return self._call_workflow(b64)

        results = self._model.predict(source=frame, conf=self.conf, verbose=False)
        return self._parse_yolo_detections(results)

    def track_video(self, source: str):
        """Stream tracking results frame-by-frame. Local mode only."""
        if self.mode == "hosted":
            raise NotImplementedError("Use scripts/track_video.py for hosted video tracking.")
        return self._model.track(source=source, tracker="bytetrack.yaml", conf=self.conf, stream=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_workflow(self, b64_image: str) -> list[dict]:
        payload = {
            "api_key": self._api_key,
            "inputs": {
                "image": {"type": "base64", "value": b64_image},
                "classes": ["volleyball"],
            },
        }
        resp = requests.post(self._url, json=payload, timeout=60)
        resp.raise_for_status()
        return self._parse_detections(resp.json())

    def _parse_detections(self, result) -> list[dict]:
        """Extract detections from a workflow API response into a flat list."""
        detections = []

        # Unwrap top-level outputs list if present
        if isinstance(result, dict) and "outputs" in result:
            items = result["outputs"]
        elif isinstance(result, list):
            items = result
        else:
            items = [result]

        for item in items:
            if not isinstance(item, dict):
                continue
            predictions = None
            pred_val = item.get("predictions")
            if isinstance(pred_val, dict):
                predictions = pred_val.get("predictions", [])
            elif isinstance(pred_val, list):
                predictions = pred_val
            elif "x" in item and "y" in item:
                predictions = [item]

            if predictions:
                for p in predictions:
                    if isinstance(p, dict) and "x" in p and "y" in p:
                        detections.append({
                            "x": float(p["x"]),
                            "y": float(p["y"]),
                            "width": float(p.get("width", 0)),
                            "height": float(p.get("height", 0)),
                            "confidence": float(p.get("confidence", 1.0)),
                            "class": p.get("class", "volleyball"),
                        })
        return detections

    def _parse_yolo_detections(self, results) -> list[dict]:
        """Convert Ultralytics Results objects to the same flat dict format."""
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "confidence": float(box.conf[0]),
                    "class": r.names[int(box.cls[0])],
                })
        return detections
