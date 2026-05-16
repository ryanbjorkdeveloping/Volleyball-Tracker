import cv2
import numpy as np

_SCALE = 0.33  # downscale factor for processing


class CVBallTracker:
    """
    Frame-by-frame volleyball tracker.

    Pipeline per frame:
      1. Frame diff     — finds moving regions (ignores static objects like logos)
      2. Size filter    — ball is small; rejects large blobs (players, spectators)
      3. Hough circles  — confirms the candidate is actually round
      4. Kalman filter  — predicts position when ball isn't detected

    Calibrated for the specific video: ~25px ball radius at 3420-wide resolution,
    which is ~8px at 0.33 scale.
    """

    def __init__(self, min_radius: int = 5, max_radius: int = 20, conf: float = 0.3):
        self.min_radius = min_radius   # in scaled pixels
        self.max_radius = max_radius   # in scaled pixels
        self.conf = conf

        self._prev_gray = None

        self._kalman = cv2.KalmanFilter(4, 2)
        self._kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self._kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self._kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self._kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
        self._kalman.errorCovPost = np.eye(4, dtype=np.float32)

        self._initialized = False
        self._frames_since_detect = 0
        self._max_coast = 5
        self._confirm_buffer = []   # recent circles pending lock-on confirmation
        self._confirm_needed = 2    # need N consecutive frames before locking on

    def detect_frame(self, frame: np.ndarray) -> list[dict]:
        inv_scale = 1.0 / _SCALE

        small = cv2.resize(frame, (0, 0), fx=_SCALE, fy=_SCALE)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 1)

        predicted = self._kalman.predict()
        pred_x = float(predicted[0][0])
        pred_y = float(predicted[1][0])

        detection = None
        if self._prev_gray is not None:
            diff = cv2.absdiff(gray, self._prev_gray)
            _, motion_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

            # Light morphology — don't inflate small blobs into large ones
            kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_DILATE, kernel3, iterations=1)

            # Run Hough on full gray (masking first breaks gradient detection),
            # then keep only circles whose center falls inside the motion mask.
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1.0,
                minDist=15,
                param1=30,
                param2=8,
                minRadius=self.min_radius,
                maxRadius=self.max_radius,
            )

            if circles is not None:
                # Filter: circle center must be in a moving region
                h, w = motion_mask.shape
                motion_circles = [
                    c for c in circles[0]
                    if 0 <= int(c[1]) < h and 0 <= int(c[0]) < w
                    and motion_mask[int(c[1]), int(c[0])] == 255
                ]
                if motion_circles:
                    detection = self._pick_best_circle(
                        np.array(motion_circles), pred_x, pred_y
                    )

        self._prev_gray = gray

        if detection is not None:
            cx_s, cy_s, r_s = detection

            if not self._initialized:
                # Require N consecutive frames before locking on (suppresses false positives)
                self._confirm_buffer.append((cx_s, cy_s, r_s))
                if len(self._confirm_buffer) >= self._confirm_needed:
                    # Check all buffered positions are close together (same object)
                    xs = [c[0] for c in self._confirm_buffer]
                    ys = [c[1] for c in self._confirm_buffer]
                    spread = max(max(xs)-min(xs), max(ys)-min(ys))
                    if spread < self.max_radius * 6:
                        # Lock on — initialise Kalman with buffered positions
                        for bx, by, _ in self._confirm_buffer:
                            self._kalman.correct(np.array([[np.float32(bx)], [np.float32(by)]]))
                        self._initialized = True
                        self._confirm_buffer = []
                        self._frames_since_detect = 0
                    else:
                        # Objects too spread — clear buffer, start fresh
                        self._confirm_buffer = [self._confirm_buffer[-1]]
                # Not yet locked — don't output a detection
                return []

            # Already locked on — correct Kalman and output
            self._kalman.correct(np.array([[np.float32(cx_s)], [np.float32(cy_s)]]))
            self._frames_since_detect = 0
            cx = cx_s * inv_scale
            cy = cy_s * inv_scale
            diameter = r_s * 2 * inv_scale
            return [{
                "x": cx, "y": cy,
                "width": diameter, "height": diameter,
                "confidence": 0.85, "class": "volleyball", "source": "detected",
            }]

        else:
            # No detection this frame
            self._confirm_buffer = []  # reset confirmation window

        if self._initialized and self._frames_since_detect < self._max_coast:
            self._frames_since_detect += 1
            cx = pred_x * inv_scale
            cy = pred_y * inv_scale
            diameter = self.max_radius * 2 * inv_scale
            conf = max(0.3, 0.8 - self._frames_since_detect * 0.1)
            return [{
                "x": cx, "y": cy,
                "width": diameter, "height": diameter,
                "confidence": conf, "class": "volleyball", "source": "kalman",
            }]

        # Too many frames without detection — reset so the ball can be re-acquired
        if self._initialized and self._frames_since_detect >= self._max_coast:
            self._initialized = False
            self._kalman.errorCovPost = np.eye(4, dtype=np.float32)

        self._frames_since_detect += 1
        return []

    def _pick_best_circle(self, circles: np.ndarray, pred_x: float, pred_y: float):
        """
        From Hough circles, pick the one closest to Kalman prediction.
        On first detection (no prior), pick the most central / largest circle.
        Returns (cx, cy, r) in scaled coords, or None.
        """
        if not self._initialized:
            # No prior — pick circle closest to the center of the frame (ball is usually mid-frame)
            return tuple(circles[0])

        max_jump = max(60, self.max_radius * 4)
        reachable = [c for c in circles if np.hypot(c[0] - pred_x, c[1] - pred_y) < max_jump]
        if not reachable:
            return None

        return min(reachable, key=lambda c: np.hypot(c[0] - pred_x, c[1] - pred_y))
