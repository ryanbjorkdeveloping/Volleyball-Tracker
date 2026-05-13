import cv2
import numpy as np


class CourtMapper:
    """Maps pixel coordinates to real-world court coordinates via homography."""

    # Standard volleyball court dimensions in meters
    COURT_WIDTH = 9.0
    COURT_LENGTH = 18.0

    def __init__(self):
        self.H = None  # homography matrix

    def calibrate(self, pixel_points: np.ndarray, court_points: np.ndarray | None = None):
        """
        pixel_points: (4, 2) array of court corner pixels in the image
        court_points: (4, 2) real-world coords in meters; defaults to standard court corners
        """
        if court_points is None:
            court_points = np.array([
                [0.0, 0.0],
                [self.COURT_WIDTH, 0.0],
                [self.COURT_WIDTH, self.COURT_LENGTH],
                [0.0, self.COURT_LENGTH],
            ], dtype=np.float32)

        self.H, _ = cv2.findHomography(pixel_points.astype(np.float32), court_points)

    def to_court_coords(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        if self.H is None:
            raise RuntimeError("Call calibrate() before converting coordinates.")
        pt = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self.H)
        return float(result[0][0][0]), float(result[0][0][1])
