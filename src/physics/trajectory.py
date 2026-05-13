import numpy as np

GRAVITY = 9.81  # m/s²


class BallPhysics:
    def __init__(self, fps: float):
        self.fps = fps
        self.dt = 1.0 / fps

    def velocity(self, pos_prev: tuple, pos_curr: tuple) -> np.ndarray:
        return (np.array(pos_curr) - np.array(pos_prev)) / self.dt

    def predict_landing(self, pos: tuple, vel: np.ndarray) -> tuple[float, float] | None:
        """
        Predict where ball lands (z=0) given current 3D position and velocity.
        pos: (x, y, z) in meters, z is height
        vel: (vx, vy, vz) in m/s
        Returns (x, y) landing point or None if ball is already on ground or moving up with no arc.
        """
        x0, y0, z0 = pos
        vx, vy, vz = vel

        # Solve: z0 + vz*t - 0.5*g*t^2 = 0
        a = -0.5 * GRAVITY
        b = vz
        c = z0
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return None

        t1 = (-b + discriminant**0.5) / (2 * a)
        t2 = (-b - discriminant**0.5) / (2 * a)
        t = max(t for t in [t1, t2] if t > 0) if any(t > 0 for t in [t1, t2]) else None
        if t is None:
            return None

        return x0 + vx * t, y0 + vy * t
