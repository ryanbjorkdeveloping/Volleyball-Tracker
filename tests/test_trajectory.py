import numpy as np
import pytest
from src.physics.trajectory import BallPhysics


def test_velocity_calculation():
    physics = BallPhysics(fps=30)
    vel = physics.velocity((0, 0), (3, 0))
    assert vel[0] == pytest.approx(90.0)


def test_landing_prediction_returns_point():
    physics = BallPhysics(fps=30)
    pos = (0.0, 0.0, 3.0)
    vel = np.array([5.0, 0.0, 0.0])
    result = physics.predict_landing(pos, vel)
    assert result is not None
    assert len(result) == 2


def test_landing_none_when_already_grounded():
    physics = BallPhysics(fps=30)
    pos = (0.0, 0.0, 0.0)
    vel = np.array([5.0, 0.0, 0.0])
    result = physics.predict_landing(pos, vel)
    assert result is None
