import pytest

from temporal_consistency.utils import compute_iou


EPS = 1e-10


@pytest.mark.parametrize(
    "box1, box2, expected_iou",
    [
        # Non-overlapping boxes
        ([0, 0, 1, 1], [3, 3, 4, 4], 0.0),
        # Fully overlapping boxes
        ([0, 0, 1, 1], [0, 0, 1, 1], 1.0),
        # Partially overlapping boxes
        ([0, 0, 1, 1], [1, 1, 2, 2], 0.0),
        # Another partially overlapping boxes
        ([0, 0, 2, 2], [1, 1, 3, 3], 1 / 7),
        # Box2 inside box1
        ([0, 0, 3, 3], [1, 1, 2, 2], 1 / 9),
        # Negative coordinates
        ([-3, -3, -1, -1], [-2, -2, -1, -1], 1 / 4),
        # Negative coordinates
        ([-3, -3, 1, 1], [-4, -4, -2, -2], 1 / 19),
    ],
)
def test_compute_iou(box1, box2, expected_iou):
    assert abs(compute_iou(box1, box2) - expected_iou) < EPS
