import pytest

from temporal_consistency.utils import (
    compute_iou,
    get_runtime_str,
    ltrb_to_ltwh,
    ltwh_to_ltrb,
)


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


def test_get_runtime_str():
    runtime_str = get_runtime_str()
    assert len(runtime_str) == 15
    assert runtime_str[8] == "-"


@pytest.mark.parametrize(
    "ltrb, ltwh",
    [
        ([1, 2, 3, 4], [1, 2, 2, 2]),
        ([1, 2, 4, 6], [1, 2, 3, 4]),
    ],
)
def test_ltwh_to_ltrb(ltrb, ltwh):
    assert ltwh_to_ltrb(ltwh) == ltrb


@pytest.mark.parametrize(
    "ltrb, ltwh",
    [
        ([1, 2, 8, 6], [1, 2, 7, 4]),
        ([1, 2, 4, 6], [1, 2, 3, 4]),
    ],
)
def test_ltrb_to_ltwh(ltrb, ltwh):
    assert ltrb_to_ltwh(ltrb) == ltwh
