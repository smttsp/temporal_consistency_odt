import copy
import sys
from datetime import datetime

import cv2


EPS = sys.float_info.epsilon


def create_video_writer(
    video_cap: cv2.VideoCapture, output_filename: str, fps: float = -1
):
    """Create a video writer object to write the output video

    Args:
        video_cap (cv2.VideoCapture): Video capture object
        output_filename (str): Output filename
        fps (float, optional): Frames per second. Defaults to None.

    Returns:
        cv2.VideoWriter: Video writer object
    """

    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS)) if fps <= 0 else fps

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_filename, fourcc, fps, (frame_width, frame_height)
    )

    return writer


def compute_iou(bbox1: list[int], bbox2: list[int]):
    """Compute the intersection over union (IoU) of two bounding boxes.

    Args:
        bbox1 (list): First bounding box in the format of [x1, y1, x2, y2]
        bbox2 (list): Second bounding box in the format of [x1, y1, x2, y2]

    Returns:
        float: Intersection over union (IoU) of bbox1 and bbox2
    """

    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Compute the area of the intersection rectangle
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Compute the area of both bounding boxes
    box1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    box2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Compute the IoU
    iou = inter_area / float(box1_area + box2_area - inter_area + EPS)

    return iou


def get_runtime_str():
    """Getting datetime as a string

    Returns:
        str: Datetime as a string
    """

    runtime_str = (
        datetime.now()
        .isoformat()
        .replace(":", "")
        .replace("-", "")
        .replace("T", "-")
        .split(".")[0]
    )
    return runtime_str


def ltwh_to_ltrb(ltwh: list[int]):
    """Converts bounding box coordinates from [left, top, width, height] to
    [left, top, right, bottom].
    """

    ltrb = copy.deepcopy(ltwh)
    ltrb[2] += ltrb[0]
    ltrb[3] += ltrb[1]
    return ltrb


def ltrb_to_ltwh(ltrb: list[int]):
    """Converts bounding box coordinates from [left, top, right, bottom] to
    [left, top, width, height].
    """

    ltwh = copy.deepcopy(ltrb)
    ltwh[2] -= ltwh[0]
    ltwh[3] -= ltwh[1]
    return ltwh
