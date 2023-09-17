import sys

import cv2


EPS = sys.float_info.epsilon


def create_video_writer(video_cap, output_filename):
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    writer = cv2.VideoWriter(
        output_filename, fourcc, fps, (frame_width, frame_height)
    )

    return writer


def compute_iou(bbox1, bbox2):
    # Determine the coordinates of the intersection rectangle
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    # Compute the area of the intersection rectangle
    inter_area = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    box1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    box2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Compute the IoU
    iou = inter_area / float(box1_area + box2_area - inter_area + EPS)

    return iou
