"""This module handles object detection and tracking in video sequences.
It performs

- real-time object detection using a YOLO model
- object tracking using Deep SORT.

The frames can be optionally augmented before processing which is
for helping with robustness of the object detection model (i.e., finding failures).

The output includes visualization of object tracking (one video with bboxes and
one video per object with the object's track).
"""

import datetime

import cv2
import numpy
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

from temporal_consistency.augmentations import get_random_augmentation
from temporal_consistency.tracked_frame import (
    TrackedFrame,
    TrackedFrameCollection,
)
from temporal_consistency.utils import create_video_writer
from temporal_consistency.vis_utils import (
    draw_bbox_around_object,
    draw_fps_on_frame,
)


def get_detected_object(data, confidence_threshold):
    """Filters out the detected object if the confidence is below the threshold.

    Args:
        data (list): List of data for a detected object.
        confidence_threshold (float): Confidence threshold for object detection.

    Returns:
        list: List of data for a detected object if the confidence is above the thresh.
    """

    confidence = data[4]

    x_min, y_min, x_max, y_max = map(int, data[:4])
    ltwh = [x_min, y_min, x_max - x_min, y_max - y_min]
    class_id = int(data[5])
    res = [ltwh, confidence, class_id]
    return res


def object_detection(model, frame, num_aug=0, confidence_threshold=0.1):
    """Performs object detection on the given frame and returns the results.

    Args:
        model (YOLO): Model used for object detection.
        frame (numpy.ndarray): Frame on which objects are detected.
        num_aug (int, optional): Number of augmentations to apply to the frame.
        confidence_threshold (float, optional): Threshold for object detection.
    """

    frame_aug = get_random_augmentation(frame, num_aug=num_aug)

    with torch.no_grad():
        detections = model(frame_aug)[0]

    all_results = [
        get_detected_object(data, confidence_threshold)
        for data in detections.boxes.data.tolist()
    ]
    results = []
    low_confidence_results = []

    for res in all_results:
        if res[1] >= confidence_threshold:
            results.append(res)
        else:
            low_confidence_results.append(res)

    return results, low_confidence_results, frame_aug


def object_tracking(
    frame: numpy.ndarray,
    results: list,
    deep_sort_tracker: DeepSort,
    classes: dict,
) -> numpy.ndarray:
    """Processes the given frame with object tracking using Deep SORT
        and visualizes the tracking results.

    The function takes an input frame, object detection results, a DeepSort
    tracker instance, and a dictionary mapping class IDs to class names.
    It updates the tracker with the new detection results and draws bounding boxes
    around the confirmed tracks on a copy of the input frame.

    Args:
        frame (numpy.ndarray): The frame on which objects are detected and tracked.
        results (list): List of object detection results for the given frame.
        deep_sort_tracker (DeepSort): Instance of the DST to update and track objects.
        classes (dict): Dictionary mapping class IDs to class names for visualization.

    Returns:
        numpy.ndarray: Frame with drawn bboxes around the confirmed tracked objects.
    """

    tracks = deep_sort_tracker.update_tracks(results, frame=frame)

    frame_after = frame.copy()
    for track in tracks:
        if not track.is_confirmed():
            continue

        voc_bbox = track.to_ltrb()
        draw_bbox_around_object(frame_after, track, voc_bbox, classes)
    return frame_after


def process_single_frame(
    model,
    video_cap,
    frame_id,
    tframe_collection,
    deep_sort_tracker,
    num_aug,
    confidence_threshold,
):
    """Processes a single frame from the video. This function does object
        detection and tracking. It also updates the TrackedFrameCollection.
        The last step is to draw the FPS on the frame.

    Args:
        model (YOLO): Model used for object detection.
        video_cap (cv2.VideoCapture): Video capture object.
        frame_id (int): Frame ID.
        tframe_collection (TrackedFrameCollection): Collection of tracked frames.
        deep_sort_tracker (DeepSort): Deep SORT tracker.
        num_aug (int): Number of augmentations to apply to the frame.
        confidence_threshold (float): Confidence threshold for object detection.

    Returns:
        bool: True if the end of the video is reached, False otherwise.
    """

    start = datetime.datetime.now()

    ret, frame = video_cap.read()
    if not ret:
        return True, None

    results, low_confidence_results, frame_aug = object_detection(
        model, frame, num_aug, confidence_threshold
    )
    frame_after = object_tracking(
        frame_aug, results, deep_sort_tracker, classes=model.names
    )
    tframe = TrackedFrame(
        frame_id,
        frame_aug,
        deep_sort_tracker.tracker,
        low_confidence_results,
        class_names=model.names,
    )
    tframe_collection.add_tracked_frame(tframe)
    end = datetime.datetime.now()

    total_time = (end - start).total_seconds() * 1000
    draw_fps_on_frame(frame_after, total_time)

    return frame_after


# def apply_detection_and_tracking(
#     model,
#     deep_sort_tracker,
#     num_aug,
#     video_cap,
#     writer,
#     out_folder,
#     out_video_fps,
#     confidence_threshold,
# ):


def apply_detection_and_tracking(
    model,
    deep_sort_tracker: DeepSort,
    num_aug: int,
    video_cap: cv2.VideoCapture,
    writer: cv2.VideoWriter,
    out_folder: str,
    out_video_fps: int,
    confidence_threshold: float,
) -> TrackedFrameCollection:
    """Applies object detection and tracking on video frames using
    the provided model and tracker.

    This function processes a video by detecting objects in its frames and
    then tracking those objects using Deep SORT. The results, including bboxes,
    are written to a video. The tracked objects are also saved as separate videos.

    Args:
        model (YOLO): Model used for object detection.
        deep_sort_tracker (DeepSort): Deep SORT tracker instance for object tracking.
        num_aug (int): Number of augmentations to apply to the frame.
        video_cap (cv2.VideoCapture): Video capture object to read frames from.
        writer (cv2.VideoWriter): Video writer object to output the processed video.
        out_folder (str): Output folder path where tracked objects will be saved.
        out_video_fps (int): Frames per second for the output video.
        confidence_threshold (float): Confidence threshold for object detection.

    Returns:
        TrackedFrameCollection: A collection of frames with tracking information.
    """

    tframe_collection = TrackedFrameCollection(
        video_cap=video_cap, class_names=model.names, out_folder=out_folder
    )

    frame_id = 0
    while True:
        frame_after = process_single_frame(
            model,
            video_cap,
            frame_id,
            tframe_collection,
            deep_sort_tracker,
            num_aug,
            confidence_threshold,
        )
        if end_of_video:
            break

        if frame_id >= 100:
            break

        writer.write(frame_after)
        frame_id += 1

    tframe_collection.export_all_objects(out_video_fps=out_video_fps)

    video_cap.release()
    writer.release()
    cv2.destroyAllWindows()

    return tframe_collection
