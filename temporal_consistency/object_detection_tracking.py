import datetime

import cv2
import torch
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
    confidence = data[4]

    res = []
    if confidence >= confidence_threshold:
        x_min, y_min, x_max, y_max = map(int, data[:4])
        ltwh = [x_min, y_min, x_max - x_min, y_max - y_min]
        class_id = int(data[5])
        res = [ltwh, confidence, class_id]
    return res


def object_detection(model, frame, num_aug=0, confidence_threshold=0.0):
    frame_aug = get_random_augmentation(frame, num_aug=num_aug)

    with torch.no_grad():
        detections = model(frame)[0]

    all_filtered_results = [
        get_detected_object(data, confidence_threshold)
        for data in detections.boxes.data.tolist()
    ]

    results = [res for res in all_filtered_results if res]

    return results, frame_aug


def object_tracking(frame, results, deep_sort_tracker, classes):
    # update the tracker with the new detections
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
    start = datetime.datetime.now()

    ret, frame = video_cap.read()
    if not ret:
        return True, None, 0

    results, frame_aug = object_detection(
        model, frame, num_aug, confidence_threshold
    )
    frame_after = object_tracking(
        frame_aug, results, deep_sort_tracker, classes=model.names
    )
    tframe = TrackedFrame(frame_id, frame_aug, deep_sort_tracker.tracker)
    tframe_collection.add_tracked_frame(tframe)
    end = datetime.datetime.now()

    total_time = (end - start).total_seconds() * 1000
    draw_fps_on_frame(frame_after, total_time)

    return False, frame_after, total_time


def object_detection_and_tracking(
    model, deep_sort_tracker, video_filepath, num_aug, confidence_threshold
):
    video_cap = cv2.VideoCapture(video_filepath)
    output_filepath = video_filepath.replace(".mp4", "_output.mp4")

    writer = create_video_writer(video_cap, output_filepath)

    tframe_collection = TrackedFrameCollection(
        video_cap=video_cap, classes=model.names
    )
    frame_id = 0

    while True:
        end_of_video, frame_after, total_time = process_single_frame(
            model,
            video_cap,
            frame_id,
            tframe_collection,
            deep_sort_tracker,
            num_aug,
            confidence_threshold,
        )
        frame_id += 1

        if end_of_video:
            break
        writer.write(frame_after)

    tframe_collection.export_all_objects()

    video_cap.release()
    writer.release()
    cv2.destroyAllWindows()

    return None
