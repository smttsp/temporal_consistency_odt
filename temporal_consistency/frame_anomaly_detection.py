"""This module contains `TemporalAnomalyDetector` class for identifying inconsistencies
in object tracking across a sequence of frames. It checks for issues like classification
inconsistencies, missing objects in frames, single-frame appearances, and low
Intersection-over-Union (IoU) values. Detected anomalies are stored in a dictionary.
"""

import sys
from collections import defaultdict

from loguru import logger

from temporal_consistency.tracked_frame import TrackedFrameCollection
from temporal_consistency.utils import compute_iou


MIN_IOU_THRESH = 0.5
EPS = sys.float_info.epsilon


class TemporalAnomalyDetector:
    """Detects anomalies in the temporal consistency of the tracked objects."""

    def __init__(self, frame_collection: TrackedFrameCollection):
        """Initializes the TemporalAnomalyDetector.

        Args:
            frame_collection (TrackedFrameCollection): A collection of frames
                containing tracked objects.
        """

        self.frame_collection = frame_collection
        self.anomalies = defaultdict(list)
        self.scan_for_anomalies()

    def scan_for_anomalies(self):
        """Scans for anomalies across all objects in the frame collection."""

        for object_id, track_info in self.frame_collection.all_objects.items():
            self.inspect_object_for_anomalies(object_id, track_info)

        return None

    def inspect_object_for_anomalies(
        self, object_id: str, track_info: dict
    ) -> bool:
        """Checks for potential anomalies for a single tracked object.

        Args:
            object_id (str): Unique identifier for the object.
            track_info (): Tracking information associated with the object.

        Returns:
            bool: True if anomalies are detected, False otherwise.
        """

        anomaly_exist = (
            self.is_class_inconsistent(object_id, track_info)
            or self.is_object_missing_in_frames(object_id, track_info)
            or self.appears_only_in_single_frame(object_id, track_info)
            or self.has_low_iou(object_id, track_info)
        )
        return anomaly_exist

    def is_class_inconsistent(self, object_id: str, track_info: dict) -> bool:
        """Verifies that an object maintains consistent classification across frames.

        Returns:
            bool: True if the class is consistent, False if there are inconsistencies.
        """

        all_classes = set(v.class_name for v in track_info.values())

        if len(all_classes) > 1:
            log = f"{object_id=} occurs as the following classes: {all_classes}"
            logger.info(log)
            self.anomalies[object_id].append(log)

        return len(all_classes) > 1

    def is_object_missing_in_frames(
        self, object_id: str, track_info: dict
    ) -> bool:
        """Checks if an object goes missing in intermediate frames.

        Returns:
            bool: True if the object is missing in some frames, False otherwise.
        """

        keys = track_info.keys()
        mn_idx, mx_idx, size = min(keys), max(keys), len(track_info)
        expected_size = mx_idx - mn_idx + 1
        if size != expected_size:
            log = f"{object_id=} is missing in {expected_size - size} frames"
            logger.info(log)
            self.anomalies[object_id].append(log)

        return expected_size != size

    def appears_only_in_single_frame(
        self, object_id: str, track_info: dict
    ) -> bool:
        """Checks if an object only appears in a single frame, potentially
            indicating a false detection.

        Returns:
            bool: True if the object appears in only one frame, False otherwise.
        """

        if len(track_info) == 1:
            log = f"{object_id=} occurs only in one frame, may indicate false detection"
            logger.info(log)
            self.anomalies[object_id].append(log)

        return len(track_info) == 1

    def has_low_iou(self, object_id: str, track_info: dict) -> bool:
        """Assesses if the Intersection over Union (IoU) is below a threshold,
            indicating potential tracking issues.

        Returns:
            bool: True if the IoU is low, False otherwise.
        """

        keys = sorted(list(track_info.keys()))
        flag = False
        for frame_i, frame_j in zip(keys, keys[1:]):
            t1 = track_info[frame_i]
            t2 = track_info[frame_j]

            # TODO (samet): Need to check if the bboxes are close to frame edges
            iou = compute_iou(t1.ltrb, t2.ltrb)

            if iou < MIN_IOU_THRESH:
                log = (
                    f"{iou=} is lower than threshold of {MIN_IOU_THRESH} "
                    f"between {frame_i=} and {frame_j=}"
                )
                logger.info(log)
                self.anomalies[object_id].append(log)
                flag = True

        return flag
