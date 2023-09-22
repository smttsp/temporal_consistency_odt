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

    def __init__(self, tframe_collection: TrackedFrameCollection):
        """Initializes the TemporalAnomalyDetector.

        Args:
            tframe_collection (TrackedFrameCollection): A collection of frames
                containing tracked objects.
        """

        self.tframe_collection = tframe_collection
        self.anomalies = defaultdict(list)
        self.scan_for_anomalies()
        self.export_anomalies()

    def export_anomalies(self):
        """Exports the frames where there is an anomaly for at least one object.
        The exported files are:
        1. Raw frame as frame{frame_id}.jpg
        2. Frame with bboxes as frame{frame_id}_bbox.jpg. The green bboxes are the
            tracked objects, and the red bboxes are the low-confidence detections.
        3. a file containing all the bboxes in the frame as frame{frame_id}_bbox.txt

        1 + 3 can be used to help with labeling the data (i.e. model assisted labeling)
        """

    def scan_for_anomalies(self):
        """Scans for anomalies across all objects in the frame collection."""

        for object_id, track_info in self.tframe_collection.all_objects.items():
            self.inspect_object_for_anomalies(object_id, track_info)

        return None

    def inspect_object_for_anomalies(
        self, object_id: str, track_info: dict
    ) -> bool:
        """Checks for potential anomalies for a single tracked object.

        Args:
            object_id (str): Unique identifier of the tracked object.
            track_info (dict): A dictionary containing frame IDs as keys and prediction
                objects as values, where each prediction has a 'class_name' attribute.

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
            bool: False if the class is consistent, True if there are inconsistencies.
        """

        all_classes = set(v.class_name for v in track_info.values())

        if len(all_classes) > 1:
            log = f"{object_id=} occurs as the following classes: {all_classes}"
            logger.info(log)
            self.get_frame_id_for_class_inconsistency(object_id, track_info)

        return len(all_classes) > 1

    def get_frame_id_for_class_inconsistency(self, object_id, track_info):
        """Find the first frame ID where a class inconsistency occurs in
            a tracking sequence.

        Returns:
            None: The function modifies the 'anomalies' attribute of the object,
                appending frame IDs where class inconsistency is detected.
        """

        prev_class = None

        for frame_id, pred in track_info.items():
            current_class = pred.class_name
            if prev_class is not None and prev_class != current_class:
                self.anomalies[object_id].append(frame_id)
                break

            prev_class = current_class

        return None

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

            # sampling only one frame where the object is missing
            frame_ids = set(range(mn_idx, mx_idx + 1)).difference(keys)
            self.anomalies[object_id].append(list(frame_ids)[0])

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
            frame_id = list(track_info.keys())[0]
            self.anomalies[object_id].append(frame_id)

        return len(track_info) == 1

    def has_low_iou(self, object_id: str, track_info: dict) -> bool:
        """Assesses if the Intersection over Union (IoU) is below a threshold,
            indicating potential tracking issues.

        Returns:
            bool: True if there is low IoU, False otherwise.
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
                self.anomalies[object_id].append(frame_i)
                flag = True

        return flag
