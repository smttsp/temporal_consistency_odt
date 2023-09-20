import sys
from collections import defaultdict

from temporal_consistency.tracked_frame import TrackedFrameCollection
from temporal_consistency.utils import compute_iou


MIN_IOU_THRESH = 0.5
EPS = sys.float_info.epsilon


def extract_track_data(tracker):
    """Extract track_id and det_class from each tracker"""
    return {track.track_id: track.det_class for track in tracker.tracks}


def extract_object_pairs(tracker1, tracker2):
    bbox_dict = defaultdict(list)
    for track in tracker1.tracks:
        bbox_dict[track.track_id].append(track.to_ltrb())

    for track in tracker2.tracks:
        bbox_dict[track.track_id].append(track.to_ltrb())

    return bbox_dict


class TemporalAnomalyDetector:
    def __init__(self, frame_collection: TrackedFrameCollection):
        """Initializes the TemporalAnomalyDetector.

        Args:
            frame_collection (TrackedFrameCollection): A collection of frames containing tracked objects.
        """
        self.frame_collection = frame_collection
        self.scan_for_anomalies()
        self.anomaly_logs = []

    def scan_for_anomalies(self):
        """Scans for anomalies across all objects in the frame collection."""
        for object_id, track_info in self.frame_collection.all_objects.items():
            print(object_id)
            anomaly_exist = self.inspect_object_for_anomalies(
                object_id, track_info
            )

    def inspect_object_for_anomalies(self, object_id, track_info) -> bool:
        """Checks for potential anomalies for a single tracked object.

        Args:
            object_id (int): Unique identifier for the object.
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

    def is_class_inconsistent(self, object_id, track_info) -> bool:
        """Verifies that an object maintains consistent classification across frames.

        Returns:
            bool: True if the class is consistent, False if there are inconsistencies.
        """

        all_classes = set(v.class_name for v in track_info.values())

        if len(all_classes) > 1:
            print(
                f"{object_id=} occurs as the following classes: {all_classes}"
            )

        return len(all_classes) > 1

    def is_object_missing_in_frames(self, object_id, track_info) -> bool:
        """Checks if an object goes missing in intermediate frames.

        Returns:
            bool: True if the object is missing in some frames, False otherwise.
        """

        keys = track_info.keys()
        mn_idx, mx_idx, size = min(keys), max(keys), len(track_info)
        expected_size = mx_idx - mn_idx + 1
        if size != expected_size:
            print(f"{object_id=} is missing in {expected_size - size} frames")

        return expected_size != size

    def appears_only_in_single_frame(self, object_id, track_info) -> bool:
        """Checks if an object only appears in a single frame, potentially indicating a false detection.

        Returns:
            bool: True if the object appears in only one frame, False otherwise.
        """

        if len(track_info) == 1:
            print(
                f"{object_id=} occurs only in one frame, potentially indicating a false detection"
            )

        return len(track_info) == 1

    def has_low_iou(self, object_id, track_info) -> bool:
        """Assesses if the Intersection over Union (IoU) is below a threshold, indicating potential tracking issues.

        Returns:
            bool: True if the IoU is low, False otherwise.
        """

        keys = sorted(list(track_info.keys()))
        flag = False
        for key1, key2 in zip(keys, keys[1:]):
            t1 = track_info[key1]
            t2 = track_info[key2]

            iou = compute_iou(t1.ltrb, t2.ltrb)
            if iou < MIN_IOU_THRESH:
                print(
                    f"{iou=} is lower than threshold of {MIN_IOU_THRESH} "
                    f"between {key1} and {key2}"
                )
                flag = True
        return flag

    # ----------------------------------------------------------------------------------------
    # 1. check if class_name is consistent
    # 2. check if there is a missing frame (object doesn't exist in one frame)
    # 3. check if there is an object occurring only in one frame
    # 4. check if there is low_iou in one frame
