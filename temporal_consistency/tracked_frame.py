import copy
import os
from collections import defaultdict

import numpy

from temporal_consistency.utils import create_video_writer
from temporal_consistency.vis_utils import put_text_on_upper_corner


class Prediction:
    """Single prediction for an object. Contains the bounding box coordinates,
    confidence, class ID, and class name.
    """

    def __init__(
        self,
        frame_id: int,
        ltrb: list[int],
        confidence: float,
        class_id: int,
        class_names: dict,
    ):
        self.frame_id = frame_id
        self.ltrb = ltrb
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_names.get(class_id, None)


class TrackedFrame:
    """A single frame together with its tracked objects."""

    def __init__(
        self, frame_id, frame, tracker, low_confidence_results, class_names
    ):
        self.frame_id = frame_id
        self.tracker = copy.deepcopy(tracker)
        self.frame = frame
        self.num_object = len(tracker.tracks)
        self.class_names = class_names
        self.object_ids = self.get_object_ids()
        # self.low_confidence_results = low_confidence_results
        self.low_confidence_objects = self.get_low_confidence_objects(
            low_confidence_results
        )

    def get_low_confidence_objects(self, low_confidence_results):
        """Returns a list of low confidence objects."""

        low_confidence_objects = []

        for res in low_confidence_results:
            cur_pred = Prediction(
                frame_id=self.frame_id,
                ltrb=res[0],
                confidence=res[1],
                class_id=res[2],
                class_names=self.class_names,
            )
            low_confidence_objects.append(cur_pred)

        return low_confidence_objects

    def get_object_ids(self):
        return set(t.track_id for t in self.tracker.tracks)


class TrackedFrameCollection:
    """A collection of TrackedFrames containing frames and the tracked objects."""

    def __init__(self, video_cap, class_names, out_folder):
        self.video_cap = video_cap
        self.out_folder = out_folder
        self.class_names = class_names
        self.tracked_frames = []
        self.all_objects = defaultdict(dict)

    def add_tracked_frame(self, tracked_frame: TrackedFrame):
        """Adds a tracked frame to the collection."""

        self.tracked_frames.append(tracked_frame)
        self.update_all_objects_dict(tracked_frame)

    def update_all_objects_dict(self, tracked_frame):
        """Updates the dictionary of objects. Each key is an object ID
        and the value is a dictionary of frame IDs and predictions.
        """

        for track in tracked_frame.tracker.tracks:
            cur_pred = Prediction(
                frame_id=tracked_frame.frame_id,
                ltrb=list(map(int, track.to_ltrb())),
                confidence=track.det_conf,
                class_id=track.det_class,
                class_names=self.class_names,
            )
            cur_dict = {tracked_frame.frame_id: cur_pred}
            self.all_objects[track.track_id].update(cur_dict)

    def export_all_objects(self, out_video_fps):
        """Exports all objects to individual videos."""

        os.makedirs(self.out_folder, exist_ok=True)

        for object_id in self.all_objects:
            filename = os.path.join(self.out_folder, f"obj_{object_id}.mp4")
            writer = create_video_writer(
                self.video_cap, filename, fps=out_video_fps
            )
            self.export_object(writer, object_id)

    def export_object(self, writer, object_id):
        """Exports a single object to a video file."""

        a_dict = self.all_objects[object_id]

        start_idx, end_idx = min(a_dict.keys()), max(a_dict.keys())

        for frame_id in range(start_idx, end_idx + 1):
            if frame_id not in a_dict:
                continue
            frame = self.tracked_frames[frame_id].frame

            black_frame = numpy.zeros_like(frame)
            cur_prediction = a_dict[frame_id]
            x1, y1, x2, y2 = cur_prediction.ltrb
            black_frame[y1 : y2 + 1, x1 : x2 + 1] = frame[
                y1 : y2 + 1, x1 : x2 + 1
            ]
            class_name = cur_prediction.class_name

            text = f"{frame_id=}, {object_id=}, {class_name=}"
            put_text_on_upper_corner(black_frame, text)
            writer.write(black_frame)

        writer.release()
