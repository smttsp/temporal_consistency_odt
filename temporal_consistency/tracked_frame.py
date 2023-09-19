import copy
import os
from collections import defaultdict

import numpy
from utils import create_video_writer
from vis_utils import put_test_on_upper_corner


OUT_FOLDER = "/users/samet/desktop/output/"


class TrackedFrame:
    def __init__(self, frame_id, frame, tracker):
        self.frame_id = frame_id
        self.tracker = copy.deepcopy(tracker)
        self.frame = frame
        self.num_object = len(tracker.tracks)
        self.object_ids = self.get_object_ids()

    def get_object_ids(self):
        return set(t.track_id for t in self.tracker.tracks)


class TrackedFrameCollection:
    def __init__(self, video_cap, classes, out_folder=OUT_FOLDER):
        self.video_cap = video_cap
        self.out_folder = out_folder
        self.classes = classes
        self.tracked_frames = []
        self.all_objects = defaultdict(dict)

    def add_tracked_frame(self, tracked_frame: TrackedFrame):
        self.tracked_frames.append(tracked_frame)
        self.update_frame_objects_dict(tracked_frame)

    def update_frame_objects_dict(self, tracked_frame):
        for track in tracked_frame.tracker.tracks:
            cur_dict = {tracked_frame.frame_id: list(map(int, track.to_ltrb()))}
            self.all_objects[track.track_id].update(cur_dict)

    def export_all_objects(self):
        os.makedirs(self.out_folder, exist_ok=True)

        for object_id in self.all_objects:
            filename = os.path.join(self.out_folder, f"obj_{object_id}.mp4")
            writer = create_video_writer(self.video_cap, filename)
            self.export_object(writer, object_id)

    def export_object(self, writer, object_id):
        a_dict = self.all_objects[object_id]

        start_idx, end_idx = min(a_dict.keys()), max(a_dict.keys())

        for frame_id in range(start_idx, end_idx + 1):
            if frame_id not in a_dict:
                continue
            frame = self.tracked_frames[frame_id].frame

            black_frame = numpy.zeros_like(frame)
            x1, y1, x2, y2 = a_dict[frame_id]
            black_frame[y1 : y2 + 1, x1 : x2 + 1] = frame[
                y1 : y2 + 1, x1 : x2 + 1
            ]
            class_name = self.__get_class_name_from_track(frame_id, object_id)

            text = f"{frame_id=}, {object_id=}, {class_name=}"
            put_test_on_upper_corner(black_frame, text)
            writer.write(black_frame)

        writer.release()

    def __get_class_name_from_track(self, frame_id, object_id):
        tracks = self.tracked_frames[frame_id].tracker.tracks
        class_ids = [t.det_class for t in tracks if t.track_id == object_id]
        class_id = class_ids[0]
        class_name = self.classes[class_id]
        return class_name