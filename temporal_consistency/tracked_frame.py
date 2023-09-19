import copy
import os
from collections import defaultdict
from utils import create_video_writer
import numpy


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
    def __init__(self, video_cap, output_folder="/users/samet/desktop/output/"):
        self.out_folder = output_folder
        os.makedirs(output_folder)

        self.tracked_frames = []
        self.all_objects = defaultdict(dict)
        self.video_cap_dict = {}  # object_id -> video_cap
        self.video_cap = video_cap

    def add_tracked_frame(self, tracked_frame: TrackedFrame):
        self.tracked_frames.append(tracked_frame)
        self.update_frame_objects_dict(tracked_frame)

    def update_frame_objects_dict(self, tracked_frame):
        for track in tracked_frame.tracker.tracks:
            cur_dict = {tracked_frame.frame_id: list(map(int, track.to_ltrb()))}
            self.all_objects[track.track_id].update(cur_dict)

    def export_all_objects(self):
        for object_id in self.all_objects:
            filename = os.path.join(self.out_folder, f"obj_{object_id}.mp4")
            writer = create_video_writer(self.video_cap, filename)
            self.export_object(writer, object_id)

    def export_object(self, writer, object_id):
        a_dict = self.all_objects[str(object_id)]

        start_idx, end_idx = min(a_dict.keys()), max(a_dict.keys())

        for idx in range(start_idx, end_idx + 1):
            if idx not in a_dict:
                continue
            frame = self.tracked_frames[idx].frame

            black_frame = numpy.zeros_like(frame)
            x1, y1, x2, y2 = a_dict[idx]
            black_frame[y1 : y2 + 1, x1 : x2 + 1] = frame[
                y1 : y2 + 1, x1 : x2 + 1
            ]
            writer.write(black_frame)

        writer.release()
