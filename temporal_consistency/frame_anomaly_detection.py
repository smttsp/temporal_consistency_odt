import copy


# Extract track_id and det_class from each tracker
def extract_track_data(tracker):
    return {track.track_id: track.det_class for track in tracker.tracks}


class FrameInfo:
    def __init__(self, frame, tracker):
        self.tracker = copy.deepcopy(tracker)
        self.frame = frame
        self.num_object = len(tracker.tracks)
        self.object_ids = self.get_object_ids()

    def get_object_ids(self):
        return set(t.track_id for t in self.tracker.tracks)


class FrameInfoList:
    def __init__(self):
        self.frame_info_list = []

    def add_frame_info(self, frame_info):
        self.frame_info_list.append(frame_info)

        if len(self.frame_info_list) > 3:
            self.frame_info_list.pop(0)

        is_abnormal = (
            self.evaluate_middle_frame()
            if len(self.frame_info_list) == 3
            else False
        )
        if is_abnormal:
            print()
        return is_abnormal

    def evaluate_middle_frame(self):
        return (
            self.check_if_missing_or_extra_object()
            or self.check_incorrect_object_classification()
        )

