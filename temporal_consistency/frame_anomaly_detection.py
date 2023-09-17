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

    def check_if_missing_or_extra_object(self):
        """This checks if there is any object that is missing in the middle
        but exists in the other two. Or middle one contains on object
        that is not in the other ones.
        """
        pass
        prev_ids, mid_ids, next_ids = [
            lst.object_ids for lst in self.frame_info_list
        ]

        # Find elements in mid_ids that aren't in prev_ids or next_ids
        extra_elements_in_mid = mid_ids - (prev_ids.union(next_ids))

        # Find elements that are in both prev_ids and next_ids but not in mid_ids
        missing_elements_in_mid = (prev_ids.intersection(next_ids)) - mid_ids

        return extra_elements_in_mid or missing_elements_in_mid

