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

    def check_incorrect_object_classification(self):
        """The classes of each object should be consistent. If an object is classified
        as a member of class_i, then class_j, this inconsistency shows there is a
        discrepancy in classifications which is a good indicator that the frame
        should be logged
        """
        prev, mid, next = self.frame_info_list

        # Extract data from the trackers
        prev_data = extract_track_data(prev.tracker)
        mid_data = extract_track_data(mid.tracker)
        next_data = extract_track_data(next.tracker)

        # Find common track_ids between the trackers
        common_track_ids = (
            set(prev_data.keys()) & set(mid_data.keys()) & set(next_data.keys())
        )

        # Identify mismatches in mid-tracker
        mismatched_tracks = {}
        for track_id in common_track_ids:
            if (
                mid_data[track_id] != prev_data[track_id]
                or mid_data[track_id] != next_data[track_id]
            ):
                mismatched_tracks[track_id] = mid_data[track_id]

        # print("Mismatched tracks in mid:", mismatched_tracks)
        return len(mismatched_tracks) > 0

    def is_low_iou(self):
        pass
