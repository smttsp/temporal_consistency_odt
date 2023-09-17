import cv2

GREEN = (0, 255, 0)
WHITE = (255, 255, 255)


def draw_bbox_around_object(frame, track, voc_bbox, classes):
    track_id = track.track_id
    cls = track.det_class
    x_min, y_min, x_max, y_max = (
        int(voc_bbox[0]),
        int(voc_bbox[1]),
        int(voc_bbox[2]),
        int(voc_bbox[3]),
    )
    # draw the bounding box and the track id
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), GREEN, 2)
    cv2.rectangle(frame, (x_min, y_min - 20), (x_min + 60, y_min), GREEN, -1)

    new_text = f"{track_id}: {classes.get(cls)}"
    cv2.putText(
        frame,
        new_text,
        (x_min + 5, y_min - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        WHITE,
        2,
    )
    return None
