import cv2


GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
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
        img=frame,
        text=new_text,
        org=(x_min + 5, y_min - 8),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=WHITE,
        thickness=2,
    )
    return None


def draw_fps_on_frame(frame, total_time):
    fps = f"FPS: {1000 / total_time:.2f}"
    cv2.putText(
        img=frame,
        text=fps,
        org=(50, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2,
        color=BLUE,
        thickness=8,
    )

    return None
