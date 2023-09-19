import cv2


GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)


def put_test_on_upper_corner(frame, text):
    cv2.putText(
        img=frame,
        text=text,
        org=(20, 40),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.9,
        color=WHITE,
        thickness=2,
    )


def draw_class_name(frame, ltrb_bbox, track_id, class_name):
    x_min, y_min, x_max, y_max = map(int, ltrb_bbox)
    # draw the bounding box and the track id
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), GREEN, 2)
    cv2.rectangle(frame, (x_min, y_min - 20), (x_min + 60, y_min), GREEN, -1)
    new_text = f"{track_id}: {class_name}"
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


def draw_bbox_around_object(frame, track, ltrb_bbox, classes):
    track_id = track.track_id
    cls = track.det_class
    class_name = classes.get(cls)
    draw_class_name(frame, ltrb_bbox, track_id, class_name)
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
