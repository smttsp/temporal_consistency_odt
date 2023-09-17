import copy
import datetime

from ultralytics import YOLO
import cv2
from helper import create_video_writer
from deep_sort_realtime.deepsort_tracker import DeepSort

CONFIDENCE_THRESHOLD = 0.7
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)


def object_detection(model, frame):
    detections = model(frame)[0]

    results = []

    for data in detections.boxes.data.tolist():
        confidence = data[4]

        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        x_min, y_min, x_max, y_max = (
            int(data[0]),
            int(data[1]),
            int(data[2]),
            int(data[3]),
        )
        class_id = int(data[5])
        # add the bounding box (x, y, w, h), confidence and class id to the results list
        results.append(
            [[x_min, y_min, x_max - x_min, y_max - y_min], confidence, class_id]
        )
    return results


def draw_bbox_around_object(frame, track, voc_bbox):
    track_id = track.track_id

    x_min, y_min, x_max, y_max = (
        int(voc_bbox[0]),
        int(voc_bbox[1]),
        int(voc_bbox[2]),
        int(voc_bbox[3]),
    )
    # draw the bounding box and the track id
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), GREEN, 2)
    cv2.rectangle(frame, (x_min, y_min - 20), (x_min + 20, y_min), GREEN, -1)
    cv2.putText(
        frame,
        str(track_id),
        (x_min + 5, y_min - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        WHITE,
        2,
    )
    return None


def object_tracking(frame, results, tracker):
    print(tracker)
    # update the tracker with the new detections
    tracks = tracker.update_tracks(results, frame=frame)
    # loop over the tracks
    for track in tracks:
        # if the track is not confirmed, ignore it
        if not track.is_confirmed():
            continue

        # get the track id and the bounding box
        voc_bbox = track.to_ltrb()
        draw_bbox_around_object(frame, track, voc_bbox)
    return frame


def object_detection_and_tracking(model, video_filepath):
    # initialize the video capture object
    video_cap = cv2.VideoCapture(video_filepath)
    output_filepath = video_filepath.replace(".mp4", "_output.mp4")

    # initialize the video writer object
    writer = create_video_writer(video_cap, output_filepath)

    # load the pre-trained YOLOv8n model
    tracker = DeepSort(max_age=50)
    tracker_list: list[DeepSort] = []
    while True:
        start = datetime.datetime.now()

        ret, frame = video_cap.read()

        if not ret:
            break

        results = object_detection(model, frame)
        frame = object_tracking(frame, results, tracker)
        tracker_list.append(copy.deepcopy(tracker))
        end = datetime.datetime.now()

        total = (end - start).total_seconds() * 1000
        print(f"Time to process 1 frame: {total:.0f} milliseconds")

        # calculate the frame per second and draw it on the frame
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(
            frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8
        )

        cv2.imshow("Frame", frame)
        writer.write(frame)
        if cv2.waitKey(1) == ord("q"):
            break

    video_cap.release()
    writer.release()
    cv2.destroyAllWindows()
    return tracker_list


if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    tracker_list = object_detection_and_tracking(
        model, video_filepath="video2.mp4"
    )
    print()
