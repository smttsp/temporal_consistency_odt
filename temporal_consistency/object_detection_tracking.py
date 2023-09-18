import datetime

import cv2
from augmentations import get_random_augmentation
from deep_sort_realtime.deepsort_tracker import DeepSort
from frame_anomaly_detection import FrameInfo, FrameInfoList
from utils import create_video_writer
from vis_utils import draw_bbox_around_object


def object_detection(model, frame, num_aug=0, confidence_threshold=0.0):
    frame_aug = get_random_augmentation(frame, num_aug=num_aug)
    detections = model(frame_aug)[0]

    results = []

    for data in detections.boxes.data.tolist():
        confidence = data[4]

        if float(confidence) < confidence_threshold:
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
    return results, frame_aug


def object_tracking(frame, results, deep_sort_tracker, classes):
    # print(tracker)
    # update the tracker with the new detections
    tracks = deep_sort_tracker.update_tracks(results, frame=frame)

    frame_after = frame.copy()
    for track in tracks:
        # if the track is not confirmed, ignore it
        if not track.is_confirmed():
            continue

        # get the track id and the bounding box
        voc_bbox = track.to_ltrb()
        draw_bbox_around_object(frame_after, track, voc_bbox, classes)
    return frame_after


def object_detection_and_tracking(model, video_filepath, confidence_threshold):
    # initialize the video capture object
    video_cap = cv2.VideoCapture(video_filepath)
    output_filepath = video_filepath.replace(".mp4", "_output.mp4")

    # initialize the video writer object
    writer = create_video_writer(video_cap, output_filepath)

    # load the pre-trained YOLOv8n model
    deep_sort_tracker = DeepSort(max_age=50)
    frame_info_list = FrameInfoList()

    while True:
        start = datetime.datetime.now()

        ret, frame = video_cap.read()

        if not ret:
            break

        results, frame_aug = object_detection(
            model, frame, confidence_threshold
        )
        frame_after = object_tracking(
            frame_aug, results, deep_sort_tracker, classes=model.names
        )
        frame_info = FrameInfo(frame_aug, deep_sort_tracker.tracker)
        frame_info_list.add_frame_info(frame_info)

        end = datetime.datetime.now()

        total = (end - start).total_seconds() * 1000
        print(f"Time to process 1 frame: {total:.0f} milliseconds")

        # calculate the frame per second and draw it on the frame
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(
            frame_after,
            fps,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            8,
        )

        cv2.imshow("Frame", frame_after)
        writer.write(frame_after)
        if cv2.waitKey(1) == ord("q"):
            break

    video_cap.release()
    writer.release()
    cv2.destroyAllWindows()
    return None
