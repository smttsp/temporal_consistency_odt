from ultralytics import YOLO
from temporal_consistency.object_detection_tracking import (
    object_detection_and_tracking,
)

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")

    object_detection_and_tracking(model, video_filepath="data/video3.mp4")
    print()
