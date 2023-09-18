import argparse

from ultralytics import YOLO

from temporal_consistency.object_detection_tracking import (
    object_detection_and_tracking,
)


CONFIDENCE_THRESHOLD = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo for parsing different types of data"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Filtering predictions with low confidence",
    )

    # Parsing a string
    parser.add_argument(
        "--details",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Indicates how much details you want in analysis, "
        "1: minimal details"
        "2: more details"
        "3: most details",
    )

    parser.add_argument(
        "--num_aug",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Each frame will go through up to 3 augmentations. 0-> no augmentation",
    )

    # Parsing an integer
    parser.add_argument(
        "--int_val", type=int, default=0, help="An integer value"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    confidence_threshold = args.confidence
    num_aug = args.num_aug

    model = YOLO("yolov8n.pt")

    object_detection_and_tracking(
        model,
        video_filepath="data/video1.mp4",
        num_aug=num_aug,
        confidence_threshold=confidence_threshold,
    )
    print()
