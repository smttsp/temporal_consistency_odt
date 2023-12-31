"""This script is an entrypoint for performing object detection and tracking
on a given video using YOLO and Deep Sort. It also includes options for
augmenting frames and adjusts tracking based on configurable settings like
confidence threshold and max age for tracks.

Detected temporal anomalies in tracking are logged for further analysis.
They will later be used for model training and fine-tuning.
"""

import os

import configargparse
from deep_sort_realtime.deepsort_tracker import DeepSort
from loguru import logger
from ultralytics import YOLO

from temporal_consistency.frame_anomaly_detection import TemporalAnomalyDetector
from temporal_consistency.object_detection_tracking import (
    run_detection_and_tracking_pipeline,
)
from temporal_consistency.utils import get_runtime_str


CONFIDENCE_THRESHOLD = 0.4
MAX_AGE = 25


def parse_args():
    parser = configargparse.ArgumentParser(
        description="Demo for parsing different types of data"
    )
    parser.add_argument(
        "--config",
        is_config_file=True,
        default="conf.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--video_filepath",
        help="Path to the input video file",
    )
    parser.add_argument(
        "--out_folder",
        help="Path to the output folder",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Filtering predictions with low confidence",
    )
    parser.add_argument(
        "--max_age",
        type=float,
        default=MAX_AGE,
        help="Filtering predictions with low confidence",
    )
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
    parser.add_argument(
        "--out_video_fps",
        type=int,
        default=25,
        help="FPS of the output videos",
    )

    args = parser.parse_args()
    return args


def main(args):
    runtime_str = get_runtime_str()
    args.out_folder = os.path.join(args.out_folder, runtime_str)
    os.makedirs(args.out_folder, exist_ok=True)

    logfile = os.path.join(args.out_folder, "output.log")
    logger.add(logfile)

    model = YOLO("yolov8n.pt")
    deep_sort_tracker = DeepSort(max_age=args.max_age)

    tframe_collection = run_detection_and_tracking_pipeline(
        model, deep_sort_tracker, args
    )
    TemporalAnomalyDetector(tframe_collection)


if __name__ == "__main__":
    args = parse_args()

    main(args)
