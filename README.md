# Temporal Video Consistency
**Data collection through temporal consistency of a video:** 
This repo uses object detection and tracking methods and finds failures in a video, i.e., when

- additional or missing frame
  - some objects are missed in a frame that existed in the previous and next frames
  - some extra objects found in a frame that weren't predicted in the previous and next frames 
- when an object is predicted as a different class than the previous one
- low_iou of an object in a frame and its previous frame

This repo uses the above heuristics and finds failures on video frames which 
will be used in model training. 

Moreover, this repo allows users to choose random augmentations to be applied on
all frames, then we will see if the augmentations cause any extra failures. 
This way, we can further evaluate the robustness of the object detection model (currently `yolo_v8`). 
And collect more samples which may help improve the model further


## References

One of the core pieces of this repo object detection and tracking is obtained from 
the following article: https://thepythoncode.com/article/real-time-object-tracking-with-yolov8-opencv

