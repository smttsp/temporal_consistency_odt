# Temporal Anomaly Detection in Videos
**Data collection through temporal consistency of a video:** 


This repository harnesses object detection and tracking techniques to identify inconsistencies 
and anomalies in video sequences.
By identifying these discrepancies, we can capture valuable data to enhance our object
detection model. 
These identified anomalies can subsequently be used in the training, validation, and 
testing phases for future iterations of the model.

The system identifies the following anomalies:

**Discrepancies in Object Count**:
- Frames missing objects that are present in adjacent frames.
- Frames with additional objects not found in adjacent frames.

**Class Mismatch**: 
- Instances where an object's predicted class differs from its previous frame classification.

**Low IOU**:
- Situations where the intersection-over-union (IOU) of an object between two consecutive frames drops below a threshold, indicating potential tracking issues.


### Examples

Below are some examples:

#### - Additional/missing objects

#### - Different classes

|                                                    class_i                                                     |                                                    class_j                                                     |
|:--------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|
| ![im1](https://github.com/smttsp/temporal_consistency_odt/assets/4594945/07a547cd-b8ad-4cfc-8c63-da80db762320) | ![im2](https://github.com/smttsp/temporal_consistency_odt/assets/4594945/9e62f950-2702-460f-852d-f5e82893e99c) |

#### - Low iou

Below is an example of low_iou (in fact, object_3 is switched to another car). Hence the iou=0.0

|                                                    frame_i                                                     |                                                  frame_(i+1)                                                   |
|:--------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|
| ![im1](https://github.com/smttsp/temporal_consistency_odt/assets/4594945/8fd14598-5b28-429b-8b25-d030ad619284) | ![im2](https://github.com/smttsp/temporal_consistency_odt/assets/4594945/120fc6c3-738a-4ac6-957f-51d73495b39a) |


## Other outputs


Here is an example where an object is cropped from the video. For the sake of visibility, it is captured with 5 FPS. 

Notice that in frame 38, the class name changes from `car` to `truck`, and in the next frame it switches back to `car`.

https://github.com/smttsp/temporal_consistency_odt/assets/4594945/88e8f4d6-2b87-4e1a-8869-664b5478fa1f


---

**Augmentation Evaluation**:
The repository also supports the application of random frame augmentations. 
By observing how these augmentations affect detection outcomes, users can 
assess the resilience and robustness of the detection model (`yolo_v8` in our case).
This may also aid in accumulating diverse samples that can further improve the model's performance.

Below is an example of random augmentations **TRIGGER ALERT: might trigger photosensitive people!!!**:

https://github.com/smttsp/temporal_consistency_odt/assets/4594945/36507731-c401-45ce-8d4a-56939619ebc9


## Installation

### Prerequisite: `pyenv`

`pyenv` simplifies Python version management, enabling you to seamlessly switch between 
Python versions for different project requirements.



https://github.com/pyenv/pyenv-installer

On macOS you can use [brew](https://brew.sh), but you may need to grab the `--HEAD` version for the latest:

```bash
brew install pyenv --HEAD
```

or

```bash
curl https://pyenv.run | bash
```

And then you should check the local `.python-version` file or `.envrc` and install the correct version which will be the basis for the local virtual environment. If the `.python-version` exists you can run:

```bash
pyenv install
```

This will show a message like this if you already have the right version, and you can just respond with `N` (No) to cancel the re-install:

```bash
pyenv: ~/.pyenv/versions/3.8.6 already exists
continue with installation? (y/N) N
```

### Prerequisite: `direnv`

`direnv` streamlines environment variable management, allowing you to isolate 
project-specific configurations and dependencies within your development environment.

https://direnv.net/docs/installation.html

```bash
curl -sfL https://direnv.net/install.sh | bash
```

### Developer Setup

If you are a new developer to this package and need to develop, test, or build -- please run the following to create a developer-ready local Virtual Environment:

```bash
direnv allow
python --version
pip install --upgrade pip
pip install poetry
poetry install
```

## References

One of the core pieces of this repo object detection and tracking is obtained from 
the following article: https://thepythoncode.com/article/real-time-object-tracking-with-yolov8-opencv

