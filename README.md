# Temporal Video Consistency
**Data collection through temporal consistency of a video:** 
This repo uses object detection and tracking methods and finds failures in a video, i.e., when

- additional or missing object in a frame
  - some objects are missed in a frame that existed in the previous and next frames
  - some extra objects found in a frame that weren't predicted in the previous and next frames 
- when an object is predicted as a different class than the previous one
- low_iou of an object in a frame and its previous frame

This repo uses the above heuristics and finds failures in video frames which 
will be used in model training. 

Moreover, this repo allows users to choose random augmentations to be applied on
all frames, then we will see if the augmentations cause any extra failures. 
This way, we can further evaluate the robustness of the object detection model (currently `yolo_v8`). 
And collect more samples which may help improve the model further

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


## Installation

### Prerequisite: `pyenv`

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

