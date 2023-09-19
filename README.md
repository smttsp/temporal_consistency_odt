# Temporal Video Consistency
**Data collection through temporal consistency of a video:** 


This repository harnesses object detection and tracking techniques to identify inconsistencies 
and anomalies in video sequences. Specifically, the system identifies:

**Discrepancies in Object Count**:

   - Frames missing objects that are present in adjacent frames.
   - Frames with additional objects not found in adjacent frames.

**Class Mismatch**: 

Instances where an object's predicted class differs from its previous frame classification.

**Low IOU**: 

Situations where the intersection-over-union (IOU) of an object between two consecutive frames drops below a threshold, indicating potential tracking issues.

---
The primary goal of this repo is enhancing model training. 
As the system captures these anomalies, those failures can then be used to refine the object detection model.

**Augmentation Evaluation**:
The repository also supports the application of random frame augmentations. 
By observing how these augmentations affect detection outcomes, users can 
assess the resilience and robustness of the detection model, specifically `yolo_v8`. 
This also aids in accumulating diverse samples that can further improve the model's performance.


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


## TODO

I will integrate the augmentations in the next iteration. Currently, the 
implementation requires some reviews. I haven't really investigated the code
nor added any unit-tests. So, I suspect the code might be buggy. 

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

