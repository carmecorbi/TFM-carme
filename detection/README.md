
# 🎯 Object Detection in SoccerNet Tracking

## 📝 Objective
The goal of this part of the project is to detect key elements in the SoccerNet Tracking dataset, namely **players**, **goalkeepers**, **referees**, and the **ball**, across all video sequences. The detection approach is based on **YOLO object detectors**.

Detection is divided into two categories:
- A **joint detector** trained on both **ball** and **person** classes.
- Two **class-specific detectors**, trained separately for **ball** and **person**.

## ⚙️ Preprocessing
Original detection annotations are converted to the **YOLO format**, which uses normalized bounding boxes:All values are **normalized** with respect to image width and height, and there is **no header** in YOLO label files. Each line corresponds to one object in one image.

The conversion involves transforming bounding boxes from the standard `(x, y, width, height)` format — where `(x, y)` is the **top-left corner** — to YOLO format, where coordinates represent the **center of the box**: `(x_center = (x + width / 2) / image_width, y_center = (y + height / 2) / image_height, width = width / image_width, height = height / image_height)`

The original annotations from the SoccerNet Tracking dataset (`det.txt` and `gt.txt`) are parsed and converted using the scripts available in the [pre-processing](https://github.com/carmecorbi/TFM-carme/tree/main/detection/pre-processing) folder:

- `labelling.py`: generates YOLO annotations for the **joint detector** (ball + person)
- `labelling_ball.py`: generates YOLO annotations for the **ball-only detector**
- `labelling_person.py`: generates YOLO annotations for the **person-only detector**

Each script creates `.txt` files in YOLO format, one per image frame.

## 🚀 Training with Ultralytics YOLO

The [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework is used to train the models.Before training, make sure to install the necessary dependencies. The recommended way is to follow the official Ultralytics installation instructions.
Here is a generic example of a training script using the Python API:

```python
from ultralytics import YOLO
# Load a pretrained YOLO model checkpoint
model = YOLO('path/to/pretrained_model.pt')

# Train the model
results = model.train(
    data="path/to/dataset.yaml",    # dataset configuration file
    epochs=50,                      # number of training epochs
    batch=16,                      # batch size
    imgsz=640,                     # input image size (pixels)
    device=0,                      # GPU device index or 'cpu'
    patience=15,                   # early stopping patience
    project='training_output',     # output directory
    freeze=0,                     # number of layers to freeze initially
    classes=[0],                   # list of class IDs to train on
    plots=True                     # enable training plots
)

```
## 🧪 Evaluation with Ultralytics YOLO

The Ultralytics YOLO framework also provides an easy way to evaluate trained models.

Below is a generic example of an evaluation script using the Python API:

```python

from ultralytics import YOLO

# Load the trained YOLO model weights
model = YOLO('path/to/best_model.pt')

# Dataset config file
DATASET_PATH = "path/to/dataset.yaml"

# Run evaluation (validation) on the specified data split
results = model.val(
    data=DATASET_PATH,    # dataset YAML file
    split='train',        # dataset split to evaluate ('train', 'val', or 'test')
    conf=0.01,            # confidence threshold for detections
    device=0,             # device to run evaluation on (GPU id or 'cpu')
    classes=[0, 1]        # list of class IDs to evaluate on
)

# Print key evaluation metrics
print("Average precision:", results.box.ap)
print("Average precision at IoU=0.50:", results.box.ap50)
print("F1 score:", results.box.f1)
print("Mean average precision at IoU=0.50:", results.box.map50)
print("Mean average precision at IoU=0.75:", results.box.map75)
print("Precision:", results.box.p)
print("Recall:", results.box.r)
```
## 📊 Results

Below are the evaluation metrics obtained for the joint detection model (ball + person classes) under different training setups:

| Model            | Class  | AP   | AP@0.5 | F1 Score | Precision | Recall | mAP@0.5 |
|------------------|--------|------|--------|----------|-----------|--------|---------|
| **Pre-Trained**  | Person | 0.49 | 0.89   | 0.88     | 0.87      | 0.88   | 0.53    |
|                  | Ball   | 0.06 | 0.17   | 0.27     | 0.47      | 0.19   | 0.53       |
| **Fully Unfrozen** | Person | 0.67 | 0.95   | 0.95     | 0.93      | 0.97   | 0.71    |
|                  | Ball   | 0.06 | 0.17   | 0.27     | 0.47      | 0.19   | 0.71      |
| **Backbone Frozen** | Person | 0.69 | 0.98   | 0.94     | 0.93      | 0.96   | 0.70    |
|                  | Ball   | 0.16 | 0.43   | 0.51     | 0.68      | 0.41   | 0.70      |
