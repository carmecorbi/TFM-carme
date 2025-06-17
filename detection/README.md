
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

The [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework is used to train the models.
