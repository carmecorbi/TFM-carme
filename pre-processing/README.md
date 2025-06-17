
# Preprocessing SoccerNet Tracking Dataset
## 📚 Dataset Overview

The **SoccerNet Tracking** dataset is designed for multi-object tracking tasks in professional soccer matches. It consists of short, annotated video clips captured from the **main broadcast camera** during the **2019 Swiss Super League** season. All videos are recorded in **1080p Full HD resolution** at **25 frames per second**.

Each video clip spans **30 seconds**, which corresponds to **750 frames** per sequence.


## 📁 Dataset Structure
- **Training Set**
  - 57 sequences
  - Collected from **3 different matches**

- **Test Set**
  - 49 sequences
  - Also collected from **3 different matches**

Each sequence directory contains the following:
- `img1/`: folder with all 750 video frames (30s at 25 FPS)
- `det/det.txt`: file with gt detections in the format: frame_id, -1, x, y, w, h, confidence_score, -1, -1, -1
- `gt/gt.txt`: ground truth tracking annotations in the format: frame_id, track_id, x, y, w, h, confidence_score, -1, -1, -1
-  `gameinfo.ini`: semantic metadata about each tracklet, mapping `trackletID` to object type and team.
