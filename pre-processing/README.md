
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


## 🛠️ Preprocessing Step 1: Creating the Validation Set

The first preprocessing step is to create a **validation set**, which is **not** part of the original SoccerNet Tracking dataset. This set is built from an extended dataset called **GameState Reconstruction**, which is an enriched version of SoccerNet Tracking with more sequences and annotations in a slightly different format.

- The GameState Reconstruction validation set includes **58 sequences**, each 30 seconds long (750 frames). These sequences are split into:
  - **38 sequences** for the extended **training set** (from 2 matches)
  - **20 sequences** for the new **validation set** (from 1 match)

This split ensures that training and validation sequences come from **different matches**, promoting better generalization.

### 🔧 Scripts Used in This Step

The following scripts are responsible for generating the required files for these sequences:

- `detections.py` → generates `det/det.txt`
- `tracking.py` → generates `gt/gt.txt`
- `tracklets_info.py` → generates `gameinfo.ini`

These scripts reformat the GameState Reconstruction annotations into the same structure used by the original SoccerNet Tracking dataset, enabling seamless integration with existing tracking pipelines.
