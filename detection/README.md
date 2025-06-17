
# 🎯 Object Detection in SoccerNet Tracking

## 📝 Objective
The goal of this part of the project is to detect key elements in the SoccerNet Tracking dataset, namely **players**, **goalkeepers**, **referees**, and the **ball**, across all video sequences. The detection approach is based on **YOLO object detectors**.

Detection is divided into two categories:
- A **joint detector** trained on both **ball** and **person** classes.
- Two **class-specific detectors**, trained separately for **ball** and **person**.

## ⚙️ Preprocessing
Original detection annotations are converted to the **YOLO format**, which uses normalized bounding boxes:
