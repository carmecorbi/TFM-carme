# Ball Detection with Player Heatmaps

## Motivation
The aim of this part of the project is to improve ball detection performance by incorporating spatial and temporal context from player positions. Instead of relying solely on RGB data, the system uses heatmaps derived from player annotations. These heatmaps are processed with a 3D convolutional network (CNN) and fused with the original video frames before being passed to a YOLO-based object detector.

## Pipeline Overview
### 1. Player-wise Gaussian Heatmap Generation (per team)
   
   Player annotations are converted into heatmaps to provide spatial context for the ball detector.  
   For each frame:
   - Ground truth bounding boxes are converted into 2D Gaussian heatmaps.
   - A separate heatmap is generated for each team (left and right).
   - Variance of the Gaussian is proportional to the player’s bounding box size.
   - Heatmaps are saved as grayscale PNG images.

This step is implemented in the script: `create_heatmaps.py`. Example usage:
```bash
python create_heatmaps.py --seqname SNMOT-060
```
To provide temporal information to the detector, heatmaps from the previous 5 frames are grouped per current frame. For each frame f, the correponding directory includes heatmaps from frame f-5 to f-1 for both teams. This step is implemented in the sript: `organize_heatmaps.py`. Example usage:

```bash
python organize_heatmaps.py
```
### 2. Temporal Encoding with 3D CNNs
Each 5-frame sequence (left and right heatmaps) is passed through a 3D custom convolutional neural network to extract temporal-spatial patterns. These patterns capture team formations, player motion, and interactions that are useful for ball detection.

Key details:
- Input: a stack of 5 grayscale heatmaps (5 x H x W) per team.
- Architecture: two identical 3D CNNs (one for the left team, one for the right), sharing weights.
- Output: one 2D feature map per team that encodes contextual information from the 5-frame window.

The CNN is implemented in [`custom_cnn.py`](https://github.com/carmecorbi/TFM-carme/blob/main/detection/heatmaps/PyTorch-YOLOv3/pytorchyolo/custom_cnn.py).

### 3.Multimodal Fusion for YOLO Input
The two encoded heatmaps (one per team) are concatenated with the original RGB frame along the channel dimension. This results in a 5-channel input tensor: H x W x 5 → [R,G,B, Left Heatmap, Right Team Heatmap]

The fused input is passed to a modified YOLO detector capable of processing 5-channel inputs. The detection model is based on YOLOv3 implemented in PyTorch, adapted from the repository [https://github.com/eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3/tree/master) to support the additional heatmap channels.

To enable this, several significant modifications were made:
- Custom Dataset Implementation:
  A new dataset class (OwnDataset) was created to load and preprocess both RGB images and their corresponding      heatmaps.

   The dataset is organized as follows:
   
   ```text
   
   ├── heatmaps/
   │   ├── train/
   │   │   └── <sequence_name>/
   │   │       └── <frame_id>/                  # e.g. 000006 to 000750
   │   │           ├── left/                    # 5 heatmaps for team left
   │   │           └── right/                   # 5 heatmaps for team right
   │   ├── val/
   │   └── test/
   │
   ├── images/
   │   ├── train/
   │   │   └── <sequence_name>/                 # All RGB frames (e.g. 000006.jpg)
   │   ├── val/
   │   └── test/
   │   ├── train_ball.txt                       # Contains relative paths to training images
   │   ├── val_ball.txt                         # Contains relative paths to validation images
   │   └── test_ball.txt                        # Contains relative paths to test images
   │
   ├── labels/
   │   ├── train/
   │   │   └── <sequence_name>/                 # YOLO-format .txt files per frame
   │   ├── val/
   │   └── test/
Each modality (image, label, heatmap) is synchronized by frame index, and the dataset files (train_ball.txt, val_ball.txt, test_ball.txt) are used to list all image paths per split. These lists serve as input to the dataloader and are referenced in the YOLO [`custom.data`](https://github.com/carmecorbi/TFM-carme/blob/main/detection/heatmaps/PyTorch-YOLOv3/config/custom.data) configuration file.

