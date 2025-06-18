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
- Input: a stack of 5 grayscale heatmaps (5 x H x w) per team.
- Architecture: two identical 3D CNNs (one for the left team, one for the right), sharing weights.
- Output: one 2D feature map per team that encodes contextual information from the 5-frame window.




4. Multimodal Fusion for YOLO Input
