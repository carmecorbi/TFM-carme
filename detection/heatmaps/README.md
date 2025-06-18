# Ball Detection with Player Heatmaps

## Motivation
The aim of this part of the project is to improve ball detection performance by incorporating spatial and temporal context from player positions. Instead of relying solely on RGB data, the system uses heatmaps derived from player annotations. These heatmaps are processed with a 3D convolutional network (CNN) and fused with the original video frames before being passed to a YOLO-based object detector.

## Pipeline Overview

1. Player-wise Gaussian Heatmap Generation (per team)
   Players annotations are converted into heatmaps to provide spatial context for the ball detector. For each     frame:
   - 
3. Temporal Encoding with 3D CNNs
4. Multimodal Fusion for YOLO Input
