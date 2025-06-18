# Ball Detection with Player Heatmaps

## Motivation
The aim of this part of the project is to improve ball detection performance by incorporating spatial and temporal context from player positions. Instead of relying solely on RGB data, the system uses heatmaps derived from player annotations. These heatmaps are processed with a 3D convolutional network (CNN) and fused with the original video frames before being passed to a YOLO-based object detector.

## Pipeline Overview
### 1. **Player-wise Gaussian Heatmap Generation (per team)**
   
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
   
3. Temporal Encoding with 3D CNNs
4. Multimodal Fusion for YOLO Input
