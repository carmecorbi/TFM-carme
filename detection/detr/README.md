# Transformer-Based Ball Detection with Player Context

## Motivation
Building upon our previous approach that leverages player heatmaps for ball detection, we further explored a transformer-based architecture that integrates player positional information directly into the decoding stage. This model is based on a modified version of DeTR (DETection TRansformer), where player context is injected into the object queries of the decoder.

## Data Conversion
To train the transformer-based detector (based on DETR), we first convert our dataset from YOLO format to COCO format. This script takes YOLO-style annotations (1 `.txt` file per image) and converts them into a single COCO-style `.json` file, as required by DETR.

Example usage:
```bash
python convert_coco.py
```

## Pipeline Overview
Instead of relying only on learnable object queries, we inject player positional information directly into the decoder of DETR, providing strong contextual cues for ball detection.
![BallDetection2](https://github.com/user-attachments/assets/13fda4ae-185a-4d7c-97f5-87cc7ac93415)

### 1. Player Position Embeddings
Player bounding boxes (from both teams) are used to extract (x, y) center positions. Coordinates are normalized to [0, 1]. Passed through a linear layer to project into the same embedding space (256-d). 

### 2. Handling Variable Number of Players

### 3. Team Embeddings

