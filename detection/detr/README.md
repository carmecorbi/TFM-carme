# Transformer-Based Ball Detection with Player Context

## Motivation
Building upon our previous approach that leverages player heatmaps for ball detection, we further explored a transformer-based architecture that integrates player positional information directly into the decoding stage. This model is based on a modified version of DeTR (DETection TRansformer), where player context is injected into the object queries of the decoder.

## Data Conversion
The original dataset uses YOLO format. However, DETR requires COCO format. 

## Pipeline Overview
![BallDetection2](https://github.com/user-attachments/assets/13fda4ae-185a-4d7c-97f5-87cc7ac93415)

