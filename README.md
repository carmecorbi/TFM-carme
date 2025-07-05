# Master Thesis - Multi-Object Tracking in Broadcast Football Videos
This repository contains the code devoloped as part of my Master's thesis in Computer Vision. The objective is to detect and track players, referees, and the ball in broadcast football footage using deep learning techniques.

## Repository Structure
The project is organized as follows:
- `pre-processing`: Scripts to preprocess the SN-Tracking and SN-GameState datasets.
- `detection`: Contains all detection-related experiments:
    - **Standard object detection** for players, goalkeeper, and referees.
    - **Heatmap-based ball detection**, leveraging spatio-temporal context from players positions.
    - **Transformer-based ball detection**, using player-aware object queries to enrich detection with contextual reasoning.
- `tracking`: Implements multi-object tracking (MOT) using SOTA algorithms given custom detections.
