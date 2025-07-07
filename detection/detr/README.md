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
To enhance the model's understanding of player context, we incorporate explicit player positional informaion into the transformer decoder. This required extending the standard COCO dataset format with custom annotations containing player bounding boxes, team affiliation, and center coordinates for each frame.

For this, we created a custom dataset loader in: `detr/datasets/coco_own.py`

This loader extends `torchvision.datasets.CocoDetection` to support both COCO object annotations and additional player-related metadata derived from tracking files.

Key Components:
- Player Team Mapping: The function `load_tracking_info()` parses the `gameinfo.ini` file for each sequence to build a dictionary mapping each `track_id` to its corresponding team: 0 → left team and 1 → right team.
  
- Player Detection per Frame: The function `load_players` reads `gt.txt` and extracts all player detections for a specific frame. Each detection includes: Normalized bounding box coordinates `(x,y,w,h)`, Center point of the bounding box and Team identifier.
  
- COCO Annotation Augmentation: The custom `COCODetection` class overrides the `__getitem__` method to:
  1. Load the original COCO image and annotations.
  2. Parse the sequence name and frame number from the image filename.
  3. Use the tracking data to retrieve player positions for that frame.
  4. Add the player information under a new key in the `target` dictionary: `target["players"] = List[Dict[bbox, center, team]]`


Player bounding boxes (from both teams) are used to extract (x, y) center positions. Coordinates are normalized to [0, 1]. Passed through a linear layer to project into the same embedding space (256-d). 

### 2. Handling Variable Number of Players

### 3. Team Embeddings

