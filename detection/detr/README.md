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
  4. Add the player information under a new key in the `target` dictionary: `target["players"] = List[Dict[bbox, center, team]]`.

### 2. Handling Variable Number of Players
To handle the variable number of playersper frame, we apply fixed-length padding so that each input has a consistent shape. Each team can have up to 11 players, so player bounding boxes are padded to a shape of `[batch_size,11,4]` per team (`4 = [x,y,w,h]`).

Key Components:
- Padding and Preprocessing: We define a function `pad_to_max_len()` that takes lists of bounding boxes for each sample and returns padded tensors with a fixed number of players slots (11). This is done separately for each team.
- Modifications in `train_one_epoch()` and `evaluate()`: Both functions in `detr/engine.py` were updated to:
  1. Check if the `dataset_file is "players".
  2. Extract player bounding boxes per team from the `targets` in each batch.
  3. Apply the `pad_to_max_len()` function to produce fixed-shape tensors `bb_a` and `bb_b`.
  4. Pass these padded tensors as new arguments to the model.
- Transformer Decoder Integration: In `detr/models/transformer.py`, the `Transformer` class is extended to support the additional inputs:
  - `bb_a` and `bb_b` are passed through a shared linear projection layer to map them to the decoder dimension: `[batch_size,11,d_model]`.
  - These are permuted to `[11, batch_size, d_model]` and concatenated with the initial decoder target (`tgt`) along the sequence dimension.
  - Final `tgt` shape: `[num_queries + 11 + 11, batch_size, d_model]`.

### 3. Team Embeddings

