import os
import json

# Base path where the sequences are located
base_path = "/data-fast/data-server/ccorbi/SoccerNetGS/gamestate-2024/valid"
output_path = "/data-fast/data-server/ccorbi/SN-Tracking/tracking/val"

# Image dimensions
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

# Iterate over each sequence
for seq_name in os.listdir(base_path):
    seq_path = os.path.join(base_path, seq_name)
    json_path = os.path.join(seq_path, "Labels-GameState.json")

    if not os.path.isfile(json_path):
        continue

    # Create the corresponding folders
    seq_output_path = os.path.join(output_path, seq_name)
    det_dir = os.path.join(seq_output_path, "det")
    gt_dir = os.path.join(seq_output_path, "gt")
    os.makedirs(det_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    det_file_path = os.path.join(det_dir, "det.txt")

    with open(json_path, 'r') as f:
        data = json.load(f)

    with open(det_file_path, 'w') as det_file:
        for annotation in data.get("annotations", []):
            image_id = annotation["image_id"]
            frame_id = int(image_id[-3:])  # Get the last 3 digits

            category_id = annotation["category_id"]
            if category_id not in [1, 2, 3, 4]:
                continue  # Ignore other categories

            bbox = annotation["bbox_image"]
            x = int(bbox["x_center"] - bbox["w"] / 2)
            y = int(bbox["y_center"] - bbox["h"] / 2)
            w = int(bbox["w"])
            h = int(bbox["h"])

            # Write the detection in MOT format
            det_file.write(f"{frame_id},-1,{x},{y},{w},{h},1,-1,-1,-1\n")
