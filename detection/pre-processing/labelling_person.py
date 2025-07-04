import os
import configparser
from collections import defaultdict

# Constants
SRC_ROOT = "/data-fast/data-server/ccorbi/SN-Tracking/tracking/test"
DST_ROOT = "/data-fast/data-server/ccorbi/dataset_tracking/person/labels/test"
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
MAX_FRAME_ID = 750

# Iterate over each subfolder in the sequence
for sequence in os.listdir(SRC_ROOT):
    sequence_path = os.path.join(SRC_ROOT, sequence)
    if not os.path.isdir(sequence_path):
        continue

    print(f"Processing {sequence}")

    # Read gameinfo.ini to determine which tracklet is the ball
    ini_path = os.path.join(sequence_path, "gameinfo.ini")
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve the case of the keys
    config.read(ini_path)

    ball_ids = set()
    for key in config['Sequence']:
        if key.startswith("trackletID_"):
            value = config['Sequence'][key].strip().lower()
            print(f"DEBUG: key={key}, value={value}")
            if "ball" in value:
                try:
                    tracklet_num = int(key.split('_')[-1])
                    print(f"  -> Found BALL with tracklet_id = {tracklet_num}")
                    ball_ids.add(tracklet_num)
                except ValueError:
                    print(f"  -> ERROR converting tracklet_num for key: {key}")

    # Read gt.txt
    gt_path = os.path.join(sequence_path, "gt", "gt.txt")
    if not os.path.isfile(gt_path):
        print(f"  gt.txt not found for {sequence}, skipping.")
        continue

    frame_data = defaultdict(list)

    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue

            frame_id = int(parts[0])
            tracklet_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])

            if frame_id > MAX_FRAME_ID:
                continue

            # Calculate YOLO-normalized coordinates
            x_center = (x + w / 2) / IMAGE_WIDTH
            y_center = (y + h / 2) / IMAGE_HEIGHT
            w_norm = w / IMAGE_WIDTH
            h_norm = h / IMAGE_HEIGHT

            class_id = 32 if tracklet_id in ball_ids else 0
            frame_data[frame_id].append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    # Write the output files
    out_dir = os.path.join(DST_ROOT, sequence)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(1, MAX_FRAME_ID + 1):
        filename = f"{i:06d}.txt"
        out_path = os.path.join(out_dir, filename)
        lines = [line for line in frame_data.get(i, []) if not line.startswith("32 ")]
        if lines:  # Only write if there are detections that are not the ball
            with open(out_path, "w") as f:
                f.write("\n".join(lines))

print("Process completed.")
