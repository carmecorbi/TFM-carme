import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from collections import defaultdict
import argparse

# --- PARSE ARGUMENTS ---
parser = argparse.ArgumentParser(description="Generate heatmaps per sequence")
parser.add_argument("--seqname", type=str, required=True, help="Name of the sequence, e.g.: SNMOT-060")
args = parser.parse_args()

seqname = args.seqname

# --- CONFIGURE PATHS AND PARAMETERS ---
base_path = f"/data-fast/data-server/ccorbi/SN-Tracking/tracking/test/{seqname}"
gt_file = os.path.join(base_path, "gt/gt.txt")
seqinfo_file = os.path.join(base_path, "gameinfo.ini")

output_dir_left = f"heatmap_test/{seqname}/left"
output_dir_right = f"heatmap_test/{seqname}/right"
log_file = f"heatmap_test/{seqname}/image_log.txt"

image_size = (1080, 1920)

os.makedirs(output_dir_left, exist_ok=True)
os.makedirs(output_dir_right, exist_ok=True)

# --- READ SEQINFO.INI ---
tracklet_teams = {}
with open(seqinfo_file, "r") as f:
    for line in f:
        if line.startswith("trackletID_"):
            parts = line.strip().split("=")
            tid = int(parts[0].split("_")[1])
            info = parts[1].strip()
            if "player" in info or "goalkeeper":
                if "team left" in info:
                    tracklet_teams[tid] = "left"
                elif "team right" in info:
                    tracklet_teams[tid] = "right"

# --- GROUP ANNOTATIONS BY FRAME ---
frame_annotations = defaultdict(list)
with open(gt_file, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        frame = int(parts[0])
        tid = int(parts[1])
        if tid not in tracklet_teams:
            continue
        bbox = list(map(int, parts[2:6]))
        frame_annotations[frame].append((tid, bbox))

# --- CREATE AND SAVE HEATMAPS PER FRAME ---
with open(log_file, "w") as log_f:
    for frame in sorted(frame_annotations.keys()):
        heatmap_left = np.zeros(image_size, dtype=np.float32)
        heatmap_right = np.zeros(image_size, dtype=np.float32)

        for tid, (x, y, w, h) in frame_annotations[frame]:
            cx, cy = x + w // 2, y + h // 2

            point_map = np.zeros(image_size, dtype=np.float32)
            point_map[cy, cx] = 1.0

            sigma_x = w / 2
            sigma_y = h / 2
            sigma = (sigma_y, sigma_x)

            player_heatmap = gaussian_filter(point_map, sigma=sigma)
            player_heatmap /= player_heatmap.max()

            if tracklet_teams[tid] == "left":
                heatmap_left = np.maximum(heatmap_left, player_heatmap)
            else:
                heatmap_right = np.maximum(heatmap_right, player_heatmap)
        
        img_l = (heatmap_left * 255).astype(np.uint8)
        out_path_l = os.path.join(output_dir_left, f"{frame:06d}.png")
        cv2.imwrite(out_path_l, img_l)
        log_f.write(out_path_l + "\n")

        img_r = (heatmap_right * 255).astype(np.uint8)
        out_path_r = os.path.join(output_dir_right, f"{frame:06d}.png")
        cv2.imwrite(out_path_r, img_r)
        log_f.write(out_path_r + "\n")

print("Process complete!")


