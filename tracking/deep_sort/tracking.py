import os
import subprocess

base_seq_dir = "/data-fast/data-server/ccorbi/SN-Tracking/tracking/test"
detections_dir = "/data-fast/data-server/ccorbi/deepsort/detections"
output_dir = "/home-net/ccorbi/tracking/deep_sort/DeepSORT_100_euclidean_02/data"

os.makedirs(output_dir, exist_ok=True)

for seq_name in os.listdir(base_seq_dir):
    seq_path = os.path.join(base_seq_dir, seq_name)
    if not os.path.isdir(seq_path):
        continue
    
    detection_file = os.path.join(detections_dir, f"{seq_name}.npy")
    if not os.path.exists(detection_file):
        print(f"Warning: No detection file for {seq_name}, skipping.")
        continue

    output_file = os.path.join(output_dir, f"{seq_name}.txt")

    cmd = [
        "python", "deep_sort_app.py",
        f"--sequence_dir={seq_path}",
        f"--detection_file={detection_file}",
        f"--output_file={output_file}",
        "--min_confidence=0.3",
        "--max_cosine_distance=0.2",
        "--nn_budget=100"
    ]

    print(f"Running tracking for sequence {seq_name}...")
    subprocess.run(cmd)
