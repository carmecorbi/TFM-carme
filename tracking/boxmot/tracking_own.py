import os
import cv2
import numpy as np
import time
import torch
from pathlib import Path
from tqdm import tqdm

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import TRACKER_CONFIGS


def write_results_to_txt(results, output_path):
    with open(output_path, 'w') as f:
        f.writelines(results)
def read_detections_from_mot_file(detection_file_path):
    detections = []
    with open(detection_file_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            x1 = float(parts[2])
            y1 = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])
            detections.append((frame_id, [x1, y1, x1 + w, y1 + h, conf]))
    return detections


def run_on_sequence(sequence_path, args):
    images_dir = sequence_path / 'img1'

    # La ruta a les deteccions ara és externa, no dins sequence_path
    detections_folder = Path("/home-net/ccorbi/detection/train_person/person_detections_05")
    detection_file = detections_folder / f"{sequence_path.name}.txt"

    # Preparem la carpeta de sortida
    output_dir = Path("/home-net/ccorbi/tracking/boxmot/results/BoTSORT_own/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{sequence_path.name}.txt"

    print(f"\nProcessing sequence: {sequence_path.name}")

    detections = read_detections_from_mot_file(detection_file)  # Ha de funcionar igual
    image_files = sorted(images_dir.glob("*.jpg"))

    # Tracker setup
    tracking_config = TRACKER_CONFIGS / (args.tracking_method + '.yaml')
    if args.config_path:
        tracking_config = args.config_path

    tracker = create_tracker(
        args.tracking_method,
        tracking_config,
        args.reid_model,
        9 if args.device == "cuda" else "cpu",
        args.half,
        args.per_class
    )

    if hasattr(tracker, 'model'):
        tracker.model.warmup()

    track_eval_format = []

    for i, img_path in enumerate(tqdm(image_files, desc=f"{sequence_path.name}")):
        frame_number = i + 1
        frame = cv2.imread(str(img_path))

        frame_detections = [d[1] for d in detections if d[0] == frame_number]
        if frame_detections:
            dets_for_tracker = np.array(frame_detections)
            class_id = np.zeros((dets_for_tracker.shape[0], 1))
            dets_for_tracker_xyxy = np.hstack((dets_for_tracker[:, :4], dets_for_tracker[:, 4:5], class_id))
        else:
            dets_for_tracker_xyxy = np.empty((0, 6))

        tracker_outputs = tracker.update(dets_for_tracker_xyxy, frame)

        for output in tracker_outputs:
            x1, y1, x2, y2, track_id = map(int, output[:5])
            w, h = x2 - x1, y2 - y1
            track_eval_format.append(f"{frame_number},{track_id},{x1},{y1},{w},{h},1,-1,-1,-1\n")

    write_results_to_txt(track_eval_format, output_file)
    print(f"Saved tracking results to {output_file}")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Batch BoxMOT tracker over sequences")
    parser.add_argument("--data_dir", type=Path, default="/data-fast/data-server/ccorbi/SN-Tracking/tracking/test",
                        help="Root directory containing test sequences.")
    parser.add_argument("-m", "--tracking_method", type=str, default="deepocsort",
                        help="Tracking method: deepocsort, botsort, strongsort, ocsort, bytetrack")
    parser.add_argument("-c", "--config_path", type=Path, help="Optional tracking config override.")
    parser.add_argument("--reid_model", type=Path, default='osnet_x0_25_msmt17.pt', help="Path to ReID model")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--half", action="store_true", help="Use FP16 inference")
    parser.add_argument("--per_class", action="store_true", help="Track classes separately")
    return parser.parse_args()


def main():
    args = parse_args()
    sequence_dirs = [p for p in args.data_dir.iterdir() if p.is_dir()]
    with torch.no_grad():
        for seq_path in sequence_dirs:
            run_on_sequence(seq_path, args)


if __name__ == "__main__":
    main()
