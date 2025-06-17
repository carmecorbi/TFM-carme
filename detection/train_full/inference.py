import argparse
import time
import os
import sys
from ultralytics import YOLO


model = YOLO('/home-net/ccorbi/detection/train_full/train_full/train/weights/best.pt')

SEQUENCE_PATH = f"/data-fast/data-server/ccorbi/dataset_tracking/full/images/test/SNMOT-116"

output_dir = "/home-net/ccorbi/detection/train_full/inference_classes"

os.makedirs(output_dir, exist_ok=True)

results_inference = model.predict(source=SEQUENCE_PATH, device=2,save=True,project=output_dir)