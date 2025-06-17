import argparse
import time
import os
import sys
from ultralytics import YOLO


model = YOLO('yolo12s')

SEQUENCE_PATH = f"/data-fast/data-server/ccorbi/dataset_tracking/full/images/test/SNMOT-116"

output_dir = "/home-net/ccorbi/detection/pre-trained/inference_classes"

os.makedirs(output_dir, exist_ok=True)

results_inference = model.predict(source=SEQUENCE_PATH, save=True,project=output_dir)