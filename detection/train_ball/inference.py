import argparse
import time
import os
import sys
from ultralytics import YOLO


model = YOLO('/home-net/ccorbi/detection/train_ball/training_models/train_ball_m_singleclass/train/weights/best.pt')

SEQUENCE_PATH = f"/data-fast/data-server/ccorbi/dataset_tracking/full/images/test/SNMOT-138"

output_dir = "inference_YOLO12m_singleclass"

os.makedirs(output_dir, exist_ok=True)

results_inference = model.predict(source=SEQUENCE_PATH, conf=0.01,device=6,save=True,project=output_dir)