import argparse
import time
import os
from ultralytics import YOLO
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = YOLO('yolo12s.pt')

DATASET_PATH = "/data-fast/data-server/ccorbi/dataset_tracking/full/data.yaml"

results = model.val(data=DATASET_PATH,split='test',classes=[0,32])
# Print specific metrics
print("Average precision for all classes:", results.box.all_ap)
print("Average precision:", results.box.ap)
print("Average precision at IoU=0.50:", results.box.ap50)
print("F1 score:", results.box.f1)
print("Mean average precision:", results.box.map)
print("Mean average precision at IoU=0.50:", results.box.map50)
print("Mean average precision at IoU=0.75:", results.box.map75)
print("Mean precision:", results.box.mp)
print("Mean recall:", results.box.mr)
print("Precision:", results.box.p)
print("Recall:", results.box.r)