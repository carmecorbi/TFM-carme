import argparse
import time
import os
import sys
from ultralytics import YOLO

log_file = open('/home-net/ccorbi/detection/train_full/results/optuna_trial1', 'w')
sys.stdout = log_file
sys.stderr = log_file

model = YOLO('/home-net/ccorbi/detection/train_full/optuna_train_full/trial_1/weights/best.pt')

DATASET_PATH = "/data-fast/data-server/ccorbi/dataset_tracking/full/data.yaml"

results = model.val(data=DATASET_PATH,split='train',conf=0.001,device=[2],classes=[0,1])
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


results = model.val(data=DATASET_PATH,split='val',conf=0.001,device=[2],classes=[0,1])
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

results = model.val(data=DATASET_PATH,split='test',conf=0.001,device=[2],classes=[0,1])
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