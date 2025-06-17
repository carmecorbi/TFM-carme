import argparse
import time
import os
import sys
from ultralytics import YOLO
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
log_file = open('/home-net/ccorbi/detection/train_ball/results/results_yolo12m_singleclass_01.txt', 'w')
sys.stdout = log_file
sys.stderr = log_file

model = YOLO('/home-net/ccorbi/detection/train_ball/training_models/train_ball_m_singleclass/train/weights/best.pt')

DATASET_PATH = "/data-fast/data-server/ccorbi/dataset_tracking/ball/data.yaml"

results = model.val(data=DATASET_PATH,split='train',device=[0],conf=0.10,plots=True,project='/home-net/ccorbi/detection/train_ball/results/yolo12m_singleclass_01')
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


results = model.val(data=DATASET_PATH,split='val',device=[0],conf=0.10,plots=True,project='/home-net/ccorbi/detection/train_ball/results/yolo12m_singleclass_01')
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


results = model.val(data=DATASET_PATH,split='test',device=[0],conf=0.10,plots=True,project='/home-net/ccorbi/detection/train_ball/results/yolo12m_singleclass_01')
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


