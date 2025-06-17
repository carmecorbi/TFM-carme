from train import DetectionTrainer
import torch.nn as nn
import sys
log_file = open('entrenament_log_ball_yolo11s_focaloss_050.txt', 'w')
sys.stdout = log_file
sys.stderr = log_file


args = dict(model="/home-net/ccorbi/detection/train_ball/yolo11s.pt",data="/data-fast/data-server/ccorbi/dataset_tracking/ball/data.yaml", single_cls=True,epochs=50,imgsz=1024,batch=0.7,device=0,patience=25,project='/home-net/ccorbi/detection/train_ball/train_yolo11_focalloss_050',freeze=10,classes=[0],plots=True)

trainer = DetectionTrainer(overrides=args)
trainer.train()
log_file.close()