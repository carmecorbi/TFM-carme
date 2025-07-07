
#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm
import sys
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb

from pytorchyolo.models import load_model
from pytorchyolo.utils.logger import Logger
from pytorchyolo.utils.utils import to_cpu, load_classes, print_environment_info, worker_seed_set
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS
#from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.test import _evaluate, _create_validation_data_loader

from terminaltables import AsciiTable

from torchsummary import summary
def main():
    conf_thresholds = [0.1, 0.01, 0.001]
    nms_thresholds = [0.7]

    for conf in conf_thresholds:
        for nms in nms_thresholds:
            args = Args()
            args.conf_thres = conf
            args.nms_thres = nms
            # Update output file name dynamically
            log_filename = f"validation_output_conf{conf}_nms{nms}.txt"
            log_path = os.path.join(
                "/home-net/ccorbi/detection/heatmaps/PyTorch-YOLOv3/YOLOv3-baseline-singleclass-validation",
                log_filename
            )
            validate_only_with_logpath(args, log_path)

# Slight modification of validate_only to accept a log file path
def validate_only_with_logpath(args, log_path):
    log_file = open(log_path, "w")
    sys.stdout = log_file
    sys.stderr = log_file
    print_environment_info()

    print(f"Command line arguments: {args}")

    logger = Logger(args.logdir)

    data_config = parse_data_config(args.data)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = load_model(args.model, args.pretrained_weights)
    model.to(device)
    model.eval()

    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']

    validation_dataloader = _create_validation_data_loader(
        valid_path,
        mini_batch_size,
        model.hyperparams['height'],
        n_cpu=args.n_cpu
    )

    print("\n---- Running validation ----")
    precision, recall, AP, f1, ap_class, avg_iou_loss, avg_obj_loss, avg_cls_loss, avg_total_loss = _evaluate(
        model,
        validation_dataloader,
        class_names,
        img_size=model.hyperparams['height'],
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        verbose=args.verbose
    )

    print("\n---- Evaluation Results ----")
    print(f"Precision: {precision.mean():.4f}")
    print(f"Recall:    {recall.mean():.4f}")
    print(f"mAP:       {AP.mean():.4f}")
    print(f"F1 Score:  {f1.mean():.4f}")

    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


class Args:
    def __init__(self):
        self.model = "/home-net/ccorbi/detection/heatmaps/PyTorch-YOLOv3/config/yolov3-singleclass.cfg"
        self.data = "/home-net/ccorbi/detection/heatmaps/PyTorch-YOLOv3/config/custom_singleclass.data"
        self.verbose = True
        self.n_cpu = 4
        self.pretrained_weights = "/home-net/ccorbi/detection/heatmaps/PyTorch-YOLOv3/YOLOv3-baseline-singleclass/yolov3_ckpt_11.pth"
        self.iou_thres = 0.5
        self.logdir = "logs"
        self.seed = 42

if __name__ == "__main__":
    main()
