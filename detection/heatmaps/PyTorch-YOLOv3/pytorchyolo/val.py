
#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm
import sys
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

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


def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader

def validate_only(args):
    log_file = open("validation_output_0001.txt", "w")
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
    model.eval()  # Important: set model to evaluation mode

    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']

    validation_dataloader = _create_validation_data_loader(
        valid_path,
        mini_batch_size,
        model.hyperparams['height'],
        n_cpu=0
    )

    print("\n---- Running validation ----")
    metrics_output = _evaluate(
        model,
        validation_dataloader,
        class_names,
        img_size=model.hyperparams['height'],
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        verbose=args.verbose
    )

    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        print("\n---- Evaluation Results ----")
        print(f"Precision: {precision.mean():.4f}")
        print(f"Recall:    {recall.mean():.4f}")
        print(f"mAP:       {AP.mean():.4f}")
        print(f"F1 Score:  {f1.mean():.4f}")

        evaluation_metrics = [
            ("validation/precision", precision.mean()),
            ("validation/recall", recall.mean()),
            ("validation/mAP", AP.mean()),
            ("validation/f1", f1.mean())]
        logger.list_of_scalars_summary(evaluation_metrics, 0)

    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print("Validation finished. Output saved to 'validation_output.txt'")
class Args:
    def __init__(self):
        self.model = "/home-net/ccorbi/detection/heatmaps/PyTorch-YOLOv3/config/yolov3-original.cfg"
        self.data = "/home-net/ccorbi/detection/heatmaps/PyTorch-YOLOv3/config/custom.data"
        self.epochs = 30
        self.verbose = True
        self.n_cpu = 4
        self.pretrained_weights = "/home-net/ccorbi/detection/heatmaps/PyTorch-YOLOv3/checkpoints_YOLOv3_3channels/yolov3_ckpt_27.pth"
        self.checkpoint_interval = 3
        self.evaluation_interval = 1
        self.multiscale_training = False
        self.iou_thres = 0.5
        self.conf_thres = 0.0001
        self.nms_thres = 0.5
        self.logdir = "logs"
        self.seed = 42
def main():
    args = Args()
    validate_only(args)

if __name__ == "__main__":
    main()
