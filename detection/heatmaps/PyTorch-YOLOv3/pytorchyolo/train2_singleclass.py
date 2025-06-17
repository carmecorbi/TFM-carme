#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm
import sys
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pytorchyolo.custom_cnn import  Temporal3DCNN
from pytorchyolo.models import load_model
from pytorchyolo.utils.logger import Logger
from pytorchyolo.utils.utils import to_cpu, load_classes, print_environment_info, worker_seed_set
from pytorchyolo.utils.datasets import OwnDataset
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS
#from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.test import _evaluate2, _create_validation_data_loader2

from terminaltables import AsciiTable

from torchsummary import summary


def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False,base_hmap=None):
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
    dataset = OwnDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS,
        base_hmap_path = base_hmap
        )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader


def run(args):
      # Redirect all prints to a file
    log_file = open("training_YOLOv3_random_5channels_confidencelow_singleclass.txt", "w")
    sys.stdout = log_file
    sys.stderr = log_file  # Optional: also redirect errors
    print_environment_info()
    parser = argparse.ArgumentParser(description="Trains the YOLO model.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--pretrained_weights", type=str, help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Interval of epochs between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument("--logdir", type=str, default="logs", help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
    #args = parser.parse_args()
    print(f"Command line arguments: {args}")

    '''if args.seed != -1:
        provide_determinism(args.seed)'''

    logger = Logger(args.logdir)  # Tensorboard logger

    # Create output directories if missing
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints_YOLOv3_channels5_random_confidencelow_singleclass", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(args.data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print('device',device)

    # ############
    # Create model
    # ############

    model = load_model(args.model, args.pretrained_weights)
    model = model.to(device)
    heatmap = Temporal3DCNN().to(device)

    # Print model
    if args.verbose:
        summary(model, input_size=(5, model.hyperparams['height'], model.hyperparams['height']))

    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']
    print(mini_batch_size)
    # #################
    # Create Dataloader
    # #################

    # Load training dataloader
    dataloader = _create_data_loader(
        train_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu,
        args.multiscale_training,
        '/data-fast/data-server/ccorbi/ball/heatmaps')

    # Load validation dataloader
    validation_dataloader = _create_validation_data_loader2(
        valid_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu,
        '/data-fast/data-server/ccorbi/ball/heatmaps'
        )

    # ################
    # Create optimizer
    # ################

    params = [p for p in model.parameters() if p.requires_grad]

    if (model.hyperparams['optimizer'] in [None, "adam"]):
        optimizer = optim.Adam(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
        )
    elif (model.hyperparams['optimizer'] == "sgd"):
        optimizer = optim.SGD(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
    # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
    # instead of: 0, 10, 20
    for epoch in range(1, args.epochs+1):

       

        model.train()  # Set model to training mode
        heatmap.train()
     
        for batch_i, (paths, imgs, targets, heatmap_lefts, heatmap_rights) in enumerate(tqdm.tqdm(dataloader, desc="Training Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.to(device, non_blocking=True)
            heatmap_lefts = heatmap_lefts.to(device)
            heatmap_rights = heatmap_rights.to(device)
            #print("imgs.shape:", imgs.shape)               # [B, 3, H, W]
            #print("heatmapleft.shape:", heatmap_lefts.shape) 
            #print("heatmapright.shape:", heatmap_rights.shape) 
            encoded_left = heatmap(heatmap_lefts)
            encoded_right = heatmap(heatmap_rights)
            #print("encoded_left.shape:", encoded_left.shape)   # [B, 1, H, W]
            #print("encoded_right.shape:", encoded_right.shape) # [B, C, H, W]
            targets = targets.to(device)
            input_tensor = torch.cat([imgs, encoded_left, encoded_right], dim=1)
            input_tensor = input_tensor.to(device)
            #print("input_tensor.shape:", input_tensor.shape)


            outputs = model(input_tensor)

            loss, loss_components = compute_loss(outputs, targets, model)

            loss.backward()

            ###############
            # Run optimizer
            ###############

            if batches_done % model.hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model.hyperparams['learning_rate']
                if batches_done < model.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                # Log the learning rate
                logger.scalar_summary("train/learning_rate", lr, batches_done)
                # Set learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lr

                # Run optimizer
                optimizer.step()
                # Reset gradients
                optimizer.zero_grad()

            # ############
            # Log progress
            # ############
           
            print(AsciiTable(
                    [
                        ["Type", "Value"],
                        ["IoU loss", float(loss_components[0])],
                        ["Object loss", float(loss_components[1])],
                        ["Class loss", float(loss_components[2])],
                        ["Loss", float(loss_components[3])],
                        ["Batch loss", to_cpu(loss).item()],
                    ]).table)

            # Tensorboard logging
            tensorboard_log = [
                ("train/iou_loss", float(loss_components[0])),
                ("train/obj_loss", float(loss_components[1])),
                ("train/class_loss", float(loss_components[2])),
                ("train/loss", to_cpu(loss).item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model.seen += imgs.size(0)

        # #############
        # Save progress
        # #############

        # Save model to checkpoint file
        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = f"checkpoints_YOLOv3_channels5_random_confidencelow/yolov3_ckpt_{epoch}.pth"
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            torch.save(model.state_dict(), checkpoint_path)

        # ########
        # Evaluate
        # ########

        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = _evaluate2(
                model,
                heatmap,
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
                print(AsciiTable(
                    [
                        ["Metric", "Value"],
                        ["Precision",  precision.mean()],
                        ["Recall",recall.mean()],
                        ["mAP", AP.mean()],
                        ["F1", f1.mean()]
                    ]).table)

                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    ("validation/f1", f1.mean())]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)

    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print("Training finished. Output saved to 'training_output.txt'")

class Args:
    def __init__(self):
        self.model = "/home-net/ccorbi/detection/heatmaps/PyTorch-YOLOv3/config/yolov3-original2.cfg"
        self.data = "/home-net/ccorbi/detection/heatmaps/PyTorch-YOLOv3/config/custom.data"
        self.epochs = 30
        self.verbose = False
        self.n_cpu = 4
        self.pretrained_weights = "/home-net/ccorbi/detection/heatmaps/PyTorch-YOLOv3/pytorchyolo/utils/yolov3.weights"
        self.checkpoint_interval = 3
        self.evaluation_interval = 1
        self.multiscale_training = False
        self.iou_thres = 0.5
        self.conf_thres = 0.00001
        self.nms_thres = 0.5
        self.logdir = "logs"
        self.seed = 42

def main():
    args = Args()
    run(args)
if __name__ == "__main__":
    main()
