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
from pytorchyolo.utils.transforms import ToTensor, PadSquare, RelativeLabels, AbsoluteLabels, ImgAug
from torchvision import transforms

from terminaltables import AsciiTable

from torchsummary import summary

AUGMENTATION = transforms.Compose([
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
])

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

def evaluate_and_log(model, dataloader, class_names, split_name, epoch, args, log_losses=True):
    precision, recall, AP, f1, _, iou_loss, obj_loss, cls_loss, total_loss = _evaluate(
        model,
        dataloader,
        class_names,
        img_size=model.hyperparams['height'],
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        verbose=args.verbose
    )

    print(f"\n---- Evaluation on {split_name.upper()} ----")
    table_data = [
        ["Metric", "Value"],
        ["Precision", precision.mean()],
        ["Recall", recall.mean()],
        ["mAP", AP.mean()],
        ["F1", f1.mean()],
    ]

    log_dict = {
        f"{split_name}/precision": precision.mean(),
        f"{split_name}/recall": recall.mean(),
        f"{split_name}/mAP": AP.mean(),
        f"{split_name}/f1": f1.mean(),
        "epoch": epoch
    }

    if log_losses:
        table_data += [
            [f"{split_name.capitalize()} IoU Loss", iou_loss],
            [f"{split_name.capitalize()} Obj Loss", obj_loss],
            [f"{split_name.capitalize()} Cls Loss", cls_loss],
            [f"{split_name.capitalize()} Total Loss", total_loss],
        ]
        log_dict.update({
            f"{split_name}/iou_loss": iou_loss,
            f"{split_name}/obj_loss": obj_loss,
            f"{split_name}/cls_loss": cls_loss,
            f"{split_name}/total_loss": total_loss,
        })

    print(AsciiTable(table_data).table)
    wandb.log(log_dict)

def run(args):
      # Redirect all prints to a file
    log_file = open("YOLOv3-baseline-singleclass-augmentation.txt", "w")
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
    
    wandb.init(project="YOLOv3-baseline-new", config=args)
    # Defineix mètriques per batch amb un step personalitzat
    wandb.define_metric("batch/*", step_metric="batches_global")

    # Defineix mètriques per epoch amb un step diferent
    for split in ["train", "validation", "test"]:
        wandb.define_metric(f"{split}/*", step_metric="epoch")

    # Create output directories if missing
    os.makedirs("output", exist_ok=True)
    os.makedirs("YOLOv3-baseline-singleclass-augmentation", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(args.data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    test_path = data_config["test"]
    class_names = load_classes(data_config["names"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device',device)

    # ############
    # Create model
    # ############
    model = load_model(args.model, args.pretrained_weights)
    model = model.to(device)

    # Print model
    if args.verbose:
        summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']

    # #################
    # Create Dataloader
    # #################

    # Load training dataloader
    dataloader = _create_data_loader(
        train_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu,
        args.multiscale_training)

    # Load validation dataloader
    validation_dataloader = _create_validation_data_loader(
        valid_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu)

    test_dataloader = _create_validation_data_loader(
        test_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu)

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
    lr = model.hyperparams['learning_rate']
    batches_global = 1
    for epoch in range(1, args.epochs+1):
        epoch_iou_loss = 0.0
        epoch_obj_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_total_loss = 0.0

        
        model.train()  # Set model to training mode
        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(dataloader) * (epoch-1) + batch_i

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)
           

            outputs = model(imgs)
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
            
            wandb.log({
                "batch/iou_loss": float(loss_components[0]),
                "batch/obj_loss": float(loss_components[1]),
                "batch/class_loss": float(loss_components[2]),
                "batch/loss": to_cpu(loss).item(),
                "batch/learning_rate": lr,
                "batches_global": batches_global 
        
            })

            epoch_iou_loss += float(loss_components[0])
            epoch_obj_loss += float(loss_components[1])
            epoch_cls_loss += float(loss_components[2])
            epoch_total_loss += to_cpu(loss).item()

            model.seen += imgs.size(0)
            
            batches_global += 1 


        # #############
        # Save progress
        # #############
        avg_iou_loss = epoch_iou_loss / len(dataloader)
        avg_obj_loss = epoch_obj_loss / len(dataloader)
        avg_cls_loss = epoch_cls_loss / len(dataloader)
        avg_total_loss = epoch_total_loss / len(dataloader)

        print(f"\nEpoch {epoch} Summary:")
        print(f"IoU Loss: {avg_iou_loss:.4f}")
        print(f"Obj Loss: {avg_obj_loss:.4f}")
        print(f"Cls Loss: {avg_cls_loss:.4f}")
        print(f"Total Loss: {avg_total_loss:.4f}\n")

        wandb.log({
            "train/iou_loss": avg_iou_loss,
            "train/obj_loss": avg_obj_loss,
            "train/class_loss": avg_cls_loss,
            "train/total_loss": avg_total_loss,
            "epoch": epoch 
        })

        # Save model to checkpoint file
        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = f"YOLOv3-baseline-singleclass-augmentation/yolov3_ckpt_{epoch}.pth"
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            torch.save(model.state_dict(), checkpoint_path)

        # ########
        # Evaluate
        # ########

        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            evaluate_and_log(model, dataloader, class_names, "train", epoch, args, log_losses=False)
            evaluate_and_log(model, validation_dataloader, class_names, "validation", epoch, args)
            evaluate_and_log(model, test_dataloader, class_names, "test", epoch, args)

    wandb.finish()

    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


class Args:
    def __init__(self):
        self.model = "/home-net/ccorbi/detection/heatmaps/PyTorch-YOLOv3/config/yolov3-singleclass_low.cfg"
        self.data = "/home-net/ccorbi/detection/heatmaps/PyTorch-YOLOv3/config/custom_singleclass.data"
        self.epochs = 50
        self.verbose = False
        self.n_cpu = 4
        self.pretrained_weights = "/home-net/ccorbi/detection/heatmaps/PyTorch-YOLOv3/pytorchyolo/utils/yolov3.weights"
        self.checkpoint_interval = 1
        self.evaluation_interval = 1
        self.multiscale_training = False
        self.iou_thres = 0.5
        self.conf_thres = 0.001
        self.nms_thres = 0.7
        self.logdir = "logs"
        self.seed = 42

def main():
    args = Args()
    run(args)
    
if __name__ == "__main__":
    main()