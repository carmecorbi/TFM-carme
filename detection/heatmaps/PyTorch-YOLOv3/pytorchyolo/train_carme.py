from __future__ import division

import os
import argparse
import tqdm
import sys

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from pytorchyolo.models import load_model
from pytorchyolo.utils.logger import Logger
from pytorchyolo.utils.utils import to_cpu, load_classes, print_environment_info, worker_seed_set
from pytorchyolo.utils.dataset_carme import SoccerDataset
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.utils.loss import compute_loss
#from pytorchyolo.test import _evaluate, _create_validation_data_loader
from pytorchyolo.custom_cnn import  Temporal3DCNN
from pytorchyolo.test import evaluate2
from terminaltables import AsciiTable
from PIL import Image, ImageOps
from torchsummary import summary

class PadToSquare:
        def __call__(self, img):
            # img és una PIL Image
            w, h = img.size
            max_wh = max(w, h)
            # Padding per fer la imatge quadrada
            padding = (
                (max_wh - w) // 2,  # left
                (max_wh - h) // 2,  # top
                (max_wh - w + 1) // 2,  # right
                (max_wh - h + 1) // 2   # bottom
            )
            return ImageOps.expand(img, padding)

def create_dataloader(image_root, label_root, heatmap_root, sequences, batch_size=16, shuffle=False,transform=None,img_size=416):
    """
    Creates a SoccerDataset and corresponding DataLoader.

    Args:
        image_root (str): Root directory containing the images.
        label_root (str): Root directory containing the label files.
        heatmap_root (str): Root directory containing heatmaps.
        sequences (list of str): List of sequence folder names to include.
        batch_size (int, optional): Number of samples per batch. Default is 16.
        shuffle (bool, optional): Whether to shuffle the dataset each epoch. Default is True.
        transform (callable, optional): Transform to apply to images and heatmaps. Defaults to ToTensor.

    Returns:
        dataset (SoccerDataset): The created dataset instance.
        dataloader (DataLoader): DataLoader for iterating over the dataset with batching.
    """
    transform = transforms.Compose([
        PadToSquare(),              # primer padding per fer quadrada
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # Instantiate the dataset with the given parameters and transform
    dataset = SoccerDataset(
        image_root=image_root,
        label_root=label_root,
        heatmap_root=heatmap_root,
        sequences=sequences,
        transform=transform,
        img_size=img_size
    )

    # Create the DataLoader using the dataset and the custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn
    )

    return dataloader

def run(args):
    log_file = open("training_output.txt", "w")
    sys.stdout = log_file
    sys.stderr = log_file  
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
    print(f"Command line arguments: {args}")


    logger = Logger(args.logdir)  # Tensorboard logger

    # Create output directories if missing
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    #data_config = parse_data_config(args.data)
    #train_path = data_config["train"]
    #valid_path = data_config["valid"]
    class_names = ["sports ball"]  #load_classes(data_config["names"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ############
    # Create model
    # ############
    model = load_model(args.model,args.pretrained_weights).to(device)

    # Print model
    if args.verbose:
        summary(model, input_size=(5, model.hyperparams['height'], model.hyperparams['height']))

    #mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']

    # #################
    # Create Dataloader
    # #################

    # Load training dataloader
    train_dataloader = create_dataloader(
          image_root = '/data-fast/data-server/ccorbi/ball/images/train',
          label_root = '/data-fast/data-server/ccorbi/ball/labels/train',
          heatmap_root = '/data-fast/data-server/ccorbi/ball/heatmaps/train',
          batch_size=8,
          img_size=608,
          sequences = ['SNMOT-060']
    )

    val_dataloader = create_dataloader(
          image_root = '/data-fast/data-server/ccorbi/ball/images/val',
          label_root = '/data-fast/data-server/ccorbi/ball/labels/val',
          heatmap_root = '/data-fast/data-server/ccorbi/ball/heatmaps/train',
          batch_size=8,
          img_size=608,
          sequences = ['SNGS-039']
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

    heatmap = Temporal3DCNN().to(device)
    for epoch in range(1, args.epochs+1):

        print("\n---- Training Model ----")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.train()  # Set model to training mode
        heatmap.train()

        for batch_i, (images, heatmaps_left, heatmaps_right, labels) in enumerate(tqdm.tqdm(train_dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(train_dataloader) * epoch + batch_i
            images = images.to(device)
            heatmaps_left = heatmaps_left.to(device)
            heatmaps_right = heatmaps_right.to(device)
            labels = labels.to(device)
            encoded_left = heatmap(heatmaps_left)   
            encoded_right = heatmap(heatmaps_right)
            print(labels)

            input_tensor = torch.cat([images, encoded_left, encoded_right], dim=1)
            input_tensor =input_tensor.to(device)

            outputs = model(input_tensor)
            loss, loss_components = compute_loss(outputs, labels, model)
            loss.backward()

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
            if args.verbose:
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

            model.seen += images.size(0)

        # #############
        # Save progress
        # #############

        # Save model to checkpoint file
        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = f"checkpoints/yolov3_ckpt_{epoch}.pth"
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            torch.save(model.state_dict(), checkpoint_path)
        if epoch % args.evaluation_interval == 0:
                print("\n---- Evaluating Model ----")
                # Evaluate the model on the validation set
                metrics_output = evaluate2(
                    model,
                    val_dataloader,
                    class_names,
                    img_size=model.hyperparams['height'],
                    iou_thres=args.iou_thres,
                    conf_thres=args.conf_thres,
                    nms_thres=args.nms_thres,
                    verbose=args.verbose
                )

                if metrics_output is not None:
                    precision, recall, AP, f1, ap_class = metrics_output
                    print(f"Precision: {precision.mean():.4f}")
                    print(f"Recall:    {recall.mean():.4f}")
                    print(f"mAP:       {AP.mean():.4f}")
                    print(f"F1 Score:  {f1.mean():.4f}")
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
        self.model = "/home-net/ccorbi/detection/heatmaps/PyTorch-YOLOv3/config/yolov3-custom.cfg"
        self.epochs = 10
        self.verbose = True
        self.n_cpu = 4
        self.pretrained_weights = "/home-net/ccorbi/detection/heatmaps/PyTorch-YOLOv3/pytorchyolo/utils/yolov3.weights"
        self.checkpoint_interval = 1
        self.evaluation_interval = 1
        self.multiscale_training = False
        self.iou_thres = 0.5
        self.conf_thres = 0.1
        self.nms_thres = 0.5
        self.logdir = "logs"
        self.seed = 42

def main():
    args = Args()
    run(args)

if __name__ == "__main__":
    main()