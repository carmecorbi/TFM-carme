#! /usr/bin/env python3

from __future__ import division

import argparse
import tqdm
import numpy as np

from terminaltables import AsciiTable

import torch
from pytorchyolo.utils.loss import compute_loss
from torch.utils.data import DataLoader
from torch.autograd import Variable

from pytorchyolo.models import load_model
from pytorchyolo.utils.utils import load_classes, ap_per_class, get_batch_statistics, non_max_suppression, to_cpu, xywh2xyxy, print_environment_info
from pytorchyolo.utils.datasets import ListDataset, OwnDataset
from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.utils.transforms import ToTensor, PadSquare, RelativeLabels, AbsoluteLabels, ImgAug
from torchvision import transforms
#from pytorchyolo.custom_cnn import  Temporal3DCNN

def evaluate_model_file(model_path, weights_path, img_path, class_names, batch_size=8, img_size=416,
                        n_cpu=8, iou_thres=0.5, conf_thres=0.5, nms_thres=0.5, verbose=True):
    """Evaluate model on validation dataset.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param class_names: List of class names
    :type class_names: [str]
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param iou_thres: IOU threshold required to qualify as detected, defaults to 0.5
    :type iou_thres: float, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :param verbose: If True, prints stats of model, defaults to True
    :type verbose: bool, optional
    :return: Returns precision, recall, AP, f1, ap_class
    """
    dataloader = _create_validation_data_loader(
        img_path, batch_size, img_size, n_cpu)
    model = load_model(model_path, weights_path)
    metrics_output = _evaluate(
        model,
        dataloader,
        class_names,
        img_size,
        iou_thres,
        conf_thres,
        nms_thres,
        verbose)
    return metrics_output


def print_eval_stats(AP,ap_class,class_names, verbose):
        
            # Prints class AP and mean AP
    ap_table = [["Index", "Class", "AP"]]
    for i, c in enumerate(ap_class):
        ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
    print(AsciiTable(ap_table).table)
    print(f"---- mAP {AP.mean():.5f} ----")
    


def _evaluate(model, dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
    """Evaluate model on validation dataset.

    :param model: Model to evaluate
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images with targets
    :type dataloader: DataLoader
    :param class_names: List of class names
    :type class_names: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param iou_thres: IOU threshold required to qualify as detected
    :type iou_thres: float
    :param conf_thres: Object confidence threshold
    :type conf_thres: float
    :param nms_thres: IOU threshold for non-maximum suppression
    :type nms_thres: float
    :param verbose: If True, prints stats of model
    :type verbose: bool
    :return: Returns precision, recall, AP, f1, ap_class
    """
    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    val_iou_loss = 0.0
    val_iou_loss = 0.0
    val_obj_loss = 0.0
    val_cls_loss = 0.0
    val_total_loss = 0.0
    num_batches = 0

    for _, imgs, targets in tqdm.tqdm(dataloader, desc="Validating"):
        # Extract labels
        labels += targets[:, 1].tolist()
        loss_targets = targets.clone()
        loss_targets = loss_targets.to(device)
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        #imgs = Variable(imgs.type(Tensor), requires_grad=False)
        imgs = imgs.to(device).float().requires_grad_(False)

        with torch.no_grad():
            outputs1, outputs2 = model(imgs)
            loss, loss_components = compute_loss(outputs2, loss_targets, model)
            val_iou_loss += float(loss_components[0])
            val_obj_loss += float(loss_components[1])
            val_cls_loss += float(loss_components[2])
            val_total_loss += float(loss)
            num_batches += 1

            outputs = non_max_suppression(outputs1, conf_thres=conf_thres, iou_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        avg_iou_loss = val_iou_loss / num_batches
        avg_obj_loss = val_obj_loss / num_batches
        avg_cls_loss = val_cls_loss / num_batches
        avg_total_loss = val_total_loss / num_batches
    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(
        true_positives, pred_scores, pred_labels, labels)

    print_eval_stats(AP,ap_class, class_names, verbose)

    return precision, recall, AP, f1, ap_class,avg_iou_loss, avg_obj_loss, avg_cls_loss, avg_total_loss

def _evaluate2(model, heatmap,dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
    """Evaluate model on validation dataset.

    :param model: Model to evaluate
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images with targets
    :type dataloader: DataLoader
    :param class_names: List of class names
    :type class_names: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param iou_thres: IOU threshold required to qualify as detected
    :type iou_thres: float
    :param conf_thres: Object confidence threshold
    :type conf_thres: float
    :param nms_thres: IOU threshold for non-maximum suppression
    :type nms_thres: float
    :param verbose: If True, prints stats of model
    :type verbose: bool
    :return: Returns precision, recall, AP, f1, ap_class
    """
    model.eval()  # Set model to evaluation mode
    heatmap.eval()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    val_iou_loss = 0.0
    val_iou_loss = 0.0
    val_obj_loss = 0.0
    val_cls_loss = 0.0
    val_total_loss = 0.0
    num_batches = 0

    for _, imgs, targets,heatmap_lefts,heatmap_rights in tqdm.tqdm(dataloader, desc="Validating"):
        # Extract labels
        labels += targets[:, 1].tolist()
        loss_targets = targets.clone()
        loss_targets = loss_targets.to(device)
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = imgs.to(device).float().requires_grad_(False)
        heatmap_lefts = heatmap_lefts.to(device).float()
        heatmap_rights = heatmap_rights.to(device).float()
        
        #print('input tensor',imgs.shape)
        with torch.no_grad():
            encoded_left = heatmap(heatmap_lefts)
            encoded_right = heatmap(heatmap_rights)

            #print(f"[DEBUG] imgs shape:           {imgs.shape}")           # e.g. [B, 3, 416, 416]
            #print(f"[DEBUG] encoded_left shape:   {encoded_left.shape}")   # e.g. [B, 1, 416, 416]
            #print(f"[DEBUG] encoded_right shape:  {encoded_right.shape}")  # e.g. [B, 1, 416, 416]

            input_tensor = torch.cat([imgs, encoded_left, encoded_right], dim=1)


            #print(f"[DEBUG] input_tensor shape:   {input_tensor.shape}")   # e.g. [B, 5, 416, 416]

            outputs1, outputs2 = model(input_tensor)
            loss, loss_components = compute_loss(outputs2, loss_targets, model)
        
            val_iou_loss += float(loss_components[0])
            val_obj_loss += float(loss_components[1])
            val_cls_loss += float(loss_components[2])
            val_total_loss += float(loss)
            num_batches += 1

            outputs = non_max_suppression(outputs1, conf_thres=conf_thres, iou_thres=nms_thres)

            #for i, out in enumerate(outputs):
                #if out is not None and out.size(0) > 0:
                    #print(f"[DEBUG] Batch {i} → {out.size(0)} boxes")
                #else:
                    #print(f"[DEBUG] Batch {i} → No boxes")
            
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        avg_iou_loss = val_iou_loss / num_batches
        avg_obj_loss = val_obj_loss / num_batches
        avg_cls_loss = val_cls_loss / num_batches
        avg_total_loss = val_total_loss / num_batches

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(
        true_positives, pred_scores, pred_labels, labels)
    precision, recall, AP, f1, ap_class = metrics_output
    print_eval_stats(AP,ap_class, class_names, verbose)

    return precision, recall, AP, f1, ap_class,avg_iou_loss, avg_obj_loss, avg_cls_loss, avg_total_loss



def evaluate2(model, dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
    """Evaluate model on validation dataset.

    :param model: Model to evaluate
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images with targets
    :type dataloader: DataLoader
    :param class_names: List of class names
    :type class_names: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param iou_thres: IOU threshold required to qualify as detected
    :type iou_thres: float
    :param conf_thres: Object confidence threshold
    :type conf_thres: float
    :param nms_thres: IOU threshold for non-maximum suppression
    :type nms_thres: float
    :param verbose: If True, prints stats of model
    :type verbose: bool
    :return: Returns precision, recall, AP, f1, ap_class
    """
    device = torch.device("cpu")
    heatmap = Temporal3DCNN().to(device)
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    heatmap.eval()

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (images, heatmaps_left, heatmaps_right, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Validating")):
        images = images.to(device)
        heatmaps_left = heatmaps_left.to(device)
        heatmaps_right = heatmaps_right.to(device)
        targets = targets.to(device)
        encoded_left = heatmap(heatmaps_left)   
        encoded_right = heatmap(heatmaps_right)


        input_tensor = torch.cat([images, encoded_left, encoded_right], dim=1)
        #print("\n[DEBUG] Raw targets:")
        #print(targets)

        # Extract labels (class index)
        #print("\n[DEBUG] Class labels (targets[:, 1]):")
        #print(targets[:, 1])

        labels += targets[:, 1].tolist()

        # BEFORE converting bbox format
        #print("\n[DEBUG] Bounding boxes in xywh (targets[:, 2:]):")
        #print(targets[:, 2:])

        # Convert format and rescale
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        # AFTER processing
        #print("\n[DEBUG] Bounding boxes in xyxy (after rescaling):")
        #print(targets[:, 2:])
        with torch.no_grad():
            outputs = model(input_tensor)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(
        true_positives, pred_scores, pred_labels, labels)

    print_eval_stats(metrics_output, class_names, verbose)

    return metrics_output

def _create_validation_data_loader(img_path, batch_size, img_size, n_cpu):
    """
    Creates a DataLoader for validation.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    AUGMENTATION = transforms.Compose([
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
    ])
    dataset = ListDataset(img_path, img_size=img_size, multiscale=False, transform=AUGMENTATION)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader

def _create_validation_data_loader2(img_path, batch_size, img_size, n_cpu,base_hmap):
    """
    Creates a DataLoader for validation.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    AUGMENTATION = transforms.Compose([
    AbsoluteLabels(),
    PadSquare(),
    RelativeLabels(),
    ToTensor(),
    ])
    dataset = OwnDataset(img_path, img_size=img_size, multiscale=False, transform=AUGMENTATION,base_hmap_path=base_hmap)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader


def run(args):
    print_environment_info()
    parser = argparse.ArgumentParser(description="Evaluate validation data.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="weights/yolov3.weights", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Size of each image batch")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the validation more verbose")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    #args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # Load configuration from data file
    data_config = parse_data_config(args.data)
    # Path to file containing all images for validation
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])  # List of class names

    precision, recall, AP, f1, ap_class = evaluate_model_file(
        args.model,
        args.weights,
        valid_path,
        class_names,
        batch_size= 64//16,
        img_size=args.img_size,
        n_cpu=args.n_cpu,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        verbose=True)

class Args:
        def __init__(self):
            self.model = "/home-net/ccorbi/detection/heatmaps/PyTorch-YOLOv3/config/yolov3-original.cfg"
            self.data = "/home-net/ccorbi/detection/heatmaps/PyTorch-YOLOv3/config/custom.data"
            self.batch_size = 8
            self.verbose = True
            self.img_size = 8
            self.n_cpu = 4
            self.weights = "/home-net/ccorbi/detection/heatmaps/PyTorch-YOLOv3/pytorchyolo/utils/yolov3.weights"
            self.iou_thres = 0.5
            self.conf_thres = 0.1
            self.nms_thres = 0.5

if __name__ == "__main__":
    args = Args()
    run(args)
