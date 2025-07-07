# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

def pad_to_max_len(batch_bboxes, max_len=11):
    # Convertim cada llista a tensor i la paddejem (amb zeros)
    batch_tensors = []
    for bboxes in batch_bboxes:
        t = torch.tensor(bboxes, dtype=torch.float32) if len(bboxes) > 0 else torch.empty((0,4))
        if t.shape[0] < max_len:
            padding = torch.zeros((max_len - t.shape[0], 4))
            t = torch.cat([t, padding], dim=0)
        elif t.shape[0] > max_len:
            t = t[:max_len]
        batch_tensors.append(t)
    # Stack batch_size x max_len x 4
    return torch.stack(batch_tensors)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,dataset_file=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        if dataset_file != "players":
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(samples)
        else:
            new_targets = []
            for t in targets:
                new_t = {}
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        new_t[k] = v.to(device)
                    else:
                        new_t[k] = v  # manté com està (llista o altres)
                new_targets.append(new_t)
            targets = new_targets

            #print(f"Batch size (samples): {samples.tensors.shape[0]}")
            #print(f"Batch size (targets): {len(targets)}") 

            #Players information
            
            bb_a_batch = []
            bb_b_batch = []
            team_a_batch = []
            team_b_batch = []

            for t in targets:
                players = t["players"]
                bb_a = []
                bb_b = []
                for p in players:
                    if p["team"] == 0:
                        bb_a.append(p['bbox'])
                    else:
                        bb_b.append(p['bbox'])
                bb_a_batch.append(bb_a)
                bb_b_batch.append(bb_b)
            #print(f"Num samples in batch: {len(bb_a_batch)}")
            #print(f"Exemple primer bb_a: {bb_a_batch[0]}")
            #print(f"Exemple primer bb_b: {bb_b_batch[0]}")

            bb_a_padded = pad_to_max_len(bb_a_batch, 11)
            bb_b_padded = pad_to_max_len(bb_b_batch, 11)
            #print(f"Shape bb_a_padded: {bb_a_padded.shape}")  # hauria de ser [batch_size, 11, 4]
            #print(f"Shape bb_b_padded: {bb_b_padded.shape}")

            # Mostra primer tensor per comprovar padding
            #print(f"Primer element bb_a_padded[0]:\n{bb_a_padded[0]}")
            #print(f"Primer element bb_b_padded[0]:\n{bb_b_padded[0]}")
            
            outputs = model(samples, bb_a = bb_a_padded.to(device),bb_b = bb_b_padded.to(device))
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir,dataset_file=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        if dataset_file != "players":
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(samples)
        else:
            new_targets = []
            for t in targets:
                new_t = {}
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        new_t[k] = v.to(device)
                    else:
                        new_t[k] = v  # manté com està (llista o altres)
                new_targets.append(new_t)
            targets = new_targets

            bb_a_batch = []
            bb_b_batch = []
            team_a_batch = []
            team_b_batch = []

            for t in targets:
                players = t["players"]
                bb_a = []
                bb_b = []
                for p in players:
                    if p["team"] == 0:
                        bb_a.append(p['bbox'])
                    else:
                        bb_b.append(p['bbox'])
                bb_a_batch.append(bb_a)
                bb_b_batch.append(bb_b)
            
            bb_a_padded = pad_to_max_len(bb_a_batch, 11)
            bb_b_padded = pad_to_max_len(bb_b_batch, 11)

            outputs = model(samples,bb_a = bb_a_padded.to(device),bb_b = bb_b_padded.to(device))
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        #print(f"[evaluate] Results (bbox) per batch: {results}")
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        #print(f"[evaluate] Predictions preparades per coco_evaluator: {res}") 
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
