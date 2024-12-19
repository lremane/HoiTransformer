# ------------------------------------------------------------------------
# Licensed under the Apache License, Version 2.0 (the "License")
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train functions used in main.py
"""
import math
import sys
from typing import Iterable

import torch

import util.misc as utils

import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k not in ['image_id']} for t in targets]

        # # Assuming 'samples.tensors' is your input image tensor
        # image_tensor = samples.tensors  # Adjust this if the attribute name is different
        #
        # # Ensure the tensor is in the right format (C, H, W)
        # if image_tensor.ndim == 4:  # If batch of images, pick one image
        #     print("Original Tensor Shape:", image_tensor.shape)
        #     image_tensor = image_tensor[0]  # Get the first image from the batch
        #
        # # Convert tensor to a NumPy array in HWC format (required by OpenCV)
        # image_array = image_tensor.permute(1, 2, 0).cpu().numpy()  # Convert CxHxW -> HxWxC
        # image_array = (image_array * 255).astype(np.uint8)  # Assuming input is in [0, 1] range
        #
        # # Convert RGB (matplotlib format) to BGR (OpenCV format)
        # image_array_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        #
        # # Loop through each target
        # for target in targets:
        #     # Draw all human_boxes
        #     if 'human_boxes' in target:
        #         for box in target['human_boxes']:
        #             x_min, y_min, x_max, y_max = box.cpu().numpy()
        #             cv2.rectangle(image_array_bgr,
        #                           (int(x_min), int(y_min)),
        #                           (int(x_max), int(y_max)),
        #                           (0, 255, 0), 2)  # Green for human_boxes
        #
        #     # Draw all object_boxes
        #     if 'object_boxes' in target:
        #         for box in target['object_boxes']:
        #             x_min, y_min, x_max, y_max = box.cpu().numpy()
        #             cv2.rectangle(image_array_bgr,
        #                           (int(x_min), int(y_min)),
        #                           (int(x_max), int(y_max)),
        #                           (255, 0, 0), 2)  # Blue for object_boxes
        #
        #     # Draw all action_boxes
        #     # if 'action_boxes' in target:
        #     #     for box in target['action_boxes']:
        #     #         x_min, y_min, x_max, y_max = box.cpu().numpy()
        #     #         cv2.rectangle(image_array_bgr,
        #     #                       (int(x_min), int(y_min)),
        #     #                       (int(x_max), int(y_max)),
        #     #                       (0, 0, 255), 2)  # Red for action_boxes
        #
        # # Convert back to RGB for visualization with Matplotlib
        # image_with_rectangles_rgb = cv2.cvtColor(image_array_bgr, cv2.COLOR_BGR2RGB)
        #
        # # Display the image with the rectangles
        # # Get the original image dimensions
        # height, width, _ = image_with_rectangles_rgb.shape
        #
        # # Set the figure size based on the image size
        # plt.figure(figsize=(width / 100, height / 100), dpi=100)  # Scale dimensions to inches (width, height)
        #
        # # Display the image with the rectangles
        # plt.imshow(image_with_rectangles_rgb)
        # plt.axis('off')  # Turn off axes
        # plt.tight_layout(pad=0)  # Remove any extra padding
        # plt.show()

        outputs = model(samples)
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
