# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics as skmetrics


def calculate_area(pred, label, num_classes, ignore_index=255):
    """
    Calculate intersect, prediction and label area

    Args:
        pred (Tensor): The prediction by model.
        label (Tensor): The ground truth of image.
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored. Default: 255.

    Returns:
        Tensor: The intersection area of prediction and the ground on all class.
        Tensor: The prediction area on all class.
        Tensor: The ground truth area on all class
    """
    if len(pred.shape) == 4:
        pred = torch.squeeze(pred, axis=1)
    if len(label.shape) == 4:
        label = torch.squeeze(label, axis=1)
    if not pred.shape == label.shape:
        raise ValueError('Shape of `pred` and `label should be equal, '
                         'but there are {} and {}.'.format(pred.shape,
                                                           label.shape))
    pred_area = []
    label_area = []
    intersect_area = []
    mask = label != ignore_index

    for i in range(num_classes):
        pred_i = torch.logical_and(pred == i, mask)
        label_i = label == i
        intersect_i = torch.logical_and(pred_i, label_i)
        pred_area.append(torch.sum(pred_i.type(torch.int64)))
        label_area.append(torch.sum(label_i.type(torch.int64)))
        intersect_area.append(torch.sum(intersect_i.type(torch.int64)))

    pred_area = torch.stack(pred_area)
    label_area = torch.stack(label_area)
    intersect_area = torch.stack(intersect_area)

    return intersect_area, pred_area, label_area


def auc_roc(logits, label, num_classes, ignore_index=None):
    """
    Calculate area under the roc curve

    Args:
        logits (Tensor): The prediction by model on testset, of shape (N,C,H,W) .
        label (Tensor): The ground truth of image.   (N,1,H,W)
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored. Default: 255.

    Returns:
        auc_roc(float): The area under roc curve
    """
    if ignore_index or len(np.unique(label)) > num_classes:
        raise RuntimeError('labels with ignore_index is not supported yet.')

    if len(label.shape) != 4:
        raise ValueError(
            'The shape of label is not 4 dimension as (N, C, H, W), it is {}'.
            format(label.shape))

    if len(logits.shape) != 4:
        raise ValueError(
            'The shape of logits is not 4 dimension as (N, C, H, W), it is {}'.
            format(logits.shape))

    N, C, H, W = logits.shape
    logits = np.transpose(logits, (1, 0, 2, 3))
    logits = logits.reshape([C, N * H * W]).transpose([1, 0])

    label = np.transpose(label, (1, 0, 2, 3))
    label = label.reshape([1, N * H * W]).squeeze()

    if not logits.shape[0] == label.shape[0]:
        raise ValueError('length of `logit` and `label` should be equal, '
                         'but they are {} and {}.'.format(logits.shape[0],
                                                          label.shape[0]))

    if num_classes == 2:
        auc = skmetrics.roc_auc_score(label, logits[:, 1])
    else:
        auc = skmetrics.roc_auc_score(label, logits, multi_class='ovr')

    return auc


def mean_iou(intersect_area, pred_area, label_area):
    """
    Calculate iou.

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.

    Returns:
        np.ndarray: iou on all classes.
        float: mean iou of all classes.
    """
    intersect_area = intersect_area.cpu().numpy()
    pred_area = pred_area.cpu().numpy()
    label_area = label_area.cpu().numpy()
    union = pred_area + label_area - intersect_area
    class_iou = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            iou = 0
        else:
            iou = intersect_area[i] / union[i]
        class_iou.append(iou)
    miou = np.mean(class_iou)
    return np.array(class_iou), miou


def dice(intersect_area, pred_area, label_area):
    """
    Calculate DICE.

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.

    Returns:
        np.ndarray: DICE on all classes.
        float: mean DICE of all classes.
    """
    intersect_area = intersect_area.cpu().numpy()
    pred_area = pred_area.cpu().numpy()
    label_area = label_area.cpu().numpy()
    union = pred_area + label_area
    class_dice = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            dice = 0
        else:
            dice = (2 * intersect_area[i]) / union[i]
        class_dice.append(dice)
    mdice = np.mean(class_dice)
    return np.array(class_dice), mdice


def class_measurement(intersect_area, pred_area, label_area):
    """
    Calculate accuracy, calss precision and class recall.

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.

    Returns:
        float: The mean accuracy.
        np.ndarray: The precision of all classes.
        np.ndarray: The recall of all classes.
    """
    intersect_area = intersect_area.cpu().numpy()
    pred_area = pred_area.cpu().numpy()
    label_area = label_area.cpu().numpy()

    mean_acc = np.sum(intersect_area) / np.sum(pred_area)
    class_precision = []
    class_recall = []
    for i in range(len(intersect_area)):
        precision = 0 if pred_area[i] == 0 \
            else intersect_area[i] / pred_area[i]
        recall = 0 if label_area[i] == 0 \
            else intersect_area[i] / label_area[i]
        class_precision.append(precision)
        class_recall.append(recall)

    return mean_acc, np.array(class_precision), np.array(class_recall)


def kappa(intersect_area, pred_area, label_area):
    """
    Calculate kappa coefficient

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes..
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.

    Returns:
        float: kappa coefficient.
    """
    intersect_area = intersect_area.cpu().numpy().astype(np.float64)
    pred_area = pred_area.cpu().numpy().astype(np.float64)
    label_area = label_area.cpu().numpy().astype(np.float64)
    total_area = np.sum(label_area)
    po = np.sum(intersect_area) / total_area
    pe = np.sum(pred_area * label_area) / (total_area * total_area)
    kappa = (po - pe) / (1 - pe)
    return kappa


class metrics():
    def __init__(self, pred, label, num_classes, ignore_index=255):
        self.pred = pred
        self.label = label
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.intersect_area = None
        self.pred_area = None
        self.label_area = None
        self.mean_iou = 0
        self.mean_acc = 0
        self.kappa = 0

    def __call__(self):
        self.intersect_area, self.pred_area, self.label_area = calculate_area(
            self.pred, self.label, self.num_classes, self.ignore_index)
        _, self.mean_iou = mean_iou(
            self.intersect_area, self.pred_area, self.label_area)
        self.mean_acc, _, _ = class_measurement(
            self.intersect_area, self.pred_area, self.label_area)
        self.kappa = kappa(self.intersect_area,
                           self.pred_area, self.label_area)

    def get_acc(self):
        return self.mean_acc

    def get_mean_iou(self):
        return self.mean_iou

    def get_kappa(self):
        return self.kappa
