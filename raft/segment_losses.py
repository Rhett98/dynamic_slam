#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.spatial import cKDTree
try:
    from itertools import ifilterfalse
except ImportError:
    from itertools import filterfalse as ifilterfalse
import torch.nn.functional as F
from knn_cuda import KNN
# import knn

def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[torch.nonzero(valid, as_tuple=False).squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

import numpy as np

def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        # do the mapping
        return lut[label]

class Lovasz_softmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None):
        super(Lovasz_softmax, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, self.classes, self.per_image, self.ignore)
    
class SegmentLoss(nn.Module):
    """ Dynamic Segmentation loss function """
    def __init__(self, config):
        super(SegmentLoss, self).__init__()
        # init unbalance loss weight
        self.nclasses = len(config["learning_map_inv"])
        self.learning_map = config["learning_map"]
        content = torch.zeros(self.nclasses, dtype=torch.float)
        for cl, freq in config["content"].items():
            x_cl = map(cl, self.learning_map)  # map actual class to xentropy class
            content[x_cl] += freq
        loss_w = 1 / (content + 0.001)  # get weights
        for x_cl, w in enumerate(loss_w):  # ignore the ones necessary to ignore
            if config["learning_ignore"][x_cl]:
                # don't weigh
                loss_w[x_cl] = 0
        print("Loss weights from content: ", loss_w.data)
        self.nll_loss = nn.NLLLoss(weight=loss_w)
        self.Ls = Lovasz_softmax(ignore=0)
        
    def forward(self, label, output):         
        wce, jacc = self.nll_loss(torch.log(output.clamp(min=1e-8)), label.long()) , self.Ls(output, label.long())
        return wce + jacc


class KDPointToPointLoss(nn.Module):
    def __init__(self):
        super(KDPointToPointLoss, self).__init__()
        self.lossMeanMSE = torch.nn.MSELoss()
        self.lossPointMSE = torch.nn.MSELoss(reduction="none")
        
    def find_target_correspondences(self, kd_tree_target, source_list_numpy):
        target_correspondence_indices = kd_tree_target[0].query(source_list_numpy)[1]
        return target_correspondence_indices

    def move_zero_point(self, pc):
        x_coords = pc[:, 0]
        y_coords = pc[:, 1]
        z_coords = pc[:, 2]

        # 找到x、y、z坐标不全为0的点的索引
        valid_indices = (x_coords != 0.0) | (y_coords != 0.0) | (z_coords != 0.0)
        # 使用索引来筛选有效点
        filtered_point_cloud = pc[valid_indices]
        return filtered_point_cloud
    
    def forward(self, source_point_cloud, target_point_cloud):
        # convert img to pointcloud
        B = len(source_point_cloud)
        loss = torch.zeros((B))
        for batch_index in range(B):
            batch_source_point_cloud = self.move_zero_point(source_point_cloud[batch_index].contiguous().view(-1, 3))
            batch_target_point_cloud = self.move_zero_point(target_point_cloud[batch_index].contiguous().view(-1, 3))
            # Build kd-tree
            target_kd_tree = [cKDTree(batch_target_point_cloud.detach().cpu().numpy())]
            # Find corresponding target points for all source points
            target_correspondences_of_source_points = \
                torch.from_numpy(self.find_target_correspondences(
                    kd_tree_target=target_kd_tree, 
                    source_list_numpy=batch_source_point_cloud.detach().cpu().numpy()))
            target_points = batch_target_point_cloud[target_correspondences_of_source_points, :]
            loss[batch_index] = self.lossMeanMSE(batch_source_point_cloud, target_points)
        return torch.mean(loss)

class knnLoss(nn.Module):
    def __init__(self, k=3):
        super(knnLoss, self).__init__()
        self.knn = KNN(k, transpose_mode=True)
        
    def forward(self, source_pc, target_pc):
        B = source_pc.shape[0]
        loss = torch.zeros((B))
        downsample_target_pc = self.get_downsample_pc(target_pc.permute(0, 2, 3, 1), 32, 512)
        downsample_source_pc = self.get_downsample_pc(source_pc, 32, 512)
        for batch_index in range(B):
            dist, _ = self.knn(self.move_zero_point(downsample_target_pc[batch_index].contiguous().view(-1, 3)).unsqueeze(0), \
                                self.move_zero_point(downsample_source_pc[batch_index].contiguous().view(-1, 3)).unsqueeze(0))
            loss[batch_index] = torch.mean(dist) 
        return torch.mean(loss)
    
    def move_zero_point(self, pc):
        x_coords = pc[:, 0]
        y_coords = pc[:, 1]
        z_coords = pc[:, 2]

        # 找到x、y、z坐标均不为0的点的索引
        valid_indices = (x_coords != 0.0) | (y_coords != 0.0) | (z_coords != 0.0)
        # 使用索引来筛选有效点
        filtered_point_cloud = pc[valid_indices]
        return filtered_point_cloud
    
    def get_downsample_pc(self, pc, out_H: int, out_W: int):
        """According to given stride and output size, return the corresponding selected points
        Args:
            array (Tensor): [any array with shape (B, H, W, 3)]
            out_H (int): [height of output array]
            out_W (int): [width of output array]
        Returns:
            Tensor: (B, outh, outw, 3) 
        """
        batch_size, H, W, C = pc.shape
        stride_H, stride_W = int(H/out_H), int(W/out_W)
        select_h_idx = torch.arange(0, out_H * stride_H, stride_H)
        select_w_idx = torch.arange(0, out_W * stride_W, stride_W)
        height_indices = (torch.reshape(select_h_idx, (1, -1, 1))).expand(batch_size, out_H, out_W)         # b out_H out_W
        width_indices = (torch.reshape(select_w_idx, (1, 1, -1))).expand(batch_size, out_H, out_W)            # b out_H out_W
        padding_indices = torch.reshape(torch.arange(batch_size), (-1, 1, 1)).expand(batch_size, out_H, out_W)   # b out_H out_W
        downsample_xyz_proj = pc[padding_indices, height_indices, width_indices, :]
        
        return downsample_xyz_proj