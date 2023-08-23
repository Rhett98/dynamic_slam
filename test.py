# import torch
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# from ops_pytorch.fused_conv_select_k.fused_conv_select_k import fused_conv_select_k
# from ops_pytorch.fused_conv_random_k.fused_conv_random_k import fused_conv_random_k
# from conv_util import cost_volume

# torch.backends.cudnn.enable =True
# torch.backends.cudnn.benchmark = True

# def get_hw_idx(B, out_H, out_W, stride_H = 1, stride_W = 1):

#     H_idx = torch.reshape(torch.arange(0, out_H * stride_H, stride_H), [1, -1, 1, 1]).expand(B, out_H, out_W, 1)
#     W_idx = torch.reshape(torch.arange(0, out_W * stride_W, stride_W), [1, 1, -1, 1]).expand(B, out_H, out_W, 1)

#     idx_n2 = torch.cat([H_idx, W_idx], dim = -1).reshape(B, -1, 2)

#     return idx_n2

# a = torch.randn(2, 4, 56, 3).cuda()
# b = torch.randn(2, 4, 56, 3).cuda()
# c = torch.randn(2, 4, 56, 64).cuda()
# d = torch.randn(2, 4, 56, 64).cuda()
# # print(a)
# # print(torch.square(a))

# # cost_volume1 = cost_volume(batch_size=2, kernel_size1=[3, 5], kernel_size2=[5, 35], nsample=4,
# #                                         nsample_q=32, \
# #                                         H=4, W=56, \
# #                                         stride_H=1, stride_W=1, distance=1,
# #                                         in_channels=[64, 64],
# #                                         mlp1=[128, 64, 64], mlp2=[128, 64], is_training=True,
# #                                         bn_decay=None,
# #                                         bn=True, pooling='max', knn=True, corr_func='concat').cuda()
# # out = cost_volume1(a, b, c, d)
# batch_size = 2
# B = batch_size
# H = 4
# W = 56
# stride_H, stride_W = 1, 1
# idx_n2 = get_hw_idx(batch_size, H, W, stride_H, stride_W)
# idx_hw = idx_n2.cuda().int().contiguous()
# kernel_size2 = [5, 35]
# kernel_total_q = kernel_size2[0]*kernel_size2[1]
# random_hw = (torch.arange(0,kernel_total_q)).cuda().int()
# nsample_q = 32
# distance = 100
# stride_h, stride_w = 1, 1

# select_b_idx = torch.zeros(B, H*W, nsample_q, 1).cuda().long().detach()             # (B N nsample_q 1)
# select_h_idx = torch.zeros(B, H*W, nsample_q, 1).cuda().long().detach()
# select_w_idx = torch.zeros(B, H*W, nsample_q, 1).cuda().long().detach()

# valid_idx = torch.zeros(B, H*W, kernel_total_q, 1).cuda().float().detach()
# valid_in_dis_idx = torch.zeros(B, H*W, kernel_total_q, 1).cuda().float().detach()
# select_mask = torch.zeros(B, H*W, nsample_q, 1).cuda().float().detach()

# print(random_hw)

# select_b_idx, select_h_idx, select_w_idx,\
# valid_idx, valid_in_dis_idx, select_mask = fused_conv_select_k(a, b, 
#                                         idx_hw, random_hw, H, W, 
#                                         H*W, kernel_size2[0], kernel_size2[1],
#                                         nsample_q, 0, float(distance), stride_h, stride_w, 
#                                         select_b_idx, select_h_idx, select_w_idx, 
#                                         valid_idx, valid_in_dis_idx, select_mask, H, W)
# print("H:", select_h_idx)
# # print("W:", select_w_idx)

import numpy as np
import csv

def write_poses_to_text_file(file_name, poses):
    with open(file_name, "w", newline="") as txt_file:
        file_writer = csv.writer(txt_file, delimiter=" ")
        for pose in poses:
            pose_list = pose
            file_writer.writerow(pose_list)
            
pose = np.load(r"/home/yu/Resp/TransLO/experiment/pwclonet_KITTI_2023-08-17_15-43/eval/translonet_01/01_eval_300/01_pred.npy")
write_poses_to_text_file(r"01_300_pred.txt", pose)