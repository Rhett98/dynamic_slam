import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.autograd import Variable

from raft.update import BasicUpdateBlock, SmallUpdateBlock
from raft.extractor import BasicEncoder, SmallEncoder, MaskDecoder
from raft.corr import CorrBlock, AlternateCorrBlock
from raft.utils.utils import bilinear_sampler, coords_grid, upflow8
from pointpillar_encoder import PillarEncoder, PillarLayer

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
        self.hidden_dim = hdim = 96
        self.context_dim = cdim = 64
        args.corr_levels = 3
        args.corr_radius = 3
        self.pillar_inchannel = 8
        self.pillar_outchannel = 32
        if 'dropout' not in self.args:
            self.args.dropout = 0

        # preprocess network
        self.pillarlayer = PillarLayer(args.voxel_size,args.point_cloud_range,args.max_num_points,args.max_voxels)
        self.pillarencoder = PillarEncoder(args.voxel_size,args.point_cloud_range,self.pillar_inchannel,self.pillar_outchannel)
        # feature network, context network, and update block
        self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
        self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        self.mask_decoder = MaskDecoder()

        
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def get_static_flow(self, pc1, Tr):
        '''
        pc1:(B,3,H,W)
        Tr:(B,4,4)
        return static_flow:(B,2,H,W)
        '''
        b,_,h,w = pc1.shape
        padp = torch.ones(b,1,h,w, device=pc1.device)
        
        hom_pc1 = torch.cat([pc1, padp],dim=1).reshape(b,4,h*w)
        # print(hom_pc1.device, Tr.device)
        trans_pc1 = torch.matmul(Tr, hom_pc1).reshape(b,4,h,w)[:,:3,:,:]
        static_flow = trans_pc1 - pc1
        return static_flow[:,:2,:,:]

    def cal_dis_matrix(self, flow, flow_ststic):
        '''
        flow:(B,2,H,W)
        flow_ststic:(B,2,H,W)
        return dis:(B,2,H,W)
        '''
        delta_f = flow - flow_ststic
        dis = torch.sum(delta_f**2, dim=1).unsqueeze(1)
        return dis

    def forward(self, pc1, pc2, T_gt=None, iters=6, flow_init=None):
        """ Estimate optical flow between pair of frames """
        pillars1, coors_batch1, npoints_per_pillar1, _, pillar_center1 = self.pillarlayer(pc1)
        pillars2, coors_batch2, npoints_per_pillar2, _, _ = self.pillarlayer(pc2)
        pillar_feature1 = self.pillarencoder(pillars1, coors_batch1, npoints_per_pillar1)
        pillar_feature2 = self.pillarencoder(pillars2, coors_batch2, npoints_per_pillar2)
        
        if T_gt is not None:
            flow_s = self.get_static_flow(pillar_center1, T_gt)
        else:
            #TODO:使用ICP计算T_gt
            flow_s = 0

        # 特征网络
        fmap1, fmap2 = self.fnet([pillar_feature1, pillar_feature2])        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.args.corr_levels ,radius=self.args.corr_radius)
        
        # 语义网络
        cnet = self.cnet(pillar_feature1)
        net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(pillar_center1)
        
        if flow_init is not None:
            coords1 = coords1 + flow_init

        dynamic_masks = []
        for itr in range(iters):
            # print("-------iter: ",itr,"-----------")
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            # 根据预测光流和静态光流来计算动态mask
            dis_matrix = self.cal_dis_matrix(flow_up, flow_s)
            dy_mask = self.mask_decoder(dis_matrix)
            # add result to list
            dynamic_masks.append(dy_mask)
            
        return dynamic_masks
    

