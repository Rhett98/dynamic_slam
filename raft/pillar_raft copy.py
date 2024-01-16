import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.autograd import Variable

from raft.update import BasicUpdateBlock, SmallUpdateBlock
from raft.extractor import BasicEncoder, SmallEncoder
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
        self.H_input, self.W_input = args.H_input, args.W_input
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

    # def warp_img(self, x, flo):
    #     """
    #     warp an image/tensor (im2) back to im1, according to the optical flow

    #     x: [B, C, H, W] (im2)
    #     flo: [B, 2, H, W] flow

    #     """
    #     B, C, H, W = x.size()
    #     # mesh grid 
    #     xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    #     yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    #     xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    #     yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    #     grid = torch.cat((xx,yy),1).float()

    #     if x.is_cuda:
    #         grid = grid.cuda()
    #     vgrid = Variable(grid) + flo

    #     # scale grid to [-1,1] 
    #     vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    #     vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    #     vgrid = vgrid.permute(0,2,3,1)        
    #     output = nn.functional.grid_sample(x, vgrid)
    #     mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    #     mask = nn.functional.grid_sample(mask, vgrid)
        
    #     mask[mask<0.999] = 0
    #     mask[mask>0] = 1
        
    #     return output*mask

    def forward(self, pc1, pc2, iters=6, flow_init=None):
        """ Estimate optical flow between pair of frames """
        pillars1, coors_batch1, npoints_per_pillar1, pillar_center1 = self.pillarlayer(pc1)
        pillars2, coors_batch2, npoints_per_pillar2, _ = self.pillarlayer(pc2)
        pillar_feature1 = self.pillarencoder(pillars1, coors_batch1, npoints_per_pillar1)
        pillar_feature2 = self.pillarencoder(pillars2, coors_batch2, npoints_per_pillar2)
        # image1 = image1.permute(0, 3, 1, 2).contiguous()
        # image2 = image2.permute(0, 3, 1, 2).contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([pillar_feature1, pillar_feature2])        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.args.corr_levels ,radius=self.args.corr_radius)
        # run the context network
        cnet = self.cnet(pillar_feature1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(pillar_center1)
        b, _, h, w = coords0.shape
        logits = torch.zeros((b,3,h,w), device=pillar_feature1.device)
        
        if flow_init is not None:
            coords1 = coords1 + flow_init

        moving_predicts = []
        for itr in range(iters):
            # print("-------iter: ",itr,"-----------")
            coords1 = coords1.detach()
            logits = logits.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            # net, up_mask, delta_flow, delta_logits = self.update_block(net, inp, corr, flow, logits)
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            # logits = logits + delta_logits
            # logits = F.softmax(logits, dim=1)
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
                # logits_up = upflow8(logits)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                # logits_up = self.upsample_flow(logits)
            
            # add result to list
            # moving_predicts.append(F.softmax(logits_up, dim=1))
            
        return moving_predicts
    


    