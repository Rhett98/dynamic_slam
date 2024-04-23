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
from model_utils import ProjectPCimg2SphericalRing

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

        if 'dropout' not in self.args:
            self.args.dropout = 0

        # feature network, context network, and update block
        self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
        self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        # self.predict_moving_net = nn.Sequential(nn.Conv2d(2, 9, 3, padding=1),
        #                                         nn.Conv2d(9, 3, 3, padding=1),)
        
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

    # def warp_img(self, img, flow):
    #     """ Warp img [B, 3, H, W] with flow [B, 2, H, W]"""
    #         # 获取输入张量的尺寸
    #     B, _, H, W = img.size()

    #     # 创建目标张量，初始化为零
    #     warped_img = torch.zeros_like(img).cuda()

    #     # 生成采样网格
    #     grid_x, grid_y = torch.meshgrid(torch.arange(H), torch.arange(W))
    #     grid_x = grid_x.float().cuda()
    #     grid_y = grid_y.float().cuda()

    #     # 添加光流位移
    #     grid_x = grid_x.unsqueeze(0).expand(B, -1, -1).contiguous()
    #     grid_y = grid_y.unsqueeze(0).expand(B, -1, -1).contiguous()

    #     grid_x += flow[:, 0, :, :]
    #     grid_y += flow[:, 1, :, :]

    #     # 归一化到[-1, 1]范围
    #     grid_x = 2.0 * (grid_x / (H - 1)) - 1.0
    #     grid_y = 2.0 * (grid_y / (W - 1)) - 1.0

    #     # 扩展维度以匹配grid的维度
    #     grid_x = grid_x.unsqueeze(3)
    #     grid_y = grid_y.unsqueeze(3)

    #     # 使用grid_sample进行双线性采样
    #     grid = torch.cat((grid_x, grid_y), dim=3)
    #     warped_img = F.grid_sample(img, grid, padding_mode='border', align_corners=True)

    #     return warped_img
    def warp_img(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)
        
        mask[mask<0.999] = 0
        mask[mask>0] = 1
        
        return output*mask

    def forward(self, pc1, pc2, iters=6, flow_init=None):
        """ Estimate optical flow between pair of frames """
        image1, _ = ProjectPCimg2SphericalRing(pc1, None, self.H_input, self.W_input)
        image2, _ = ProjectPCimg2SphericalRing(pc2, None, self.H_input, self.W_input)
        image1 = image1.permute(0, 3, 1, 2).contiguous()
        image2 = image2.permute(0, 3, 1, 2).contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        
        # corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.args.corr_levels ,radius=self.args.corr_radius)
        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)
        b, _, h, w = coords0.shape
        logits = torch.zeros((b,3,h,w), device=image1.device)
        
        if flow_init is not None:
            coords1 = coords1 + flow_init
        # print(coords1.permute(0, 2, 3, 1))
        warp_image1s = []
        moving_predicts = []
        for itr in range(iters):
            # print("-------iter: ",itr,"-----------")
            coords1 = coords1.detach()
            logits = logits.detach()
            corr = corr_fn(coords1) # index correlation volume
            # print(corr.shape)
            flow = coords1 - coords0
            net, up_mask, delta_flow, delta_logits = self.update_block(net, inp, corr, flow, logits)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            # print(coords1.permute(0, 2, 3, 1))
            logits = logits + delta_logits
            # logits = F.softmax(logits, dim=1)
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
                logits_up = upflow8(logits)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            warp_image1 = self.warp_img(image1, flow_up)
            # print(flow_up)
            # print(warp_image1.permute(0, 2, 3, 1))
            # add result to list
            warp_image1s.append(warp_image1)
            moving_predicts.append(F.softmax(logits_up, dim=1))
            
        return warp_image1s, moving_predicts
    


    