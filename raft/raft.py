import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from raft.update import BasicUpdateBlock, SmallUpdateBlock
from raft.extractor import BasicEncoder, SmallEncoder
from raft.corr import CorrBlock, AlternateCorrBlock
from raft.utils.utils import bilinear_sampler, coords_grid, upflow8
from translo_model_utils import ProjectPCimg2SphericalRing

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
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 3
        args.corr_radius = 3
        self.H_input, self.W_input = args.H_input, args.W_input

        if 'dropout' not in self.args:
            self.args.dropout = 0

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        self.predict_moving_net = nn.Conv2d(2, 3, 1)

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


    def forward(self, pc1, pc2, iters=8, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        image1, mask_proj1 = ProjectPCimg2SphericalRing(pc1, None, self.H_input, self.W_input)
        image2, mask_proj2 = ProjectPCimg2SphericalRing(pc2, None, self.H_input, self.W_input)
        image1 = image1.permute(0, 3, 1, 2).contiguous()
        image2 = image2.permute(0, 3, 1, 2).contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])        
        # print("fmap1:",fmap1.shape)
        # print("fmap2:",fmap2)
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        
        # corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        corr_fn = CorrBlock(fmap1, fmap2,num_levels=self.args.corr_levels ,radius=self.args.corr_radius)

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        
        coords0, coords1 = self.initialize_flow(image1)
        # print("coords0:",coords0.shape)
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        moving_predict = []
        for itr in range(iters):
            # print("-------iter: ",itr,"-----------")
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0

            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)
            m_feature = self.predict_moving_net(flow_up)
            moving_predict.append(F.softmax(m_feature, dim=1))

        if test_mode:
            return coords1 - coords0, flow_up
            
        return moving_predict#flow_predictions
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()
    
    pc1 = np.fromfile("demo_pc/velodyne/000000.bin", dtype=np.float32).reshape(-1, 4)
    pc2 = np.fromfile("demo_pc/velodyne/000001.bin", dtype=np.float32).reshape(-1, 4)
    pc1 = torch.from_numpy(pc1[:, :3].astype(np.float32)).float().unsqueeze(0).cuda()
    pc2 = torch.from_numpy(pc2[:, :3].astype(np.float32)).float().unsqueeze(0).cuda()

    # pc1, pc2 = torch.randn(1,50000,3).cuda(), torch.randn(1,50000,3).cuda()
    # proj1, mask_proj1 = ProjectPCimg2SphericalRing(pc1, None, 64, 1792)
    # print(proj1.shape, mask_proj1.shape)
    model = RAFT(args).cuda()
    output = model(pc1, pc2)
    print(output[0])
    print(output[0].shape)