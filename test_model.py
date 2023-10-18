import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from raft.raft import RAFT


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--num_points', type=int, default=150000, help='Point Number [default: 2048]')

    parser.add_argument('--H_input', type=int, default=64, help='H Number [default: 64]')
    parser.add_argument('--W_input', type=int, default=1536, help='W Number [default: 1800]')
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
    # print(output[0])
