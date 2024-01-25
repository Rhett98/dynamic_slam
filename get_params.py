import torch
from raft.pillar_raft import RAFT
from configs import dynamic_seg_args
from thop import profile

args = dynamic_seg_args()
model = RAFT(args)
input1 = [torch.randn(15000, 3)]
input2 = [torch.randn(15000, 3)]
input3 = [torch.randn(4, 4)]
# flops, params = profile(model, inputs=([input1, input2 ,input3],))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchsummary import summary

summary(model)