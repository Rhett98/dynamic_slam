import torch
from raft.pillar_raft import RAFT
from thop import profile
model = RAFT()
input1 = [torch.randn(15000, 3)]
input2 = [torch.randn(15000, 3)]
input3 = [torch.randn(4, 4)]
flops, params = profile(model, inputs=(input1, input2 ,input3,))
print('flops:{}'.format(flops))
print('params:{}'.format(params))
