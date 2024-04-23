import torch
from raft.extractor import MaskDecoder
def get_static_flow(pc1, Tr):
    '''
    pc1:(B,3,H,W)
    Tr:(B,4,4)
    return static_flow:(B,3,H,W)
    '''
    b,_,h,w = pc1.shape
    padp = torch.ones(b,1,h,w)
    hom_pc1 = torch.cat([pc1, padp],dim=1).reshape(b,4,h*w)
    trans_pc1 = torch.matmul(Tr, hom_pc1).reshape(b,4,h,w)[:,:3,:,:]
    static_flow = trans_pc1 - pc1
    return static_flow

def cal_dis_matrix(flow, flow_ststic):
    '''
    flow:(B,3,H,W)
    flow_ststic:(B,3,H,W)
    return dis:(B,3,H,W)
    '''
    delta_f = flow - flow_ststic
    dis = torch.sum(delta_f**2, dim=1)
    return dis

if __name__ == '__main__':
    pc1 = torch.randn([2,3,5,5])
    Tr = torch.tensor([[[1,0,0,10],[0,1,0,10],[0,0,1,10],[0,0,0,1]],[[1,0,0,8],[0,1,0,8],[0,0,1,8],[0,0,0,1]]]).float()
    f_s = get_static_flow(pc1, Tr)
    # print(f_s)
    flow = f_s[0].unsqueeze(0)
    flow_s = f_s[1].unsqueeze(0)
    dis = cal_dis_matrix(flow, flow_s)
    print(dis)
    
    # a = torch.randn([1,1,10,10])
    # net = MaskDecoder()
    # b = net(a)
    # print(b)
    
    