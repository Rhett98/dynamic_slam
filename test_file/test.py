from scipy.spatial import cKDTree
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import ProjectPCimg2SphericalRing
# from knn_cuda import KNN
import numpy as np
import time
from knn_cuda import KNN

# class KDPointToPointLoss(nn.Module):
#     def __init__(self):
#         super(KDPointToPointLoss, self).__init__()
#         self.lossMeanMSE = torch.nn.MSELoss()
#         self.lossPointMSE = torch.nn.MSELoss(reduction="none")
        
#     def find_target_correspondences(self, kd_tree_target, source_list_numpy):
#         target_correspondence_indices = kd_tree_target[0].query(source_list_numpy)[1]
#         return target_correspondence_indices

#     def move_zero_point(self, pc):
#         x_coords = pc[:, 0]
#         y_coords = pc[:, 1]
#         z_coords = pc[:, 2]

#         # 找到x、y、z坐标不全为0的点的索引
#         valid_indices = (x_coords != 0.0) | (y_coords != 0.0) | (z_coords != 0.0)
#         # 使用索引来筛选有效点
#         filtered_point_cloud = pc[valid_indices]
#         return filtered_point_cloud
    
#     def forward(self, source_point_cloud, target_point_cloud):
#         # convert img to pointcloud
#         B, _, _ = source_point_cloud.shape
#         loss = torch.zeros((B))
#         for batch_index in range(B):
#             # batch_source_point_cloud = self.move_zero_point(source_point_cloud[batch_index].contiguous().view(-1, 3))
#             # batch_target_point_cloud = self.move_zero_point(target_point_cloud[batch_index].contiguous().view(-1, 3))
#             batch_source_point_cloud = source_point_cloud[batch_index].contiguous().view(-1, 3)
#             batch_target_point_cloud = target_point_cloud[batch_index].contiguous().view(-1, 3)
#             # print(batch_source_point_cloud.shape, batch_target_point_cloud.shape)
#             # Build kd-tree
#             target_kd_tree = [cKDTree(batch_target_point_cloud.detach().cpu().numpy())]
#             # Find corresponding target points for all source points
#             target_correspondences_of_source_points = \
#                 torch.from_numpy(self.find_target_correspondences(
#                     kd_tree_target=target_kd_tree, 
#                     source_list_numpy=batch_source_point_cloud.detach().cpu().numpy()))
#             target_points = batch_target_point_cloud[target_correspondences_of_source_points, :]
#             loss[batch_index] = self.lossMeanMSE(batch_source_point_cloud, target_points)
#         return torch.mean(loss)
       
# class knnLoss(nn.Module):
#     def __init__(self, k=1):
#         super(knnLoss, self).__init__()
#         self.knn = KNN(k, transpose_mode=True)
#         self.lossPointMSE = torch.nn.MSELoss()

#     def forward(self, source_pc, target_pc):
#         source_pc = self.move_zero_point(source_pc)
#         target_pc = self.move_zero_point(target_pc)
#         indxs = self.get_idx_dists_for_knn(source_pc, target_pc)
#         bsize = len(indxs)
#         total_loss = torch.zeros(bsize)
#         for batch_index in range(bsize):
#             nearest_source_pc = torch.gather(source_pc[batch_index], dim=0, index=indxs[batch_index][0].repeat(1,3))
#             loss = self.lossPointMSE(nearest_source_pc.unsqueeze(0), target_pc[batch_index].unsqueeze(0))
#             total_loss[batch_index] = loss
#         return torch.mean(total_loss)
        
#     def get_idx_dists_for_knn(self, source_pc, target_pc):
#         bsize = len(source_pc)
#         indices = []
#         for batch_index in range(bsize):
#             _, indice = self.knn(source_pc[batch_index].unsqueeze(0), \
#                                 target_pc[batch_index].unsqueeze(0))
#             indices.append(indice)
#         return indices
    
#     def move_zero_point(self, pc):
#         bsize = pc.shape[0]
#         filtered_point_cloud = []
#         for batch_index in range(bsize):
#             # x_coords = pc[batch_index, :, 0]
#             # y_coords = pc[batch_index, :, 1]
#             # z_coords = pc[batch_index, :, 2]

#             # # 找到x、y、z坐标均不为0的点的索引
#             # valid_indices = (x_coords != 0.0) | (y_coords != 0.0) | (z_coords != 0.0)
#             # # 使用索引来筛选有效点
#             # filtered_point_cloud.append(pc[batch_index, valid_indices])
#             filtered_point_cloud.append(pc[batch_index])

#         return filtered_point_cloud
    

# def get_downsample_pc(pc, batch_size, out_H: int, out_W: int, stride_H: int, stride_W: int):
#     """According to given stride and output size, return the corresponding selected points

#     Args:
#         array (tf.Tensor): [any array with shape (B, H, W, 3)]
#         stride_H (int): [stride in height]
#         stride_W (int): [stride in width]
#         out_H (int): [height of output array]
#         out_W (int): [width of output array]
#     Returns:
#         Tensor: (B, outh, outw, 3) 
#     """
#     select_h_idx = torch.arange(0, out_H * stride_H, stride_H)
#     select_w_idx = torch.arange(0, out_W * stride_W, stride_W)
#     height_indices = (torch.reshape(select_h_idx, (1, -1, 1))).expand(batch_size, out_H, out_W)         # b out_H out_W
#     width_indices = (torch.reshape(select_w_idx, (1, 1, -1))).expand(batch_size, out_H, out_W)            # b out_H out_W
#     padding_indices = torch.reshape(torch.arange(batch_size), (-1, 1, 1)).expand(batch_size, out_H, out_W)   # b out_H out_W
#     downsample_xyz_proj = pc[padding_indices, height_indices, width_indices, :]
    
#     return downsample_xyz_proj


# def move_zero_point(pc):
#     x_coords = pc[:, 0]
#     y_coords = pc[:, 1]
#     z_coords = pc[:, 2]

#     # 找到x、y、z坐标均不为0的点的索引
#     valid_indices = (x_coords != 0.0) | (y_coords != 0.0) | (z_coords != 0.0)
#     # 使用索引来筛选有效点
#     filtered_point_cloud = pc[valid_indices]
#     print(filtered_point_cloud.shape)
#     return filtered_point_cloud
    
# if __name__ == '__main__':
    
#     # pc1 = np.fromfile("demo_pc/velodyne/000000.bin", dtype=np.float32).reshape(4, -1)
#     # pc2 = np.fromfile("demo_pc/velodyne/000001.bin", dtype=np.float32).reshape(4, -1)
#     # pc1 = torch.from_numpy(pc1[:3, :].astype(np.float32)).float().unsqueeze(0)
#     # pc2 = torch.from_numpy(pc2[:3, :].astype(np.float32)).float().unsqueeze(0)
#     # print(pc1.shape,pc2.shape)
#     # t1 = time.time()
#     # loss_fn = KDPointToPointLoss()
#     # print(loss_fn.forward(pc1,pc2), "spend:",time.time()-t1, "s")
#     # t2 = time.time()
#     # # print(pc1.squeeze(0).transpose(0,1).shape)
#     # dist = knn.knn_loss(pc1.squeeze(0).transpose(0,1),pc2.squeeze(0).transpose(0,1),3)
#     # print(dist, pos2.cuda()"spend:",time.time()-t2, "s")
    
#     # loss_fn = knnLoss()
#     # print(loss_fn.forward(pc1.cuda(),pc2.cuda()))
    
#     from kitti_pytorch import semantic_points_dataset
#     from configs import translonet_args
    
#     args = translonet_args()
#     train_dir_list = [1]
    
#     train_dataset = semantic_points_dataset(
#         is_training = 1,
#         num_point=args.num_points,
#         data_dir_list=train_dir_list,
#         config=args
#     )
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=args.workers,
#         pin_memory=True,
#         drop_last=True,
#         worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
#     )
#     # loss_fn = KDPointToPointLoss()
#     loss_fn = knnLoss(k=1)

#     for i, data in enumerate(train_loader, 0):
#         pos2, pos1, label2, sample_id, T_gt, T_trans, T_trans_inv, Tr = data
#         print("----------",sample_id,"--------------")
#         if i==15:
#             break
#         # print("before:", loss_fn.forward(pos2.cuda(), pos1.cuda()))

#         T_inv = torch.linalg.inv(T_gt.to(torch.float32)).cuda()
#         # print(T_gt)
        
#         # # 利用变换矩阵T_gt将pos1转换到pos2
#         # padp = torch.ones([pos1.shape[0], pos1.shape[1], 1])
#         # hom_p1 = torch.cat([pos1.cpu(), padp], dim=2)
#         # trans_pos1 = torch.mm(T_gt[0].float(), hom_p1[0].transpose(1,0).float())[:-1,:].unsqueeze(0)
#         # print("trans after:", loss_fn.forward(pos2, trans_pos1.permute(0,2,1)))
        
#         ###############################################################################
#         image1, img1_mask = ProjectPCimg2SphericalRing(pos1.cuda(), None, 64, 1536)
#         image2, img2_mask = ProjectPCimg2SphericalRing(pos2.cuda(), None, 64, 1536)
#         image1 = get_downsample_pc(image1,1,32,512,2,3)
#         image2 = get_downsample_pc(image2,1,32,512,2,3)
#         # print(img1_mask.shape)
#         # print(image1_)
#         flat_img1 = image1.view(1, -1, 3)
#         flat_img2 = image2.view(1, -1, 3)
#         t1 = time.time()
#         print("img before:", loss_fn.forward(flat_img2, flat_img1))
#         print("spend time:", time.time()- t1)
#         # 利用变换矩阵T_gt将pos1转换到pos2
#         # ex_img2 = torch.zeros_like(flat_img1)
#         flat_img1 = move_zero_point(flat_img1.squeeze()).unsqueeze(0)
#         padp = torch.ones([flat_img1.shape[0], flat_img1.shape[1], 1]).cuda()
#         hom_p1 = torch.cat([flat_img1, padp], dim=2)
#         trans_pos1 = torch.mm(T_inv[0].float().cuda(), hom_p1[0].transpose(1,0).float())[:-1,:].unsqueeze(0)
#         # homogeneous_points = torch.cat((flat_img1, torch.ones((flat_img1.shape[0], flat_img1.shape[1], 1), device=flat_img1.device)), dim=2)

#         # # 使用矩阵乘法计算变换后的坐标
#         # transformed_points = torch.matmul(homogeneous_points.float(), T_gt.permute(0, 2, 1).cuda().float())  # 注意需要转置变换矩阵

#         # # 从变换后的坐标中提取前三列作为三维点坐标
#         # trans_pos1 = transformed_points[:, :, :3]

#         # print(T_gt[0].float())
#         # print(trans_pos1.shape)
#         t2 = time.time()
#         print("img after:", loss_fn.forward(flat_img2, trans_pos1.transpose(2,1)))
#         print("spend time:", time.time()- t2)
from torch.autograd import Variable
def warp(x, flo):
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
    output = F.grid_sample(x, vgrid, align_corners=False)
    
    mask = torch.autograd.Variable(torch.ones(x.size()))
    mask = F.grid_sample(mask, vgrid, align_corners=False)
    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    return output*mask

def warp_img(img, flow):
    flow = flow.permute(0,2,3,1)
    return F.grid_sample(img, flow, align_corners=False)

# def warp_img(img, flow):
#     """ Warp img [B, 3, H, W] with flow [B, 2, H, W]"""
#         # 获取输入张量的尺寸
#     B, _, H, W = img.size()

#     # 创建目标张量，初始化为零
#     warped_img = torch.zeros_like(img)

#     # 生成采样网格
#     grid_x, grid_y = torch.meshgrid(torch.arange(H, device=img.device), torch.arange(W, device=img.device))
#     grid_x = grid_x.float()
#     grid_y = grid_y.float()

#     # 添加光流位移
#     grid_x = grid_x.unsqueeze(0).expand(B, -1, -1).contiguous() + flow[:, 0, :, :]
#     grid_y = grid_y.unsqueeze(0).expand(B, -1, -1).contiguous() + flow[:, 1, :, :]

#     # 归一化到[-1, 1]范围
#     grid_x = 2.0 * (grid_x / (H - 1)) - 1.0
#     grid_y = 2.0 * (grid_y / (W - 1)) - 1.0
#     print(grid_x,grid_y)
#     # 扩展维度以匹配grid的维度
#     grid_x = grid_x.unsqueeze(3)
#     grid_y = grid_y.unsqueeze(3)
#     print(grid_x.shape,grid_y.shape)
#     # 使用grid_sample进行双线性采样
#     grid = torch.cat((grid_x, grid_y), dim=3)
#     print("grid:", grid)
#     warped_img = F.grid_sample(img, grid, align_corners=True)

#     return warped_img

if __name__ == '__main__':
    img = torch.randn((1,3,2,2))
    flow = torch.ones((1,2,2,2))*1
    print(img.transpose(1,3))
    # print(flow.transpose(1,3))
    print(warp(img,flow).transpose(1,3))