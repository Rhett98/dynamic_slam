# import numpy as np
# import torch
# from scipy.spatial import cKDTree

# # 准备两个点云数据集
# # point_cloud1 = np.random.rand(100, 3)  # 第一个点云，包含100个点，每个点有3个坐标
# # point_cloud2 = np.random.rand(150, 3)  # 第二个点云，包含150个点，每个点有3个坐标
# pc1 = np.fromfile("demo_pc/velodyne/000000.bin", dtype=np.float32).reshape(-1, 4)
# pc2 = np.fromfile("demo_pc/velodyne/000001.bin", dtype=np.float32).reshape(-1, 4)
# point_cloud1 = pc1[:, :3].astype(np.float32)
# point_cloud2 = pc2[:, :3].astype(np.float32)
# # 构建KNN树
# kdtree1 = cKDTree(point_cloud1)
# kdtree2 = cKDTree(point_cloud2)

# # 设置K值，即要计算的最近邻的数量
# K = 5

# # 计算每个点与其K个最近邻点之间的平均距离
# print(kdtree2.query(point_cloud1, k=K)[0])
# average_distance1 = np.mean(kdtree2.query(point_cloud1, k=K)[0])
# average_distance2 = np.mean(kdtree1.query(point_cloud2, k=K)[0])

# # 汇总损失，例如取平均值
# loss = 0.5 * (average_distance1 + average_distance2)

# print("平均距离损失:", loss)

import torch
import time
# Make sure your CUDA is available.
assert torch.cuda.is_available()

from knn_cuda import KNN
"""
if transpose_mode is True, 
    ref   is Tensor [bs x nr x dim]
    query is Tensor [bs x nq x dim]
    
    return 
        dist is Tensor [bs x nq x k]
        indx is Tensor [bs x nq x k]
else
    ref   is Tensor [bs x dim x nr]
    query is Tensor [bs x dim x nq]
    
    return 
        dist is Tensor [bs x k x nq]
        indx is Tensor [bs x k x nq]
"""

knn = KNN(k=1, transpose_mode=True)

ref = torch.randn(1, 20, 3).cuda()
query = torch.randn(1, 15, 3).cuda()
t1 = time.time()
dist, indx = knn(ref, query)  # 32 x 50 x 10
print("spend time:", time.time()- t1)
# print(dist)
print(indx)
ep_indx = indx.repeat(1,1,3)
print(ep_indx.shape)
print(ref)
print(torch.gather(ref, 1, ep_indx)) 


# def move_zero_point(pc):
#     x_coords = pc[:, 0]
#     y_coords = pc[:, 1]
#     z_coords = pc[:, 2]

#     # 找到x、y、z坐标均不为0的点的索引
#     valid_indices = (x_coords != 0.0) | (y_coords != 0.0) | (z_coords != 0.0)
#     # 使用索引来筛选有效点
#     filtered_point_cloud = pc[valid_indices]
#     return filtered_point_cloud

# if __name__ == '__main__':
#     pc1 = torch.randn((2,1000,3))
#     pc2 = torch.zeros((2,1000,3))
#     pc = torch.cat([pc1,pc2],dim=1)
#     print(move_zero_point(pc).shape)