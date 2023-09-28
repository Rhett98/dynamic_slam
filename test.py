import scipy.spatial
import torch
import torch.nn as nn
import torch.nn.functional as F
from translo_model_utils import ProjectPCimg2SphericalRing
from knn_cuda import KNN

class KDPointToPointLoss(nn.Module):
    def __init__(self):
        super(KDPointToPointLoss, self).__init__()
        self.lossMeanMSE = torch.nn.MSELoss()
        self.lossPointMSE = torch.nn.MSELoss(reduction="none")
        
    def find_target_correspondences(self, kd_tree_target, source_list_numpy):
        target_correspondence_indices = kd_tree_target[0].query(source_list_numpy[0])[1]
        return target_correspondence_indices

    def forward(self, source_point_cloud, target_point_cloud):
        # convert img to pointcloud
        B, _, _ = source_point_cloud.shape
        loss_list = []
        loss = torch.Tensor()
        for batch_index in range(B):
            batch_source_point_cloud = source_point_cloud[batch_index].view(1, 3, -1)
            batch_target_point_cloud = target_point_cloud[batch_index].view(1, 3, -1)
            # Build kd-tree
            target_kd_tree = [scipy.spatial.cKDTree(batch_target_point_cloud[0].permute(1, 0).detach().cpu().numpy())]

            # Find corresponding target points for all source points
            target_correspondences_of_source_points = \
                torch.from_numpy(self.find_target_correspondences(
                    kd_tree_target=target_kd_tree,
                    source_list_numpy=batch_source_point_cloud.permute(0, 2, 1).detach().cpu().numpy()))
            target_points = batch_target_point_cloud[:, :, target_correspondences_of_source_points]
            loss_list.append(self.lossMeanMSE(batch_source_point_cloud, target_points))
            # loss += self.lossMeanMSE(batch_source_point_cloud, target_points)
        return loss_list
    
    
class knnLoss(nn.Module):
    def __init__(self, k=5):
        super(knnLoss, self).__init__()
        self.knn = KNN(k, transpose_mode=True)
        
    def forward(self, source_pc, target_pc):
        dist, _ = self.knn(self.move_zero_point(target_pc), self.move_zero_point(source_pc))
        return torch.mean(dist)
    
    def move_zero_point(self, pc):
        x_coords = pc[:, :, 0]
        y_coords = pc[:, :, 1]
        z_coords = pc[:, :, 2]

        # 找到x、y、z坐标均不为0的点的索引
        valid_indices = (x_coords != 0.0) | (y_coords != 0.0) | (z_coords != 0.0)
        # 使用索引来筛选有效点
        filtered_point_cloud = pc[valid_indices]
        return filtered_point_cloud.unsqueeze(0)
    
if __name__ == '__main__':
    import numpy as np
    
    # pc1 = np.fromfile("demo_pc/velodyne/000000.bin", dtype=np.float32).reshape(4, -1)
    # pc2 = np.fromfile("demo_pc/velodyne/000001.bin", dtype=np.float32).reshape(4, -1)
    # pc1 = torch.from_numpy(pc1[:3, :].astype(np.float32)).float().unsqueeze(0)
    # pc2 = torch.from_numpy(pc2[:3, :].astype(np.float32)).float().unsqueeze(0)
    # # print(pc1.shape,pc2.shape)
    # loss_fn = KDPointToPointLoss()
    # print(loss_fn.forward(pc1,pc2))
    
    from kitti_pytorch import semantic_points_dataset
    from configs import translonet_args
    
    args = translonet_args()
    train_dir_list = [4]
    
    train_dataset = semantic_points_dataset(
        is_training = 1,
        num_point=args.num_points,
        data_dir_list=train_dir_list,
        config=args
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )
    loss_fn = knnLoss()

    for i, data in enumerate(train_loader, 0):
        pos2, pos1, label2, sample_id, T_gt, T_trans, T_trans_inv, Tr = data
        print("----------",sample_id,"--------------")
        if i==10:
            break
        # print("before:", loss_fn.forward(pos1.cuda(), pos2.cuda()))
        # T_trans = T_trans.to(torch.float32)
        # T_trans_inv = T_trans_inv.to(torch.float32)
        # # T_gt = T_gt.to(torch.float32)
        # T_gt = torch.linalg.inv(T_gt.to(torch.float32))
        # # print(T_gt)
        image1, _ = ProjectPCimg2SphericalRing(pos1.cuda(), None, 64, 1792)
        image2, _ = ProjectPCimg2SphericalRing(pos2.cuda(), None, 64, 1792)
        flat_img1 = image1.view(1, -1, 3)
        flat_img2 = image2.view(1, -1, 3)
        # print(flat_img1)
        print("img before:", loss_fn.forward(flat_img1, flat_img2))
        # # 利用变换矩阵T_gt将pos1转换到pos2
        # # print(pos1.transpose(1,2).shape)
        # padp = torch.ones([pos1.shape[0], 1, flat_img1.shape[2]])
        # hom_p1 = torch.cat([flat_img1, padp], dim=1)
        # # print(flat_img1)
        # # print(hom_p1)
        # trans_pos1 = torch.mm(T_gt[0], hom_p1[0])[:-1,:].unsqueeze(0)
        # # print("ps1 after transform:", loss_fn.forward(trans_pos1, pos1.transpose(1,2)))
        # print("img after:", loss_fn.forward(trans_pos1, flat_img2))
        # # print("after:", loss_fn.forward(trans_pos1, pos2.transpose(1,2)))
        # print(trans_pos1.shape, flat_img2.shape)

            