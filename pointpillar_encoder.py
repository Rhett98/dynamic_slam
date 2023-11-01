import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.voxel_module import Voxelization
import numpy as np
import matplotlib.pyplot as plt

class PillarLayer(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels):
        super().__init__()
        self.voxel_layer = Voxelization(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

    @torch.no_grad()
    def forward(self, batched_pts, batch_labels=None):
        '''
        batched_pts: list[tensor], len(batched_pts) = bs
        return: 
               pillars: (p1 + p2 + ... + pb, num_points, c), True
               num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        '''
        pillars, coors, npoints_per_pillar = [], [], []
        for i, pts in enumerate(batched_pts):
            if batch_labels is not None:
                # print(pts.shape,batch_labels[i].unsqueeze(1).shape)
                voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(torch.cat([pts,batch_labels[i]],dim=1).contiguous())
                # voxels_out: (max_voxel, num_points, c+1) 
            else:
                voxels_out, coors_out, num_points_per_voxel_out = self.voxel_layer(pts.contiguous()) 
            # voxels_out: (max_voxel, num_points, c) 
            # coors_out: (max_voxel, 3), num_points_per_voxel_out: (max_voxel, )
            pillars.append(voxels_out)
            coors.append(coors_out.long())
            npoints_per_pillar.append(num_points_per_voxel_out)
        
        pillars = torch.cat(pillars, dim=0) # (p1 + p2 + ... + pb, num_points, c)
        npoints_per_pillar = torch.cat(npoints_per_pillar, dim=0) # (p1 + p2 + ... + pb, )
        coors_batch = []
        for i, cur_coors in enumerate(coors):
            coors_batch.append(F.pad(cur_coors, (1, 0), value=i))
        coors_batch = torch.cat(coors_batch, dim=0) # (p1 + p2 + ... + pb, 1 + 3)
        # calculate center point & label
        center_pt = torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None] # (p1 + p2 + ... + pb, 1, 3)
        batched_pillar_center = []
        if batch_labels is not None:
            pillar_label = pillars[:, :, 3]
            pillar_label_max,_ = torch.mode(pillar_label.int(), dim=1)
            batched_pillar_label = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            # center pillar scatter
            curr_pillar_center = center_pt.squeeze()[cur_coors_idx]
            pillar_center = torch.zeros((self.x_l, self.y_l, 3), dtype=torch.float32, device=pillars.device)
            pillar_center[cur_coors[:, 1], cur_coors[:, 2]] = curr_pillar_center
            pillar_center = pillar_center.permute(2, 1, 0).contiguous()
            batched_pillar_center.append(pillar_center)
            if batch_labels is not None:
                # center pillar scatter
                curr_pillar_label = pillar_label_max.unsqueeze(1)[cur_coors_idx]
                pillar_label = torch.zeros((self.x_l, self.y_l, 1), dtype=torch.int32, device=pillars.device)
                pillar_label[cur_coors[:, 1], cur_coors[:, 2]] = curr_pillar_label
                pillar_label = pillar_label.permute(2, 1, 0).contiguous()
                batched_pillar_label.append(pillar_label)
        # batch stack
        if batch_labels is not None:
            batched_pillar_label = torch.stack(batched_pillar_label, dim=0) # (bs, 1, self.y_l, self.x_l)
            batched_pillar_center = torch.stack(batched_pillar_center, dim=0) # (bs, 3, self.y_l, self.x_l)
            return pillars, coors_batch, npoints_per_pillar, batched_pillar_center, batched_pillar_label
        batched_pillar_center = torch.stack(batched_pillar_center, dim=0) # (bs, 3, self.y_l, self.x_l)
        return pillars, coors_batch, npoints_per_pillar, batched_pillar_center


class PillarEncoder(nn.Module):
    def __init__(self, voxel_size, point_cloud_range, in_channel, out_channel):
        super().__init__()
        self.out_channel = out_channel
        self.vx, self.vy = voxel_size[0], voxel_size[1]
        self.x_offset = voxel_size[0] / 2 + point_cloud_range[0]
        self.y_offset = voxel_size[1] / 2 + point_cloud_range[1]
        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

        self.conv = nn.Conv1d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channel, eps=1e-3, momentum=0.01)

    def forward(self, pillars, coors_batch, npoints_per_pillar):
        '''
        pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        npoints_per_pillar: (p1 + p2 + ... + pb, )
        return:  (bs, out_channel, y_l, x_l)
        '''
        device = pillars.device
        # 1. calculate offset to the points center (in each pillar)
        center_pt = torch.sum(pillars[:, :, :3], dim=1, keepdim=True) / npoints_per_pillar[:, None, None] # (p1 + p2 + ... + pb, 1, 3)
        offset_pt_center = pillars[:, :, :3] - center_pt  # (p1 + p2 + ... + pb, num_points, 3)
        # 2. calculate offset to the pillar center
        x_offset_pi_center = pillars[:, :, :1] - (coors_batch[:, None, 1:2] * self.vx + self.x_offset) # (p1 + p2 + ... + pb, num_points, 1)
        y_offset_pi_center = pillars[:, :, 1:2] - (coors_batch[:, None, 2:3] * self.vy + self.y_offset) # (p1 + p2 + ... + pb, num_points, 1)

        # 3. encoder
        features = torch.cat([pillars[:, :, :4], offset_pt_center, x_offset_pi_center, y_offset_pi_center], dim=-1) # (p1 + p2 + ... + pb, num_points, 9)
        features[:, :, 0:1] = x_offset_pi_center # tmp
        features[:, :, 1:2] = y_offset_pi_center # tmp

        # 4. find mask for (0, 0, 0) and update the encoded features
        voxel_ids = torch.arange(0, pillars.size(1)).to(device) # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :] # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        features *= mask[:, :, None]

        # 5. embedding
        features = features.permute(0, 2, 1).contiguous() # (p1 + p2 + ... + pb, 9, num_points)
        features = F.relu(self.bn(self.conv(features)))  # (p1 + p2 + ... + pb, out_channels, num_points)
        pooling_features = torch.max(features, dim=-1)[0] # (p1 + p2 + ... + pb, out_channels)
        
        # 6. pillar scatter
        batched_pillar_feature = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            # pillar feature scatter
            cur_features = pooling_features[cur_coors_idx]
            pillar_feature = torch.zeros((self.x_l, self.y_l, self.out_channel), dtype=torch.float32, device=device)
            pillar_feature[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            pillar_feature = pillar_feature.permute(2, 1, 0).contiguous()
            batched_pillar_feature.append(pillar_feature)
        # batch stack
        batched_pillar_feature = torch.stack(batched_pillar_feature, dim=0) # (bs, in_channel, self.y_l, self.x_l)
        return batched_pillar_feature
    

def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:            # for i in range(pillar_label_max.shape[0]):
            #     print(pillar_label_max[i])
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        # do the mapping
        return lut[label]


def vis_label(input_tensor):
    """tensor:(1,H,W)"""
    _,h,w = input_tensor.shape
    output_tensor = torch.zeros((3,h,w))
    non_label_index = torch.where(input_tensor == 0) 
    # print(non_label_index[0].shape,non_label_index[1].shape,non_label_index[2].shape) 
    # print(output_tensor.shape)
    # static_index = torch.where(input_tensor == 1)  
    moving_index = torch.where(input_tensor == 2)  
    # 使用索引tensor将对应的RGB颜色数值赋值给输出tensor  
    # white
    output_tensor[0,:,:].index_put_(non_label_index[1:], torch.tensor([1.0])) 
    output_tensor[1,:,:].index_put_(non_label_index[1:], torch.tensor([1.0])) 
    output_tensor[2,:,:].index_put_(non_label_index[1:], torch.tensor([1.0]))    
    # red
    output_tensor[0,:,:].index_put_(moving_index[1:], torch.tensor([1.0]))

    img = np.transpose(output_tensor, (1,2,0))# [H, W, C]
    plt.imshow(img)
    plt.pause(0.1)



if __name__ == '__main__':
    import yaml
    from utils1.collate_functions import collate_pair
    voxel_size=[0.2, 0.2, 9]
    point_cloud_range=[-40, -40, -3, 40, 40, 6]#[-48,-48,-3,48,48,6]
    max_num_points=4
    max_voxels=(16000, 40000)
    # batch_pts = []
    # batch_labels = []
    # dataset_config = yaml.load(open('dataset_config.yaml'), Loader=yaml.FullLoader)
    # learning_map = dataset_config["learning_map"]
    # pc1 = np.fromfile("demo_pc/velodyne/000000.bin", dtype=np.float32).reshape(-1,4)
    # pc1 = torch.from_numpy(pc1[:,:4].astype(np.float32)).float()
    # pc2 = np.fromfile("demo_pc/velodyne/000001.bin", dtype=np.float32).reshape(-1,4)
    # pc2 = torch.from_numpy(pc2[:,:4].astype(np.float32)).float()
    
    # label1 = np.fromfile("demo_pc/labels/000000.label", dtype=np.int32).reshape((-1))& 0xFFFF
    # label2 = np.fromfile("demo_pc/labels/000001.label", dtype=np.int32).reshape((-1))& 0xFFFF
    # label1 = map(label1, learning_map).reshape(-1, 1)
    # label2 = map(label2, learning_map).reshape(-1, 1)
    # label1 = torch.from_numpy(label1.astype(np.float32)).float()
    # label2 = torch.from_numpy(label2.astype(np.float32)).float()
    
    # batch_pts.append(pc1)
    # batch_labels.append(label1)
    # batch_labels.append(label2)
    # batch_pts.append(pc2)
    layer = PillarLayer(voxel_size,point_cloud_range,max_num_points,max_voxels)
    de = PillarEncoder(voxel_size,point_cloud_range,9,64)
    # pillars, coors_batch, npoints_per_pillar, pillar_center, pillar_label= layer(batch_pts, batch_labels)
    # print(pillars.shape, coors_batch.shape, npoints_per_pillar.shape)
    # feature = de(pillars, coors_batch, npoints_per_pillar)
    # print(feature.shape)
    # print(pillar_center.permute(0,2,3,1))   
    # print(pillar_label.permute(0,2,3,1)) 
    # print(pillar_center.shape, pillar_label.shape)  
    # vis_label(pillar_label[1])
    
    
    from kitti_pytorch import semantic_points_dataset
    from configs import pillar_raftnet_args
    
    args = pillar_raftnet_args()
    train_dir_list = [7]
    
    train_dataset = semantic_points_dataset(
        is_training = 1,
        num_point=args.num_points,
        data_dir_list=train_dir_list,
        config=args
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_pair,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )
    # plt.ion()
    for i, data in enumerate(train_loader, 0):
        pos2, pos1, label2, sample_id, T_gt, T_trans, T_trans_inv, Tr = data
        _,_,_, pillar_center, pillar_label= layer(pos2, label2)
        # print(pillar_center.shape, pillar_label.shape)  
        vis_label(pillar_label[0])
    