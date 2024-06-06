# -*- coding:UTF-8 -*-

import os
import yaml
import argparse
import torch
import numpy as np
import torch.utils.data as data
from tools.points_process import aug_matrix


def load_poses(pose_path):
    """ Load ground truth poses (T_w_cam0) from file.
        Args:
            pose_path: (Complete) filename for the pose file
        Returns:
            A numpy array of size nx4x4 with n poses as 4x4 transformation
            matrices
    """
    # Read and parse the poses
    poses = []
    try:
        if '.txt' in pose_path:
            with open(pose_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)
        else:
            poses = np.load(pose_path)['arr_0']
    except FileNotFoundError:
        print('Ground truth poses:'+ pose_path +' are not avaialble.')
    return np.array(poses)

def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
        if isinstance(data, list):
            nel = len(data)
        else:
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


"""
     Reading data from KITTI

"""

class points_dataset(data.Dataset):

    def __init__(self, is_training: int = 1, num_point: int = 24000, data_dir_list: list = [0, 1, 2, 3, 4, 5, 6],
                 config: argparse.Namespace = None):
        """
        :param data_dir_list
        :param config
        """
        self.args = config
        data_dir_list.sort()
        self.is_training = is_training

        self.data_list = data_dir_list
        self.lidar_root = config.lidar_root
        self.data_len_sequence = [4540, 1100, 4660, 800, 270, 2760, 1100, 1100, 4070, 1590, 1200]

        Tr_tmp = []
        data_sum = [0]
        vel_to_cam_Tr = []

        with open('./tools/calib.yaml', "r") as f:
            con = yaml.load(f, Loader=yaml.FullLoader)
        for i in range(11):
            vel_to_cam_Tr.append(np.array(con['Tr{}'.format(i)]))
        for i in self.data_list:
            data_sum.append(data_sum[-1] + self.data_len_sequence[i] + 1)
            Tr_tmp.append(vel_to_cam_Tr[i])

        self.Tr_list = Tr_tmp
        self.data_sum = data_sum
        self.lidar_path = self.lidar_root
        # self.image_path = self.image_root

    def se3_transform(self, pose, xyz):
        """Apply rigid transformation to points

        Args:
            pose: ([B,] 3, 4)
            xyz: ([B,] N, 3)

        Returns:

        """

        assert xyz.shape[-1] == 3 and pose.shape[:-2] == xyz.shape[:-2]

        rot, trans = pose[..., :3, :3], pose[..., :3, 3:4]
        transformed = np.einsum('...ij,...bj->...bi', rot, xyz) + trans.transpose(-1, -2)  # Rx + t

        return transformed


    def __len__(self):
        return self.data_sum[-1]

    def __getitem__(self, index):

        sequence_str_list = []

        for item in self.data_list:
            sequence_str_list.append('{:02d}'.format(item))

        if index in self.data_sum:
            index_index = self.data_sum.index(index)
            index_ = 0
            fn1 = index_
            fn2 = index_
            sample_id = index_

        # data sequence
        else:
            index_index, data_begin, data_end = self.get_index(index, self.data_sum)
            index_ = index - data_begin
            fn1 = index_ - 1
            fn2 = index_
            sample_id = index_

        pose_path = 'pose/' + sequence_str_list[index_index] + '_diff.npy'
        pose = np.load(pose_path)
        lidar_path = os.path.join(self.lidar_path, sequence_str_list[index_index], 'velodyne')


        fn1_dir = os.path.join(lidar_path, '{:06d}.bin'.format(fn1))
        fn2_dir = os.path.join(lidar_path, '{:06d}.bin'.format(fn2))

        point1 = np.fromfile(fn1_dir, dtype=np.float32).reshape(-1, 4)
        point2 = np.fromfile(fn2_dir, dtype=np.float32).reshape(-1, 4)


        T_diff = pose[index_:index_ + 1, :]
        T_diff = T_diff.reshape(3, 4)
        filler = np.array([0.0, 0.0, 0.0, 1.0])
        filler = np.expand_dims(filler, axis=0)  #1*4
        T_diff_add = np.concatenate([T_diff, filler], axis=0)  # 4*4
        

        Tr = self.Tr_list[index_index]
        # Tr_inv = np.linalg.inv(Tr)
        # T_gt = np.matmul(Tr_inv, T_diff_add)
        # T_gt = np.matmul(T_gt, Tr)
        T_gt = T_diff_add

        # if self.is_training:
        #     T_trans = aug_matrix()
        # else:
        #     T_trans = np.eye(4).astype(np.float32)
        T_trans = np.eye(4).astype(np.float32)
        T_trans_inv = np.linalg.inv(T_trans)


        pos1 = point1[:, :3].astype(np.float32)
        pos2 = point2[:, :3].astype(np.float32)


        return  torch.from_numpy(pos2).float(), torch.from_numpy(pos1).float(), sample_id, T_gt, T_trans, T_trans_inv, Tr

    def get_index(self, value, mylist):
        mylist.sort()
        for i, num in enumerate(mylist):
            if num > value:
                return i - 1, mylist[i - 1], num


class semantic_points_dataset(data.Dataset):

    def __init__(self, is_training: int = 1, num_point: int = 24000, data_dir_list: list = [0, 1, 2, 3, 4, 5, 6],
                 config: argparse.Namespace = None):
        """

        :param data_dir_list
        :param config
        """
        self.args = config
        data_dir_list.sort()
        self.is_training = is_training

        self.data_list = data_dir_list
        self.lidar_root = config.lidar_root
        self.data_len_sequence = [4540, 1100, 4660, 800, 270, 1000, 1100, 1100, 4070, 1590, 1200]
        
        fw = open('./tools/dataset_config.yaml')
        dataset_config = yaml.load(fw, Loader=yaml.FullLoader)
        self.learning_map = dataset_config["learning_map"]

        Tr_tmp = []
        data_sum = [0]
        vel_to_cam_Tr = []

        with open('./tools/calib.yaml', "r") as f:
            con = yaml.load(f, Loader=yaml.FullLoader)
        for i in range(11):
            vel_to_cam_Tr.append(np.array(con['Tr{}'.format(i)]))
        for i in self.data_list:
            data_sum.append(data_sum[-1] + self.data_len_sequence[i] + 1)
            Tr_tmp.append(vel_to_cam_Tr[i])

        self.Tr_list = Tr_tmp
        self.data_sum = data_sum
        self.lidar_path = self.lidar_root

    def se3_transform(self, pose, xyz):
        """Apply rigid transformation to points

        Args:
            pose: ([B,] 3, 4)
            xyz: ([B,] N, 3)

        Returns:

        """
        assert xyz.shape[-1] == 3 and pose.shape[:-2] == xyz.shape[:-2]

        rot, trans = pose[..., :3, :3], pose[..., :3, 3:4]
        transformed = np.einsum('...ij,...bj->...bi', rot, xyz) + trans.transpose(-1, -2)  # Rx + t
        return transformed


    def __len__(self):
        return self.data_sum[-1]

    def __getitem__(self, index):

        sequence_str_list = []

        for item in self.data_list:
            sequence_str_list.append('{:02d}'.format(item))

        if index in self.data_sum:
            index_index = self.data_sum.index(index)
            index_ = 0
            fn1 = index_
            fn2 = index_
            sample_id = index_

        # data sequence
        else:
            index_index, data_begin, data_end = self.get_index(index, self.data_sum)
            index_ = index - data_begin
            fn1 = index_ - 1
            fn2 = index_
            sample_id = index_

        path_seq = int(sequence_str_list[index_index])
        pose_path = 'pose/' + sequence_str_list[index_index] + '_diff.npy'
        pose = np.load(pose_path)
        lidar_path = os.path.join(self.lidar_path, sequence_str_list[index_index], 'velodyne')
        label_path = os.path.join(self.lidar_path, sequence_str_list[index_index], 'labels')


        fn1_dir = os.path.join(lidar_path, '{:06d}.bin'.format(fn1))
        fn2_dir = os.path.join(lidar_path, '{:06d}.bin'.format(fn2))
        lb1_dir = os.path.join(label_path, '{:06d}.label'.format(fn1))

        point1 = np.fromfile(fn1_dir, dtype=np.float32).reshape(-1, 4)
        point2 = np.fromfile(fn2_dir, dtype=np.float32).reshape(-1, 4)
        label1 = np.fromfile(lb1_dir, dtype=np.int32).reshape((-1))& 0xFFFF
        label1 = map(label1, self.learning_map).reshape(-1, 1)
        
        T_diff = pose[index_:index_ + 1, :]
        T_diff = T_diff.reshape(3, 4)
        filler = np.array([0.0, 0.0, 0.0, 1.0])
        filler = np.expand_dims(filler, axis=0)  #1*4
        T_diff_add = np.concatenate([T_diff, filler], axis=0)  # 4*4
        T_gt = T_diff_add
        Tr = self.Tr_list[index_index]
        # Tr_inv = np.linalg.inv(Tr)
        # T_gt = np.matmul(Tr_inv, T_diff_add)
        # T_gt = np.matmul(T_gt, Tr)

        if self.is_training:
            T_trans = aug_matrix()
        else:
            T_trans = np.eye(4).astype(np.float32)

        T_trans_inv = np.linalg.inv(T_trans)

        pos1 = point1[:, :3].astype(np.float32)
        pos2 = point2[:, :3].astype(np.float32)

        return  torch.from_numpy(pos1).float(), \
                torch.from_numpy(pos2).float(), \
                torch.from_numpy(label1).float(),\
                path_seq, sample_id, T_gt, T_trans, T_trans_inv, Tr

    def get_index(self, value, mylist):
        mylist.sort()
        for i, num in enumerate(mylist):
            if num > value:
                return i - 1, mylist[i - 1], num

class semantic_school_points_dataset(data.Dataset):

    def __init__(self, is_training: int = 1, num_point: int = 24000, data_dir_list: list = [0, 1, 2, 3, 4, 5, 6],
                 config: argparse.Namespace = None):
        """

        :param data_dir_list
        :param config
        """
        self.args = config
        data_dir_list.sort()
        self.is_training = is_training

        self.data_list = data_dir_list
        self.lidar_root = config.lidar_root
        self.data_len_sequence = [5836,4814,3736,5448]
        
        fw = open('./tools/dataset_config_school.yaml')
        dataset_config = yaml.load(fw, Loader=yaml.FullLoader)
        self.learning_map = dataset_config["learning_map"]

        Tr_tmp = []
        data_sum = [0]
        vel_to_cam_Tr = []

        with open('./tools/calib.yaml', "r") as f:
            con = yaml.load(f, Loader=yaml.FullLoader)
        for i in range(11):
            vel_to_cam_Tr.append(np.array(con['Tr{}'.format(i)]))
        for i in self.data_list:
            data_sum.append(data_sum[-1] + self.data_len_sequence[i] + 1)
            Tr_tmp.append(vel_to_cam_Tr[i])

        self.Tr_list = Tr_tmp
        self.data_sum = data_sum
        self.lidar_path = self.lidar_root


    def __len__(self):
        return self.data_sum[-1]

    def __getitem__(self, index):

        sequence_str_list = []

        for item in self.data_list:
            sequence_str_list.append('{:02d}'.format(item))

        if index in self.data_sum:
            index_index = self.data_sum.index(index)
            index_ = 0
            fn1 = index_
            fn2 = index_
            sample_id = index_

        # data sequence
        else:
            index_index, data_begin, data_end = self.get_index(index, self.data_sum)
            index_ = index - data_begin
            fn1 = index_ - 1
            fn2 = index_
            sample_id = index_

        pose_path = os.path.join(self.lidar_path, sequence_str_list[index_index], "poses.txt")
        pose = load_poses(pose_path)
        lidar_path = os.path.join(self.lidar_path, sequence_str_list[index_index], 'velodyne')
        label_path = os.path.join(self.lidar_path, sequence_str_list[index_index], 'labels')

        fn1_dir = os.path.join(lidar_path, '{:06d}.bin'.format(fn1))
        fn2_dir = os.path.join(lidar_path, '{:06d}.bin'.format(fn2))
        lb1_dir = os.path.join(label_path, '{:06d}.label'.format(fn1))

        point1 = np.fromfile(fn1_dir, dtype=np.float32).reshape(-1, 4)
        point2 = np.fromfile(fn2_dir, dtype=np.float32).reshape(-1, 4)
        label1 = np.fromfile(lb1_dir, dtype=np.int32).reshape((-1))& 0xFFFF
        label1 = map(label1, self.learning_map).reshape(-1, 1)
        
        T_t1 = pose[fn1, :].reshape(4, 4)
        T_t2 = pose[fn2, :].reshape(4, 4)
        T_diff = np.matmul(np.linalg.inv(T_t1),T_t2)
        Tr = self.Tr_list[index_index]
        Tr_inv = np.linalg.inv(Tr)
        T_gt = T_diff

        if self.is_training:
            T_trans = aug_matrix()
        else:
            T_trans = np.eye(4).astype(np.float32)
        T_trans_inv = np.linalg.inv(T_trans)
        
        pos1 = point1[:, :3].astype(np.float32)
        pos2 = point2[:, :3].astype(np.float32)

        return  torch.from_numpy(pos1).float(), \
                torch.from_numpy(pos2).float(), \
                torch.from_numpy(label1).float(),\
                sample_id, T_gt, T_trans, T_trans_inv, Tr

    def get_index(self, value, mylist):
        mylist.sort()
        for i, num in enumerate(mylist):
            if num > value:
                return i - 1, mylist[i - 1], num

class tracking_dataset(data.Dataset):

    def __init__(self, is_training: int = 1, num_point: int = 24000, data_dir_list: list = [0, 1, 2, 3, 4, 5, 6],
                 config: argparse.Namespace = None):
        """
        :param data_dir_list
        :param config
        """
        self.args = config
        data_dir_list.sort()
        self.is_training = is_training

        self.data_list = data_dir_list
        self.lidar_root = config.lidar_root
        self.data_len_sequence = [153,442,232,143,313,296,269,799,389,802,293,372,77,339,105,375,208,144,338,1058,836]
        
        Tr_tmp = []
        data_sum = [0]
        vel_to_cam_Tr = []

        with open('./tools/calib_tracking.yaml', "r") as f:
            con = yaml.load(f, Loader=yaml.FullLoader)
        for i in range(21):
            vel_to_cam_Tr.append(np.array(con['Tr{}'.format(i)]))
        for i in self.data_list:
            data_sum.append(data_sum[-1] + self.data_len_sequence[i] + 1)
            Tr_tmp.append(vel_to_cam_Tr[i])

        self.Tr_list = Tr_tmp
        self.data_sum = data_sum
        self.lidar_path = self.lidar_root

    def se3_transform(self, pose, xyz):
        """Apply rigid transformation to points
        Args:
            pose: ([B,] 3, 4)
            xyz: ([B,] N, 3)
        Returns:
        """
        assert xyz.shape[-1] == 3 and pose.shape[:-2] == xyz.shape[:-2]
        rot, trans = pose[..., :3, :3], pose[..., :3, 3:4]
        transformed = np.einsum('...ij,...bj->...bi', rot, xyz) + trans.transpose(-1, -2)  # Rx + t
        return transformed

    def __len__(self):
        return self.data_sum[-1]

    def __getitem__(self, index):

        sequence_str_list = []

        for item in self.data_list:
            sequence_str_list.append('{:0>4d}'.format(item))

        if index in self.data_sum:
            index_index = self.data_sum.index(index)
            index_ = 0
            fn1 = index_
            fn2 = index_
            sample_id = index_

        # data sequence
        else:
            index_index, data_begin, data_end = self.get_index(index, self.data_sum)
            index_ = index - data_begin
            fn1 = index_ - 1
            fn2 = index_
            sample_id = index_
            
        pose_path = os.path.join(self.lidar_path, "pose", sequence_str_list[index_index],"pose.txt")
        pose = load_poses(pose_path)
        
        lidar_path = os.path.join(self.lidar_path, 'velodyne', sequence_str_list[index_index] )

        fn1_dir = os.path.join(lidar_path, '{:06d}.bin'.format(fn1))
        fn2_dir = os.path.join(lidar_path, '{:06d}.bin'.format(fn2))

        point1 = np.fromfile(fn1_dir, dtype=np.float32).reshape(-1, 4)
        point2 = np.fromfile(fn2_dir, dtype=np.float32).reshape(-1, 4)

        T_t1 = pose[fn1, :].reshape(4, 4)
        T_t2 = pose[fn2, :].reshape(4, 4)
        T_diff = np.matmul(np.linalg.inv(T_t1),T_t2)
        Tr = self.Tr_list[index_index]
        Tr_inv = np.linalg.inv(Tr)
        # T_gt = np.matmul(Tr, T_diff)
        # T_gt = np.matmul(T_gt, Tr_inv)
        T_gt = T_diff
        
        if self.is_training:
            T_trans = aug_matrix()
        else:
            T_trans = np.eye(4).astype(np.float32)

        T_trans_inv = np.linalg.inv(T_trans)

        pos1 = point1[:, :3].astype(np.float32)
        pos2 = point2[:, :3].astype(np.float32)

        return  torch.from_numpy(pos2).float(), torch.from_numpy(pos1).float(), sample_id, T_gt, T_trans, T_trans_inv, Tr

    def get_index(self, value, mylist):
        mylist.sort()
        for i, num in enumerate(mylist):
            if num > value:
                return i - 1, mylist[i - 1], num
            
class school_dataset(data.Dataset):

    def __init__(self, is_training: int = 1, num_point: int = 24000, data_dir_list: list = [0, 1, 2, 3, 4, 5, 6],
                 config: argparse.Namespace = None):
        """
        :param data_dir_list
        :param config
        """
        self.args = config
        data_dir_list.sort()
        self.is_training = is_training

        self.data_list = data_dir_list
        self.lidar_root = config.lidar_root
        self.data_len_sequence = [5836,4814,3736,5448]
        
        Tr_tmp = []
        data_sum = [0]
        vel_to_cam_Tr = []

        with open('./tools/calib_tracking.yaml', "r") as f:
            con = yaml.load(f, Loader=yaml.FullLoader)
        for i in range(21):
            vel_to_cam_Tr.append(np.array(con['Tr{}'.format(i)]))
        for i in self.data_list:
            data_sum.append(data_sum[-1] + self.data_len_sequence[i] + 1)
            Tr_tmp.append(vel_to_cam_Tr[i])

        self.Tr_list = Tr_tmp
        self.data_sum = data_sum
        self.lidar_path = self.lidar_root

    def __len__(self):
        return self.data_sum[-1]

    def __getitem__(self, index):

        sequence_str_list = []

        for item in self.data_list:
            sequence_str_list.append('{:0>2d}'.format(item))

        if index in self.data_sum:
            index_index = self.data_sum.index(index)
            index_ = 0
            fn1 = index_
            fn2 = index_
            sample_id = index_

        # data sequence
        else:
            index_index, data_begin, data_end = self.get_index(index, self.data_sum)
            index_ = index - data_begin
            fn1 = index_ - 1
            fn2 = index_
            sample_id = index_
            
        pose_path = os.path.join(self.lidar_path, sequence_str_list[index_index], "poses.txt")
        pose = load_poses(pose_path)
        
        lidar_path = os.path.join(self.lidar_path, sequence_str_list[index_index], 'velodyne')

        fn1_dir = os.path.join(lidar_path, '{:06d}.bin'.format(fn1))
        fn2_dir = os.path.join(lidar_path, '{:06d}.bin'.format(fn2))

        point1 = np.fromfile(fn1_dir, dtype=np.float32).reshape(-1, 4)
        point2 = np.fromfile(fn2_dir, dtype=np.float32).reshape(-1, 4)

        T_t1 = pose[fn1, :].reshape(4, 4)
        T_t2 = pose[fn2, :].reshape(4, 4)
        T_diff = np.matmul(np.linalg.inv(T_t1),T_t2)
        Tr = self.Tr_list[index_index]
        Tr_inv = np.linalg.inv(Tr)
        T_gt = T_diff
        
        if self.is_training:
            T_trans = aug_matrix()
        else:
            T_trans = np.eye(4).astype(np.float32)
        # T_trans = np.eye(4).astype(np.float32)

        T_trans_inv = np.linalg.inv(T_trans)

        pos1 = point1[:, :3].astype(np.float32)
        pos2 = point2[:, :3].astype(np.float32)

        return  torch.from_numpy(pos2).float(), torch.from_numpy(pos1).float(), sample_id, T_gt, T_trans, T_trans_inv, Tr

    def get_index(self, value, mylist):
        mylist.sort()
        for i, num in enumerate(mylist):
            if num > value:
                return i - 1, mylist[i - 1], num

if __name__ == '__main__':
    import argparse
    import tqdm
    from utils1.collate_functions import collate_pair, collate_pair_wo_label
    from configs import dynamic_seg_school_args, odometry_tracking_args
    global args
    # args = odometry_tracking_args()
    args = dynamic_seg_school_args()
    
    train_dir_list = [3]
    train_dataset = semantic_school_points_dataset(
        is_training = 1,
        num_point=150000,
        data_dir_list=train_dir_list,
        config=args
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_pair_wo_label,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )
    # total_unlabel_p = 0
    total_static_p = 0
    total_moving_p = 0
    # for i, data in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
    for id, data in enumerate(train_dataset):
        p1, p2, label1, sample_id, T_gt, T_trans, T_trans_inv, Tr = data
        ca, cb, cc = 0, 0, 0
        for i in label1:
            if i == 0:
                ca+=1
            if i == 1:
                cb+=1
            if i == 2:
                cc+=1
        # if cb == 0:
        #     print("no static:",sample_id)
        # if cc == 0:
        #     print("no dynamic:",sample_id)
        total_static_p+=cb
        total_moving_p+=cc
        # print("num of point: ", ca, cb, cc)
        # T_gt = torch.from_numpy(T_gt).float()
        # print(T_gt)
        # padp = torch.ones(p1.shape[0]).unsqueeze(1)
        # hom_pc1 = torch.cat([p1, padp], dim=1).transpose(0,1)
        # trans_pc1 = torch.mm(T_gt, hom_pc1).transpose(0,1)[:,:-1]
        # print(trans_pc1)
        # print("__________________")
        # break
        print(sample_id)
    print("s: ",total_static_p)
    print("m: ",total_moving_p)