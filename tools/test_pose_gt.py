import os
import yaml
import argparse
import numpy as np

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


pose_path = '/home/yu/Resp/dynamic_slam/pose/01_diff.npy'
pose = np.load(pose_path)
T_diff = pose[2, :]
T_diff = T_diff.reshape(3, 4)
filler = np.array([0.0, 0.0, 0.0, 1.0])
filler = np.expand_dims(filler, axis=0)  #1*4
T_diff_add = np.concatenate([T_diff, filler], axis=0)  # 4*4
print("from diff.npy:",T_diff_add)

pose_path = "/home/yu/Resp/dynamic_slam/ground_truth_pose/01.txt"
pose = load_poses(pose_path)
T_t1 = pose[1, :].reshape(4, 4)
T_t2 = pose[2, :].reshape(4, 4)
T_diff = np.matmul(np.linalg.inv(T_t1),T_t2)
T_gt = T_diff
print("from gt.txt:", T_gt)