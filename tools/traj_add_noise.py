import numpy as np
import matplotlib.pyplot as plt

def loadPoses(file_name):
    '''
        Each line in the file should follow one of the following structures
        (1) idx pose(3x4 matrix in terms of 12 numbers)
        (2) pose(3x4 matrix in terms of 12 numbers)
    '''
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    file_len = len(s)
    poses = {}
    pose_diff = {}
    frame_idx = 0
    for cnt, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split()]
        withIdx = int(len(line_split)==13)
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row*4 + col + withIdx]
        if withIdx:
            frame_idx = line_split[0]
        else:
            frame_idx = cnt
        poses[frame_idx] = P
        if frame_idx==0:
            pose_diff[frame_idx] = poses[frame_idx]
        else:
            pose_diff[frame_idx] = np.matmul(np.linalg.inv(poses[frame_idx-1]),poses[frame_idx])
    return pose_diff

def aug_matrix():
    
    anglex = 0#np.clip(0.01 * np.random.randn(), -0.02, 0.02).astype(float) * np.pi / 4.0
    angley = 0#np.clip(0.01 * np.random.randn(), -0.02, 0.02).astype(float) * np.pi / 4.0
    anglez = 0#np.clip(0.05 * np.random.randn(), -0.1, 0.1).astype(float) * np.pi / 4.0

    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    
    Rx = np.array([[1, 0, 0],
                    [0, cosx, -sinx],
                    [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny],
                    [0, 1, 0],
                    [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0],
                    [sinz, cosz, 0],
                    [0, 0, 1]])

    scale = np.diag(np.random.uniform(1.00, 1.00, 3).astype(float))
    R_trans = Rx.dot(Ry).dot(Rz).dot(scale.T)

    xx = 0#np.clip(0.5*np.random.rand()-0.25, -0.5, 0.5).astype(float)
    yy = 0#np.clip(0.2*np.random.rand()-0.1, -0.2, 0.2).astype(float)
    zz = np.clip(0.05 * np.random.randn()-0.025, -0.15, 0.15).astype(float)

    add_xyz = np.array([[xx], [yy], [zz]])

    T_trans = np.concatenate([R_trans, add_xyz], axis=-1)
    filler = np.array([0.0, 0.0, 0.0, 1.0])
    filler = np.expand_dims(filler, axis=0)  ##1*4
    T_trans = np.concatenate([T_trans, filler], axis=0)  # 4*4

    return T_trans

if __name__ == '__main__':
    import os
    origin_traj_path = "/home/yu/毕设/毕设/数据/4-kitti-tracking优化前后位姿/after/07_opti.txt"
    pose1 = loadPoses(origin_traj_path)
    # print(type(pose1))
    for i, TT in pose1.items():
        if i%40 == 0:
            TT = np.matmul(TT,aug_matrix())
        if i == 0:
            T_final = TT 
            T = T_final[:3, :]
            T = T.reshape(1, 1, 12)
        else:
            T_final = np.matmul(T_final, TT)
            T_current = T_final[:3, :]
            T_current = T_current.reshape(1, 1, 12)
            T = np.append(T, T_current, axis=0)
        
    Tt_list = T.reshape(-1, 12)
    fname_txt = os.path.join('test_data/07_no.txt')
    np.savetxt(fname_txt, Tt_list)