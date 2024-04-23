import numpy as np
import glob
import argparse
import os, os.path

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
    poses = []
    frame_idx = 0
    for cnt, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split()]
        withIdx = int(len(line_split)==13)
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row*4 + col + withIdx]
        poses.append(convert2CameraCoord(P))
    return np.array(poses)

def convert2CameraCoord(pose_mat):
    '''
        Convert the pose of lidar coordinate to camera coordinate
    '''
    R_C2L = np.array([[0,   0,   1,  0],
                        [-1,  0,   0,  0],
                        [0,  -1,   0,  0],
                        [0,   0,   0,  1]])
    inv_R_C2L = np.linalg.inv(R_C2L)            
    R = np.dot(inv_R_C2L, pose_mat)
    rot = np.dot(R, R_C2L)
    return rot[:3,:] 

                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KITTI Evaluation toolkit')
    parser.add_argument('--file_path', type=str, default='/home/yu/Resp/dynamic_slam/07_loam.txt',  help='Directory path of pose')

    args = parser.parse_args()
    output_path = "./07_loam.txt"
    pose = loadPoses(args.file_path)
    T = pose.reshape(-1,12)
    np.savetxt(output_path, T)