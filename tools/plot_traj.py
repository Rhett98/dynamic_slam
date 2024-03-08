import glob
import argparse
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
# choose other backend that not required GUI (Agg, Cairo, PS, PDF or SVG) when use matplotlib
plt.switch_backend('agg')
import matplotlib.backends.backend_pdf
import tools.transformations as tr
from tools.pose_evaluation_utils import quat_pose_to_mat

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
    return rot 

def loadPoses(file_name, toCameraCoord=True):
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
        if toCameraCoord:
            poses[frame_idx] = convert2CameraCoord(P)
        else:
            poses[frame_idx] = P
    return poses

def plotPath_3D(seq, poses_gt, poses_result, plot_path_dir, compare_pose):
    """
        plot the path in 3D space
    """
    from mpl_toolkits.mplot3d import Axes3D

    start_point = [[0], [0], [0]]
    fontsize_ = 8
    style_pred = 'r-'
    style_gt = 'k-'
    style_p1 = 'b-'
    style_p2 = 'g-'
    style_p3 = 'm-'
    style_O = 'ko'

    poses_dict = {}      
    poses_dict["Ours"] = poses_result
    if poses_gt:
        poses_dict["Ground Truth"] = poses_gt
        
    if compare_pose:
        poses_dict["LOAM w/o mapping"] = compare_pose[0]
        poses_dict["LOAM"] = compare_pose[1]
        # poses_dict["LOAMx"] = compare_pose[2]

    fig = plt.figure(figsize=(8,8), dpi=110)
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection = '3d')

    for key,_ in poses_dict.items():
        plane_point = []
        for frame_idx in sorted(poses_dict[key].keys()):
            pose = poses_dict[key][frame_idx]
            plane_point.append([pose[0,3], pose[2,3], pose[1,3]])
        plane_point = np.asarray(plane_point)
        if key == 'Ours':
            style = style_pred 
        if poses_gt:
            if key == 'Ground Truth':
                style = style_gt 
        if compare_pose:
            if key == 'LOAM w/o mapping':
                style = style_p1
            if key == 'LOAM':
                style = style_p2 
            if key == 'LOAMx':
                style = style_p3
        plt.plot(plane_point[:,0], plane_point[:,1], plane_point[:,2], style, label=key)  
    plt.plot(start_point[0], start_point[1], start_point[2], style_O, label='Start Point')

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    xmean = np.mean(xlim)
    ymean = np.mean(ylim)
    zmean = np.mean(zlim)
    plot_radius = max([abs(lim - mean_)
                    for lims, mean_ in ((xlim, xmean),
                                        (ylim, ymean),
                                        (zlim, zmean))
                    for lim in lims])
    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])

    # ax.legend()
    # plt.legend(loc="upper right", prop={'size':fontsize_}) 
    ax.set_xlabel('x (m)', fontsize=fontsize_)
    ax.set_ylabel('z (m)', fontsize=fontsize_)
    ax.set_zlabel('y (m)', fontsize=fontsize_)
    ax.view_init(elev=20., azim=-35)

    png_title = "{}_path_3D".format(seq)
    plt.savefig(plot_path_dir+"/"+png_title+".png", bbox_inches='tight', pad_inches=0.1)
    # pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir +  "/" + png_title + ".pdf")        
    fig.tight_layout()
    # pdf.savefig(fig)  
    # plt.show()
    plt.close()
        
        
def plotPath_2D_3(seq, poses_gt, poses_result, plot_path_dir, compare_pose):
    '''
        plot path in XY, XZ and YZ plane
    '''
    fontsize_ = 10
    start_point = [0, 0]
    style_pred = 'r-'
    style_gt = 'k-'
    style_p1 = 'b-'
    style_p2 = 'g-'
    style_p3 = 'm-'
    style_O = 'ko'

    poses_dict = {}      
    poses_dict["Ours"] = poses_result
    if poses_gt:
        poses_dict["Ground Truth"] = poses_gt
        
    if compare_pose:
        poses_dict["LOAM w/o mapping"] = compare_pose[0]
        poses_dict["LOAM"] = compare_pose[1]
        # poses_dict["LOAMx"] = compare_pose[2]
        
    fig = plt.figure(figsize=(20,6), dpi=100)
    
    for key,_ in poses_dict.items():
        plane_point = []
        for frame_idx in sorted(poses_dict[key].keys()):
            pose = poses_dict[key][frame_idx]
            plane_point.append([pose[0,3], pose[2,3], pose[1,3]])
        plane_point = np.asarray(plane_point)
        if key == 'Ours':
            style = style_pred 
        if poses_gt:
            if key == 'Ground Truth':
                style = style_gt 
        if compare_pose:
            if key == 'LOAM w/o mapping':
                style = style_p1
            if key == 'LOAM':
                style = style_p2 
            if key == 'LOAMx':
                style = style_p3
                
        ### plot the figure
        plt.subplot(1,3,1)
        ax = plt.gca()
        plt.plot(plane_point[:,0], plane_point[:,1], style, label=key) 
        # plt.legend(loc="upper right", prop={'size':fontsize_})
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        ### set the range of x and y
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        plot_radius = max([abs(lim - mean_)
                            for lims, mean_ in ((xlim, xmean),
                                                (ylim, ymean))
                            for lim in lims])
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

        plt.subplot(1,3,2)
        ax = plt.gca()
        plt.plot(plane_point[:,0], plane_point[:,2], style, label=key) 
        # plt.legend(loc="upper right", prop={'size':fontsize_})
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('y (m)', fontsize=fontsize_)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

        plt.subplot(1,3,3)
        ax = plt.gca()
        plt.plot(plane_point[:,1], plane_point[:,2], style, label=key) 
        # plt.legend(loc="upper right", prop={'size':fontsize_})
        plt.xlabel('y (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius]) 
    
    #plot start point
    plt.subplot(1,3,1)
    ax = plt.gca()
    plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
    plt.subplot(1,3,2)
    ax = plt.gca()
    plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
    plt.subplot(1,3,3)
    ax = plt.gca()
    plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
    
    png_title = "{}_path".format(seq)
    plt.savefig(plot_path_dir +  "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
    # pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir +  "/" + png_title + ".pdf")        
    fig.tight_layout()
    # pdf.savefig(fig)  
    # plt.show()
    plt.close()
    
    
if __name__ == '__main__':
    seq = 77
    gt_file_name = "/home/yu/Resp/dataset/data_tracking_velodyne/training/pose/0007/pose.txt"
    pred_file_name = "/home/yu/毕设/毕设/数据/4-kitti-tracking优化前后位姿/before/07_output.txt"
    pose1_path = "/home/yu/毕设/毕设/数据/4-kitti-tracking优化前后位姿/after/07_opti.txt"
    pose2_path = "/home/yu/毕设/毕设/数据/tracking-loam-traj/LOAM-traj-mapping-tracking-07.txt"
    # pose3_path = "/home/yu/Nutstore Files/Nutstore/毕设/数据/3-seq08不同层位姿输出/08_l3.txt"
    eva_seq_dir = "./plot"
    poses_gt = loadPoses(gt_file_name)
    poses_result = loadPoses(pred_file_name)
    pose1 = loadPoses(pose1_path)
    pose2 = loadPoses(pose2_path,False)
    # pose3 = loadPoses(pose3_path)
    plotPath_3D(seq, poses_gt, poses_result, eva_seq_dir, [pose1, pose2])
    plotPath_2D_3(seq, poses_gt, poses_result, eva_seq_dir, [pose1, pose2])