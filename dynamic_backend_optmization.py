from functools import partial
from typing import List, Optional

import numpy as np
import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt

from kitti_tracking_reader import tData, tracking_dataset, seq_length

def skew_symmetric(w):
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]])
    
def Btw3Factor(poseKey1, poseKey2, poseKey3, noise_model):
    def error_func(this: gtsam.CustomFactor, v: gtsam.Values, H: Optional[List[np.ndarray]]) -> float:
        """
        Error function that mimics a Between3Factor
        :param this: reference to the current CustomFactor being evaluated
        :param v: Values object
        :param H: list of references to the Jacobian arrays
        :return: the non-linear error
        """
        key0 = this.keys()[0]
        key1 = this.keys()[1]
        key2 = this.keys()[2]
        
        p1, p2, p3 = v.atPose3(key0), v.atPose3(key1), v.atPose3(key2)
        error = gtsam.Pose3.Logmap(p1.inverse() * p2 * p3.inverse())
        
        J = np.zeros((6, 6))
        J[:3, :3] = skew_symmetric(error[:3])
        J[:3, 3:] = skew_symmetric(error[3:])
        J[3:, 3:] = skew_symmetric(error[:3])
        J = J * 0.5 + np.eye(6)
        
        if H is not None:
            H[0] = (-1) * J * ((p2 * p3.inverse()).inverse()).AdjointMap()
            H[1] = J * ((p2 * p3.inverse()).inverse()).AdjointMap()
            H[2] = (-1) * J
        return error
    return gtsam.CustomFactor(noise_model, gtsam.KeyVector([poseKey1, poseKey2, poseKey3]), error_func)


class backend_optimization():
    def __init__(self):
        self.graph_init()
        self.noise_init()
        self.odom_prior_init()
        
    def noise_init(self):
        # 设置噪声
        egoP_egoP = 1e-4
        egoP_objP = 1e-2
        objP_objP_chgP = 1e-2
        chgP_chgP = 0.1
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6,1e-6,
                                                        1e-6,1e-6,
                                                        1e-6,1e-6]))
        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([egoP_egoP,egoP_egoP,
                                                        egoP_egoP,egoP_egoP,
                                                        egoP_egoP,egoP_egoP]))
        self.obj_ego_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([egoP_objP,egoP_objP,
                                                        egoP_objP,egoP_objP,
                                                        egoP_objP,egoP_objP]))
        self.frame_motion_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([objP_objP_chgP,objP_objP_chgP,
                                                        objP_objP_chgP,objP_objP_chgP,
                                                        objP_objP_chgP,objP_objP_chgP]))
        self.smoothing_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([chgP_chgP,chgP_chgP,
                                                        chgP_chgP,chgP_chgP,
                                                        chgP_chgP,chgP_chgP]))
    
    def graph_init(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.result = gtsam.Values()
        self.parameters = gtsam.ISAM2Params()
        self.parameters.setRelinearizeThreshold(0.1)
        self.parameters.relinearizeSkip = 1
        self.isam2 = gtsam.ISAM2(self.parameters)
        # 因子序号
        self.f_id = 0
        # 帧序号
        self.frame_index = 0
        # 里程计因子序号
        self.odom_id = [0]
        self.obj_ids = {}
        self.pose_change_ids = {}
        self.ego_pose = []
        self.obj_traj = {}
        
    def add_keyframe(self):
        a=0
    
    def odom_prior_init(self):
        self.ego_pose.append(gtsam.Pose3(np.eye(4)))
        self.graph.add(gtsam.PriorFactorPose3(self.f_id, self.ego_pose[-1], self.prior_noise))
        self.initial_estimate.insert(self.f_id, self.ego_pose[-1])
        self.f_id+=1
    
    def add_odom_factor(self):
        # 里程计运动因子
        self.graph.add(gtsam.BetweenFactorPose3(self.odom_id[-1], self.f_id, self.ego_pose[-2].inverse()*self.ego_pose[-1], self.odom_noise))
        # 里程计位姿
        self.initial_estimate.insert(self.f_id, self.ego_pose[-1])
        self.odom_id.append(self.f_id)
        self.f_id+=1
        
    def add_dyobj_factor(self, new_obj, history_obj):
        # 新加入obj
        for k, v in new_obj.items():
            # 动态目标与里程计相对运动因子
            self.graph.add(gtsam.BetweenFactorPose3(self.odom_id[-1], self.f_id, self.ego_pose[-1].inverse()*v, self.obj_ego_noise))
            self.initial_estimate.insert(self.f_id, v)
            self.obj_ids[k]=[self.f_id]
            self.f_id+=1
        # 已存在历史obj
        for k, v in history_obj.items():
            self.graph.add(gtsam.BetweenFactorPose3(self.odom_id[-1], self.f_id, self.ego_pose[-1].inverse()*v, self.obj_ego_noise))
            self.initial_estimate.insert(self.f_id, v)
            # 记录obj节点id
            self.obj_ids[k].append(self.f_id)
            self.f_id+=1
            # pose_chage_node
            self.initial_estimate.insert(self.f_id, self.obj_traj[k][-2].inverse()*v)
            # 记录pose_change节点id
            if k in self.pose_change_ids.keys():
                self.pose_change_ids[k].append(self.f_id)
            else:
                self.pose_change_ids[k]=[self.f_id]
            self.f_id+=1  
            # 帧间动态目标相对运动因子
            self.graph.add(Btw3Factor(self.obj_ids[k][-2], self.obj_ids[k][-1], self.pose_change_ids[k][-1], self.frame_motion_noise))
            # 平滑运动因子
            if len(self.pose_change_ids[k])>1:
                self.graph.add(gtsam.BetweenFactorPose3(self.pose_change_ids[k][-2], self.pose_change_ids[k][-1], gtsam.Pose3(np.eye(4)), self.smoothing_noise))
            
    def load_frame_tracking_data(self, ego_pose:gtsam.Pose3, data_list:List[tData]):
        new_obj=dict()
        history_obj = dict()
        for obj_data in data_list:
            obj_pose = gtsam.Pose3(obj_data.T)
            global_obj_pose = ego_pose * obj_pose
            if obj_data.track_id in self.obj_traj.keys():
                self.obj_traj[obj_data.track_id].append(global_obj_pose)
                history_obj[obj_data.track_id] = global_obj_pose
            else:
                self.obj_traj[obj_data.track_id] = [global_obj_pose]
                new_obj[obj_data.track_id] = global_obj_pose
        return new_obj, history_obj

    def update_frame_message(self, ego_pose, dyobj_data):
        # print(self.frame_index)
        # 更新当前帧里程计位姿
        self.ego_pose.append(ego_pose)
        # 加载当前帧数据
        new_obj, history_obj = self.load_frame_tracking_data(ego_pose, dyobj_data)
        # 添加新因子
        self.add_odom_factor()
        self.add_dyobj_factor(new_obj, history_obj)
        # 因子图优化
        self.isam2.update(self.graph, self.initial_estimate)
        self.graph.resize(0)
        self.initial_estimate.clear()
        current_estimate = self.isam2.calculateEstimate()
        # 优化后数据更新到self.ego_pose与self.obj_traj

        # 更新当前帧序号
        self.frame_index+=1
        return current_estimate


def aug_matrix():
    
    anglex = 0#np.clip(0.01 * np.random.randn(), -0.02, 0.02).astype(float) * np.pi / 4.0
    angley = 0#np.clip(0.01 * np.random.randn(), -0.02, 0.02).astype(float) * np.pi / 4.0
    anglez = np.clip(0.05 * np.random.randn(), -0.1, 0.1).astype(float) * np.pi / 4.0

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

    xx = np.clip(0.5*np.random.rand()-0.25, -0.5, 0.5).astype(float)
    yy = np.clip(0.2*np.random.rand()-0.1, -0.2, 0.2).astype(float)
    zz = 0#np.clip(0.05 * np.random.randn(), -0.15, 0.15).astype(float)

    add_xyz = np.array([[xx], [yy], [zz]])

    T_trans = np.concatenate([R_trans, add_xyz], axis=-1)
    filler = np.array([0.0, 0.0, 0.0, 1.0])
    filler = np.expand_dims(filler, axis=0)  ##1*4
    T_trans = np.concatenate([T_trans, filler], axis=0)  # 4*4

    return T_trans


if __name__ == '__main__':
    import os
    eval_list = [4,7,8,9,15,18,19]
    local_graph = backend_optimization() 
    seq_index = 18
    dataset = tracking_dataset("/home/yu/Resp/dataset/data_tracking_velodyne/training",seq_index,"experiment/odometry_EVAL_KITTI_2024-01-11_17-21")
    fig = plt.figure(0)
    if not fig.axes:
        axes = fig.add_subplot(projection='3d')
    else:
        axes = fig.axes[0]
    plt.cla()
    # print(gtsam.Pose3(aug_matrix()))
    ######### 原始obj轨迹 ########
    origin_obj_traj = dict()
    first = True
    frame_ego_pose = gtsam.Pose3(gtsam.Rot3.RzRyRx(0, 0, 0), gtsam.Point3(0, 0, 0))
    for frame in range(seq_length[seq_index]):
        if first:
            frame_ego_pose = gtsam.Pose3(dataset.get_pose(frame))
            frame_obj_data = dataset.get_label(frame)
            current_estimate = local_graph.update_frame_message(frame_ego_pose, frame_obj_data)
            # 保存原始obj轨迹
            for obj_data in frame_obj_data:
                obj_pose = gtsam.Pose3(obj_data.T)
                global_obj_pose = frame_ego_pose * obj_pose
                if obj_data.track_id in origin_obj_traj.keys():
                    origin_obj_traj[obj_data.track_id].append(global_obj_pose)
                else:
                    origin_obj_traj[obj_data.track_id] = [global_obj_pose]
            first = False
        else:
            if frame_ego_pose.range(gtsam.Pose3(dataset.get_pose(frame)))>0:
                # print(frame_ego_pose.range(gtsam.Pose3(dataset.get_pose(frame))))
                frame_ego_pose = gtsam.Pose3(dataset.get_pose(frame))#*gtsam.Pose3(aug_matrix())
                frame_obj_data = dataset.get_label(frame)
                current_estimate = local_graph.update_frame_message(frame_ego_pose, frame_obj_data)
                # 保存原始obj轨迹
                for obj_data in frame_obj_data:
                    obj_pose = gtsam.Pose3(obj_data.T)
                    global_obj_pose = frame_ego_pose * obj_pose
                    if obj_data.track_id in origin_obj_traj.keys():
                        origin_obj_traj[obj_data.track_id].append(global_obj_pose)
                    else:
                        origin_obj_traj[obj_data.track_id] = [global_obj_pose]
        # if frame==150:
        #     break
     
    # ######## 绘制原始动态目标轨迹 ########
    # obj_value = gtsam.Values()   
    # for id, traj in origin_obj_traj.items():
    #     obj_value.clear()
    #     flag = 0 
    #     for t in traj:
    #         obj_value.insert(flag, t.translation())
    #         flag+=1
        # gtsam_plot.plot_3d_points(0, obj_value, 'g*')
        
    # ######### 优化后动态目标轨迹 ########
    # obj_traj_value = gtsam.Values()
    # for i in local_graph.obj_ids.keys():
    #     obj_traj_value.clear()
    #     f = 0
    #     for vv in local_graph.obj_ids[i]:
    #         obj_traj_value.insert(f, current_estimate.atPose3(vv).translation())
    #         f+=1
    #     gtsam_plot.plot_3d_points(0, obj_traj_value,'r+')
       
    ######## 得到优化后的里程计轨迹 ########
    plot_value = gtsam.Values()
    i=0
    for v in local_graph.odom_id:
        plot_value.insert(i, current_estimate.atPose3(v))
        T_final = current_estimate.atPose3(v).matrix()
        if i==0:
            T = T_final[:3, :]
            T = T.reshape(1, 1, 12)
        else:
            T_current = T_final[:3, :]
            T_current = T_current.reshape(1, 1, 12)
            T = np.append(T, T_current, axis=0)
        i+=1
    # gtsam_plot.plot_trajectory(0,plot_value)
    
    ####### 保存优化后的里程计轨迹到txt #########
    T_list = T.reshape(-1, 12)
    fname_txt = os.path.join('test_data/'+str(seq_index)+'_optimazation_noise.txt')
    np.savetxt(fname_txt, T_list)
    
    ####### 绘制原始里程计轨迹 ########
    origin_odom_value = gtsam.Values()
    f = 0
    for vv in local_graph.ego_pose:
        origin_odom_value.insert(f, vv)
        f+=1
    # gtsam_plot.plot_trajectory(0, origin_odom_value,5)
    ####### 保存优化后的里程计轨迹到txt #########
    print(len(local_graph.ego_pose))
    i=0
    for tj in local_graph.ego_pose:
        T_final = tj.matrix()
        if i==0:
            Tt = T_final[:3, :]
            Tt = Tt.reshape(1, 1, 12)
        else:
            T_current = T_final[:3, :]
            T_current = T_current.reshape(1, 1, 12)
            Tt = np.append(Tt, T_current, axis=0)
        i+=1
        
    Tt_list = Tt.reshape(-1, 12)
    fname_txt = os.path.join('test_data/'+str(seq_index)+'_optimazation_noise_origin.txt')
    np.savetxt(fname_txt, Tt_list)
    
    ### 绘制动态目标的速度(帧间运动距离) ###
    obj_speed_value = {}
    f = 0
    for i in local_graph.pose_change_ids.keys():
        for vv in local_graph.pose_change_ids[i]:
            if f in obj_speed_value.keys():
                obj_speed_value[f].append(current_estimate.atPose3(vv).range(gtsam.Pose3(np.eye(4))))
            else:
                obj_speed_value[f] = [current_estimate.atPose3(vv).range(gtsam.Pose3(np.eye(4)))]
        f+=1
    # print(obj_speed_value)
    
    
    # plt.axis('equal')
    # plt.ioff()
    # plt.show()