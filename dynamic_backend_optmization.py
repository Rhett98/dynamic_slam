from functools import partial
from typing import List, Optional

import numpy as np
import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt

from kitti_tracking_reader import tData, tracking_dataset

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


class FactorGraph(object):
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.result = gtsam.Values()
        self.parameters = gtsam.ISAM2Params()
        self.parameters.setRelinearizeThreshold(0.1)
        self.parameters.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(self.parameters)
        self.f_id = 0
        
    def AddBTW2factor(self, id1:int, id2:int, p:gtsam.Pose3, noiseScore:float):
        Noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([noiseScore,noiseScore,
                                                        noiseScore,noiseScore,
                                                        noiseScore,noiseScore]))
        self.graph.add(gtsam.BetweenFactorPose3(id1, id2, p, Noise))
        self.f_id+=1
        
    def AddBTW3factor(self, id1:int, id2:int, id3:int, noiseScore:float):
        Noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([noiseScore,noiseScore,
                                                        noiseScore,noiseScore,
                                                        noiseScore,noiseScore]))
        self.graph.add(Btw3Factor(id1, id2, id3, Noise))
        self.f_id+=1

    def AddPriorFactor(self, id:int, p:gtsam.Pose3, noiseScore:float):
        Noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([noiseScore,noiseScore,
                                                        noiseScore,noiseScore,
                                                        noiseScore,noiseScore]))
        self.graph.add(gtsam.PriorFactorPose3(id, p, Noise))
        self.f_id+=1
      
    def AddMarginalPrior(self, id:int, p:gtsam.Pose3):
        marginals = gtsam.Marginals(self.graph, self.result)
        updatenoise = gtsam.noiseModel.Gaussian.Covariance(marginals.marginalCovariance(id))
        self.graph.add(gtsam.PriorFactorPose3(id, p, updatenoise))
        self.f_id+=1

    def SetInitialEstimate(self, id:int, p:gtsam.Pose3):
        self.initial_estimate.insert(id, p)

    def Removefactor(self, id:int):
        self.graph.remove(id)

    def StartOptimiz(self):
        self.isam.update(self.graph, self.initial_estimate)
        current_estimate = self.isam.calculateEstimate()
        self.initial_estimate.clear()   
        return current_estimate 


def read_tracking_data(data:List[tData]):
    obj_num = len(data)
    obj_track_ids = []
    obj_poses = []
    for i in range(obj_num):
        track_id = data[i].track_id
        x,y,z = data[i].x, data[i].y, data[i].z
        pose = gtsam.Pose3(gtsam.Rot3.RzRyRx(0, 0, 0), gtsam.Point3(X, Y, Z))
        obj_track_ids.append(track_id)
        obj_poses.append(pose)
    return obj_track_ids, obj_poses


class backend_optimization():
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.result = gtsam.Values()
        self.parameters = gtsam.ISAM2Params()
        self.parameters.setRelinearizeThreshold(0.1)
        self.parameters.relinearizeSkip = 1
        self.isam2 = gtsam.ISAM2(self.parameters)
        self.f_id = 0
        # 设置噪声
        noiseScore = 0.001
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([noiseScore,noiseScore,
                                                        noiseScore,noiseScore,
                                                        noiseScore,noiseScore]))
        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([noiseScore,noiseScore,
                                                        noiseScore,noiseScore,
                                                        noiseScore,noiseScore]))
        self.obj_ego_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([noiseScore,noiseScore,
                                                        noiseScore,noiseScore,
                                                        noiseScore,noiseScore]))
        self.frame_motion_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([noiseScore,noiseScore,
                                                        noiseScore,noiseScore,
                                                        noiseScore,noiseScore]))
        self.smoothing_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([noiseScore,noiseScore,
                                                        noiseScore,noiseScore,
                                                        noiseScore,noiseScore]))
        
    def add_odom_factor(self, pose):
        if self.f_id != 0:
            # 里程计运动因子
            self.graph.add(gtsam.BetweenFactorPose3(self.f_id, self.f_id+1, pose, self.odom_noise))
            # 里程计位姿
            self.initial_estimate.insert(self.f_id+1, pose)
        else:
            # 里程计先验因子
            self.graph.add(gtsam.PriorFactorPose3(self.f_id, pose, self.prior_noise))
            self.initial_estimate.insert(self.f_id, pose)
        self.f_id+=1
        
    def add_dyobj_factor(self, dyobj_data):
        # 解析动态目标数据
        ids, poses = read_tracking_data(dyobj_data)
        #管理跟踪队列
        
        for i in range(len(ids)):
            # 动态目标与里程计相对运动因子
            self.graph.add(gtsam.BetweenFactorPose3(self.f_id-1, self.f_id, poses[i], self.obj_ego_noise))
            # 动态目标位姿
            self.initial_estimate.insert(self.f_id, poses[i])
            self.f_id+=1
            # 帧间动态目标相对运动因子
            self.graph.add(Btw3Factor(self.f_id, self.f_id+1, self.f_id+2, self.frame_motion_noise))
            # 平滑运动因子
            self.graph.add(gtsam.BetweenFactorPose3(self.f_id, self.f_id+1, pose, self.smoothing_noise))
            # 动态目标位姿变化量
            self.initial_estimate.insert(self.f_id, pose)
            self.f_id+=1
        
    def update_frame_message(self, ego_pose, dyobj_data):
        self.add_odom_factor(ego_pose)
        self.add_dyobj_factor(dyobj_data)
        self.isam2.update(self.graph, self.initial_estimate)
        self.isam2.update()
        self.graph.resize(0)
        self.initial_estimate.clear()
        current_estimate = self.isam2.calculateEstimate()



if __name__ == '__main__':
    egoP_egoP = 1e-6
    egoP_objP = 1e-2
    objP_objP_chgP = 1.0
    chgP_chgP = 1e-2
    local_graph = FactorGraph() 
    # add prior factors
    local_graph.AddPriorFactor(0,gtsam.Pose3(gtsam.Rot3.RzRyRx(0, 0, 0), gtsam.Point3(0, 0, 0)), egoP_egoP)
    local_graph.AddPriorFactor(1,gtsam.Pose3(gtsam.Rot3.RzRyRx(0, 0, 0), gtsam.Point3(0, -1, 0)), egoP_egoP)
    local_graph.AddPriorFactor(4,gtsam.Pose3(gtsam.Rot3.RzRyRx(0, 0, 0), gtsam.Point3(1, 0, 0)), egoP_egoP)
    # add btw2 factors
    local_graph.AddBTW2factor(0,2,gtsam.Pose3(gtsam.Rot3.RzRyRx(0, 0, 0), gtsam.Point3(2, 0, 0)), egoP_egoP)
    local_graph.AddBTW2factor(2,5,gtsam.Pose3(gtsam.Rot3.RzRyRx(0, 0, 0), gtsam.Point3(2, 0, 0)), egoP_egoP)
    local_graph.AddBTW2factor(0,1,gtsam.Pose3(gtsam.Rot3.RzRyRx(0, 0, 0), gtsam.Point3(0, -1, 0)), egoP_objP)
    local_graph.AddBTW2factor(2,3,gtsam.Pose3(gtsam.Rot3.RzRyRx(0, 0, 0), gtsam.Point3(-1, -1, 0)), egoP_objP)
    local_graph.AddBTW2factor(5,6,gtsam.Pose3(gtsam.Rot3.RzRyRx(0, 0, 0), gtsam.Point3(-2, -1, 0)), egoP_egoP)
    local_graph.AddBTW2factor(4,7,gtsam.Pose3(gtsam.Rot3.RzRyRx(0, 0, 0), gtsam.Point3(0, 0, 0)), chgP_chgP)
    # add btw3 factors
    local_graph.AddBTW3factor(1, 3, 4, objP_objP_chgP); 
    local_graph.AddBTW3factor(3, 6, 7, objP_objP_chgP); 
    
    # set initial value
    local_graph.SetInitialEstimate(0, gtsam.Pose3(gtsam.Rot3.RzRyRx(0.1, 0, 0), gtsam.Point3(0.1, -0.1, 0)))
    local_graph.SetInitialEstimate(1, gtsam.Pose3(gtsam.Rot3.RzRyRx(-0.1, 0, 0), gtsam.Point3(-0.1, -1.1, 0)))
    local_graph.SetInitialEstimate(2, gtsam.Pose3(gtsam.Rot3.RzRyRx(0.1, 0, 0), gtsam.Point3(2.2, 0.2, 0)))
    local_graph.SetInitialEstimate(3, gtsam.Pose3(gtsam.Rot3.RzRyRx(-0.1, 0, 0), gtsam.Point3(-1.1, -0.9, 0)))#wrong initial value
    local_graph.SetInitialEstimate(4, gtsam.Pose3(gtsam.Rot3.RzRyRx(0.1, 0, 0), gtsam.Point3(0.9, 0.15, 0)))
    local_graph.SetInitialEstimate(5, gtsam.Pose3(gtsam.Rot3.RzRyRx(-0.1, 0, 0), gtsam.Point3(4.2, -0.2, 0)))
    local_graph.SetInitialEstimate(6, gtsam.Pose3(gtsam.Rot3.RzRyRx(0.1, 0, 0), gtsam.Point3(-2.1, -1.0, 0)))#wrong initial value
    local_graph.SetInitialEstimate(7, gtsam.Pose3(gtsam.Rot3.RzRyRx(-0.1, 0, 0), gtsam.Point3(0.05, -0.05, 0)))
    # start optimize
    result = local_graph.StartOptimiz()
    print("\nFactor Graph:\n{}".format(local_graph.graph))
    print("\nFinal Result:\n{}".format(result))

    