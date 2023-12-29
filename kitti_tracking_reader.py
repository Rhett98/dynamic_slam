# -*- coding:UTF-8 -*-

import sys,os,copy,math
import numpy as np
import gtsam

"""
     Reading data from KITTI tracking

"""

seq_length = [153,442,232,143,313,298,271,799,389,802,293,372,77,339,105,375,208,144,338,1058,836]

object_type_to_idx = {'Car': 0,
                      'Van' : 1,
                      'Truck' : 2,
                      'Pedestrian' : 3,
                      'Person (sitting)' : 4,
                      'Cyclist' : 5,
                      'Tram' : 6,
                      'Misc' : 7
                    }


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
        print('Ground truth poses are not avaialble.')
    return np.array(poses)


def load_calib(calib_path):
    """ Load calibrations (T_cam_velo) from file.
    """
    # Read and parse the calibrations
    T_cam_velo = []
    try:
        with open(calib_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Tr_velo_cam' in line:
                    line = line.replace('Tr_velo_cam', '')
                    T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))
    except FileNotFoundError:
        print('Calibrations are not avaialble.')
    return np.array(T_cam_velo)


class tData:
    """
        Utility class to load tracking data.
    """
    
    def __init__(self,frame=-1,obj_type="unset",truncation=-1,occlusion=-1,\
                 obs_angle=-10,x1=-1,y1=-1,x2=-1,y2=-1,w=-1,h=-1,l=-1,\
                 X=-1000,Y=-1000,Z=-1000,yaw=-10,score=-1000,track_id=-1):
        """
            Constructor, initializes the object given the parameters.
        """
        
        # init object data
        self.frame      = frame
        self.track_id   = track_id
        self.obj_type   = obj_type
        self.truncation = truncation
        self.occlusion  = occlusion
        self.obs_angle  = obs_angle
        self.x1         = x1
        self.y1         = y1
        self.x2         = x2
        self.y2         = y2
        self.w          = w
        self.h          = h
        self.l          = l
        self.X          = X
        self.Y          = Y
        self.Z          = Z
        self.yaw        = yaw
        self.score      = score
        self.ignored    = False
        self.valid      = False
        self.tracker    = -1
        self.T          = -1

    def __str__(self):
        """
            Print read data.
        """
        attrs = vars(self)
        return '\n'.join("%s: %s" % item for item in attrs.items())


class tracking_dataset(object):
    def __init__(self, dataset_path, seq_index, cls="car"):
        self.sequence_index = "{:0>4d}".format(seq_index)
        self.lidar_path = os.path.join(dataset_path, "velodyne",self.sequence_index)
        self.gt_path = os.path.join(dataset_path, "label_02")
        self.pose_path = os.path.join(dataset_path, "pose",self.sequence_index,"pose.txt")
        self.calib_path = os.path.join(dataset_path, "calib")
        self.seq_length = seq_length[seq_index]
        self._load_pose()
        self._load_label(cls)
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def _load_pose(self):
        # load poses
        poses = np.array(load_poses(self.pose_path))
        self.inv_frame0 = np.linalg.inv(poses[0])

        # load calibrations
        T_cam_velo = load_calib(os.path.join(self.calib_path, "%s.txt" % self.sequence_index))
        self.T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
        self.T_velo_cam = np.linalg.inv(self.T_cam_velo)

        # convert kitti poses from camera coord to LiDAR coord
        new_poses = []
        for pose in poses:
            # new_poses.append(self.T_velo_cam.dot(self.inv_frame0).dot(pose).dot(self.T_cam_velo))
            new_poses.append(self.inv_frame0.dot(pose))
        self.poses = np.array(new_poses)

    def _load_label(self, cls="car"):
        # construct objectDetections object to hold detection data
        t_data  = tData()
        self.seq_data           = []
        self.n_trajectories_seq = []
        n_trajectories     = 0
        filename = os.path.join(self.gt_path, "%s.txt" % self.sequence_index)
        f = open(filename, "r")
        f_data = [[] for x in range(self.seq_length)]
        ids = []
        n_in_seq = 0
        id_frame_cache = []
        
        for line in f:
            # KITTI tracking benchmark data format:
            # (frame,tracklet_id,objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry)
            line = line.strip()
            fields = line.split(" ")
            # classes that should be loaded (ignored neighboring classes)
            if "car" in cls.lower():
                classes = ["car","van"]
            elif "pedestrian" in cls.lower():
                classes = ["pedestrian","person_sitting"]
            else:
                classes = [cls.lower()]
            classes += ["dontcare"]
            if not any([s for s in classes if s in fields[2].lower()]):
                continue
            # get fields from table
            t_data.frame        = int(float(fields[0]))     # frame
            t_data.track_id     = int(float(fields[1]))     # id
            t_data.obj_type     = fields[2].lower()         # object type [car, pedestrian, cyclist, ...]
            t_data.truncation   = int(float(fields[3]))     # truncation [-1,0,1,2]
            t_data.occlusion    = int(float(fields[4]))     # occlusion  [-1,0,1,2]
            t_data.obs_angle    = float(fields[5])          # observation angle [rad]
            t_data.x1           = float(fields[6])          # left   [px]
            t_data.y1           = float(fields[7])          # top    [px]
            t_data.x2           = float(fields[8])          # right  [px]
            t_data.y2           = float(fields[9])          # bottom [px]
            t_data.h            = float(fields[10])         # height [m]
            t_data.w            = float(fields[11])         # width  [m]
            t_data.l            = float(fields[12])         # length [m]
            t_data.X            = float(fields[13])         # X [m]
            t_data.Y            = float(fields[14])         # Y [m]
            t_data.Z            = float(fields[15])         # Z [m]
            t_data.yaw          = float(fields[16])         # yaw angle [rad]
            t_data.T = self.T_velo_cam.dot(np.array(gtsam.Pose3(gtsam.Rot3.RzRyRx(0, 0, t_data.yaw), gtsam.Point3(t_data.X, t_data.Y, t_data.Z)).matrix())).dot(self.T_cam_velo)
            # do not consider objects marked as invalid
            if t_data.track_id == -1 and t_data.obj_type == "dontcare":
                continue

            idx = t_data.frame
            # check if length for frame data is sufficient
            if idx >= len(f_data):
                print("extend f_data", idx, len(f_data))
                f_data += [[] for x in range(max(500, idx-len(f_data)))]
            try:
                id_frame = (t_data.frame,t_data.track_id)
                id_frame_cache.append(id_frame)
                f_data[t_data.frame].append(copy.copy(t_data))
            except:
                print(len(f_data), idx)
                raise

        if t_data.track_id not in ids and t_data.obj_type!="dontcare":
            ids.append(t_data.track_id)
            n_trajectories +=1
            n_in_seq +=1

        # only add existing frames
        self.n_trajectories_seq.append(n_in_seq)
        self.seq_data=f_data
        f.close()
    
    def get_label(self, index):
        return self.seq_data[index]
    
    def get_pose(self, index):
        return self.poses[index]

    def get_pc(self, index):
        fn_dir = os.path.join(self.lidar_path, '{:06d}.bin'.format(index))
        point = np.fromfile(fn_dir, dtype=np.float32).reshape(-1, 4)
        pc = point[:, :3].astype(np.float32)
        return  pc
                
def plot_2d_points_proj(fignum, values, label,title="Global Traj"):
    keys = values.keys()
    fig = plt.figure(fignum)
    # Plot points and covariance matrices
    x, y = [],[]
    for key in keys:
        point = values.atPoint3(key)
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, label=label)
    plt.legend(loc = 'upper right')
    # plt.axis([-70, 70, -20, 20])
    fig.suptitle(title)

        
if __name__ == '__main__': 
    import gtsam.utils.plot as gtsam_plot
    import matplotlib.pyplot as plt
    seq_index = 0
    dataset = tracking_dataset("/home/yu/Resp/dataset/data_tracking_velodyne/training",seq_index)
    obj_traj = dict()
    fig = plt.figure(0)
    if not fig.axes:
        axes = fig.add_subplot(projection='3d')
    else:
        axes = fig.axes[0]
    plt.cla()
    plt.xlabel('x')
    plt.ylabel('y')
    
    # plot ego pose
    ego_value = gtsam.Values()
    for i in range(seq_length[seq_index]):
        ego_pose = gtsam.Pose3(dataset.get_pose(i))
        ego_value.insert(i,ego_pose.translation())
    # plot_2d_points_proj(0, ego_value,'ego')
    gtsam_plot.plot_trajectory(0,ego_value)
        
    # plot obj global_pose
    for i in range(seq_length[seq_index]):
        ego_pose = gtsam.Pose3(dataset.get_pose(i))
        for obj_data in dataset.get_label(i):
            obj_pose = gtsam.Pose3(obj_data.T)
            global_obj_pose = ego_pose * obj_pose
            if obj_data.track_id in obj_traj.keys():
                obj_traj[obj_data.track_id].append(global_obj_pose)
            else:
                obj_traj[obj_data.track_id] = [global_obj_pose]
    
    obj_value = gtsam.Values()   
    for id, traj in obj_traj.items():
        obj_value.clear()
        flag = 0 
        cl = np.random.random(), np.random.random(), np.random.random()
        for t in traj:
            obj_value.insert(flag, t.translation())
            flag+=1
        # plot_2d_points_proj(0, obj_value, id)
        # gtsam_plot.plot_3d_points(0, obj_value)


    # for i in range(seq_length[seq_index]):
    #     ego_pose = gtsam.Pose3(dataset.get_pose(i))
    #     last_obj_pose = gtsam.Pose3(np.eye(4))
    #     for obj_data in dataset.get_label(i):
    #         obj_pose = gtsam.Pose3(obj_data.T)
    #         global_obj_pose = ego_pose * obj_pose
    #         global_obj_pose = last_obj_pose.inverse() * obj_pose
    #         if obj_data.track_id in obj_traj.keys():
    #             obj_traj[obj_data.track_id].append(global_obj_pose)
    #         else:
    #             obj_traj[obj_data.track_id] = [global_obj_pose]
    #         last_obj_pose = ego_pose * obj_pose
            
    plt.axis('equal')
    plt.ioff()
    plt.show()
