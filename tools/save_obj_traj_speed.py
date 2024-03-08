import sys,os,copy,math
import numpy as np

def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion'''
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < 1e-8:
        return np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    return np.array(
        [[1.0 - (yY + zZ), xY - wZ, xZ + wY],
         [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
         [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])

def euler2quat(z=0, y=0, x=0, isRadian=True):
    if not isRadian:
        z = ((np.pi) / 180.) * z
        y = ((np.pi) / 180.) * y
        x = ((np.pi) / 180.) * x
    z = z / 2.0
    y = y / 2.0
    x = x / 2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    return np.array([
        cx * cy * cz - sx * sy * sz,
        cx * sy * sz + cy * cz * sx,
        cx * cz * sy - sx * cy * sz,
        cx * cy * sz + sx * cz * sy])
    
def euler2mat(z=0, y=0, x=0, isRadian=True):
    return quat2mat(euler2quat(z,y,x))



def loadPoses(file_name):
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
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
    return poses, pose_diff


def load_label(filename, track_id):
    f = open(filename, "r")
    s = f.readlines()
    f.close()
    poses = {}
    frame_idx = 0
    filler = np.array([0.0, 0.0, 0.0, 1.0])
    filler = np.expand_dims(filler, axis=0)  ##1*4
    for cnt, line in enumerate(s):
        line_split = [i for i in line.split()]
        if int(line_split[1])!= track_id:
            continue
        frame_idx = int(line_split[0])
        x, y, z, yaw = float(line_split[13]), float(line_split[14]), float(line_split[15]), float(line_split[16])
        R = euler2mat(yaw, 0, 0)
        t = np.array([x,y,z]).reshape(3,-1)
        T = np.concatenate([np.concatenate([R, t], axis=-1), filler], axis=0)
        poses[frame_idx] = T
    return poses
        
def aug_matrix():
    anglex = 0#np.clip(0.01 * np.random.randn(), -0.02, 0.02).astype(float) * np.pi / 4.0
    angley = 0#np.clip(0.01 * np.random.randn(), -0.02, 0.02).astype(float) * np.pi / 4.0
    anglez = np.clip(0.02 * np.random.randn(), -0.1, 0.1).astype(float) * np.pi / 4.0

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

    xx = np.clip(0.2*np.random.rand()-0.1, -0.5, 0.5).astype(float)
    yy = np.clip(0.1*np.random.rand()-0.05, -0.2, 0.2).astype(float)
    zz = 0#np.clip(0.05 * np.random.randn(), -0.15, 0.15).astype(float)

    add_xyz = np.array([[xx], [yy], [zz]])

    T_trans = np.concatenate([R_trans, add_xyz], axis=-1)
    filler = np.array([0.0, 0.0, 0.0, 1.0])
    filler = np.expand_dims(filler, axis=0)  ##1*4
    T_trans = np.concatenate([T_trans, filler], axis=0)  # 4*4
    return T_trans


if __name__ == '__main__':
    origin_traj_path = "/home/yu/Resp/dataset/data_tracking_velodyne/training/pose/0010/pose.txt"
    label_file = "/home/yu/Resp/dataset/data_tracking_velodyne/training/label_02/0010.txt"
    obj_pose = load_label(label_file,track_id=0)
    ego_pose, ego_pose_diff = loadPoses(origin_traj_path)
    # save obj traj
    for i, TT in obj_pose.items():
        # if i/50 == 0:
        #     TT = np.matmul(TT,aug_matrix())
        if i == 0:
            T_final = np.matmul(ego_pose[i], TT) 
            T = T_final[:3, :]
            T = T.reshape(1, 1, 12)
        else:
            T_final = np.matmul(ego_pose[i], TT)
            T_current = T_final[:3, :]
            T_current = T_current.reshape(1, 1, 12)
            T = np.append(T, T_current, axis=0)
        
    Tt_list = T.reshape(-1, 12)
    fname_txt = os.path.join('test_data/10_obj0_pose.txt')
    np.savetxt(fname_txt, Tt_list)
    
    # save obj traj with noise
    for i, TT in obj_pose.items():
        # print(i%5)
        if i == 0:
            T_final = np.matmul(ego_pose[i], TT) 
            T = T_final[:3, :]
            T = T.reshape(1, 1, 12)
        else:   
            T_final = np.matmul(ego_pose[i], TT)
            T_final = np.matmul(T_final, aug_matrix())
            T_current = T_final[:3, :]
            T_current = T_current.reshape(1, 1, 12)
            T = np.append(T, T_current, axis=0)
        
    Tt_noise_list = T.reshape(-1, 12)
    fname_txt = os.path.join('test_data/10_obj0_pose_noise_opti.txt')
    np.savetxt(fname_txt, Tt_noise_list)
    
    # save obj velocity
    first = True
    vel_list = []
    for i, TT in obj_pose.items():
        # if i/50 == 0:
        #     TT = np.matmul(TT,aug_matrix())
        if first == True:
            T_last = np.matmul(ego_pose[i], TT)
            first = False
            continue
            
        T_current = np.matmul(ego_pose[i], TT)
        vel = np.linalg.norm(T_current[:3,3]  - T_last[:3,3])*36
        T_last = T_current
        vel_list.append(vel)
        
    V_list = np.array(vel_list).reshape(-1, 1)
    fname_txt = os.path.join('test_data/10_obj0_vel.txt')
    np.savetxt(fname_txt, V_list)