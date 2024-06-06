import os
import numpy as np
import torch

# voxel pc:[1,3,n,x,y]
# predict label [1,x,y]
# output pc [N,3]
# output label [N]

def save_seg_result(logdir, path_seq, path_name, voxel_pc, pre_label):
    n_point = voxel_pc.shape[2]
    pre_label = pre_label.repeat(n_point, 1, 1).unsqueeze(0).unsqueeze(0) #label [1,1,n,x,y]

    output_pc_np = voxel_pc.cpu().numpy()
    output_pc_np = output_pc_np.reshape((-1,3)).astype(np.float32)

    pred_np = pre_label.cpu().numpy()
    pred_np = pred_np.reshape((-1)).astype(np.int32)

    mask = np.any(output_pc_np != 0, axis=1)
    pred_np = pred_np[mask]
    output_pc_np = output_pc_np[mask]

    # save scan
    pc_folder = os.path.join(logdir, "sequences", path_seq, "velodyne")
    label_folder = os.path.join(logdir, "sequences", path_seq, "labels")
    if os.path.isdir(label_folder) is False:
        os.makedirs(label_folder)
        os.makedirs(pc_folder)
    path_pc = os.path.join(pc_folder, path_name+".bin")
    path_label = os.path.join(label_folder, path_name+".label")
    pred_np.tofile(path_label)
    output_pc_np.tofile(path_pc)


# p1 = torch.rand([1,3,5,400,400])
# l1 = torch.rand([1,400,400])
# logdir = "experiment"
# path_seq = "00"
# path_name = "000001"
# path = os.path.join(logdir, "sequences", path_seq, "predictions")
# if os.path.isdir(path) is False:
#     os.makedirs(path)
# save_seg_result(logdir, path_seq, path_name, p1, l1)

# pc1 = np.fromfile("experiment/sequences/00/predictions/000001.bin", dtype=np.float32).reshape(-1,3)
# label1 = np.fromfile("experiment/sequences/00/predictions/000001.label", dtype=np.int32).reshape((-1))& 0xFFFF
# print(pc1.shape, label1.shape)

# pc1 = np.fromfile("demo_pc/velodyne/000000.bin", dtype=np.float32).reshape(-1,4)
# label1 = np.fromfile("demo_pc/labels/000000.label", dtype=np.int32).reshape((-1))& 0xFFFF
# print(pc1.shape, label1.shape)