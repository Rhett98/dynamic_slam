import torch
import knn
import time
import numpy as np
    
    

if __name__ == '__main__':
    pc1 = np.fromfile("/home/yu/Resp/dynamic_slam/demo_pc/velodyne/000000.bin", dtype=np.float32).reshape(4, -1)
    pc2 = np.fromfile("/home/yu/Resp/dynamic_slam/demo_pc/velodyne/000001.bin", dtype=np.float32).reshape(4, -1)
    pc1 = torch.from_numpy(pc1[:3, :].astype(np.float32)).float()
    pc2 = torch.from_numpy(pc2[:3, :].astype(np.float32)).float()
    # print(pc2.shape)
    t = time.time()
    dist = knn.knn_loss(pc1.transpose(0,1), pc2.transpose(0,1), 3)
    print('fw time', time.time()-t, 's')
    print(dist)
