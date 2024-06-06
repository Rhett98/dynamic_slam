# dynamic_slam

## sceneflow-guided moving segment network
在kitti数据集训练
python train_pillar_raft.py --ckpt model_path

在自定义校园数据集训练
python train_pillar_raft_school.py --ckpt model_path

## dynamic lidar odometry network
在kitti-odometry数据集
python train_dylo.py --ckpt model_path 

在kitti-tracking数据集
python train_dylo_tracking_odom.py --ckpt model_path 

在自定义校园数据集
python train_dylo_school.py --ckpt model_path 

## global optimization
python dynamic_backend_optmization.py 

