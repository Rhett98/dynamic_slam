# -*- coding:UTF-8 -*-

import argparse


"""
  config
"""
  
  
def dynamic_seg_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 3]')
    parser.add_argument('--multi_gpu', type=str, default=None, help='The gpu [default : null]')
    parser.add_argument('--limit_or_filter', type=bool, default=True, help='if False, filter will reserve 40m~50m points')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 16]')
    parser.add_argument('--eval_batch_size', type=int, default=2, help='Batch Size during evaling [default: 64]')
    parser.add_argument('--eval_before', type=int, default=0, help='if 1, eval before train')

    parser.add_argument('--lidar_root', default='/home/yu/Resp/dataset/sequences', help='Dataset directory [default: /dataset]')
    parser.add_argument('--image_root', default='/dataset/data_odometry_color', help='Dataset directory [default: /dataset]')
    parser.add_argument('--log_dir', default='log_train', help='Log dir [default: log_train]')

    parser.add_argument('--num_points', type=int, default=150000, help='Point Number [default: 2048]')

    parser.add_argument('--H_input', type=int, default=64, help='H Number [default: 64]')
    parser.add_argument('--W_input', type=int, default=1536, help='W Number [default: 1800]')

    parser.add_argument('--max_epoch', type=int, default=251, help='Epoch to run [default: 151]')
    parser.add_argument('--weight_decay', type=int, default=0.0001, help='The Weight decay [default : 0.0001]')
    parser.add_argument('--workers', type=int, default=2,
                        help='Sets how many child processes can be used [default : 16]')
    parser.add_argument('--model_name', type=str, default='raftnet', help='base_dir_name [default: pwclonet]')
    parser.add_argument('--task_name', type=str, default=None, help='who can replace model_name ')

    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--optimizer', default='Adam', help='adam or momentum [default: adam]')
    
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--initial_lr', type=bool, default=False, help='Initial learning rate or not [default: False]')
    parser.add_argument('--learning_rate_clip', type=float, default=1e-6, help='learning_rate_clip [default : 1e-5]')
    parser.add_argument('--lr_stepsize', type=int, default=16, help="lr_stepsize")
    parser.add_argument('--lr_gamma', type=float, default=0.8, help="lr_gamma")
    parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
    parser.add_argument('--is_training', type=bool, default=True, help='is_training [default : True]')
    # pillar parameter
    parser.add_argument('--voxel_size', type=float, default=[0.2, 0.2, 9], help=' [default: 0.9]')
    parser.add_argument('--point_cloud_range', type=float, default=[-40, -40, -3, 40, 40, 6], help=' [default: 0.9]')
    parser.add_argument('--max_num_points', type=float, default=5, help=' [default: 0.9]')
    parser.add_argument('--max_voxels', type=float, default=(16000, 40000), help=' [default: 0.9]')
    
    args = parser.parse_args()
    return args
  
  
def odometry_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 3]')
    parser.add_argument('--multi_gpu', type=str, default=None, help='The gpu [default : null]')
    parser.add_argument('--limit_or_filter', type=bool, default=True, help='if False, filter will reserve 40m~50m points')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 16]')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='Batch Size during evaling [default: 64]')
    parser.add_argument('--eval_before', type=int, default=0, help='if 1, eval before train')

    parser.add_argument('--lidar_root', default='/home/yu/Resp/dataset/sequences', help='Dataset directory [default: /dataset]')
    parser.add_argument('--image_root', default='/dataset/data_odometry_color', help='Dataset directory [default: /dataset]')
    parser.add_argument('--log_dir', default='log_train', help='Log dir [default: log_train]')

    parser.add_argument('--num_points', type=int, default=150000, help='Point Number [default: 2048]')

    parser.add_argument('--H_input', type=int, default=64, help='H Number [default: 64]')
    parser.add_argument('--W_input', type=int, default=1536, help='W Number [default: 1800]')

    parser.add_argument('--max_epoch', type=int, default=351, help='Epoch to run [default: 151]')
    parser.add_argument('--weight_decay', type=int, default=0.0001, help='The Weight decay [default : 0.0001]')
    parser.add_argument('--workers', type=int, default=2,
                        help='Sets how many child processes can be used [default : 16]')
    parser.add_argument('--model_name', type=str, default='odometry', help='base_dir_name [default: odometry]')
    parser.add_argument('--task_name', type=str, default=None, help='who can replace model_name ')

    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--optimizer', default='Adam', help='adam or momentum [default: adam]')
    
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--initial_lr', type=bool, default=False, help='Initial learning rate or not [default: False]')
    parser.add_argument('--learning_rate_clip', type=float, default=1e-6, help='learning_rate_clip [default : 1e-5]')
    parser.add_argument('--lr_stepsize', type=int, default=16, help="lr_stepsize")
    parser.add_argument('--lr_gamma', type=float, default=0.8, help="lr_gamma")
    parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
    parser.add_argument('--is_training', type=bool, default=True, help='is_training [default : True]')

    args = parser.parse_args()
    return args
  
def odometry_tracking_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 3]')
    parser.add_argument('--multi_gpu', type=str, default=None, help='The gpu [default : null]')
    parser.add_argument('--limit_or_filter', type=bool, default=True, help='if False, filter will reserve 40m~50m points')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 16]')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='Batch Size during evaling [default: 64]')
    parser.add_argument('--eval_before', type=int, default=0, help='if 1, eval before train')

    parser.add_argument('--lidar_root', default='/home/yu/Resp/dataset/data_tracking_velodyne/training', help='Dataset directory [default: /dataset]')
    parser.add_argument('--image_root', default='/dataset/data_odometry_color', help='Dataset directory [default: /dataset]')
    parser.add_argument('--log_dir', default='log_train', help='Log dir [default: log_train]')

    parser.add_argument('--num_points', type=int, default=150000, help='Point Number [default: 2048]')

    parser.add_argument('--H_input', type=int, default=64, help='H Number [default: 64]')
    parser.add_argument('--W_input', type=int, default=1536, help='W Number [default: 1800]')

    parser.add_argument('--max_epoch', type=int, default=351, help='Epoch to run [default: 151]')
    parser.add_argument('--weight_decay', type=int, default=0.0001, help='The Weight decay [default : 0.0001]')
    parser.add_argument('--workers', type=int, default=2,
                        help='Sets how many child processes can be used [default : 16]')
    parser.add_argument('--model_name', type=str, default='odometry', help='base_dir_name [default: odometry]')
    parser.add_argument('--task_name', type=str, default=None, help='who can replace model_name ')

    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--optimizer', default='Adam', help='adam or momentum [default: adam]')
    
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--initial_lr', type=bool, default=False, help='Initial learning rate or not [default: False]')
    parser.add_argument('--learning_rate_clip', type=float, default=1e-6, help='learning_rate_clip [default : 1e-5]')
    parser.add_argument('--lr_stepsize', type=int, default=16, help="lr_stepsize")
    parser.add_argument('--lr_gamma', type=float, default=0.8, help="lr_gamma")
    parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
    parser.add_argument('--is_training', type=bool, default=True, help='is_training [default : True]')

    args = parser.parse_args()
    return args