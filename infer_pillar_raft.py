# -*- coding:UTF-8 -*-

import os
import sys
import torch
import datetime
import torch.utils.data
import numpy as np
import time
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs import dynamic_seg_infer_args
from tools.logger_tools import log_print, creat_logger
from kitti_pytorch import semantic_points_dataset
from raft.pillar_raft import RAFT
from utils1.collate_functions import collate_pair
from ioueval import iouEval
from pointpillar_encoder import PillarLayer
from tools.save_seg_result import save_seg_result

f = open('tools/dataset_config.yaml')
dataset_config = yaml.load(f, Loader=yaml.FullLoader)

args = dynamic_seg_infer_args()

'''CREATE DIR'''
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
experiment_dir = os.path.join(base_dir, 'experiment')
if not os.path.exists(experiment_dir): os.makedirs(experiment_dir)
if not args.task_name:
    file_dir = os.path.join(experiment_dir, '{}_KITTI_{}'.format(args.model_name, str(
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))))
else:
    file_dir = os.path.join(experiment_dir, args.task_name)
if not os.path.exists(file_dir): os.makedirs(file_dir)
eval_dir = os.path.join(file_dir, 'eval')
if not os.path.exists(eval_dir): os.makedirs(eval_dir)
log_dir = os.path.join(file_dir, 'logs')
if not os.path.exists(log_dir): os.makedirs(log_dir)

'''LOG'''
tb_writer = SummaryWriter(log_dir)

def sequence_loss(pred_list, gt, loss_fn, gamma=0.8, gap=1):
    """ Loss function defined over sequence of predictions """
    n_predictions = len(pred_list)    
    seq_loss = 0.0
    # label_gt = label_gt.unsqueeze(0)
    for i in range(int(n_predictions/gap)):
        i_weight = gamma**(n_predictions - i - 1)
        loss = loss_fn(gt, pred_list[i])
        seq_loss += i_weight * (loss)
    return seq_loss

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():

    global args, dataset_config, tb_writer

    test_dir_list =[6,7,8]

    logger = creat_logger(log_dir, args.model_name)
    logger.info('----------------------------------------EVALING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(args.gpu)

    model = RAFT(args)
    model.cuda()
    log_print(logger, 'just one gpu is:' + str(args.gpu))

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                    momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=args.weight_decay)
        
    optimizer.param_groups[0]['initial_lr'] = args.learning_rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_stepsize,
                                                gamma=args.lr_gamma, last_epoch=-1)

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['opt_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        log_print(logger, 'load model {}'.format(args.ckpt))
    evaluator = iouEval(3, 'cuda', [0])
    eval(model, test_dir_list, 0, logger, tb_writer, evaluator)


def eval(model, test_list, epoch, logger, tb_writer, evaluator):
    global args
    bev_proj_fn = PillarLayer(args.voxel_size,args.point_cloud_range,args.max_num_points,args.max_voxels)
    for item in test_list:
        acc = AverageMeter()
        static_iou = AverageMeter()
        moving_iou = AverageMeter()
        test_dataset = semantic_points_dataset(
            is_training = 0,
            num_point = args.num_points,
            data_dir_list = [item],
            config = args
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_pair,
            pin_memory=False,
            worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
        )
        evaluator.reset()
        with torch.no_grad():
            for batch_id, data in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
                pos1, pos2, label1, path_seq, sample_id, T_gt, T_trans, T_trans_inv, Tr = data
                pos1 = [b.cuda() for b in pos1]
                pos2 = [b.cuda() for b in pos2]
                label1 = [b.cuda() for b in label1]

                _, _, _, batched_pillar_all, img1, img_label1 = bev_proj_fn(pos1, label1)
                
                # forward
                moving_masks = model(pos1, pos2, T_gt.cuda().to(torch.float32))
                argmax = moving_masks[-1].argmax(dim=1)
                save_seg_result(eval_dir, str("{:0>2d}".format(path_seq.tolist()[0])), str("{:0>6d}".format(sample_id.tolist()[0])), batched_pillar_all, argmax)
                evaluator.addBatch(argmax.long(), img_label1.squeeze().long())
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoU()
                acc.update(accuracy.item(), len(pos2))
                static_iou.update(class_jaccard[1].item(), len(pos2))
                moving_iou.update(class_jaccard[2].item(), len(pos2))
                
            log_print(logger,'EVAL: EPOCH {} accuracy: {:04f} static iou: {:04f} \
            moving iou: {:04f} '.format(epoch, float(acc.avg), static_iou.avg, moving_iou.avg))
            # write to tensorboard
            tb_writer.add_scalar("eval_accuracy", acc.avg, epoch)
            tb_writer.add_scalar("eval_static_iou", static_iou.avg, epoch)
            tb_writer.add_scalar("eval_moving_iou", moving_iou.avg, epoch)

    return 0


if __name__ == '__main__':
    main()
