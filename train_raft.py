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

from configs import dynamic_seg_args
from tools.excel_tools import SaveExcel
from tools.euler_tools import quat2mat
from tools.logger_tools import log_print, creat_logger
from kitti_pytorch import semantic_points_dataset
from raft.raft import RAFT
from utils1.collate_functions import collate_pair
from raft.segment_losses import SegmentLoss, KDPointToPointLoss, knnLoss, Lovasz_softmax
from model_utils import ProjectPCimg2SphericalRing
from ioueval import iouEval


f = open('dataset_config.yaml')
dataset_config = yaml.load(f, Loader=yaml.FullLoader)

args = dynamic_seg_args()

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
checkpoints_dir = os.path.join(file_dir, 'checkpoints/raftseg')
if not os.path.exists(checkpoints_dir): os.makedirs(checkpoints_dir)

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
        # print(i," ", loss)

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

    train_dir_list = [1]#[0, 1, 2, 3, 4, 5, 6]
    test_dir_list = [4]#[7, 8, 9, 10]

    logger = creat_logger(log_dir, args.model_name)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # excel_eval = SaveExcel(test_dir_list, log_dir)sequence_flow_loss
    model = RAFT(args)
    loss_fn1 = SegmentLoss(dataset_config).cuda()
    loss_fn2 = knnLoss().cuda()
    # loss_fn2 = KDPointToPointLoss().cuda()
    # train set
    train_dataset = semantic_points_dataset(
        is_training = 1,
        num_point=args.num_points,
        data_dir_list=train_dir_list,
        config=args
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_pair,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )#collate_fn=collate_pair,

    if args.multi_gpu is not None:
        device_ids = [int(x) for x in args.multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.cuda(device_ids[0])
        log_print(logger, 'multi gpu are:' + str(args.multi_gpu))
    else:

        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.gpu)
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
        init_epoch = checkpoint['epoch']
        log_print(logger, 'load model {}'.format(args.ckpt))

    else:
        init_epoch = 0
        log_print(logger, 'Training from scratch')

    evaluator = iouEval(3, 'cuda', [0])
    # eval once before training
    if args.eval_before == 1:
        eval(model, test_dir_list, init_epoch, logger, tb_writer, evaluator)
        # excel_eval.update(eval_dir)

    for epoch in range(init_epoch + 1, args.max_epoch):
        total_loss = 0
        total_seen = 0
        acc = AverageMeter()
        static_iou = AverageMeter()
        moving_iou = AverageMeter()
        sematic_loss = AverageMeter()
        icp_loss = AverageMeter()
        
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        model = model.train()
        print("lr now: ", scheduler.get_last_lr())
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
        # for i, data in enumerate(train_loader, 0):  

            pos2, pos1, label2, sample_id, T_gt, T_trans, T_trans_inv, Tr = data
            pos2 = [b.cuda() for b in pos2]
            pos1 = [b.cuda() for b in pos1]
            label2 = [b.cuda() for b in label2]
            # print(sample_id)
            image2, img_label2 = ProjectPCimg2SphericalRing(pos2, label2, args.H_input, args.W_input)
            # print(img_label2.squeeze().shape)
            # T_trans = T_trans.cuda().to(torch.float32)
            # T_trans_inv = T_trans_inv.cuda().to(torch.float32)

            # T_inv = torch.linalg.inv(T_gt.cuda().to(torch.float32))
            
            # # 利用变换矩阵T_gt将pos1转换到pos2
            # trans_pos1 = []
            # for i, p1 in enumerate(pos1, 0):
            #     padp = torch.ones(p1.shape[0]).unsqueeze(1).cuda()
            #     hom_p1 = torch.cat([p1, padp], dim=1).transpose(0,1)
            #     trans_pos1.append(torch.mm(T_inv[i], hom_p1).transpose(0,1)[:,:-1])
            
            # forward
            warp_image1s, moving_predicts = model(pos1, pos2)
            # movinglabel loss
            loss1 = sequence_loss(moving_predicts, img_label2.squeeze(), loss_fn1)
            # knn loss
            loss2 = sequence_loss(warp_image1s, image2, loss_fn2, gap=1)
            # print("movinglabel loss : ", loss1, "knln loss :", loss2)
            # total loss
            loss = loss1.cuda() + 0.5*loss2.cuda()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("infer time:", t2-t1, "loss1 time:",t3-t2, "loss2 time:", t4-t3, "backward time:", time.time()-t4)
            with torch.no_grad():
                evaluator.reset()
                argmax = moving_predicts[-1].argmax(dim=1)
                evaluator.addBatch(argmax.long(), img_label2.squeeze().long())
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoU()
                
            acc.update(accuracy.item(), len(pos2))
            static_iou.update(class_jaccard[1].item(), len(pos2))
            moving_iou.update(class_jaccard[2].item(), len(pos2))
            sematic_loss.update(loss1.cpu().data, len(pos2))
            icp_loss.update(loss2.cpu().data, len(pos2))

            total_loss += loss.cpu().data * args.batch_size
            total_seen += args.batch_size

        scheduler.step()
        
        lr = max(optimizer.param_groups[0]['lr'], args.learning_rate_clip)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_loss = total_loss / total_seen
        log_print(logger,'EPOCH {} train mean loss: {:04f} sematic loss: {:04f} icp loss: {:04f} accuracy: {:04f} static iou: {:04f} \
        moving iou: {:04f}'.format(epoch, float(train_loss), sematic_loss.avg, icp_loss.avg, float(acc.avg), static_iou.avg, moving_iou.avg))
        # write to tensorboard
        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("train_sematic_loss", sematic_loss.avg, epoch)
        tb_writer.add_scalar("train_icp_loss", icp_loss.avg, epoch)
        tb_writer.add_scalar("train_accuracy", acc.avg, epoch)
        tb_writer.add_scalar("train_static_iou", static_iou.avg, epoch)
        tb_writer.add_scalar("train_moving_iou", moving_iou.avg, epoch)

        if epoch % 5 == 0:
            save_path = os.path.join(checkpoints_dir,
                                     '{}_{:03d}_{:04f}.pth.tar'.format(model.__class__.__name__, epoch, float(train_loss)))
            torch.save({
                'model_state_dict': model.module.state_dict() if args.multi_gpu else model.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch
            }, save_path)
            log_print(logger, 'Save {}...'.format(model.__class__.__name__))

            eval(model, test_dir_list, epoch, logger, tb_writer, evaluator)
            # excel_eval.update(eval_dir)


def eval(model, test_list, epoch, logger, tb_writer, evaluator):
    # loss_ls = Lovasz_softmax(ignore=0).to('cuda')
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
        # switch to evaluate mode
        model = model.eval()
        evaluator.reset()
        with torch.no_grad():
            for batch_id, data in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
                pos2, pos1, label2, sample_id, T_gt, T_trans, T_trans_inv, Tr = data
                pos2 = [b.cuda() for b in pos2]
                pos1 = [b.cuda() for b in pos1]
                label2 = [b.cuda() for b in label2]
                
                _, img_label2 = ProjectPCimg2SphericalRing(pos2, label2, args.H_input, args.W_input)
                # T_inv = torch.linalg.inv(T_gt.cuda().to(torch.float32))
                # # 利用变换矩阵T_gt将pos1转换到pos2,实现静态点场景流置零
                # trans_pos1 = []
                # for i, p1 in enumerate(pos1, 0):
                #     padp = torch.ones(p1.shape[0]).unsqueeze(1).cuda()
                #     hom_p1 = torch.cat([p1, padp], dim=1).transpose(0,1)
                #     trans_pos1.append(torch.mm(T_inv[i], hom_p1).transpose(0,1)[:,:-1])

                # infer
                _, output = model(pos1, pos2)
                argmax = output[-1].argmax(dim=1)
                # print(output)
                # print(img_label2.squeeze())
                # jacc = loss_ls( output[-1], img_label2.squeeze().long())
                # print(jacc)
                evaluator.addBatch(argmax.long(), img_label2.squeeze().long())
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoU()
                # print(class_jaccard)
                acc.update(accuracy.item(), len(pos2))
                static_iou.update(class_jaccard[1].item(), len(pos2))
                moving_iou.update(class_jaccard[2].item(), len(pos2))
                
            log_print(logger,'EVAL: EPOCH {} Seq: {} accuracy: {:04f} static iou: {:04f} \
            moving iou: {:04f} '.format(epoch, item, float(acc.avg), static_iou.avg, moving_iou.avg))
            # write to tensorboard
            tb_writer.add_scalar("eval_accuracy", acc.avg, epoch)
            tb_writer.add_scalar("eval_static_iou", static_iou.avg, epoch)
            tb_writer.add_scalar("eval_moving_iou", moving_iou.avg, epoch)

    return 0


if __name__ == '__main__':
    main()
