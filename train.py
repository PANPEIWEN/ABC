# @Time    : 2022/4/6 15:23
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : train.py
# @Software: PyCharm
import argparse
import os
import time


import torch.distributed
import torch.nn
from torch.utils.tensorboard import SummaryWriter
from mmcv import Config, DictAction

from utils.metric import *
from utils.logs import *
from utils.save_model import *
from utils.drawing import *
from utils.tools import *

from build.build_model import build_model
from build.build_criterion import build_criterion
from build.build_optimizer import build_optimizer
from build.build_dataset import build_dataset
from build.build_scheduler import build_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


class Train(object):
    def __init__(self, args, cfg):
        super(Train, self).__init__()
        self.num_gpus = torch.cuda.device_count() if args.local_rank != -1 else 1
        self.cfg = cfg
        self.resume = args.resume_from
        self.deep_supervision = 'deep_supervision' in self.cfg.model['decode_head']
        self.cfg.gpus = self.num_gpus
        if args.local_rank != -1:
            device = torch.device('cuda', args.local_rank)
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend=self.cfg.dist_params['backend'])
            random_seed(42)
        else:
            self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
            random_seed(42)

        data = build_dataset(args, self.cfg)
        if args.local_rank != -1:
            self.train_sample, self.train_data, self.test_data, self.train_data_len, self.test_data_len = data
        else:
            self.train_data, self.test_data, self.train_data_len, self.test_data_len = data

        # TODO Initialized inside each model
        self.model = build_model(self.cfg)
        if args.load_from:
            self.cfg.load_from = args.load_from
            checkpoint = torch.load(args.load_from)
            self.model.load_state_dict(checkpoint)

        # FIXME Loss Accuracy Decreases When Use resume_from
        if args.resume_from:
            self.cfg.resume_from = args.resume_from
            checkpoint = torch.load(args.resume_from)
            self.model.load_state_dict(checkpoint['state_dict'])
        print("Model Initializing")

        if args.local_rank <= 0:
            self.save_dir = args.config.split('/')[-1][:-3]
            self.train_log_file = train_log_file()
            make_log_dir(self.save_dir, self.train_log_file)
            save_config_log(self.cfg, self.save_dir, self.train_log_file)
            self.write = SummaryWriter(log_dir='work_dirs/' + self.save_dir + '/' + self.train_log_file + '/tf_logs')

        if args.local_rank != -1:
            self.model.to(device)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[args.local_rank], output_device=args.local_rank,
                find_unused_parameters=True)
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        else:
            self.model = self.model.to(self.device)

        self.criterion = build_criterion(self.cfg)
        self.optimizer = build_optimizer(self.model, self.cfg)
        if self.cfg.lr_config['policy']:
            self.scheduler = build_scheduler(self.optimizer, self.cfg)

        self.iou_metric = SigmoidMetric()
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
        self.ROC = ROCMetric(1, 10)

        self.optimizer.load_state_dict(checkpoint['optimizer']) if args.resume_from else do_nothing()
        self.best_mIoU = checkpoint['best_mIoU'] if args.resume_from else 0.0
        self.best_nIoU = checkpoint['best_nIoU'] if args.resume_from else 0.0
        self.best_f1 = checkpoint['best_f1'] if args.resume_from else 0.0
        self.train_loss = checkpoint['train_loss'] if args.resume_from else []
        self.test_loss = checkpoint['test_loss'] if args.resume_from else []
        self.mIoU = checkpoint['mIoU'] if args.resume_from else []
        self.nIoU = checkpoint['nIoU'] if args.resume_from else []
        self.f1 = checkpoint['f1'] if args.resume_from else []
        self.num_epoch = checkpoint['num_epoch'] if args.resume_from else []

    def training(self, epoch):
        self.model.train()
        losses = []
        if args.local_rank != -1:
            self.train_sample.set_epoch(epoch)
        if not self.resume and self.cfg.lr_config['policy']:
            self.scheduler.step(epoch - 1)
        for i, (img, mask) in enumerate(self.train_data):
            since = time.time()
            if args.local_rank != -1:
                img = img.cuda()
                mask = mask.cuda()
            else:
                img = img.to(self.device)
                mask = mask.to(self.device)
            preds = self.model(img)
            # TODO when use deep supervision, should log pred loss, not all loss sum
            if self.deep_supervision and self.cfg.model['decode_head']['deep_supervision']:
                loss = []
                for pre in preds:
                    loss.append(self.criterion(pre, mask))
                loss = sum(loss)
            else:
                loss = self.criterion(preds, mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            time_elapsed = time.time() - since
            if args.local_rank <= 0:
                msg = 'Epoch %d/%d, Iter %d/%d, train loss %.4f, lr %.5f, time %.5f' % (
                    epoch, self.cfg.runner['max_epochs'], i + 1,
                    self.train_data_len / self.cfg.data['train_batch'] / self.num_gpus,
                    np.mean(losses), self.optimizer.state_dict()['param_groups'][0]['lr'], time_elapsed)
                print(msg)
                if (i + 1) % self.cfg.log_config['interval'] == 0:
                    save_train_log(self.save_dir, self.train_log_file, epoch, self.cfg.runner['max_epochs'], i + 1,
                                   self.train_data_len / self.cfg.data['train_batch'] / self.num_gpus,
                                   np.mean(losses), self.optimizer.state_dict()['param_groups'][0]['lr'], time_elapsed)
        if args.local_rank <= 0:
            ckpt_info = {
                'epoch': epoch,
                'state_dict': self.model.module.state_dict() if args.local_rank != -1 else self.model.state_dict(),
                'loss': np.mean(losses),
                'optimizer': self.optimizer.state_dict(),
                'train_loss': self.train_loss,
                'test_loss': self.test_loss,
                'num_epoch': self.num_epoch,
                'best_mIoU': self.best_mIoU,
                'best_nIoU': self.best_nIoU,
                'best_f1': self.best_f1,
                'mIoU': self.mIoU,
                'nIoU': self.nIoU,
                'f1': self.f1
            }
            save_ckpt(ckpt_info, save_path='work_dirs/' + self.save_dir + '/' + self.train_log_file,
                      filename='last.pth.tar')
            if self.cfg.checkpoint_config['by_epoch'] and epoch % self.cfg.checkpoint_config['interval'] == 0:
                save_ckpt(ckpt_info, save_path='work_dirs/' + self.save_dir + '/' + self.train_log_file,
                          filename='epoch_%d' % epoch + '.pth.tar')

            self.train_loss.append(np.mean(losses))
            self.num_epoch.append(epoch)
            self.write.add_scalar('train/train_loss', np.mean(losses), epoch)
            self.write.add_scalar('train/lr', self.optimizer.state_dict()['param_groups'][0]['lr'], epoch)

    def testing(self, epoch):
        self.model.eval()
        self.iou_metric.reset()
        self.nIoU_metric.reset()
        self.ROC.reset()
        eval_losses = []
        # tbar = tqdm(self.test_data)
        with torch.no_grad():
            for i, (img, mask) in enumerate(self.test_data):
                if args.local_rank != -1:
                    img = img.cuda()
                    mask = mask.cuda()
                else:
                    img = img.to(self.device)
                    mask = mask.to(self.device)
                preds = self.model(img)
                if self.deep_supervision and self.cfg.model['decode_head']['deep_supervision']:
                    loss = []
                    for pre in preds:
                        loss.append(self.criterion(pre, mask))
                    loss = sum(loss)
                    preds = preds[-1]
                else:
                    loss = self.criterion(preds, mask)
                eval_losses.append(loss.item())
                self.iou_metric.update(preds, mask)
                self.nIoU_metric.update(preds, mask)
                self.ROC.update(preds, mask)
                _, IoU = self.iou_metric.get()
                _, nIoU = self.nIoU_metric.get()
                _, _, _, _, F1_score = self.ROC.get()
                if args.local_rank <= 0:
                    msg = 'Epoch %d/%d, test loss %.4f, mIoU %.4f, nIoU %.4f, F1-score %.4f, ' \
                          'best_mIoU %.4f, ' \
                          'best_nIoU %.4f, best_F1-score %.4f' % (
                              epoch, self.cfg.runner['max_epochs'], np.mean(eval_losses), IoU, nIoU, F1_score,
                              self.best_mIoU,
                              self.best_nIoU, self.best_f1)
            if args.local_rank <= 0:
                print(msg)
                save_test_log(self.save_dir, self.train_log_file, epoch, self.cfg.runner['max_epochs'],
                              np.mean(eval_losses), IoU, nIoU, F1_score, self.best_mIoU, self.best_nIoU, self.best_f1)
                self.test_loss.append(np.mean(eval_losses))
                self.mIoU.append(IoU)
                self.nIoU.append(nIoU)
                self.f1.append(F1_score)
                # TODO Optimize saving model code
                if IoU > self.best_mIoU or nIoU > self.best_nIoU:
                    # FIXME save model
                    save_ckpt({
                        'epoch': epoch,
                        'state_dict': self.model.module.state_dict()
                        if args.local_rank != -1 else self.model.state_dict(),
                        'loss': np.mean(eval_losses),
                        'mIoU': IoU,
                        'nIoU': nIoU,
                        'f1': F1_score
                    }, save_path='work_dirs/' + self.save_dir + '/' + self.train_log_file, filename='best.pth.tar')
                if IoU > self.best_mIoU:
                    save_ckpt({
                        'epoch': epoch,
                        'state_dict': self.model.module.state_dict()
                        if args.local_rank != -1 else self.model.state_dict(),
                        'loss': np.mean(eval_losses),
                        'mIoU': IoU,
                        'nIoU': nIoU,
                        'f1': F1_score
                    }, save_path='work_dirs/' + self.save_dir + '/' + self.train_log_file, filename='best_mIoU.pth.tar')
                if nIoU > self.best_nIoU:
                    save_ckpt({
                        'epoch': epoch,
                        'state_dict': self.model.module.state_dict()
                        if args.local_rank != -1 else self.model.state_dict(),
                        'loss': np.mean(eval_losses),
                        'mIoU': IoU,
                        'nIoU': nIoU,
                        'f1': F1_score
                    }, save_path='work_dirs/' + self.save_dir + '/' + self.train_log_file, filename='best_nIoU.pth.tar')
                drawing_loss(self.num_epoch, self.train_loss, self.test_loss, self.save_dir, self.train_log_file)
                drawing_iou(self.num_epoch, self.mIoU, self.nIoU, self.save_dir, self.train_log_file)
                drawing_f1(self.num_epoch, self.f1, self.save_dir, self.train_log_file)
                self.best_mIoU = max(IoU, self.best_mIoU)
                self.best_nIoU = max(nIoU, self.best_nIoU)
                self.best_f1 = max(F1_score, self.best_f1)
                self.write.add_scalar('train/test_loss', np.mean(eval_losses), epoch)
                self.write.add_scalar('test/mIoU', IoU, epoch)
                self.write.add_scalar('test/nIoU', nIoU, epoch)
                self.write.add_scalar('test/F1-score', F1_score, epoch)


def main(args):
    cfg = Config.fromfile(args.config)
    trainer = Train(args, cfg)
    if args.local_rank != -1:
        torch.distributed.barrier()
    start = 1
    if args.resume_from:
        checkpoint = torch.load(args.resume_from)
        start = checkpoint['epoch'] + 1
    for i in range(start, cfg.runner['max_epochs'] + 1):
        trainer.training(i)
        trainer.testing(i)


if __name__ == '__main__':
    args = parse_args()
    main(args)
