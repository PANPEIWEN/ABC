# @Time    : 2022/9/14 20:31
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : build_dataset.py
# @Software: PyCharm
from utils.data import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def build_dataset(args, cfg):
    trainset = DatasetLoad(mode='train', **cfg.data)
    testset = DatasetLoad(mode='test', **cfg.data)
    if args.local_rank != -1:
        train_sample = DistributedSampler(trainset)
        train_data = DataLoader(dataset=trainset, batch_size=cfg.data['train_batch'], sampler=train_sample,
                                pin_memory=True, num_workers=cfg.data['num_workers'], drop_last=True)
        test_data = DataLoader(dataset=testset, batch_size=cfg.data['test_batch'], shuffle=False,
                               num_workers=cfg.data['num_workers'], drop_last=False)
    else:
        train_data = DataLoader(dataset=trainset, batch_size=cfg.data['train_batch'], shuffle=True,
                                num_workers=cfg.data['num_workers'], drop_last=True)
        test_data = DataLoader(dataset=testset, batch_size=cfg.data['test_batch'], shuffle=False,
                               num_workers=cfg.data['num_workers'], drop_last=False)
    return [train_sample, train_data, test_data, trainset.__len__(), testset.__len__()] \
        if args.local_rank != -1 else [train_data, test_data, trainset.__len__(), testset.__len__()]
