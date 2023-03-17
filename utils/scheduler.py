# @Time    : 2022/5/31 17:19
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : scheduler.py
# @Software: PyCharm
import math


def linear(optimizer, epoch, base_lr, warmup_epoch=5):
    if epoch == 0:
        lr = base_lr / warmup_epoch
    else:
        lr = epoch * (base_lr / warmup_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class PolyLR(object):
    def __init__(self, optimizer, num_epochs, base_lr, warmup, power=0.9, warmup_epochs=5, **kwargs):
        super(PolyLR, self).__init__()
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.base_lr = base_lr
        self.warmup = warmup
        self.warmup_epoch = warmup_epochs if self.warmup else 0
        self.power = power

    def step(self, epoch):
        if self.warmup and epoch <= self.warmup_epoch:
            globals()[self.warmup](self.optimizer, epoch, self.base_lr, self.warmup_epoch)
        else:
            lr = self.base_lr * (1 - (epoch - self.warmup_epoch) / self.num_epochs) ** self.power
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr


class CosineAnnealingLR(object):
    def __init__(self, optimizer, num_epochs, base_lr, warmup, min_lr=1e-4, warmup_epochs=5, **kwargs):
        super(CosineAnnealingLR, self).__init__()
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.base_lr = base_lr
        self.warmup = warmup
        self.warmup_epoch = warmup_epochs if self.warmup else 0
        self.min_lr = min_lr

    def step(self, epoch):
        if self.warmup and epoch <= self.warmup_epoch:
            globals()[self.warmup](self.optimizer, epoch, self.base_lr, self.warmup_epoch)
        else:
            lr = self.min_lr + ((self.base_lr - self.min_lr) / 2) * (
                    1 + math.cos((epoch - self.warmup_epoch) / self.num_epochs * math.pi))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr


class StepLR(object):
    def __init__(self, optimizer, step, base_lr, warmup, gamma=0.1, warmup_epochs=5, **kwargs):
        super(StepLR, self).__init__()
        self.optimizer = optimizer
        self.step = step
        self.gamma = gamma
        self.base_lr = base_lr
        self.warmup = warmup
        self.warmup_epoch = warmup_epochs if self.warmup else 0

    def step(self, epoch):
        if self.warmup and epoch <= self.warmup_epoch:
            globals()[self.warmup](self.optimizer, epoch, self.base_lr, self.warmup_epoch)
        else:
            if epoch in self.step:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.gamma
