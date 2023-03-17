# @Time    : 2022/4/6 14:58
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : loss.py
# @Software: PyCharm
import torch
import torch.nn as nn


class SoftIoULoss(nn.Module):
    def __init__(self, **kwargs):
        super(SoftIoULoss, self).__init__()

    def forward(self, pred, target):
        # Old One
        pred = torch.sigmoid(pred)
        smooth = 1

        # print("pred.shape: ", pred.shape)
        # print("target.shape: ", target.shape)

        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)

        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()

        return loss


class CrossEntropy(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean',
                 label_smoothing=0.0, **kwargs):
        super(CrossEntropy, self).__init__()
        self.crit = nn.CrossEntropyLoss(weight, size_average, ignore_index, reduce, reduction, label_smoothing)

    def forward(self, pred, target):
        target.squeeze(dim=1)
        loss = self.crit(pred, target)
        return loss


class BCEWithLogits(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None, **kwargs):
        super(BCEWithLogits, self).__init__()
        self.crit = nn.BCEWithLogitsLoss(weight, size_average, reduce, reduction, pos_weight)

    def forward(self, pred, target):
        loss = self.crit(pred, target)
        return loss
