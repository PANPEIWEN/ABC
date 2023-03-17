# @Time    : 2022/9/14 20:12
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : build_criterion.py
# @Software: PyCharm
from utils.loss import *

__all__ = ['build_criterion', 'SoftIoULoss', 'BCEWithLogits', 'CrossEntropy']


#  TODO Multiple loss functions
def build_criterion(cfg):
    criterion_name = cfg.model['loss']['type']
    criterion_class = globals()[criterion_name]
    criterion = criterion_class(**cfg.model['loss'])
    return criterion
