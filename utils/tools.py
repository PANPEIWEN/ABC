# @Time    : 2022/9/14 22:11
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : tools.py
# @Software: PyCharm

import numpy as np
import torch
import random


def random_seed(n):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)


def do_nothing():
    pass


def model_wrapper(model_dict):
    new_dict = {}
    for k, v in model_dict.items():
        new_dict['decode_head.' + k] = v
    return new_dict
