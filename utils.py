# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib


def select_device(no_cuda=False):
    """Honor --no_cuda and report unavailable CUDA before loading data/models."""
    import torch

    if no_cuda:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check nvidia-smi and the CUDA-enabled PyTorch "
            "installation in the active conda environment, or pass --no_cuda "
            "to run on CPU."
        )
    return torch.device("cuda")


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class ChainedScheduler(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mul=1, eta_min=0, last_epoch=-1, verbose=False, warmup_steps=0, decay_rate=1., max_lr=None, gamma=None):

        self.T_0 = T_0
        self.T_mul = T_mul
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        
        self.T_cur = last_epoch
        self.T_i = T_0
        self.T_total = 0

        super(ChainedScheduler, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        
        return [self.eta_min + (base_lr * self.decay_rate - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mul
                self.T_total = self.T_total + self.T_i
                # self.decay_rate *= self.decay_rate # Original logic might vary, usually fixed or step decay
        else:
            if epoch < 0:
                raise ValueError("Epoch number should be non-negative, but got {}".format(epoch))
            if epoch < self.warmup_steps:
                self.T_cur = epoch
            else:
                self.T_cur = epoch - self.warmup_steps
                self.T_i = self.T_0
                while self.T_cur >= self.T_i:
                    self.T_cur -= self.T_i
                    self.T_i *= self.T_mul
        
        self.last_epoch = math.floor(epoch)
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
