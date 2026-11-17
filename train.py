# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import numpy as np
import random
from trainer import Trainer
from options import RMSFM2Options
import torch
import os

if __name__ == "__main__":
    options = RMSFM2Options()
    opts = options.parse()
    trainer = Trainer(opts)
    trainer.train()

