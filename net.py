# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:09:34 2019

@author: Pi
"""
#import torch
import torch.nn as nn
#import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, in_num = 3, out_num = 31):
        super(Net, self).__init__()
        self.FC = nn.Sequential(
            # ------------------------------------------------------
            nn.Linear(in_num, 50),
            nn.ReLU(),
            # ------------------------------------------------------
            nn.Linear(50, 500),
            nn.ReLU(),
            # ------------------------------------------------------
            nn.Linear(500, 500),
            nn.ReLU(),
            # ------------------------------------------------------
            nn.Linear(500, 50),
            nn.ReLU(),
            # ------------------------------------------------------
            nn.Linear(50, out_num)
        )

    def forward(self, x):
        x = self.FC(x)
        return x