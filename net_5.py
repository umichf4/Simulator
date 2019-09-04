# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch.nn as nn
import torch


class GeneratorNet(nn.Module):
    def __init__(self, in_num=3, out_num=29, d=64):
        super().__init__()
        self.in_num = in_num
        self.deconv_block = nn.Sequential(
            # ------------------------------------------------------
            nn.ConvTranspose1d(in_num, d * 8, 4, 1, 0),
            nn.BatchNorm1d(d * 8),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.ConvTranspose1d(d * 8, d * 4, 4, 2, 1),
            nn.BatchNorm1d(d * 4),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.ConvTranspose1d(d * 4, d * 2, 4, 2, 1),
            nn.BatchNorm1d(d * 2),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.ConvTranspose1d(d * 2, d, 4, 2, 1),
            nn.BatchNorm1d(d),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.ConvTranspose1d(d, int(d / 2), 4, 2, 1),
            nn.BatchNorm1d(int(d / 2)),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.ConvTranspose1d(int(d / 2), int(d / 4), 4, 2, 1),
            nn.BatchNorm1d(int(d / 4)),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.ConvTranspose1d(int(d / 4), int(d / 8), 4, 2, 1),
            nn.BatchNorm1d(int(d / 8)),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.ConvTranspose1d(int(d / 8), int(d / 16), 4, 2, 1),
            nn.BatchNorm1d(int(d / 16)),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.ConvTranspose1d(int(d / 16), int(d / 32), 4, 2, 1),
             nn.BatchNorm1d(int(d / 32)),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.ConvTranspose1d(int(d / 32), int(d / 64), 4, 2, 1),
            # ------------------------------------------------------
            nn.Tanh()
        )

        self.fc_block = nn.Sequential(
            # ------------------------------------------------------
            nn.Linear(2048, 2000),
            nn.BatchNorm1d(2000),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Linear(2000, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Linear(1000, out_num),
            nn.ReLU6()
        )

    def forward(self, net):
        net = net.view(-1, self.in_num, 1)
        net = self.deconv_block(net)
        print(net.size())
        net = net.view(net.size(0), -1)
        net = self.fc_block(net)

        return net


if __name__ == '__main__':
    import torchsummary

    if torch.cuda.is_available():
        generator = GeneratorNet().cuda()
    else:
        generator = GeneratorNet()

    torchsummary.summary(generator, tuple([3]))
