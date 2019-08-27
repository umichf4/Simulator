# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-08-05 12:45:28
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-08-26 15:36:27

import os
import sys
import numpy as np
import torch.nn as nn
import torch


def index_along(tensor, key, axis):
    indexer = [slice(None)] * len(tensor.shape)
    indexer[axis] = key
    return tensor[tuple(indexer)]


def pad_periodic(inputs, padding: int, axis: int, center: bool = True):

    if padding == 0:
        return inputs
    if center:
        if padding % 2 != 0:
            raise ValueError(
                'cannot do centered padding if padding is not even')
        inputs_list = [index_along(inputs, slice(-padding // 2, None), axis),
                       inputs,
                       index_along(inputs, slice(None, padding // 2), axis)]
    else:
        inputs_list = [inputs, index_along(inputs, slice(None, padding), axis)]
    return torch.cat(inputs_list, dim=axis)


def pad1d_meta(inputs, padding: int):
    return pad_periodic(inputs, padding, axis=-1, center=True)


class ConvTranspose1d_meta(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 groups=1, bias=True, dilation=1):
        super().__init__()
        self.padding = kernel_size - 1
        self.trim = self.padding * stride // 2
        pad = (kernel_size - stride) // 2
        self.output_padding = (kernel_size - stride) % 2
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=pad,
                                                   output_padding=0, groups=groups, bias=bias, dilation=dilation)

    def forward(self, inputs):
        padded_inputs = pad1d_meta(inputs, self.padding)
        padded_outputs = self.conv1d_transpose(padded_inputs)
        if self.output_padding:
            padded_outputs = padded_outputs[:, :, 1:]

        if self.trim:
            return padded_outputs[:, :, self.trim:-self.trim]
        else:
            return padded_outputs


class SimulatorNet(nn.Module):
    def __init__(self, in_num=2, out_num=28):
        super().__init__()

        self.FC = nn.Sequential(
            # ------------------------------------------------------
            nn.Linear(in_num, 448),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            nn.Linear(448, 896, bias=False),
            nn.BatchNorm1d(896),
            nn.LeakyReLU(0.2),
        )

        self.CONV = nn.Sequential(
            # ------------------------------------------------------
            ConvTranspose1d_meta(128, 32, 5, stride=2, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            ConvTranspose1d_meta(32, 8, 5, stride=2, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            ConvTranspose1d_meta(8, 4, 5, stride=1, bias=False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            ConvTranspose1d_meta(4, 2, 5, stride=1, bias=False),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2),
            # ------------------------------------------------------
            ConvTranspose1d_meta(2, 1, 5, stride=1, bias=False),
            # nn.BatchNorm1d(1),
            # nn.LeakyReLU(0.2),
        )

        self.shortcut = nn.Sequential()

    def forward(self, z):

        net = self.FC(z)
        net = net.view(-1, 128, 7)
        net = self.CONV(net)
        # net += self.shortcut(z[:, 2:].view_as(net))

        return net


if __name__ == '__main__':
    import torchsummary

    if torch.cuda.is_available():
        simulator = SimulatorNet().cuda()
    else:
        simulator = SimulatorNet()

    torchsummary.summary(simulator, tuple([2]))
