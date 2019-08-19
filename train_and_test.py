# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-08-17 15:20:26
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-08-19 03:05:28

import torch
import torch.nn as nn
import visdom
import numpy as np
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
from torch.utils.data import DataLoader, TensorDataset
from net import SimulatorNet
from utils import *
from tqdm import tqdm


def train_simulator(params):
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Training starts, using %s' % (device))

    # Visualization configuration
    make_figure_dir()
    viz = visdom.Visdom()
    cur_epoch_loss = None
    cur_epoch_loss_opts = {
        'title': 'Epoch Loss Trace',
        'xlabel': 'Epoch Number',
        'ylabel': 'Loss',
        'width': 1200,
        'height': 600,
        'showlegend': True,
    }

    # Data configuration
    x = torch.rand(params.all_num, params.in_num)
    train_x = x[:int(params.all_num * params.ratio), :]
    valid_x = x[int(params.all_num * params.ratio):, :]

    y = torch.rand(params.all_num, params.out_num)
    train_y = y[:int(params.all_num * params.ratio), :]
    valid_y = y[int(params.all_num * params.ratio):, :]

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=True)

    valid_dataset = TensorDataset(valid_x, valid_y)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=valid_x.shape[0], shuffle=True)

    # Net configuration
    net_real = SimulatorNet(in_num=params.in_num, out_num=params.out_num)
    net_real.to(device)
    net_imag = SimulatorNet(in_num=params.in_num, out_num=params.out_num)
    net_imag.to(device)

    optimizer_real = torch.optim.SGD(net_real.parameters(), lr=params.lr)
    optimizer_imag = torch.optim.SGD(net_imag.parameters(), lr=params.lr)
    criterion = nn.MSELoss()
    train_loss_list, val_loss_list, epoch_list = [], [], []

    if params.restore_from is not None:
        load_checkpoint(params.restore_from, net_real, net_imag, optimizer_real, optimizer_imag)

    # Start training
    for k in range(params.epochs):
        epoch = k + 1
        epoch_list.append(epoch)

        # Train
        net_real.train()
        net_imag.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer_real.zero_grad()
            optimizer_imag.zero_grad()

            real_outputs = net_real(inputs)
            imag_outputs = net_imag(inputs)
            train_real_loss = criterion(real_outputs, labels)
            train_imag_loss = criterion(imag_outputs, labels)
            train_real_loss.backward()
            train_imag_loss.backward()

            train_loss = train_imag_loss + train_real_loss

            optimizer_real.step()
            optimizer_imag.step()

        train_loss_list.append(train_loss)

        # Validation
        net_real.eval()
        net_imag.eval()
        val_loss = 0
        for i, data in enumerate(valid_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            real_outputs = net_real(inputs)
            imag_outputs = net_imag(inputs)
            val_real_loss = criterion(real_outputs, labels)
            val_imag_loss = criterion(imag_outputs, labels)
            val_loss += (val_real_loss + val_imag_loss)

        val_loss /= (i + 1)
        val_loss_list.append(val_loss)

        print('Epoch=%d  train_real: %.7f train_imag: %.7f val_real: %.7f val_imag: %.7f' %
              (epoch, train_real_loss, train_imag_loss, val_real_loss, val_imag_loss))

        # Update Visualization
        if viz.check_connection():
            cur_epoch_loss = viz.line(torch.Tensor(train_loss_list), torch.Tensor(epoch_list),
                                      win=cur_epoch_loss, name='Train Loss',
                                      update=(None if cur_epoch_loss is None else 'replace'),
                                      opts=cur_epoch_loss_opts)
            cur_epoch_loss = viz.line(torch.Tensor(val_loss_list), torch.Tensor(epoch_list),
                                      win=cur_epoch_loss, name='Validation Loss',
                                      update=(None if cur_epoch_loss is None else 'replace'),
                                      opts=cur_epoch_loss_opts)

        if epoch % params.save_epoch == 0 and epoch != params.epochs:
            path = params.save_model_dir
            name = os.path.join(path, 'Epoch' + str(epoch) + '.pth')
            save_checkpoint({'real_net_state_dict': net_real.state_dict(),
                             'real_optim_state_dict': optimizer_real.state_dict(),
                             'imag_net_state_dict': net_imag.state_dict(),
                             'imag_optim_state_dict': optimizer_imag.state_dict(),
                             },
                            path=path, name=name)

        if epoch == params.epochs:
            path = params.save_model_dir
            name = os.path.join(path, 'Epoch' + str(epoch) + '_final.pth')
            save_checkpoint({'real_net_state_dict': net_real.state_dict(),
                             'real_optim_state_dict': optimizer_real.state_dict(),
                             'imag_net_state_dict': net_imag.state_dict(),
                             'imag_optim_state_dict': optimizer_imag.state_dict(),
                             },
                            path=path, name=name)

    print('Finished Training')


def test_simulator(params):
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Test starts, using %s' % (device))

    # Visualization configuration
    make_figure_dir()

    # Data configuration
    x = torch.rand(params.all_num, params.in_num)
    test_x = x[int(params.all_num * params.ratio):, :]

    y = torch.rand(params.all_num, params.out_num)
    test_y = y[int(params.all_num * params.ratio):, :]

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    # Net configuration
    net_real = SimulatorNet(in_num=params.in_num, out_num=params.out_num)
    net_imag = SimulatorNet(in_num=params.in_num, out_num=params.out_num)
    net_real.to(device)
    net_imag.to(device)
    if params.restore_from is not None:
        load_checkpoint(params.restore_from, net_real, net_imag)

    net_real.eval()
    net_imag.eval()

    with tqdm(total=200, ncols=70) as t:
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            real_outputs = net_real(inputs)
            imag_outputs = net_imag(inputs)
            real_pred = real_outputs.view(-1).cpu().detach().numpy()
            imag_pred = imag_outputs.view(-1).cpu().detach().numpy()
            # plot_single_part(real_pred, str(i) + '.png')
            plot_both_parts(real_pred, imag_pred, str(100 * i) + '.png')
            t.update()

    print('Finished Testing')
