# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-08-17 15:20:26
# @Last Modified by:   BrandonHanx
# @Last Modified time: 2019-08-20 18:20:15

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
from torch.optim import lr_scheduler


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
    TT_list, TT_array = load_mat(os.path.join(current_dir, params.T_path))
    np.random.shuffle(TT_array)   
    all_num = TT_array.shape[0]
    TT_tensor = torch.from_numpy(TT_array)
    TT_tensor = TT_tensor.double()

    x = TT_tensor[:, :-1]
    train_x = x[:int(all_num * params.ratio), :]
    valid_x = x[int(all_num * params.ratio):, :]

    y = TT_tensor[:, -1]
    train_y = y[:int(all_num * params.ratio)]
    valid_y = y[int(all_num * params.ratio):]

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=True)

    valid_dataset = TensorDataset(valid_x, valid_y)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=valid_x.shape[0], shuffle=True)

    # Net configuration
    net = SimulatorNet(in_num=params.in_num, out_num=params.out_num)
    net = net.double()
    net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=params.lr)

    criterion = nn.MSELoss()
    train_loss_list, val_loss_list, epoch_list = [], [], []

    if params.restore_from:
        load_checkpoint(params.restore_from, net, optimizer)

    # Start training
    for k in range(params.epochs):
        epoch = k + 1
        epoch_list.append(epoch)
        
        # Train
        net.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            train_loss = criterion(outputs, labels)
            train_loss.backward()

            optimizer.step()

        train_loss_list.append(train_loss)

        # Validation
        net.eval()
        val_loss = 0
        for i, data in enumerate(valid_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

            val_loss += criterion(outputs, labels).sum()

        val_loss /= (i + 1)
        val_loss_list.append(val_loss)

        print('Epoch=%d  train_loss: %.7f valid_loss: %.7f lr: %.7f' %
              (epoch, train_loss, val_loss, scheduler.get_lr()[0]))
        
        scheduler.step()
        
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
            path = os.path.join(current_dir, params.save_model_dir)
            name = os.path.join(current_dir, params.save_model_dir, 'Epoch' + str(epoch) + '.pth')
            save_checkpoint({'net_state_dict': net.state_dict(),
                             'optim_state_dict': optimizer.state_dict(),
                             },
                            path=path, name=name)

        if epoch == params.epochs:
            path = os.path.join(current_dir, params.save_model_dir)
            name = os.path.join(current_dir, params.save_model_dir, 'Epoch' + str(epoch) + '_final.pth')
            save_checkpoint({'net_state_dict': net.state_dict(),
                             'optim_state_dict': optimizer.state_dict(),
                             },
                            path=path, name=name)

    print('Finished Training')


def find_spectrum(thickness, radius, TT_array):
    rows, _ = TT_array.shape
    wavelength, spectrum = [], []
    for row in range(rows):
        if TT_array[row, 1] == thickness and TT_array[row, 2] == radius:
            wavelength.append(TT_array[row, 0])
            spectrum.append(TT_array[row, -1])
        else:
            continue
    wavelength = np.array(wavelength)
    spectrum = np.array(spectrum)
    index_order = np.argsort(wavelength)
    return wavelength[index_order], spectrum[index_order]


def test_simulator(params):
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Test starts, using %s' % (device))

    # Visualization configuration
    make_figure_dir()

    _, TT_array = load_mat(params.T_path)

    net = SimulatorNet(in_num=params.in_num, out_num=params.out_num)
    net = net.double()
    net.to(device)
    if params.restore_from:
        load_checkpoint(params.restore_from, net, None)
    net.eval()
    thickness = 600
    radius = 60
    spectrum_fake = []

    wavelength_real, spectrum_real = find_spectrum(thickness, radius, TT_array)
    for wavelength in wavelength_real:
        test_data = [wavelength, thickness, radius]
        input_tensor = torch.from_numpy(np.array(test_data)).double().view(1, -1)
        output_tensor = net(input_tensor)
        spectrum_fake.append(output_tensor.view(-1).detach().cpu().numpy())

    spectrum_fake = np.array(spectrum_fake).squeeze()
    plot_both_parts(wavelength_real, spectrum_real, spectrum_fake, 'hhh.png')

    print('Finished Testing')
