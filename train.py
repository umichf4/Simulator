# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:12:01 2019

@author: Pi
"""
import torch
import torch.nn as nn
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
from torch.utils.data import DataLoader, TensorDataset
from net import SimulatorNet
from utils import *

def train_simulator(params, args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using %s'%(device))
    
    x = torch.rand(params['all_num'], params['in_num'])
    train_x = x[:int(params['all_num'] * params['ratio']), :]
    valid_x = x[int(params['all_num'] * params['ratio']):, :]
    
    y = torch.rand(params['all_num'], params['out_num'])
    train_y = y[:int(params['all_num'] * params['ratio']), :]
    valid_y = y[int(params['all_num'] * params['ratio']):, :]
    
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(dataset = train_dataset, batch_size = params['batchsize'], shuffle=True)
    
    valid_dataset = TensorDataset(valid_x, valid_y)
    valid_loader = DataLoader(dataset = valid_dataset, batch_size = valid_x.shape[0], shuffle=True)
    
    net = SimulatorNet(in_num = params['in_num'], out_num = params['out_num'])
    net.to(device)
    
    optimizer = torch.optim.SGD(net.parameters(), lr=params['lr'])
    criterion = nn.MSELoss()
    
    if args.restore_from is not None:
        load_checkpoint(args.restore_from, net, optimizer)
    
    for k in range(params['epochs']):
        epoch = k + 1 
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad()
    
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            
        train_loss = loss.item()      
                            
        for i, data in enumerate(valid_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            
            loss = criterion(outputs, labels)
            
        valid_loss = loss.item()     
        
        print('Epoch=%d  train_loss: %.7f valid_loss: %.7f' %
                      (epoch, train_loss, valid_loss))
        
        if epoch % params['save_epoch'] == 0 and epoch != params['epochs']:
            path = args.save_model_dir
            name = os.path.join(args.save_model_dir, 'Epoch'+str(epoch)+'.pth')
            save_checkpoint({'net_state_dict': net.state_dict(),
                             'optim_state_dict': optimizer.state_dict(),
                             },
                             path = path, name = name)
              
        if epoch == params['epochs']:
            path = args.save_model_dir
            name = os.path.join(args.save_model_dir, 'Epoch'+str(epoch)+'_final.pth')
            save_checkpoint({'net_state_dict': net.state_dict(),
                             'optim_state_dict': optimizer.state_dict(),
                             },
                             path = path, name = name)
    
    print('Finished Training')

