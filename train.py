# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:12:01 2019

@author: Pi
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from net import Net


wavelength = [400, 680] #THz
gap = 360 #nm
shape = 'circle'
in_num = 3
out_num = 32
all_num = 1000
batch_size = 50
lr = 0.1
epochs = 10
ratio = 0.8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

x = torch.rand(all_num, in_num)
train_x = x[:int(all_num * ratio), :]
valid_x = x[int(all_num * ratio):, :]

y = torch.rand(all_num, out_num)
train_y = y[:int(all_num * ratio), :]
valid_y = y[int(all_num * ratio):, :]

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)

valid_dataset = TensorDataset(valid_x, valid_y)
valid_loader = DataLoader(dataset = valid_dataset, batch_size = valid_x.shape[0], shuffle=True)

net = Net(in_num = in_num, out_num = out_num)
net.to(device)

optimizer = torch.optim.SGD(net.parameters(), lr=lr)
criterion = nn.MSELoss()

for epoch in range(epochs):
    
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
    
    print('Epoch=%d:  train_loss: %.7f valid_loss: %.7f' %
                  (epoch + 1, train_loss, valid_loss))

print('Finished Training')

