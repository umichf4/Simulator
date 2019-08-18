# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-08-17 15:20:26
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-08-18 12:49:18
import torch
import os
import json
import matplotlib.pyplot as plt
from cmath import phase, rect


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def save_checkpoint(state, path, name):
    if not os.path.exists(path):
        print("Checkpoint Directory does not exist! Making directory {}".format(path))
        os.mkdir(path)

    torch.save(state, name)

    print('Model saved')


def load_checkpoint(path, net, optimizer):
    if not os.path.exists(path):
        raise("File doesn't exist {}".format(path))

    state = torch.load(path)
    net.load_state_dict(state['net_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optim_state_dict'])

    print('Model loaded')


def make_figure_dir():
    os.makedirs('figures/loss_curves', exist_ok=True)
    os.makedirs('figures/test_output', exist_ok=True)


def plot_spectrum(data, name):
    save_dir = os.path.join('figures/test_output', name)
    plt.figure()
    plt.plot(range(len(data)), data, 'o')
    plt.legend(('Spectrum',), loc='best')
    plt.title('Spectrum')
    plt.savefig(save_dir)
    plt.close()


def rect2polar(real, imag):
    complex_number = complex(real, imag)
    return abs(complex_number), phase(complex_number)


def polar2rect(modu, phase):
    complex_number = rect(modu, phase)
    return complex_number.real, complex_number.imag
