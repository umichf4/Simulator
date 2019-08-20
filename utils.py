# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-08-17 15:20:26
# @Last Modified by:   BrandonHanx
# @Last Modified time: 2019-08-20 18:16:35
import torch
import os
import json
import matplotlib.pyplot as plt
import cmath
import numpy as np
from scipy import interpolate
import scipy.io as scio


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

    if torch.cuda.is_available():
        state = torch.load(path, map_location='cuda:0')
    else:
        state = torch.load(path, map_location='cpu')
    net.load_state_dict(state['net_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optim_state_dict'])

    print('Model loaded')


def interploate(org_data, points=1000):
    org = np.linspace(400, 680, len(org_data))
    new = np.linspace(400, 680, points)
    inter_func = interpolate.interp1d(org, org_data, kind='cubic')
    return inter_func(new)


def make_figure_dir():
    os.makedirs('figures/loss_curves', exist_ok=True)
    os.makedirs('figures/test_output', exist_ok=True)


def plot_single_part(wavelength, spectrum, name, legend='spectrum', interpolate=True):
    save_dir = os.path.join('figures/test_output', name)
    plt.figure()
    plt.plot(wavelength, spectrum, 'ob')
    plt.grid()
    if interpolate:
        new_spectrum = interploate(spectrum)
        new_wavelength = interploate(wavelength)
        plt.plot(new_wavelength, new_spectrum, '-b')
    plt.title(legend)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel(legend)
    plt.savefig(save_dir)
    plt.close()


def plot_both_parts(wavelength, real, fake, name, interpolate=True):

    color_left = 'blue'
    color_right = 'red'
    save_dir = os.path.join('figures/test_output', name)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Real', color=color_left)
    ax1.plot(wavelength, real, 'o', color=color_left, label='Real')
    if interpolate:
        new_real = interploate(real)
        new_wavelength = interploate(wavelength)
        ax1.plot(new_wavelength, new_real, color=color_left)
    ax1.legend(loc='upper left')
    ax1.tick_params(axis='y', labelcolor=color_left)
    ax1.grid()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Fake', color=color_right)  # we already handled the x-label with ax1
    ax2.plot(wavelength, fake, 'o', color=color_right, label='Fake')
    if interpolate:
        new_fake = interploate(fake)
        ax2.plot(new_wavelength, new_fake, color=color_right)
    ax2.legend(loc='upper right')
    ax2.tick_params(axis='y', labelcolor=color_right)
    ax2.spines['left'].set_color(color_left)
    ax2.spines['right'].set_color(color_right)
    plt.title('Real and Fake')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(save_dir)


def rect2polar(real, imag):
    complex_number = complex(real, imag)
    return abs(complex_number), cmath.phase(complex_number)


def polar2rect(modu, phase):
    complex_number = cmath.rect(modu, phase)
    return complex_number.real, complex_number.imag


def rect2polar_parallel(real_que, imag_que):
    assert len(real_que) == len(imag_que), "Size mismatch"
    modu_que, phase_que = np.zeros(len(real_que)), np.zeros(len(real_que))
    for i, real, imag in zip(range(len(real_que)), real_que, imag_que):
        modu_que[i], phase_que[i] = rect2polar(real, imag)
    return modu_que, phase_que


def polar2rect_parallel(modu_que, phase_que):
    assert len(modu_que) == len(phase_que), "Size mismatch"
    real_que, imag_que = np.zeros(len(modu_que)), np.zeros(len(modu_que))
    for i, modu, phase in zip(range(len(modu_que)), modu_que, phase_que):
        real_que[i], imag_que[i] = polar2rect(modu, phase)
    return real_que, imag_que


def load_mat(path):
    variables = scio.whosmat(path)
    target = variables[0][0]
    data = scio.loadmat(path)
    TT_array = data[target]
    TT_list = TT_array.tolist()
    return TT_list, TT_array


if __name__ == "__main__":

    current_dir = os.path.abspath(os.path.dirname(__file__))
    data_path = current_dir + '\\data'
    files = os.listdir(data_path)

    data_list_all = []

    for file in files:

        path = os.path.join(data_path, file)
        data_list = load_mat(path)
        data_list_all.extend(data_list)

    print('done')
