# -*- coding: utf-8 -*-

import torch
import os
import json
import matplotlib.pyplot as plt
import cmath
import numpy as np
from scipy import interpolate
import scipy.io as scio
import matlab.engine

current_dir = os.path.abspath(os.path.dirname(__file__))
Tensor = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor


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
    os.makedirs(current_dir + '/figures/loss_curves', exist_ok=True)
    os.makedirs(current_dir + '/figures/test_output', exist_ok=True)


def plot_single_part(wavelength, spectrum, name, legend='spectrum', interpolate=True):
    save_dir = os.path.join(current_dir, 'figures/test_output', name)
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
    plt.ylim((0, 1))
    plt.savefig(save_dir)
    plt.close()


def plot_both_parts(wavelength, real, fake, name, legend='Real and Fake', interpolate=True):

    color_left = 'blue'
    color_right = 'red'
    save_dir = os.path.join(current_dir, 'figures/test_output', name)

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
    plt.ylim((0, 1))

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
    plt.ylim((0, 1))
    plt.title(legend)

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


def find_spectrum(thickness, radius, gap, TT_array):
    rows, _ = TT_array.shape
    wavelength, spectrum = [], []
    for row in range(rows):
        if TT_array[row, 1] == thickness and TT_array[row, 2] == radius and TT_array[row, 3] == gap:
            wavelength.append(TT_array[row, 0])
            spectrum.append(TT_array[row, -1])
        else:
            continue
    wavelength = np.array(wavelength)
    spectrum = np.array(spectrum)
    index_order = np.argsort(wavelength)
    return wavelength[index_order], spectrum[index_order]


def load_mat(path):
    variables = scio.whosmat(path)
    target = variables[0][0]
    data = scio.loadmat(path)
    TT_array = data[target]
    TT_list = TT_array.tolist()
    return TT_list, TT_array


def data_pre(list_all, wlimit):
    dtype = [('wave_length', int), ('thickness', int), ('radius', int), ('gap', int), ('efficiency', float)]
    values = [tuple(single_device) for single_device in list_all]
    array_temp = np.array(values, dtype)
    array_all = np.sort(array_temp, order=['thickness', 'radius', 'gap', 'wave_length'])

    thickness_list = np.unique(array_all['thickness'])
    radius_list = np.unique(array_all['radius'])
    gap_list = np.unique(array_all['gap'])
    reformed = []

    for thickness in thickness_list:
        for radius in radius_list:
            for gap in gap_list:
                pick_index = np.intersect1d(np.argwhere(array_all['radius'] == radius), np.argwhere(
                    array_all['thickness'] == thickness))
                pick_index = np.intersect1d(pick_index, np.argwhere(array_all['gap'] == gap))
                picked = array_all[pick_index]
                picked = np.sort(picked, order=['wave_length'])
                cur_ref = [thickness, radius, gap]
                for picked_single in picked:
                    cur_ref.append(picked_single[4])

                reformed.append(cur_ref)

    return np.array(reformed), array_all


def inter(inputs, device):
    inputs_inter = torch.ones(inputs.shape[0], inputs.shape[1], 224)
    x = np.linspace(0, 223, num=inputs.shape[2])
    new_x = np.linspace(0, 223, num=224)

    for index_j, j in enumerate(inputs):
        for index_jj, jj in enumerate(j):
            y = jj
            f = interpolate.interp1d(x, y, kind='cubic')
            jj = f(new_x)
            inputs_inter[index_j, index_jj, :] = torch.from_numpy(jj)

    inputs_inter = inputs_inter.double().to(device)

    return inputs_inter


def RCWA_parallel(eng, w_list, thick_list, r_list, gap, acc=5):
    batch_size = len(thick_list)
    spec = np.ones((batch_size, len(w_list)))
    gap = matlab.double([gap])
    acc = matlab.double([acc])
    for i in range(batch_size):
        thick = thick_list[i]
        thick = matlab.double([thick])
        r = r_list[i]
        r = matlab.double([r])
        for index, w in enumerate(w_list):
            w = matlab.double([w])
            spec[i, index] = eng.RCWA_solver(w, gap, thick, r, acc)

    return spec


def RCWA(eng, w_list, thick, r, gap, acc=5, medium=1, shape=0):
    spec = np.ones(len(w_list))
    gap = matlab.double([gap])
    acc = matlab.double([acc])
    thick = matlab.double([thick])
    r = matlab.double([r])
    medium = matlab.double([medium])
    shape = matlab.double([shape])
    for index, w in enumerate(w_list):
        w = matlab.double([w])
        spec[index] = eng.RCWA_solver(w, gap, thick, r, acc, medium, shape)

    return spec


def RCWA_call(eng, w_list, g_list, t_list, r_list, acc=5):
    batch_size = len(t_list)
    spec = np.ones((batch_size, len(w_list)))
    for i in range(batch_size):
        thickness = t_list[i]
        thickness = matlab.double([thickness])
        gap = g_list[i]
        gap = matlab.double([gap])
        radius = r_list[i]
        radius = matlab.double([radius])
        acc = matlab.double([acc])
        wavelength = matlab.double(w_list)
        result = eng.RCWA_solver_par(wavelength, gap, thickness, radius, acc)
        result2 = np.asarray(result).tolist()
        spec[i] = result2[0]

    return spec


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def keep_range(data):
    for i in range(len(data)):
        if data[i] < 0:
            data[i] = 0
        elif data[i] > 1:
            data[i] = 1
        else:
            continue
    return data


def gauss_spec_valley(f, mean, var, depth=0.2):
    return 1 - (1 - depth) * np.exp(-(f - mean) ** 2 / (2 * (var ** 2)))


def gauss_spec_peak(f, mean, var, depth=0.2):
    return (1 - depth) * np.exp(-(f - mean) ** 2 / (2 * (var ** 2)))


def random_gauss_spec(f):
    depth = np.random.uniform(low=0.0, high=0.05)
    mean = np.random.uniform(low=460, high=490)
    var = np.random.uniform(low=20, high=40)
    return 1 - (1 - depth) * np.exp(-(f - mean) ** 2 / (2 * (var ** 2)))


def random_gauss_spec_combo(f, valley_num):
    spec = np.zeros(len(f))
    for i in range(valley_num):
        spec += random_gauss_spec(f)
    return normalization(spec)


def spec_jitter(spec, amp):
    return normalization(spec + np.random.uniform(low=-amp, high=amp, size=spec.size))


def unique_device(list_all):
    dtype = [('wave_length', int), ('thickness', int), ('radius', int), ('efficiency', float)]
    values = [tuple(single_device) for single_device in list_all]
    array_temp = np.array(values, dtype)
    array_all = np.sort(array_temp, order = ['thickness', 'radius', 'wave_length'])
    
    thickness_list = np.unique(array_all['thickness'])
    radius_list = np.unique(array_all['radius'])

    reformed = []

    for thickness in thickness_list:
        for radius in radius_list:
            pick_index = np.intersect1d(np.argwhere(array_all['radius'] == radius), np.argwhere(array_all['thickness'] == thickness))
            if pick_index.size == 0:
                continue
            pair = [thickness, radius]

            reformed.append(pair)

    return reformed


if __name__ == "__main__":

    data_path = current_dir + '\\data'
    save_path = current_dir + '\\device_id'
    files = os.listdir(data_path)

    data_list_all = []

    for file in files:

        path = os.path.join(data_path, file)
        data_list, _ = load_mat(path)
        data_list_all.extend(data_list)

    # data_prepared, data_sorted = data_pre(data_list_all, None)
    # scio.savemat(save_path, {'value': data_prepared})

    # pick_index = np.intersect1d(np.argwhere(data_sorted['radius'] == 20), np.argwhere(data_sorted['thickness'] == 200))

    # picked_data = aray[pick_index]

    # sliced = array_slice(data_sorted)

    device_id = unique_device(data_list_all)
    scio.savemat(save_path, {'device_id': device_id})
    
    print('done')
