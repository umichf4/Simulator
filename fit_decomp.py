# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit
from utils import *
import math as m
import simplejson as sj

def gaussian(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9):
    return a0 * np.exp(-np.power(x - m0, 2.) / (2 * np.power(s0, 2.))) + \
           a1 * np.exp(-np.power(x - m1, 2.) / (2 * np.power(s1, 2.))) + \
           a2 * np.exp(-np.power(x - m2, 2.) / (2 * np.power(s2, 2.))) + \
           a3 * np.exp(-np.power(x - m3, 2.) / (2 * np.power(s3, 2.))) + \
           a4 * np.exp(-np.power(x - m4, 2.) / (2 * np.power(s4, 2.))) + \
           a5 * np.exp(-np.power(x - m5, 2.) / (2 * np.power(s5, 2.))) + \
           a6 * np.exp(-np.power(x - m6, 2.) / (2 * np.power(s6, 2.))) + \
           a7 * np.exp(-np.power(x - m7, 2.) / (2 * np.power(s7, 2.))) + \
           a8 * np.exp(-np.power(x - m8, 2.) / (2 * np.power(s8, 2.))) + \
           a9 * np.exp(-np.power(x - m9, 2.) / (2 * np.power(s9, 2.)))


def gauss_decomp(wavelength, spectrum):
    a_min = [0] * 10
    a_max = [1] * 10
    m_min = [400] * 10
    m_max = [680] * 10
    s_min = [0] * 10
    s_max = [1000] * 10
    min_limit = a_min + m_min + s_min
    max_limit = a_max + m_max + s_max

    popt, pcov = curve_fit(gaussian, wavelength, spectrum, bounds=(min_limit, max_limit))

    return popt


def if_comp(wavelength, spectrum):
    right_half = np.asarray(spectrum)
    left_half = np.flip(right_half)
    whole_spectrum = np.append(left_half, right_half)
    t_series = np.fft.ifft(whole_spectrum)

    return np.abs(t_series), np.real(t_series), np.imag(t_series)


def f_decomp(spectrum, item_num):
    f_series = np.fft.fftshift(np.fft.fft(spectrum))
    all_length = len(spectrum)
    crop_start = m.floor((all_length - item_num) / 2)
    crop_end = m.floor((all_length + item_num) / 2)
    f_new = np.zeros((len(spectrum)), dtype=complex)
    f_new[crop_start:crop_end] = f_series[crop_start:crop_end]

    return f_new, f_series[crop_start:crop_end]


def decomp_fit(decomp_choice, wavelength, spectrum):
    if decomp_choice == 'gaussian':
        sample_fit = gauss_decomp(wavelength, spectrum)
        print(sample_fit, '\n')
        plt.plot(wavelength, spectrum, 'b+:', label='data')
        plt.plot(wavelength, gaussian(wavelength, *sample_fit), 'ro:', label='fit')

    elif decomp_choice == 'fourier':
        f_series, _ = f_decomp(spectrum, 15)
        plt.plot(wavelength, spectrum, 'b+:', label='data')
        plt.plot(wavelength,np.fft.ifft(np.fft.ifftshift(f_series)), 'ro:', label='fit')

    plt.legend()
    plt.show()


def visual_rep(twod_list):
    target = np.asarray(twod_list)
    min_index = np.argsort(target[:,14])
    index_temp = np.tile(min_index, (29, 1))
    index = np.transpose(index_temp)
    sorted_list = np.take_along_axis(target, index, 0)

    # TE_spec = cv2.resize(src=TE_spec, dsize=(1000, 1881), interpolation=cv2.INTER_CUBIC)

    plt.figure()
    plt.pcolor(sorted_list, cmap=plt.cm.jet)
    plt.xlabel('Index')
    plt.ylabel('Index of Devices')
    plt.title('Fourier transform spectrum')
    # plt.xticks(np.arange(len(wavelength), step=4), np.uint16(wavelength[::4]))
    # plt.yticks([])
    cb = plt.colorbar()
    cb.ax.set_ylabel('Amplitude')
    plt.show()


def save_file():
    
    data_path = current_dir + '\\data\\circle_Si.mat'
    data_list, _ = load_mat(data_path)
    data_prepared, data_sorted = data_pre(data_list, None)

    # fts_abs = open("fts_abs.txt","w+")
    # fts_real = open("fts_real.txt","w+")
    # fts_imag = open("fts_imag.txt","w+")
    # ifts_abs = open("ifts_abs.txt","w+")
    # ifts_real = open("ifts_real.txt","w+")
    # ifts_imag = open("ifts_imag.txt","w+")
    f_train = open("f_train.txt", "w+")

    abs_all = []
    real_all = []
    imag_all = []
    t_series_all = []
    t_real = []
    t_imag = []
    save_all = []

    for index in range(6765):
        spectrum = data_prepared[index][3:]
        wavelength = np.linspace(400, 680, 29)
        _, f_single = f_decomp(spectrum, 29)
        save_all.append(np.real(f_single).tolist() + np.imag(f_single).tolist())
        # abs_all.append(np.abs(f_single).tolist())
        # real_all.append(np.real(f_single).tolist())
        # imag_all.append(np.imag(f_single).tolist())
        # absv, realv, imagv = if_comp(wavelength, spectrum)
        # t_series_all.append(absv.tolist())
        # t_real.append(realv.tolist())
        # t_imag.append(imagv.tolist())

    # sj.dump(abs_all, fts_abs)
    # sj.dump(real_all, fts_real)
    # sj.dump(imag_all, fts_imag)
    # sj.dump(t_series_all, ifts_abs)
    # sj.dump(t_real, ifts_real)
    # sj.dump(t_imag, ifts_imag)
    sj.dump(save_all, f_train)

    # fts_abs.close()
    # fts_real.close()
    # fts_imag.close()
    # ifts_abs.close()
    # ifts_real.close()
    # ifts_imag.close()
    f_train.close()

    print('Done calculating and saving files\n')


def load_file(filename):

    with open(filename, 'r') as f:
        twod_list = sj.load(f)

    visual_rep(twod_list)

if __name__ == "__main__":
    
    load_file('fts_abs.txt')
