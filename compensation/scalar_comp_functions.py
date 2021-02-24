"""
Functions for magnetic compensation of a scalar magnetometer.

This file is part of MagComPy.
Copyright (C) 2021  Leon Kaub <lkaub@geophysik.uni-muenchen.de>
"""

from compensation import ridge_regression as rr
from shared import filter as ft
import numpy as np
from numpy.lib.recfunctions import append_fields
import matplotlib.pyplot as plt
from scipy.signal import welch
import configparser
from datetime import datetime

# configuration file
config = configparser.ConfigParser(interpolation=None)
config.read('magcompy.ini')

# get basic settings from config file
taper_percentage = float(config['scalar compensation']['taper_percentage'])  # default 0.2
ridge_parameter = float(config['scalar compensation']['ridge_parameter'])  # default 2e-5
comp_lp_freq = int(config['scalar compensation']['comp_lp_freq'])  # lp cutting freq to lp filter result. default: 5
sampling_rate = int(config['MA']['sampling_rate'])

# some more settings
np.set_printoptions(linewidth=1000)
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 6, 4


def read_data(filename):
    """
    reads in data that is needed for compensation
    :param filename: file path (string)
    :return: structured array
    """
    column_names = config['scalar compensation']['read_col_names'].split(',')
    column_numbers = [int(n) for n in config['scalar compensation']['read_col_numbers'].split(',')]
    dt = (float,) * 6  # data type of the columns specified above

    try:
        data = np.loadtxt(filename, dtype={'names': column_names, 'formats': dt}, delimiter=',', skiprows=1,
                          usecols=column_numbers)
    except IOError:
        print(f'{datetime.now()}:\tError while loading file. Please check file path')
        return None

    print(f'{datetime.now()}:\tLoaded data from file {filename}')
    return data


def save_to_file(original_file, new_file, comp, comp_lp):
    """
    writes all data into a new file. Make sure that formats in config file fits!
    :param original_file: file_s (or file_b)
    :param comp: compensated data
    :param comp_lp: compensated + lp filtered data
    :param new_file: file path
    """
    if new_file[-4:] != '.csv':
        new_file += '.csv'
    # load data from original file
    # noinspection PyTypeChecker
    data = np.genfromtxt(original_file, names=True, dtype=None, delimiter=',', encoding='ascii')
    # append compensated data and low-pass filtered compensated data
    append_names = ('comp_nT', f'comp_lp{config["scalar compensation"]["comp_lp_freq"]}_nT')
    data = append_fields(base=data, names=append_names, data=(comp, comp_lp), usemask=False)
    # save data in new file
    formats = config['scalar compensation']['save_col_format']
    try:
        np.savetxt(fname=new_file, X=data, fmt=formats, delimiter=',', newline='\n',
                   header=', '.join([i for i in data.dtype.names]))
        print(f'{datetime.now()}:\tData saved as: {new_file}')
    except (IOError, AttributeError):
        print(f'{datetime.now()}:\tCan not save data. Check filepath')


def regression(x, y, z, m, hp, lp):
    """
    calculates regression parameters from compensation maneuver
    :param x: flux gate data from compensation
    :param y: flux gate data from compensation
    :param z: flux gate data from compensation
    :param m: total field data from compensation
    :param hp: hp cutting frequency
    :param lp: lp cutting frequency
    :return: theta, compensated total field from compensation maneuver
    """
    # get model matrix G using x, y, z from compensation flight
    g = rr.get_g(x=x, y=y, z=z,
                 hp_freq=hp,
                 lp_freq=lp,
                 tap_p=taper_percentage,
                 sampling_rate=sampling_rate)

    # get theta (model coefficient vector) using G and total field signal
    t = rr.rid_reg(g=g,
                   data=m,
                   ridge_par=ridge_parameter,
                   hp_freq=hp,
                   tap_p=taper_percentage,
                   sampling_rate=sampling_rate)

    g_std = ft.standardize_matrix(g)

    # return theta, compensated field (measured field minus perturbation field)
    return t, m - g_std @ t


def prediction(x, y, z, m, hp, lp, t):
    """
    applies parameters from compensation maneuver on survey data
    :param x: flux gate data from survey
    :param y: flux gate data from survey
    :param z: flux gate data from survey
    :param m: total field data from survey
    :param hp: hp cutting frequency
    :param lp: lp cutting frequency
    :param t: theta from regression()
    :return: compensated total field from survey
    """
    # get model matrix G using x, y, z from survey flight
    g = rr.get_g(x=x, y=y, z=z,
                 hp_freq=hp,
                 lp_freq=lp,
                 tap_p=taper_percentage,
                 sampling_rate=sampling_rate)

    g_std = ft.standardize_matrix(g)

    # return compensated field (measured field minus perturbation field)
    return m - g_std @ t


def sweep_frequencies(calib_data, survey_data, cut_freqs, const_freq, sweep_mode):
    """
    parameter sweep for lp or hp cutting frequencies
    :param calib_data: data from compensation flight containing 'X', 'Y', 'Z', 'F'
    :param survey_data: data from survey flight containing 'X', 'Y', 'Z', 'F'
    :param cut_freqs: list of cutting frequencies
    :param const_freq: cutting frequency for hp (lp) if lp (hp) sweep
    :param sweep_mode: lp or hp
    :return: improvement ratios of compensation data, improvement ratios of survey data
    """
    print(f'{datetime.now()}:\tList of {sweep_mode} cutting frequencies:\n{cut_freqs}')
    ir_b, ir_s = [], []
    if sweep_mode == 'lp':
        for lp_value in cut_freqs:
            th, calib_com = regression(x=calib_data['X'], y=calib_data['Y'], z=calib_data['Z'], m=calib_data['F'],
                                       hp=const_freq, lp=lp_value)

            survey_com = prediction(survey_data['X'], survey_data['Y'], survey_data['Z'], survey_data['F'],
                                    hp=const_freq, lp=lp_value, t=th)

            print('For lp cut freq: {}\nParameters:\n{}\n-----------'.format(lp_value, th))
            print('results compensation data:\n')
            im_ratio_calib = ft.improvement_ratio(calib_data['F'],
                                                  calib_com,
                                                  const_freq,
                                                  tap_p=taper_percentage,
                                                  df=sampling_rate,
                                                  mode=config['scalar compensation']['improvement_ratio_mode'])

            print('results survey data\n')
            im_ratio_sur = ft.improvement_ratio(survey_data['F'],
                                                survey_com,
                                                const_freq,
                                                tap_p=taper_percentage,
                                                df=sampling_rate,
                                                mode=config['scalar compensation']['improvement_ratio_mode'])

            ir_b.append(im_ratio_calib)
            ir_s.append(im_ratio_sur)

    elif sweep_mode == 'hp':
        for hp_value in cut_freqs:
            th, calib_com = regression(x=calib_data['X'], y=calib_data['Y'], z=calib_data['Z'], m=calib_data['F'],
                                       hp=hp_value, lp=const_freq)

            survey_com = prediction(survey_data['X'], survey_data['Y'], survey_data['Z'], survey_data['F'],
                                    hp=hp_value, lp=const_freq, t=th)

            print('For hp cut freq: {:.4f}\nParameters:\n{}\n-----------'.format(hp_value, th))
            print('results compensation data:')
            im_ratio_calib = ft.improvement_ratio(calib_data['F'],
                                                  calib_com,
                                                  hp_value,
                                                  tap_p=taper_percentage,
                                                  df=sampling_rate,
                                                  mode=config['scalar compensation']['improvement_ratio_mode'])

            print('results survey data')
            im_ratio_sur = ft.improvement_ratio(survey_data['F'],
                                                survey_com,
                                                hp_value,
                                                tap_p=taper_percentage,
                                                df=sampling_rate,
                                                mode=config['scalar compensation']['improvement_ratio_mode'])

            ir_b.append(im_ratio_calib)
            ir_s.append(im_ratio_sur)

    else:
        print('Incorrect sweep mode')

    ir_b, ir_s = np.array(ir_b), np.array(ir_s)

    print_sweep_results(ir_b, ir_s, cut_freqs, const_freq, sweep_mode)
    plot_sweep_results(ir_b, ir_s, cut_freqs)

    return np.array(ir_b), np.array(ir_s)


def print_sweep_results(ir_b, ir_s, cut_freqs, const_freq, sweep_mode):
    def get_other_mode(mode):
        if mode == 'lp':
            return 'hp'
        elif mode == 'hp':
            return 'lp'

    print('-----------\n'
          f'{datetime.now()}\n'
          f'RESULTS {sweep_mode} SWEEP\n{get_other_mode(sweep_mode)} cutting frequency:\t{const_freq}\n'
          'freq:\tIR compensation:\tIR survey:')

    for i in range(len(cut_freqs)):
        print('{:.4f}\t{:.4f}\t{:.4f}'.format(cut_freqs[i], ir_b[i], ir_s[i]))


def plot_sweep_results(ir_b, ir_s, cut_freqs):
    fig, ax1 = plt.subplots()

    ax1.plot(cut_freqs, ir_b / ir_b[0], label='compensation')
    ax1.set_xlabel('frequency')
    ax1.set_ylabel('IR compensation')
    ax1.plot(cut_freqs, ir_s / ir_s[0], label='survey')

    fig.legend(bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.show()


class ScalarComp:
    def __init__(self, calib_fname, survey_fname):
        self.theta = None
        self.calib_comp, self.calib_comp_lp = None, None
        self.survey_comp, self.survey_comp_lp = None, None
        self.lp_cut, self.hp_cut = 0, 0
        self.ir_calib, self.ir_sur = 0, 0

        self.ridge_parameter = ridge_parameter
        self.taper_percentage = taper_percentage

        self.calib_fname = calib_fname
        self.survey_fname = survey_fname
        self.calib_data = read_data(self.calib_fname)
        self.survey_data = read_data(self.survey_fname)

    def compensation(self, lp, hp):
        # best fitting coefficients for compensation pattern
        self.lp_cut = lp
        self.hp_cut = hp
        bx, by, bz, bm = self.calib_data['X'], self.calib_data['Y'], self.calib_data['Z'], self.calib_data['F']

        self.theta, self.calib_comp = regression(bx, by, bz, bm, lp=lp, hp=hp)

        # use coefficients on survey data
        x, y, z, m = self.survey_data['X'], self.survey_data['Y'], self.survey_data['Z'], self.survey_data['F']

        self.survey_comp = prediction(x, y, z, m, lp=lp, hp=hp, t=self.theta)

        # LP filter results
        comp_lp_f = float(config['scalar compensation']['comp_lp_freq'])
        self.calib_comp_lp = ft.filter_lp_no_detrend(data=self.calib_comp,
                                                     freq=comp_lp_f,
                                                     tap_p=taper_percentage,
                                                     df=sampling_rate)

        self.survey_comp_lp = ft.filter_lp_no_detrend(data=self.survey_comp,
                                                      freq=comp_lp_f,
                                                      tap_p=taper_percentage,
                                                      df=sampling_rate)

        # print results and calculate improvement ratios
        print(f'{datetime.now()}:\tParameters:\n{self.theta}\n-----------')
        print('results compensation data:\n')
        self.ir_calib = ft.improvement_ratio(self.calib_data['F'],
                                             self.calib_comp, hp,
                                             tap_p=taper_percentage,
                                             df=sampling_rate,
                                             mode=config['scalar compensation']['improvement_ratio_mode'])

        print('results survey data:\n')
        self.ir_sur = ft.improvement_ratio(self.survey_data['F'],
                                           self.survey_comp, hp,
                                           tap_p=taper_percentage,
                                           df=sampling_rate,
                                           mode=config['scalar compensation']['improvement_ratio_mode'])

    def start_sweep(self, cut, freqs, mode):
        return sweep_frequencies(self.calib_data, self.survey_data, cut_freqs=freqs, const_freq=cut, sweep_mode=mode)

    def save(self, fname, rec_type):
        if rec_type == 'compensation':
            save_to_file(original_file=self.calib_fname, new_file=fname, comp=self.calib_comp,
                         comp_lp=self.calib_comp_lp)
        elif rec_type == 'survey':
            save_to_file(original_file=self.survey_fname, new_file=fname, comp=self.survey_comp,
                         comp_lp=self.survey_comp_lp)
        else:
            return

    def time_plot(self, rec_type='compensation'):
        if rec_type == 'compensation':
            data = self.calib_data
            comp = self.calib_comp
            comp_lp = self.calib_comp_lp
        elif rec_type == 'survey':
            data = self.survey_data
            comp = self.survey_comp
            comp_lp = self.survey_comp_lp
        else:
            return

        t = np.arange(len(data['F'])) / sampling_rate

        fig, ax = plt.subplots()
        ax.plot(t, data['F'], label='raw')
        if comp is not None:
            ax.plot(t, comp, label='compensated')
            ax.plot(t, comp_lp, label='compensated + lp filtered')
        ax.set_title(f'field in time ({rec_type})')
        ax.set(xlabel='time in seconds', ylabel='field in nT')
        plt.legend()
        plt.show()

    def scatter_plot(self, rec_type='compensation'):
        if rec_type == 'compensation':
            data = self.calib_data
            comp = self.calib_comp
        elif rec_type == 'survey':
            data = self.survey_data
            comp = self.survey_comp
        else:
            return

        x_ = data['X_crd'] - min(data['X_crd'])
        y_ = data['Y_crd'] - min(data['Y_crd'])

        fig, ax = plt.subplots()
        if comp is None:
            im = ax.scatter(x_, y_, c=data['F'], s=1, cmap='jet')
            ax.set(xlabel='X [m]', ylabel='Y [m]', title=f'{rec_type} (raw)')
        else:
            im = ax.scatter(x_, y_, c=comp, s=1, cmap='jet')
            ax.set(xlabel='X [m]', ylabel='Y [m]', title=f'{rec_type} (compensated)')
        plt.colorbar(im, ax=ax)
        plt.show()

    def spectrum_plot(self, rec_type='compensation'):
        if rec_type == 'compensation':
            data = self.calib_data
            comp = self.calib_comp
        elif rec_type == 'survey':
            data = self.survey_data
            comp = self.survey_comp
        else:
            return

        fig, ax = plt.subplots()
        f, p_xx = welch(data['F'], sampling_rate)
        ax.semilogy(f, p_xx, label='raw field')
        if comp is not None:
            fc, p_xxc = welch(comp, sampling_rate)
            ax.semilogy(fc, p_xxc, label='compensated field')
        ax.set(xlabel='Frequency [Hz]', ylabel='power spectral density', title=f'spectrum ({rec_type})')
        plt.legend()
        plt.show()

    def show_par(self):
        if self.theta is None:
            print('compute parameters first')
            return

        if len(self.theta) == 16:  # all parameters
            labels = ['P1', 'P2', 'P3', 'I1', 'I2', 'I3', 'I4', 'I5',
                      'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8']
            title = 'Parameters\n P=permanent, I=induced, E=Eddy currents'
            x = np.arange(1, 17)
        elif len(self.theta) == 8:  # no eddy currents
            labels = ['P1', 'P2', 'P3', 'I1', 'I2', 'I3', 'I4', 'I5']
            x = np.arange(1, 9)
            title = 'Parameters\n P=permanent, I=induced'
        else:
            return

        fig, ax = plt.subplots()
        ax.bar(x, self.theta)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        plt.show()

    def save_par(self, filename):
        if filename[-4:] != '.csv':
            filename += '.csv'

        comment = f'parameters computed from file {self.calib_fname}\n' \
                  f'lp cutting frequency: {self.lp_cut}\n' \
                  f'hp cutting frequency: {self.hp_cut}'

        try:
            np.savetxt(fname=filename, X=self.theta, fmt='%.6f', delimiter=',', newline='\n', header=comment)
            print(f'{datetime.now()}:\tParameters saved in: {filename}')
        except (IOError, AttributeError):
            print(f'{datetime.now()}:\tCan not save parameters. Check filepath')


if __name__ == '__main__':
    # files
    test_calibfile = 'ACQU3.csv'
    test_surfile = 'ACQU2.csv'

    # set filter frequencies
    lp_cutting = 2.0
    hp_cutting = 0.015

    # initialize
    mc = ScalarComp(test_calibfile, test_surfile)

    # compute compensation
    mc.compensation(lp=lp_cutting, hp=hp_cutting)

    # plotting
    mc.time_plot(rec_type='compensation')
    mc.time_plot(rec_type='survey')
