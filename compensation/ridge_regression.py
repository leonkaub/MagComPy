"""
computes the best fitting parameters for Leliak's magnetic compensation model using ridge regression

This file is part of MagComPy.
Copyright (C) 2021  Leon Kaub <lkaub@geophysik.uni-muenchen.de>
"""

import numpy as np
from shared import filter as ft


def get_derivative(v, freq, tap_p, sampling_rate):
    """
    returns the derivative of a given vector. LP filter to prevent derivative to amplify noise.
    """
    v_lp = ft.filter_lp_no_detrend(v, freq=freq, tap_p=tap_p, df=sampling_rate)
    return np.gradient(v_lp)


def get_g(x, y, z, hp_freq, lp_freq, tap_p, sampling_rate, eddy=True):
    """
    computes matrix G for ridge regression following model from Leliak (1961)
    :param x: component of magnetic field vector
    :param y: component of magnetic field vector
    :param z: component of magnetic field vector
    :param hp_freq: highpass filter cutting frequency
    :param lp_freq: lowpass filter cutting frequency
    :param tap_p: taper percentage
    :param sampling_rate: sampling rate
    :param eddy: include eddy current parameters or not
    :return: matrix G for ridge regression
    """
    g = dx = dy = dz = []

    f = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    nx = x / f  # cosine
    ny = y / f
    nz = z / f

    if eddy:
        dx = get_derivative(nx, freq=lp_freq, tap_p=tap_p, sampling_rate=sampling_rate)  # derivative of cosine
        dy = get_derivative(ny, freq=lp_freq, tap_p=tap_p, sampling_rate=sampling_rate)
        dz = get_derivative(nz, freq=lp_freq, tap_p=tap_p, sampling_rate=sampling_rate)

    nx = ft.filter_hp(nx, freq=hp_freq, tap_p=tap_p, df=sampling_rate)  # hp filter Fluxgate data
    ny = ft.filter_hp(ny, freq=hp_freq, tap_p=tap_p, df=sampling_rate)
    nz = ft.filter_hp(nz, freq=hp_freq, tap_p=tap_p, df=sampling_rate)

    if eddy:
        g = np.array([nx, ny, nz,
                      f * nx * nx, f * nx * ny, f * nx * nz,
                      f * ny * nz,
                      f * nz * nz,
                      f * nx * dx, f * nx * dy, f * nx * dz,
                      f * ny * dx, f * ny * dz,
                      f * nz * dx, f * nz * dy, f * nz * dz])
    else:
        g = np.array([nx, ny, nz,
                      f * nx * nx, f * nx * ny, f * nx * nz, f * ny * nz, f * nz * nz])

    return g.T


def least_sqr(g, data, hp_freq):
    g_t = np.transpose(g)
    data_hp_filtered = ft.filter_hp(data, freq=hp_freq, tap_p=taper_percentage, df=df)
    return (np.linalg.inv(g_t.dot(g)).dot(g_t)).dot(data_hp_filtered)


def rid_reg(g, data, ridge_par, hp_freq, tap_p, sampling_rate):
    g = ft.standardize_matrix(g)
    g_t = np.transpose(g)
    lam_i = ridge_par * np.identity(g.shape[1])
    data_hp_filtered = ft.filter_hp(data, freq=hp_freq, tap_p=tap_p, df=sampling_rate)
    return (np.linalg.inv(g_t.dot(g) + lam_i).dot(g_t)).dot(data_hp_filtered)


if __name__ == '__main__':
    taper_percentage = 0.1  # 0.1
    lp_cut = 0.3  # 0.3
    hp_cut = 0.05  # 0.05
    df = 1000  # 1000 for MagArrow
    ridge_parameter = 2e-3  # 2e-3

    # load data and call functions here
