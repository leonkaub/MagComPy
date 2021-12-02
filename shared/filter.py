"""
signal processing functions, mainly filters

This file is part of MagComPy.
Copyright (C) 2021  Leon Kaub <lkaub@geophysik.uni-muenchen.de>
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, iirfilter, sosfilt, decimate


def cos_taper(npts, p=0.1):
    """
    partly from ObsPy:
    https://docs.obspy.org/packages/autogen/obspy.signal.invsim.cosine_taper.html
    (Moritz Beyreuther, Robert Barsch, Lion Krischer, Tobias Megies, Yannik Behr and Joachim Wassermann (2010),
    ObsPy: A Python Toolbox for Seismology, SRL, 81(3), 530-533, doi:10.1785/gssrl.81.3.530.)
    """
    if p < 0 or p > 1:
        msg = "Decimal taper percentage must be between 0 and 1."
        raise ValueError(msg)

    if p == 0.0 or p == 1.0:
        frac = int(npts * p / 2.0)
    else:
        frac = int(npts * p / 2.0 + 0.5)

    idx1 = 0
    idx2 = frac - 1
    idx3 = npts - frac
    idx4 = npts - 1

    # Very small data lengths or small decimal taper percentages can result in idx1 == idx2 and idx3 == idx4
    if idx1 == idx2:
        idx2 += 1
    if idx3 == idx4:
        idx3 -= 1

    # the taper at idx1 and idx4 equals zero and at idx2 and idx3 equals one
    cos_win = np.zeros(npts)
    cos_win[idx1:idx2 + 1] = 0.5 * (1.0 - np.cos((np.pi * (np.arange(idx1, idx2 + 1) - float(idx1)) / (idx2 - idx1))))
    cos_win[idx2 + 1:idx3] = 1.0
    cos_win[idx3:idx4 + 1] = 0.5 * (1.0 + np.cos((np.pi * (float(idx3) - np.arange(idx3, idx4 + 1)) / (idx4 - idx3))))

    # if indices are identical division by zero causes NaN values in cos_win
    if idx1 == idx2:
        cos_win[idx1] = 0.0
    if idx3 == idx4:
        cos_win[idx3] = 0.0

    return cos_win


def decimate_data(data, factor):
    """
    down-sample signal after applying anti-aliasing filter
    """
    return decimate(data, q=factor)


def demean(data):
    return data - np.mean(data)


def filter_hp(data, freq, tap_p, df):
    """
    calls filt function to do a highpass filter
    """
    return filt(data, freq, tap_p, df, filter_type='hp')


def filter_lp(data, freq, tap_p, df):
    """
    calls filt function to do a lowpass filter
    """
    return filt(data, freq, tap_p, df, filter_type='lp')


def filt(data, freq, tap_p, df, filter_type):
    """
    combines the full filter process: 1. demean 2. pad with first/last values 3. taper 4. filter
    """
    npts = npts_for_taper = len(data)
    p = tap_p / (1 + tap_p)

    data = demean(data)  # remove the mean of the signal
    padded = padding(data, tap_p)  # increase length of signal by tap_p

    # getting the correct taper length (mismatch possible because of rounding errors)
    while True:
        taper = cos_taper(int(round((1 + tap_p) * npts_for_taper)), p)
        if len(taper) == len(padded):
            break
        elif len(taper) < len(padded):
            npts_for_taper += 1
        else:
            npts_for_taper -= 1

    tapered = padded * taper

    # filter
    if filter_type == 'lp':
        filtered = lowpass(tapered, cut=freq, df=df, zero_phase=True)
    elif filter_type == 'hp':
        filtered = highpass(tapered, cut=freq, df=df, zero_phase=True)
    else:
        print('Incorrect filter type (lp or hp')
        filtered = None

    # signal needs to get cut back to original length
    return filtered[int(0.5 * tap_p * npts):int((1 + 0.5 * tap_p) * npts)]


def filter_lp_no_detrend(data, freq, tap_p, df):
    """
    combines the full LP-filter process but without detrending: 1. pad with first/last values 2. taper 3. filter
    """
    npts = npts_for_taper = len(data)
    p = tap_p / (1 + tap_p)

    padded = padding(data, tap_p)

    # getting the correct taper length (mismatch possible because of rounding errors)
    while True:
        taper = cos_taper(int(round((1 + tap_p) * npts_for_taper)), p)
        if len(taper) == len(padded):
            break
        elif len(taper) < len(padded):
            npts_for_taper += 1
        else:
            npts_for_taper -= 1

    tapered = padded * taper
    lp = lowpass(tapered, cut=freq, df=df, zero_phase=True)

    # signal needs to get cut back to original length
    return lp[int(0.5 * tap_p * npts):int((1 + 0.5 * tap_p) * npts)]


def highpass(data, cut, df, order=4, zero_phase=False):
    f = cut / (0.5 * df)  # corner frequency in percentage of Nyquist frequency

    if f > 1:
        msg = "Selected corner frequency is above Nyquist frequency."
        raise ValueError(msg)

    sos = iirfilter(order, f, btype='highpass', ftype='butter', output='sos')

    if zero_phase:
        first_pass = sosfilt(sos, data)
        return sosfilt(sos, first_pass[::-1])[::-1]
    else:
        return sosfilt(sos, data)


def improvement_ratio(data1, data2, hp_freq, df, tap_p, mode='rms'):
    if mode == 'rms':
        v1 = rms(data1, hp_freq=hp_freq, tap_p=tap_p, df=df)
        v2 = rms(data2, hp_freq=hp_freq, tap_p=tap_p, df=df)
    elif mode == 'std':
        v1 = np.std(data1, ddof=1)
        v2 = np.std(data2, ddof=1)
    else:
        print('mode needs to be rms or std')
        return

    ir = v1 / v2
    print(f'raw: {v1:6.3f},\tnew: {v2:6.3f},\tIR: {ir:6.3f}\n-----------')
    return ir


def lowpass(data, cut, df, order=4, zero_phase=False):
    f = cut / (0.5 * df)  # corner frequency in percentage of Nyquist frequency

    if f > 1:
        msg = "Selected corner frequency is above Nyquist frequency."
        raise ValueError(msg)

    sos = iirfilter(order, f, btype='lowpass', ftype='butter', output='sos')

    if zero_phase:
        first_pass = sosfilt(sos, data)
        return sosfilt(sos, first_pass[::-1])[::-1]
    else:
        return sosfilt(sos, data)


def padding(data, tap_p):
    # version 1/18/19
    # add_len = int(0.5 * tap_p * len(data))  # length of array added to front/back, therefore 0.5*tap_p
    add_len = int(round(0.5 * tap_p * len(data)))  # length of array added to front/back, therefore 0.5*tap_p
    front = np.ones(add_len) * data[0]  # array added to front: (0.5*tap_p) times the 1st value of data
    end = np.ones(add_len) * data[-1]  # array added to end: (0.5*tap_p) times the last value of data
    padded = np.append(front, data)  # pre-pending front array
    padded = np.append(padded, end)  # appending end array
    return padded


def plot_spectrum(data1, data2, df, xmin=0, xmax=0):
    """
    plots power spectrum using scipy.signal.periodogram.
    plots two datasets in one figure so that original and filtered data can be compared
    :param data1: first signal
    :param data2: second signal
    :param df: sampling rate
    :param xmin: can be set as lower plotting limit.
    :param xmax: can be set as upper plotting limit. default Nyquist frequency
    """
    if xmax == 0:
        xmax = df / 2
    if xmin == 0:
        xmin = -0.01 * df
    f1, p_xx1 = periodogram(x=data1, fs=df, scaling='spectrum')
    f2, p_xx2 = periodogram(x=data2, fs=df, scaling='spectrum')
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(data1)
    ax[0].plot(data2)
    ax[0].set(xlabel='time', ylabel='nT')
    ax[1].plot(f1, np.sqrt(p_xx1), label='original')
    ax[1].plot(f2, np.sqrt(p_xx2), label='filtered')
    ax[1].set(yscale='symlog', xlabel='frequency', ylabel='power', xlim=(xmin, xmax))
    ax[1].legend()
    plt.suptitle('Power spectrum')
    plt.show()


def plot_spectrum_simple(data, df, title=''):
    """
    plots the power spectrum of one signal
    :param data: signal array
    :param df: sampling frequency
    :param title: gets attached to figure's title
    """
    f, p_xx = periodogram(x=data, fs=df, scaling='spectrum')
    fig, ax = plt.subplots()
    ax.plot(f, np.sqrt(p_xx))
    ax.set(yscale='symlog', xlabel='frequency', ylabel='power', ylim=(0, 1.5), title=f'Power spectrum {title}')
    plt.show()


def rms(data, hp_freq, tap_p, df):
    """
    returns root-mean-square of data after applying high-pass filter
    """
    hp = filter_hp(data, freq=hp_freq, tap_p=tap_p, df=df)
    return np.sqrt(np.mean(np.square(hp)))


def standardize_matrix(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0, ddof=1)  # std deviation column-wise using 1/(N-ddof)


def fourth_diff_filter(data):
    fdiff = []
    for i in range(2, len(data) - 2):
        fdiff.append(-(data[i - 2] - 4 * data[i - 1] + 6 * data[i] - 4 * data[i + 1] + data[i + 2]) / 16)
    return fdiff


if __name__ == '__main__':
    t = np.linspace(-np.pi, np.pi, 1001)
    sig = np.sin(10 * t) + 0.5 * np.sin(100 * t) + np.random.rand(len(t)) * 0.5
    # plot_spectrum(sig, 1001)
    n = len(sig)
    plt.plot(sig)
    plt.plot(cos_taper(n))
    sig1 = demean(sig)
    sig1 *= cos_taper(n)
    sig1 = lowpass(sig1, 50, 1001, zero_phase=True)
    plt.plot(sig)
    plt.show()
    plot_spectrum(sig, sig1, 1001)

    sig2 = padding(sig, 0.2)
    plt.plot(np.arange(100, 1101), sig, linewidth='2')
    plt.plot(sig2)
    sig3 = sig2 * cos_taper(int((1 + 0.2) * len(sig)), 0.2 / (1 + 0.2))
    plt.plot(sig3)
    plt.plot(cos_taper(int((1 + 0.2) * len(sig)), 0.2 / (1 + 0.2)))
    plt.show()

    sig4 = filter_hp(sig, 20, 0.2, 1001)
    plt.plot(sig4)
    plt.show()
