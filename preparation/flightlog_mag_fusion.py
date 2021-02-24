#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flightlog_mag_fusion.py

Master script/module for performing the data fusion of flightlog and magnetometer
for testing whether this improves accuracy (assessed via tieline analysis).

Input files:
    
    1. Flight log file converted from DatCon -- use the DatCon specifications in
        DatCon.cfg for setting the parameters within DatCon to keep output file
        consistently formatted.
    2. Magnetometer file -- either from the MagArrow (default) or the MagDrone

Sequence of processes : 
    1. GPS_time_fusion: the three GPS sensors aboard the Matrice 600 are out of
        sync, and our best guess for the accurate time is an average between the
        three.
    2. Resampling of files: since the sample rate is different for the flightlog
        and 
    
    
TO-DO:
    - Make error catching more robust (file type/parsing,etc.)
    - Change the ret val of gps time averaging to series rather than another df
    - Simplify the time computation by computing to epoch and averaging
        rather than the original computation
    - Change argument option -o to -F so that -o can be for final output rather
        than optional intermediate file output
    - Reimplement 'target columns' from original GPS time fusion file if necessary
    - Similar to get_sec, other time conversion and formatting function can be
        changed to work with pandas Datetime rather than string processing
    - Add option to leave out path argument and assume that it is PWD.
    - Add correction for argument check in get_sec (removed because int registering
        as int64)
    - rewrite the resampling function to use decimate function rather than 
        pandas dataframe resample (no filtering)
    - use the epoch time within resampling as well
    - make v2 with numpy/scipy decimation and resampling
    - correct for patching up line where we combine date from mag with time from
        flightlog (get date from GPS(0) date probably)
    - switch to argparse instead of getopt (simpler)
    - add column selection option to make program quicker and produce smaller files
    - find better solution to odd-number of sample bug with decimate (asfreq and decimate return different
        lengths), where currently we exclude the last couple values when doing
        asfreq
    
        
@author: gordonkeller

This file is part of MagComPy.
Copyright (C) 2021  Leon Kaub <lkaub@geophysik.uni-muenchen.de>
"""

# ============================================================================
# Imports
# ============================================================================
import sys
import getopt
import os
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.signal as sig

from pyproj import Transformer

# ============================================================================
# Global Variables
# ============================================================================
param_dict = {
    'error': False,  # set to True if there is a problem like an argument left out
    'fl_fn': None,  # flightlog filename
    'mag_fn': None,  # magnetometer filename
    'data_dir': None,  # directory containing the flightlog and magnetometer file
    'mag_type': 'MagArrow',  # mag filetype; choice between (def) MagArrow and MagDrone
    'new_fl_fn': None,  # optional output file with averaged GPS times
    'new_fn': None  # the final file output
}

valid_exts = ['csv', 'CSV']  # only allow reads of these filetypes

start_time = 0.0
end_time = 0.0
interp_freq = 10  # hz
per = int(1.0 / interp_freq * 1000)
fper = str(per) + 'L'


# ============================================================================
# Helper Functions
# ============================================================================


def arg_parsing(argv):
    """
    Helper function for checking the argument list.

    Returns
    -------
    Dictionary of argument keys and values.

    """
    # instantiate the dictionary to be populated
    ret_dict = {
        'error': False,
        'fl_fn': None,
        'mag_fn': None,
        'data_dir': None,
        'mag_type': 'MagArrow',
        'new_fl_fn': None,
        'new_fn': None
    }

    try:
        opts, args = getopt.getopt(argv, "hf:m:t:d:o:n:", ["magfile=", "flfile=",
                                                           "magtype=", "dir=", "output=",
                                                           "newfile="])
    except getopt.GetoptError:
        print("flightlog_mag_fusion.py flightlog_mag_fusion.py -t <magtype> -f "
              "<flightlog_filename> -m <magdata_filename>"
              "-d <path_to_files> -o <new_flightlog_filename> "
              "-n <combined_file_filename>")
        ret_dict['error'] = True
        return ret_dict
    for opt, arg in opts:
        if opt == '-h':
            print("flightlog_mag_fusion.py flightlog_mag_fusion.py -t <magtype> -f"
                  "<flightlog_filename> -m <magdata_filename>"
                  "-d <path_to_files> -o <new_flightlog_filename> -n <combined_file_filename>")
            sys.exit()
        elif opt in ("-t", "--magtype"):
            ret_dict['mag_type'] = arg
        elif opt in ("-m", "--magfile"):
            ret_dict['mag_fn'] = arg
        elif opt in ("-f", "--flfile"):
            ret_dict['fl_fn'] = arg
        elif opt in ("-d", "--dir"):
            ret_dict['data_dir'] = arg
        elif opt in ("-o", "--output"):
            ret_dict['new_fl_fn'] = arg
        elif opt in ("-n", "--new"):
            ret_dict['new_fn'] = arg

    print('MagData type: ', ret_dict['mag_type'])
    print('MagData filename: ', ret_dict['mag_fn'])
    print('Flightlog filename: ', ret_dict['fl_fn'])
    print('Directory containing files: ', ret_dict['data_dir'])
    print('Output filename for new flightlog: ', ret_dict['new_fl_fn'])
    return ret_dict


def check_flightlog(fn):
    """
    Helper function for checking the validity of the filename and file for
    the desired flightlog.

    Parameters
    ----------
    fn : TYPE
        DESCRIPTION.

    Returns
    -------
    True if valid, False if not.

    """
    # check argument type
    if not isinstance(fn, str):
        print('ERROR: Filename is not a string')
        return False

    # check the extension
    if not fn.split('.')[-1] in valid_exts:
        print('ERROR: Filename extension ' + str(fn.split('.')[-1]) + ' is not valid.')
        return False

    return True


def check_mag(fn):
    """
    Helper function for checking the validity of the filename and file for
    the desired magnetometer file.

    Parameters
    ----------
    fn : string
        Filename for the file of interest.

    Returns
    -------
    True if valid, False if not.

    """
    # check argument type
    if not isinstance(fn, str):
        print('ERROR: Filename is not a string')
        return False

    # check the extension
    if not fn.split('.')[-1] in valid_exts:
        print('ERROR: Filename extension ' + str(fn.split('.')[-1]) + ' is not valid.')
        return False

    return True


def check_path(path):
    """
    Check the path argument passed 

    Parameters
    ----------
    path : str
        The path to the directory containing the flightlog and mag csvs.

    Returns
    -------
    bool
        Returns True if the string for the path is valid and False if not.

    """
    # check argument type
    if not isinstance(path, str):
        print('ERROR: Path is not a string')
        return False

    # check if the path exists
    if not os.path.isdir(path):
        print('ERROR: The path provided does not exist.')
        return False

    return True


def check_new_fl_filename(fn):
    """
    Helper function for checking the validity of the filename for the new flightlog
    file.

    Parameters
    ----------
    fn : string
        Filename for the file of interest.

    Returns
    -------
    True if valid, False if not.

    """
    # check argument type
    if not isinstance(fn, str):
        print('ERROR: Filename is not a string')
        return False

    # check the extension
    if not fn.split('.')[-1] in valid_exts:
        print('ERROR: Filename extension ' + str(fn.split('.')[-1]) + ' is not valid.')
        return False

    # check that the filename isn't the same as the original flightlog filename
    if fn == param_dict['fl_fn']:
        print('ERROR: New flightlog file cannot be saved with the same name as'
              'original flightlog file (no overwriting allowed)')
        return False

    return True


def get_sec(time):
    """
    Helper function to convert an integer representation of time as HHMMSS to 
    seconds.

    Parameters
    ----------
    time : int
        Time of day represented as HHMMSS (24 hour).

    Returns
    -------
    integer
        The day time in seconds if successful, and -1 if not.

    """
    timestr = str(int(time)).zfill(6)
    hh = int(timestr[0:2])
    mm = int(timestr[2:4])
    ss = int(timestr[4:6])

    return int(hh) * 3600 + int(mm) * 60 + int(ss)


def get_hh_mm_ss_ddd(time):
    """from float seconds, returns string"""
    hhmmss = str(time).zfill(6).split('.')[0]

    hh = int(float(hhmmss) / 3600)
    mm = int((float(hhmmss) - hh * 3600) / 60)
    ss = int(float(hhmmss) - hh * 3600 - mm * 60)

    ddd = str(time).split('.')[1]

    return str(hh) + ':' + str(mm) + ':' + str(ss) + '.' + ddd


def hh_mm_ss_to_sec(timestr):
    """
    Converts the time provided in the format HH:MM:SS.DDD, where HH is the hour,
    MM is the minute, and SS.DDD is the millisecond-precision second

    Parameters
    ----------
    timestr : string representing the time

    Returns
    -------
    None.

    """
    hh = float(timestr[0:2])
    mm = float(timestr[3:5])
    ss = float(timestr[6:])
    return hh * 3600 + mm * 60 + ss


# ============================================================================
# Primary Functions
# ============================================================================


def avg_gps_timing(df, sensors_available=None):
    """
    Computes the average GPS time and retimestamps for the flightlog.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame resulting from CSV read of the flightlog data.
    sensors_available: list of which GPS sensors are available (0, 1, 2)

    Returns
    -------
    A new Pandas DataFrame with an added column. -- eventually change to a Series

    """

    # ---------------|--------------------------- GPS0
    # ---------|--------------------------------- GPS1
    # -----------------------------|--------------GPS2

    # at the index, time is theoretically prev_t + 1, so every tick back from that
    # is worth prev_t + 1 - 1/freq, hence the equation for approximate start should
    # be prev_t + 1 - i/freq

    if sensors_available is None:
        sensors_available = [0, 1, 2]

    approx_freq = 200.0  # Hz

    if len(sensors_available) == 3:
        # find the approximate start time for each of the three GPSs
        change_index = [0, 0, 0]
        gps0_firstval = df['GPS(0):Time'][0]
        for i, t in enumerate(df['GPS(0):Time']):
            if t != gps0_firstval:
                # save the index
                change_index[0] = i
                break
        gps1_firstval = df['GPS(1):Time'][0]
        for i, t in enumerate(df['GPS(1):Time']):
            if t != gps1_firstval:
                # save the index
                change_index[1] = i
                break
        gps2_firstval = df['GPS(2):Time'][0]
        for i, t in enumerate(df['GPS(2):Time']):
            if t != gps2_firstval:
                # save the index
                change_index[2] = i
                break

        avg_gps_start = (get_sec(gps0_firstval) + 1 - (change_index[0] / approx_freq)) + \
                        (get_sec(gps1_firstval) + 1 - (change_index[1] / approx_freq)) + \
                        (get_sec(gps2_firstval) + 1 - (change_index[2] / approx_freq))
        avg_gps_start /= 3.0

    elif len(sensors_available) == 2:
        # find the approximate start time for each of the two GPSs
        change_index = [0, 0]
        gps0_firstval = df[f'GPS({sensors_available[0]}):Time'][0]
        for i, t in enumerate(df[f'GPS({sensors_available[0]}):Time']):
            if t != gps0_firstval:
                # save the index
                change_index[0] = i
                break
        gps1_firstval = df[f'GPS({sensors_available[1]}):Time'][0]
        for i, t in enumerate(df[f'GPS({sensors_available[1]}):Time']):
            if t != gps1_firstval:
                # save the index
                change_index[1] = i
                break

        avg_gps_start = (get_sec(gps0_firstval) + 1 - (change_index[0] / approx_freq)) + \
                        (get_sec(gps1_firstval) + 1 - (change_index[1] / approx_freq))
        avg_gps_start /= 2.0

    else:
        # approximate start time is equal to only available GPS time
        change_index = [0]
        gps0_firstval = df[f'GPS({sensors_available[0]}):Time'][0]
        for i, t in enumerate(df[f'GPS({sensors_available[0]}):Time']):
            if t != gps0_firstval:
                # save the index
                change_index[0] = i
                break
        avg_gps_start = get_sec(gps0_firstval)

    # create new time out of gps_avg_start + offsetTime - offsetTime(0)
    fuse_gps_time = [avg_gps_start + (x - df['offsetTime'][0]) for x in df['offsetTime']]

    # add the column to our original dataframe
    df.insert(len(df.columns), "GPS(Fuse):Time", [get_hh_mm_ss_ddd(x) for x in fuse_gps_time], True)

    return df  # new dataframe with added column


def avg_gps_loc(df, sensors=None):
    """
    Averages the GPS location from three GPS sources.

    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame resulting from CSV read of the flightlog data.
    sensors: list of which GPS sensors are available (0, 1, 2)

    Returns
    -------
    A new Pandas DataFrame with an added column. -- eventually change to a Series

    """
    # 'GPS(0):Lat[degrees]'

    if sensors is None:
        sensors = [0, 1, 2]

    if not isinstance(df, pd.DataFrame):
        print('ERROR: argument passed is not a pandas DataFrame.')
        return False

    if len(sensors) == 3:
        df['GPS(Fuse):Lat[degrees]'] = df[['IMU_ATTI(0):Latitude[degrees [-180;180]]',
                                           'IMU_ATTI(1):Latitude[degrees [-180;180]]',
                                           'IMU_ATTI(2):Latitude[degrees [-180;180]]']].mean(axis=1)

        df['GPS(Fuse):Long[degrees]'] = df[['IMU_ATTI(0):Longitude[degrees [-180;180]]',
                                            'IMU_ATTI(1):Longitude[degrees [-180;180]]',
                                            'IMU_ATTI(2):Longitude[degrees [-180;180]]']].mean(axis=1)
    elif len(sensors) == 2:
        df['GPS(Fuse):Lat[degrees]'] = df[[f'IMU_ATTI({sensors[0]}):Latitude[degrees [-180;180]]',
                                           f'IMU_ATTI({sensors[1]}):Latitude[degrees [-180;180]]']].mean(axis=1)

        df['GPS(Fuse):Long[degrees]'] = df[[f'IMU_ATTI({sensors[0]}):Longitude[degrees [-180;180]]',
                                            f'IMU_ATTI({sensors[1]}):Longitude[degrees [-180;180]]']].mean(axis=1)

    else:
        df['GPS(Fuse):Lat[degrees]'] = df[[f'IMU_ATTI({sensors[0]}):Latitude[degrees [-180;180]]']]
        df['GPS(Fuse):Long[degrees]'] = df[[f'IMU_ATTI({sensors[0]}):Longitude[degrees [-180;180]]']]

    return df


def resample_fl_mag(fl_df, mag_df):
    """
    Resample either one or both of the dataframes to align timing.
    

    Parameters
    ----------
    fl_df : Pandas DataFrame
        flightlog dataframe.
    mag_df : Pandas DataFrame
        mag dataframe

    Returns
    -------
    bool
        True if successfully generated, False if an error was encountered.

    """
    # check that both are pandas dataframes
    if not isinstance(fl_df, pd.DataFrame):
        print('ERROR: flightlog data structure is not a DataFrame.')
        return False
    if not isinstance(mag_df, pd.DataFrame):
        print('ERROR: mag data structure is not a DataFrame.')
        return False

    # at this point, code will break if there is a day transition
    fl_df.index = pd.to_datetime(mag_df[' Date'][0] + " " + fl_df['GPS(Fuse):Time'])
    mag_df.index = pd.to_datetime(mag_df[' Date'][0] + " " + mag_df[' Time'])

    # get the limits of the plot
    dt_start = max(fl_df.index[0], mag_df.index[0])
    dt_end = min(fl_df.index[-1], mag_df.index[-1])

    # perform interpolation on both the files for a prescribed frequency and round off the error
    flightlog_downsampled = fl_df.asfreq(fper).interpolate(method='linear')
    flightlog_downsampled.index = flightlog_downsampled.index.round(fper)
    sensorlog_resampled = mag_df.asfreq(fper).interpolate(method='linear')
    sensorlog_resampled.index = sensorlog_resampled.index.round(fper)

    # take from the start and stop times before merging
    trunc_fl = flightlog_downsampled[dt_start:dt_end]
    trunc_sl = sensorlog_resampled[dt_start:dt_end]

    # combine the frames and write to new file
    new_dataframe = pd.concat([trunc_fl, trunc_sl], axis=1)

    return new_dataframe


def resample_fl_mag_2(fl_df, mag_df, factor):
    """
    UNFINISHED -- DO NOT USE YET
    An alternative method for resampling the flightlog and mag data to (a) be
    artificially synchronized and (b) filtered.
    
    This method utilizes the scipy decimate function. This is preferred to alt-
    ernative methods as it filters the data using an eighth order Chebychev filter
    prior to downsampling.
    
    The pandas dataframe is still used despite this being a numpy-focused algorithm.
    This is because pandas was implemented with 

    Parameters
    ----------
    fl_df : TYPE
        DESCRIPTION.
    mag_df : TYPE
        DESCRIPTION.
    factor : TYPE
        DESCRIPTION.

    Returns
    -------
    new_dataframe : TYPE
        DESCRIPTION.

    """
    print(fl_df)
    # only decimate the mag file which is at 1000 hz
    mag_df = mag_df.apply(sig.decimate, args=[factor]).set_index(mag_df.asfreq('100L').index)
    print(mag_df)

    # align timestamps by rounding to the nearest 5 ms (i.e. 200 hz)
    new_dataframe = None

    return new_dataframe


def decimate_magarrow(mag_df, factor=10, orig_per='none'):
    """
    Decimate the mag data down to a factor of 'factor' and reset the index. The columns
    related to 

    Parameters
    ----------
    mag_df : pandas DataFrame
        The magnetometer data to be decimated. Each data column representing
        sensed data (i.e. excluding the localization and timestamping) has the
        decimate scipy function applied in the returned dataframe.
    factor : int
        The factor by which to decimate the dataframe
    orig_per : str
        The original frequency of the dataframe -- if not provided, will attempt
        inferring the value but otherwise will fallback on the default value '1L'
        (which translates to one millisecond).

    Returns
    -------
    new_df : pandas DataFrme
        A new dataframe that is at the frequency resulting from decimation, which
        can then be merged with other dataframes at the same frequency.
        

    """
    if not isinstance(factor, int):
        print('ERROR: Decimation must be a whole factor (default: 10).')
        return None

    if orig_per == 'none':
        # no original period provided -- will attempt to infer the frequency
        orig_per = pd.infer_freq(mag_df.index)
        if orig_per is None:
            print('WARNING: inferring the period for the mag dataframe failed,'
                  ' so 1L will be used.')
            orig_per = '1L'

    millis = int(orig_per[:-1])  # will fail or return incorrect res currently if not L

    s_newper = f"{millis * factor}L"

    # list of the columns we want to use scipy.signal.decimate for
    # the columns which we should attempt to decimate
    dec_cols = [' Total_Field_nT', ' Flux_X_nT', ' Flux_Y_nT', ' Flux_Z_nT', ' base_nT', ' diurnal_nT']
    # list of columns we will grab every nth
    # the columns which we should simply select every nth val
    skip_cols = ['# Counter', ' Date', ' Time', ' X_NAD83UTM10N_m', ' Y_NAD83UTM10N_m', ' Altitude_m']
    new_df_1 = mag_df[dec_cols].apply(sig.decimate, args=[factor]).set_index(mag_df.asfreq(s_newper).index)
    new_df_2 = mag_df[skip_cols].asfreq(s_newper)

    return pd.concat([new_df_1, new_df_2], axis=1)


def decimate_magdrone(mag_df, factor=10, orig_per='none'):
    """
    Decimate the mag data down to a factor of 'factor' and reset the index. The columns
    related to 

    Parameters
    ----------
    mag_df : pandas DataFrame
        The magnetometer data to be decimated. Each data column representing
        sensed data (i.e. excluding the localization and timestamping) has the
        decimate scipy function applied in the returned dataframe.
    factor : int
        The factor by which to decimate the dataframe
    orig_per : str
        The original frequency of the dataframe -- if not provided, will attempt
        inferring the value but otherwise will fallback on the default value '1L'
        (which translates to one millisecond).

    Returns
    -------
    new_df : pandas DataFrme
        A new dataframe that is at the frequency resulting from decimation, which
        can then be merged with other dataframes at the same frequency.
        

    """
    if not isinstance(factor, int):
        print('ERROR: Decimation must be a whole factor (default: 10).')
        return None

    if orig_per == 'none':
        # no original period provided -- will attempt to infer the frequency
        orig_per = pd.infer_freq(mag_df.index)
        if orig_per is None:
            print('WARNING: inferring the period for the mag dataframe failed,'
                  ' so 1L will be used.')
            orig_per = '1L'

    millis = int(orig_per[:-1])  # will fail or return incorrect res currently if not L

    s_newper = f"{millis * factor}L"

    # list of the columns we want to use scipy.signal.decimate for
    # the columns which we should attempt to decimate
    dec_cols = [' B1x_nT', ' B1y_nT', ' B1z_nT', ' B2x_nT', ' B2y_nT',
                ' B2z_nT', ' Bt1_nT', ' Bt2_nT', ' Btavg_nT',
                ' base_nT', ' diurnal_1_nT', ' diurnal_2_nT', ' diurnal_avg_nT']
    # list of columns we will grab every nth
    # the columns which we should simply select every nth val
    skip_cols = ['# Timestamp_ms', ' Date', ' Time', ' X_NAD83UTM10N_m', ' Y_NAD83UTM10N_m', ' Altitude_m']

    # index to compute out to (to agree with decimate)
    end = mag_df.index[len(mag_df) - (len(mag_df) % factor) - 1]

    new_df_1 = mag_df[dec_cols].apply(sig.decimate, args=[factor]).set_index(mag_df[:end].asfreq(s_newper).index)
    new_df_2 = mag_df[skip_cols][:end].asfreq(s_newper)

    return pd.concat([new_df_1, new_df_2], axis=1)


def set_timestamp_flightlog(fl_df, mag_df):
    """
    Takes the expected columns of the magnetometer dataframe that represents
    date and time and combines to give new index.
    Currently relies on the 'date' from mag -- want to change eventually to only
    rely on flightlog. (or ignore date altogether)

    Parameters
    ----------
    fl_df : pandas DataFrame
        Flightlog data.
    mag_df : pandas DataFrame
        Mag data.

    Returns
    -------
    fl_df : pandas DataFrame
        the new flightlog data.

    """
    if not isinstance(fl_df, pd.DataFrame):
        print('ERROR: argument 0 must be a pandas DataFrame.')
        return None

    fl_df.index = pd.to_datetime(mag_df[' Date'][0] + " " + fl_df['GPS(Fuse):Time'])

    return fl_df


def set_timestamp_mag(mag_df):
    """
    Takes the expected columns of the magnetometer dataframe that represents
    date and time and combines to give new index.

    Parameters
    ----------
    mag_df : pandas DataFrame
        Mag data.

    Returns
    -------
    mag_df : pandas DataFrame
        the new mag data.

    """
    if not isinstance(mag_df, pd.DataFrame):
        print('ERROR: argument 0 must be a pandas DataFrame.')
        return None

    mag_df.index = pd.to_datetime(mag_df[' Date'][0] + " " + mag_df[' Time'])

    return mag_df


def timestamp_cleanup(df):
    """
    Makes an irregular timestamp regular by process of removing repeat values
    and then interpolating the data associated missing timestamps.

    Parameters
    ----------
    df : pandas DataFrame
        Pandas dataframe with DatetimeIndex timestamps and with associated columns
        to correct for.

    Returns
    -------
    df.

    """
    # check that the input argument
    if not isinstance(df, pd.DataFrame):
        print('ERROR: not a pandas dataframe')
        return None

    # reject multiple values
    df = df.loc[~df.index.duplicated(keep='first')]

    return df


def merge(fl_df, mag_df, round_per='1L'):
    """
    Join the two dataframes: flight log and mag data (magarrow/magdrone agnostic).
    The data should have been brought to the same frequency at this point.

    Parameters
    ----------
    fl_df : pandas DataFrame
        Flightlog dataframe.
    mag_df : pandas DataFrame
        Mag dataframe.
    round_per : str
        round percentage

    Returns
    -------
    new_df : pandas Dataframe
        The conjoined dataframes.

    """
    if not isinstance(fl_df, pd.DataFrame):
        print('ERROR: the first argument must be a pandas DataFrame')
        return None
    if not isinstance(mag_df, pd.DataFrame):
        print('ERROR: the second argument must be a pandas DataFrame')
        return None

    # round the indexes to the specified period
    fl_df.index = fl_df.index.round(round_per)
    mag_df.index = mag_df.index.round(round_per)

    # check that the indices of the dataframe overlap
    if not overlap(fl_df.index, mag_df.index):
        print('ERROR: the two dataframes passed do not have overlapping time periods')
        print(f"fl start/end: {fl_df.index[0]} - {fl_df.index[-1]}")
        print(f"mag start/end: {mag_df.index[0]} - {mag_df.index[-1]}")
        return None

    # clean up the timestamps (for which there might be repeats after rounding)
    fl_df = timestamp_cleanup(fl_df)
    mag_df = timestamp_cleanup(mag_df)

    # find the overlap of the dataframes 
    start = max(fl_df.index[0], mag_df.index[0])
    end = min(fl_df.index[-1], mag_df.index[-1])

    n_fl = fl_df[start:end]
    n_mg = mag_df[start:end]

    return pd.concat([n_fl, n_mg], axis=1)


def overlap(ser1, ser2):
    """
    Look for overlap in the two timeseries provided.
    To implement: Should frequencies be the same?
    Simplify this function later.

    Parameters
    ----------
    ser1, ser2 : pandas Timeseries

    Returns
    -------
    Bool : True if the series do overlap, False if they do not.

    """
    if ser1.min() <= ser2.min():
        if ser1.max() <= ser2.min():
            print('here')  # s1 -----------
            return False  # s2               -------------
        else:
            return True  # s2       -----------
    elif ser2.min() <= ser1.min():
        if ser2.max() <= ser1.min():
            print('there')  # s1                -----------
            return False  # s2     --------
        else:
            return True  # s2          ----------
    else:
        # shouldn't ever reach here...
        return True


def project_coordinates(df, geo_crs, proj_crs):
    """
    adds channels with projected coordinates (x, y) to dataframe
    :param df: dataframe
    :param geo_crs: epsg number of geographic coordinate system. 4326 for WGS84
    :param proj_crs: epsg number of projected coordinate system. e.g., 26910 for NAD83UTM10N
    :return: dataframe
    """
    # create transformer
    transformer = Transformer.from_crs(geo_crs, proj_crs)
    # do transformation and add to df
    df['X_flightlog_m'], df['Y_flightlog_m'] = transformer.transform(df['GPS(Fuse):Lat[degrees]'].values,
                                                                     df['GPS(Fuse):Long[degrees]'].values)
    return df


def correct_time_format(df):
    """
    timestamps used so far omitted leading 0s. returns timestamps in correct format.
    (file should use datetime objects for more stable processing)
    :param df: dataframe
    :return: dataframe with updated timestamps
    """
    n_t = []
    timestamp = df['GPS(Fuse):Time'].values

    for t in timestamp:
        # 1 MagArrow test file resulted in ca. 10 nan values in middle of GPS(Fuse):Time (why?). workaround:
        if isinstance(t, float):
            n_t.append(n_t[-1])
            continue

        # split into hours, minutes, seconds
        h, m, s = t.split(':')
        ss, ddd = s.split('.')
        ddd = round(float('0.' + ddd), 7)
        # combine to correct time format
        n_t.append(f'{int(h):02d}:{int(m):02d}:{int(ss):02d}.{str(ddd)[2:]}')

    n_t = np.array(n_t)
    df['GPS(Fuse):Time'] = n_t

    return df


# ============================================================================
# Main (if run as script instead of imported as a module)
# ============================================================================


def main(argv):
    # parse arguments
    param_dict = arg_parsing(argv)

    if param_dict['error'] is True:
        print("ERROR: arguments passed failed.")
        sys.exit(2)

    # perform file checks -- update this 
    if not check_flightlog(param_dict['fl_fn']):
        print("ERROR: flightlog file check failed.")
        sys.exit(2)
    if not check_mag(param_dict['mag_fn']):
        print("ERROR: magnetometer file check failed.")
        sys.exit(2)
    if not check_path(param_dict['data_dir']):
        print("ERROR: problem with path provided to data files")
        sys.exit(2)

    # create a path object for use with referencing data files
    path = Path(param_dict['data_dir'])
    fl_full_path = path / param_dict['fl_fn']
    mag_full_path = path / param_dict['mag_fn']

    # check flightlog file exists
    if not os.path.isfile(fl_full_path):
        print('ERROR: the flightlog file does not exist')
        sys.exit(2)

    # read the original flightlog file
    data = pd.read_csv(fl_full_path)

    # average the GPS values to generate a new pandas DataFrame
    # * this function will produce a new flightlog file with added column
    # * if the argument has been provided to create it (-o)
    if data is not None:
        fused_data = avg_gps_timing(data)
    else:
        print('ERROR: failed to load flightlog as dataframe')
        sys.exit(2)

    print(fused_data)

    # if output file desired, create that output file with the new time
    if param_dict['new_fl_fn'] is not None:
        # don't terminate if filename bad -- just don't save it
        new_fl_full_path = path / param_dict['new_fl_fn']
        if new_fl_full_path == fl_full_path:
            print("ERROR: problem with the new flightlog filename...")
            print("will not write out new file but will continue with program.")
        else:
            print("Writing new file to ", new_fl_full_path)
            fused_data.to_csv(new_fl_full_path, index=False)

    # check mag file exists
    if not os.path.isfile(mag_full_path):
        print('ERROR: the mag data file does not exist.')
        sys.exit(2)

    # load the mag data as a DataFrame
    mag_data = pd.read_csv(mag_full_path)

    # create a new dataframe that is the two merged
    new_df = None
    if mag_data is None:
        print('ERROR: failed to load mag as dataframe')
        sys.exit(2)
    elif fused_data is None:
        print('ERROR: fused gps flight log dataframe is None')
        sys.exit(2)
    else:
        print(mag_data.columns.values.tolist())
        print(fused_data.columns.values.tolist())
        # columns to use for processing -- gps time at the least
        fl_cols = ['General:absoluteHeight[meters]', 'GPS(Fuse):Lat[degrees]', 'GPS(Fuse):Long[degrees]',
                   'GPS(Fuse):Time']
        # columns for processing -- time and date at the least
        mag_cols = [' Date', ' Time', ' diurnal_nT', ' X_NAD83UTM10N_m', ' Y_NAD83UTM10N_m']
        new_df = resample_fl_mag(fused_data[fl_cols], mag_data[mag_cols])

    if new_df is None:
        print("ERROR: failed merging the flight log and mag data")
        sys.exit(2)

    # create the new file
    # if output file desired, create that output file with the new time
    if param_dict['new_fn'] is not None:
        # don't terminate if filename bad -- just don't save it
        new_full_path = path / param_dict['new_fn']
        print("Writing new file to ", new_full_path)
        new_df.to_csv(new_full_path, index=True)


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])
