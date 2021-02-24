"""
This code will process .csv files from one of the two mag sensors: Geometrics MagArrow or Sensys MagDrone
- extraction from the raw files of all data relevant for magnetic prospecting
- crop out the correct segment from recordings
- interpolation if necessary to fit frequency of magnetic recordings
- gps coordinate projection to a given CRS (coordinate system)
- diurnal correction by using data from a base station
- saving relevant data into a new .csv file that can be used for further analysis

This file is part of MagComPy.
Copyright (C) 2021  Leon Kaub <lkaub@geophysik.uni-muenchen.de>
"""

import numpy as np
from numpy.lib.recfunctions import append_fields, drop_fields
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from pyproj import CRS, Transformer
import configparser

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 12, 8

# configuration file
config = configparser.ConfigParser(interpolation=None)
config.read('magcompy.ini')

# get projected crs from config file
proj = config['general']['proj_crs']
x_proj, y_proj = 'X_' + proj + '_m', 'Y_' + proj + '_m'

ma_col_names = config['MA']['col_names'].split(',')
ma_col_dtype = config['MA']['col_dtype'].split(',')
ma_col_names[3], ma_col_names[4] = x_proj, y_proj


def get_datetime(dates, times):
    """
    creates datetime objects from string representations of dates and times
    :param dates: array containing dates as strings (either YYYY/MM/DD or YYYY-MM-DD)
    :param times: array containing times as strings (HH:MM:SS.sss)
    :return: array containing datetime objects
    """
    t = []
    if dates[0][4] == '/':
        for dat, tim in zip(dates, times):
            t.append(datetime.strptime(dat + ' ' + tim, '%Y/%m/%d %H:%M:%S.%f'))
    elif dates[0][4] == '-':
        for dat, tim in zip(dates, times):
            t.append(datetime.strptime(dat + ' ' + tim, '%Y-%m-%d %H:%M:%S.%f'))
    else:
        print('Something wrong with the date format')
    return t


def get_posix_time(times):
    """
    calculate posix time stamps from an array of datetime objects
    :param times: array of datetime timestamps
    :return: numpy array of posix time stamps
    """
    return np.array([t.timestamp() for t in times])


def ma_interp_data(time, data):
    """
    MA records some data with lower f than 1000Hz. Missing values are filled with np.nan
    This function takes in low-sampled data, removes np.nan values, and interpolates to given timestamps
    time: array of timestamps from 1000Hz mag data
    data: data that needs to be upsampled
    """
    # find indices of nonan data
    idx = []
    for i in range(len(data)):
        if not np.isnan(data[i]):
            idx.append(i)
    # Interpolate data
    interpolate = interp1d(time[idx], data[idx], kind='linear', fill_value='extrapolate')
    return interpolate(time)


def plot_to_select(xdata, ydata, delta=100):
    """
    plots flightpath of full file. Min and Max can be changed to outcrop only relevant parts of the recordings.
    :param xdata: longitude
    :param ydata: latitude
    :param delta: value step of the two sliders
    :return: values of sliders when window gets closed
    """

    def update(val):
        mini, maxi = int(smin.val), int(smax.val)
        if mini > maxi:
            mini = maxi - 1
            print(f'{datetime.now()}:\tWarning: Set Min to a value smaller than Max')
        l.set_xdata(xdata[mini:maxi])
        l.set_ydata(ydata[mini:maxi])
        ax.set_xlim(min(xdata[mini:maxi]), max(xdata[mini:maxi]))
        ax.set_ylim(min(ydata[mini:maxi]), max(ydata[mini:maxi]))
        fig.canvas.draw_idle()

    def on_close(event):
        print(f'{datetime.now()}:\tSelected segment of flight path: start index: {smin.val}, end index: {smax.val}')

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.title('Select the segment you want to process with Min/Max sliders\nClose window to continue')

    min_0, max_0 = 0, len(xdata)
    l, = plt.plot(xdata[min_0:max_0], ydata[min_0:max_0], color='g')
    plt.plot(xdata[min_0:max_0], ydata[min_0:max_0], color='r', alpha=0.15)

    # create axis for two sliders
    axmin = plt.axes([0.25, 0.1, 0.65, 0.04], facecolor='yellow')
    axmax = plt.axes([0.25, 0.15, 0.65, 0.04], facecolor='yellow')

    # create 1 min slider (start value) and 1 max slider (end value)
    smin = Slider(axmin, 'Min', 0, len(xdata), valinit=min_0, valstep=delta, valfmt='%i')
    smax = Slider(axmax, 'Max', 0, len(xdata), valinit=max_0, valstep=delta, valfmt='%i')
    smin.on_changed(update)
    smax.on_changed(update)

    fig.canvas.mpl_connect('close_event', on_close)
    plt.show()

    return int(smin.val), int(smax.val)


def is_data_in_crs_bounds(crs_p, lat, lon):
    """
    return True if proj crs covers data, otherwise False
    :param crs_p: coordinate system for projected coordinates
    :param lat: latitude
    :param lon: longitude
    :return: boolean
    """
    if min(lat) > crs_p.area_of_use.south and \
            max(lat) < crs_p.area_of_use.north and \
            min(lon) < crs_p.area_of_use.east and \
            max(lon) > crs_p.area_of_use.west:
        return True
    else:
        return False


class MagArrow:
    def __init__(self, filename, base):
        self.filename = filename
        self.base_data = base
        self.valid = True
        self.lat, self.lon, self.start, self.end, self.length = None, None, 0, 0, 0
        self.data = None

    def __del__(self):
        pass

    def start_after_check(self):
        # read in all coordinates
        self.lat, self.lon = np.loadtxt(self.filename, dtype=float, delimiter=',', unpack=True,
                                        skiprows=int(config['MA']['header_rows']),
                                        usecols=(int(config['MA']['c_lat']), int(config['MA']['c_lon'])))

        # select based on interactive scatter plot
        self.start, self.end = plot_to_select(self.lon, self.lat)
        self.length = self.end - self.start

        # create structured array based on settings in ini file and fill with data
        self.data = np.zeros(self.length, dtype=np.dtype({'names': ma_col_names, 'formats': ma_col_dtype}))
        self.fill_data()

    def fill_data(self):
        """read in data from MA file and put it into data array"""
        self.data['Counter'] = [str(i).zfill(6) for i in range(self.length)]

        self.data['Date'], self.data['Time'] = np.loadtxt(self.filename, dtype=str, delimiter=',',
                                                          skiprows=int(config['MA']['header_rows']) + self.start,
                                                          usecols=(int(config['MA']['c_date']),
                                                                   int(config['MA']['c_time'])),
                                                          unpack=True, max_rows=self.length)

        self.data['Total_Field_nT'] = np.loadtxt(self.filename, dtype=float, delimiter=',',
                                                 skiprows=int(config['MA']['header_rows']) + self.start,
                                                 usecols=int(config['MA']['c_mag']), max_rows=self.length)

        self.get_proj_coord()

        # altitude data
        ma_timestamp = get_posix_time(get_datetime(self.data['Date'], self.data['Time']))
        self.data['Altitude_m'] = self.get_altitude(ma_timestamp)
        print(f'{datetime.now()}:\tAltitude data processed')

        # get interpolated fluxgate data
        self.data['Flux_X_nT'], self.data['Flux_Y_nT'], self.data['Flux_Z_nT'] = self.get_fluxgate(ma_timestamp)
        print(f'{datetime.now()}:\tFluxgate data processed')

        # diurnal correction if there is data from a basestation
        if self.base_data:
            b_timestamp = get_posix_time(self.base_data[0])
            base_interp = interp1d(b_timestamp, self.base_data[1], kind='linear')
            try:
                self.data['base_nT'] = base_interp(ma_timestamp)
                self.data['diurnal_nT'] = self.data['Total_Field_nT'] - self.data['base_nT']
                print(f'{datetime.now()}:\tDiurnal correction done')
            except ValueError:
                print(f'{datetime.now()}:\tERROR: Base-station data do not overlap properly with survey.'
                      '\nCan not do diurnal correction.')
                return

    def get_proj_coord(self):
        # transforms coordinates to projected coordinates. Coordinate system needs to be defined
        geo_crs = int(config['MA']['crs_epsg'])
        proj_crs = int(config['general']['proj_crs_epsg'])

        # check data
        if not is_data_in_crs_bounds(CRS.from_epsg(proj_crs), self.lat[self.start:self.end],
                                     self.lon[self.start:self.end]):
            # print error message
            print(f'{datetime.now()}:\tData is not within bounds of projected CRS. Please check the crs in the '
                  f'magcompy.ini file.')
            # fill x, y with 0
            self.data[x_proj] = np.zeros(len(self.lon[self.start:self.end]))
            self.data[y_proj] = np.zeros(len(self.lat[self.start:self.end]))
            return
        else:
            print(f'{datetime.now()}:\tData is within bounds of proj CRS')

        # create transformer
        transformer = Transformer.from_crs(geo_crs, proj_crs, always_xy=True)

        # coordinate projection
        self.data[x_proj], self.data[y_proj] = transformer.transform(self.lon[self.start:self.end],
                                                                     self.lat[self.start:self.end])

        print(f'{datetime.now()}:\tCoordinates projected from {config["MA"]["crs"]} to {config["general"]["proj_crs"]}')

    def check_mag(self):
        m_valid = np.loadtxt(self.filename, dtype=float, delimiter=',', skiprows=int(config['MA']['header_rows']),
                             usecols=int(config['MA']['c_mag_valid']))
        j = k = 0
        for val in m_valid:
            if val == 0:
                j += 1
                print(f'{datetime.now()}:\tWarning: mag value invalid! Position {k}')
            k += 1
        print(f'{datetime.now()}:\tNumber of warnings in magnetic recordings: {j}')
        return j

    def get_altitude(self, timestamp):
        # noinspection PyTypeChecker
        alt = np.genfromtxt(self.filename, dtype=float, delimiter=',',
                            skip_header=int(config['MA']['header_rows']) + self.start,
                            usecols=int(config['MA']['c_alt']), max_rows=self.length)

        return ma_interp_data(timestamp, alt)

    def get_fluxgate(self, timestamp):
        """
        reads in fluxgate data from MA file, interpolates it and returns all 3 components.
        """
        fx, fy, fz = np.genfromtxt(self.filename,
                                   dtype=float,
                                   skip_header=int(config['MA']['header_rows']) + self.start,
                                   delimiter=',',
                                   usecols=(int(config['MA']['c_fx']), int(config['MA']['c_fy']),
                                            int(config['MA']['c_fz'])),
                                   unpack=True,
                                   max_rows=self.length)

        return ma_interp_data(timestamp, fx), ma_interp_data(timestamp, fy), ma_interp_data(timestamp, fz)

    def save_data(self, new_file):
        # save data into new file
        formats = config['MA']['save_col_format']

        try:
            # noinspection PyTypeChecker
            np.savetxt(fname=new_file, X=self.data, fmt=formats, delimiter=',', newline='\n',
                       header=', '.join([i for i in self.data.dtype.names]))
            return True
        except (IOError, AttributeError):
            return False


class MagDrone:
    def __init__(self, filename, base):
        self.filename = filename
        self.base_data = base

        with open(self.filename) as f:
            file_header = f.readline()

        # read in gps to select segment from recording
        lat, lon = np.genfromtxt(filename, dtype=float, delimiter=';', skip_header=int(config['MD']['header_rows']),
                                 usecols=(int(config['MD']['c_lat']), int(config['MD']['c_lon'])), unpack=True)

        idx = np.nonzero(lat)
        start, end = plot_to_select(lon[idx], lat[idx], delta=1)

        # date is taken from file header
        # assuming 1st line of file header to start with YYYYMMDD_HHMMSS
        self.date = file_header[:4] + '-' + file_header[4:6] + '-' + file_header[6:8]

        # read in all data. cutting to start in next line to get names. ratio of mag frequency to gps frequency is 40.
        # noinspection PyTypeChecker
        self.data = np.genfromtxt(filename, dtype=None, delimiter=';', skip_header=int(config['MD']['header_rows']),
                                  skip_footer=len(lat) - end * 40, names=True, encoding='ascii')
        self.data = self.data[start * 40:]

        self.process_data()

    def __del__(self):
        pass

    def process_data(self):
        bt1, bt2, btd = self.get_total_field()

        lon_proj, lat_proj, alt = self.get_proj_coord()

        # create timestamps. Time in file is given in ms from beginning of recording. needs to be changed to hh:mm:ss
        md_time = self.create_timestamps()

        # check if recording covered a day transition (UTC time)
        midnight_index = -1
        for i, t in zip(range(len(md_time)), md_time):
            if not md_time[0].startswith('00') and t.startswith('00'):
                midnight_index = i
                break

        if midnight_index >= 0:  # recording covered day transition
            # get date string of next day
            next_day = self.date[:-2] + str(int(self.date[-2:]) + 1)
            # date array for original day
            md_date1 = np.tile(self.date, midnight_index)
            # date array for next day
            md_date2 = np.tile(next_day, len(self.data['Timestamp_ms']) - midnight_index)
            # combine to a single date array
            md_date = np.concatenate((md_date1, md_date2))
        else:  # otherwise just take one date

            # check if start of recording and start of trimmed data are on same day or not
            # assuming 1st line of file header to start with YYYYMMDD_HHMMSS
            with open(self.filename) as f:
                file_header = f.readline()
                if file_header[9] == '2' and md_time[0].startswith('0'):
                    # recording started on last day, flight was on next day
                    self.date = self.date[:-2] + str(int(self.date[-2:]) + 1)

            md_date = np.tile(self.date, len(self.data['Timestamp_ms']))

        # drop fields that are no longer needed.
        drop_names = ('AccX_g', 'AccY_g', 'AccZ_g', 'Temp_Deg', 'Latitude_Decimal_Degrees', 'Longitude_Decimal_Degrees',
                      'Altitude_m', 'Satellites', 'Quality', 'GPSTime', 'f0')
        self.data = drop_fields(base=self.data, drop_names=drop_names, usemask=False)

        # append newly created data.
        append_names = ('Bt1_nT', 'Bt2_nT', 'Btavg_nT', 'Date', 'Time', x_proj, y_proj, 'Altitude_m')
        self.data = append_fields(base=self.data, names=append_names,
                                  data=(bt1, bt2, btd, md_date, md_time, lon_proj, lat_proj, alt), usemask=False)

        # diurnal correction if there is data from a basestation
        if self.base_data:
            md_timestamp = get_posix_time(get_datetime(md_date, md_time))
            b_timestamp = get_posix_time(self.base_data[0])
            bmag = self.base_data[1]

            base_interp = interp1d(b_timestamp, bmag, kind='linear')

            try:
                bmag_interp = base_interp(md_timestamp)
                diu1 = bt1 - bmag_interp
                diu2 = bt2 - bmag_interp
                diud = btd - bmag_interp
                print(f'{datetime.now()}:\tDiurnal correction done')
            except ValueError:
                print(f'{datetime.now()}:\tERROR: Base-station data do not overlap properly with survey.'
                      '\nCan not do diurnal correction.')
                bmag_interp, diu1, diu2, diud = np.zeros(self.data.shape[0]), np.zeros(self.data.shape[0]), \
                                                np.zeros(self.data.shape[0]), np.zeros(self.data.shape[0])

        else:
            bmag_interp, diu1, diu2, diud = np.zeros(self.data.shape[0]), np.zeros(self.data.shape[0]), \
                                            np.zeros(self.data.shape[0]), np.zeros(self.data.shape[0])

        append_names_base = ('base_nT', 'diurnal_1_nT', 'diurnal_2_nT', 'diurnal_avg_nT')
        self.data = append_fields(base=self.data, names=append_names_base, data=(bmag_interp, diu1, diu2, diud),
                                  usemask=False)

    def get_proj_coord(self):
        """
        projects coordinates into given CRS. Returns zero if data is out of bounds of CRS.
        :return: X, Y, altitude; all in meters
        """
        lat, lon, alt = self.interp_gps()
        geo_crs = int(config['MD']['crs_epsg'])
        proj_crs = int(config['general']['proj_crs_epsg'])

        # check data
        if not is_data_in_crs_bounds(CRS.from_epsg(proj_crs), lat, lon):
            # print error message
            print(f'{datetime.now()}:\tData is not within bounds of projected CRS. Please check the crs in the '
                  f'magcompy.ini file.')
            # fill x, y with 0 when data is not within bounds of CRS
            x = np.zeros(len(lon))
            y = np.zeros(len(lat))
            return x, y, alt

        print(f'{datetime.now()}:\tData is within bounds of proj CRS')

        # create transformer
        transformer = Transformer.from_crs(geo_crs, proj_crs, always_xy=True)

        # coordinate projection
        x, y = transformer.transform(lon, lat)

        print(f'{datetime.now()}:\tCoordinates projected from {config["MD"]["crs"]} to {config["general"]["proj_crs"]}')
        return x, y, alt

    def interp_gps(self):
        """
        MD gives GPS every 0.2s. This functions returns lat/lon and altitude with 200Hz after linear interpolation.
        :return: latitude, longitude, altitude
        """
        lat, lon, alt = self.data['Latitude_Decimal_Degrees'], self.data['Longitude_Decimal_Degrees'], \
                        self.data['Altitude_m']

        idx = np.nonzero(lat)  # indexing GPS stamps
        x = np.arange(len(lat))

        # create interpolators
        interp_lat = interp1d(x[idx], lat[idx], kind='linear', fill_value='extrapolate')
        interp_lon = interp1d(x[idx], lon[idx], kind='linear', fill_value='extrapolate')
        interp_alt = interp1d(x[idx], alt[idx], kind='linear', fill_value='extrapolate')

        # return interpolated data
        return interp_lat(x), interp_lon(x), interp_alt(x)

    def create_timestamps(self):
        md_time = []
        idx = np.nonzero(self.data['GPSTime'])  # index for nonzero values in raw gps time readings
        gps_t = self.data['GPSTime'][idx]  # nonzero values in raw gps time readings
        # deal with 0 values until the first gps reading:
        for i in range(-idx[0][0], 0):
            md_time.append(gps_t[0] + 5e-3 * i)
        # remaining timestamps: between each gps reading, add difference given in column timestamp[ms] to last gps value
        for i in range(idx[0][0], len(self.data['GPSTime'])):
            # check also for something else (e.g., satellites) in case of midnight
            if self.data['GPSTime'][i] == 0 and self.data['Satellites'][i] == 0:
                md_time.append(
                    md_time[i - 1] + (self.data['Timestamp_ms'][i] - self.data['Timestamp_ms'][i - 1]) * 1e-3)
            else:
                md_time.append(self.data['GPSTime'][i])
        # format to nice timestamps
        md_time = ['{:.3f}'.format(t) for t in md_time]  # cut to 3 decimals
        md_time = [t.zfill(10) for t in md_time]  # raw time readings have no preceding 0s (e.g. 00:01:11 = 111)
        return [t[:2] + ':' + t[2:4] + ':' + t[4:] for t in md_time]  # add colons

    def get_total_field(self):
        # computing total field of each sensor; mean total field
        bt1 = np.sqrt(self.data['B1x_nT'] ** 2 + self.data['B1y_nT'] ** 2 + self.data['B1z_nT'] ** 2)
        bt2 = np.sqrt(self.data['B2x_nT'] ** 2 + self.data['B2y_nT'] ** 2 + self.data['B2z_nT'] ** 2)
        btd = (bt1 + bt2) / 2
        print(f'{datetime.now()}:\tTotal field computed')
        return bt1, bt2, btd

    def save_data(self, new_file):
        # save data into new file
        formats = config['MD']['save_col_format']
        try:
            # noinspection PyTypeChecker
            np.savetxt(fname=new_file, X=self.data, fmt=formats, delimiter=',', newline='\n',
                       header=', '.join([i for i in self.data.dtype.names]))
            return True
        except (IOError, AttributeError):
            return False


class Base:
    """
    this applies to base station files from Geometrics G858 (.stn files)
    file format: *, 0, date_number, time, counter, magnetic recording
    """

    def __init__(self, filename, utc):
        self.filename = filename
        self.utc = utc

    def __del__(self):
        pass

    def read_file(self):
        """
        reads in basestation data. designed for .stn files. column indices can be chagned in config file
        :return: datetime object, magnetic field strength
        """
        if not self.filename:
            return False

        # get column indices from config file
        date_column = config['Base'].getint('c_date')
        time_column = config['Base'].getint('c_time')
        mag_column = config['Base'].getint('c_mag')
        # get year from config file
        year = config['Base'].getint('year')

        # sometimes there is no whitespace between * and 2nd column so it is treated as a single column
        with open(self.filename) as f:
            if f.readline()[1] != ' ':
                date_column -= 1
                time_column -= 1
                mag_column -= 1

        # search for ? in file and replace with whitespace. changes file permanently
        with open(self.filename, 'r') as fr:
            basetext = fr.read()
            if '?' in basetext:
                basetext = basetext.replace('?', ' ')
                with open(self.filename, 'w') as fw:
                    fw.write(basetext)

        # read in data column-wise
        date, mag = np.loadtxt(self.filename, dtype=float, comments='', usecols=(date_column, mag_column),
                               unpack=True)
        time = np.loadtxt(self.filename, dtype=str, comments='', usecols=time_column)

        # process data
        mag = [x / 10 for x in mag]  # mag data given without decimal mark
        date_timedelta = [timedelta(x - 1) for x in date]  # date given in days since Jan 1
        time = [datetime(year=year, month=1, day=1, hour=int(x[:2]), minute=int(x[2:4]),
                         second=int(x[4:6])) for x in time]  # parse time as datetime object for Jan 1 and add day later
        utc_timedelta = timedelta(hours=self.utc)

        date_time = []
        for i in range(len(time)):
            date_time.append(time[i] + date_timedelta[i] + utc_timedelta)  # add day of the year and utc offset to time

        return date_time, mag

    def plot_base(self, dtime, field):
        """
        creates a simple plot of the full data in the specified basestation file
        :param dtime: datetime object
        :param field: magnetic field strength
        """
        fig, ax = plt.subplots()
        ax.plot(dtime, field)
        ax.set_title(f'Base station data from file {self.filename}')
        ax.set_xlabel('time [UTC]')
        ax.set_ylabel('B [nT]')
        fig.autofmt_xdate(rotation=45)
        plt.show()


if __name__ == '__main__':
    ba = Base('base_data.stn', utc=7)
    ba_data = ba.read_file()

    ma = MagArrow(filename='MagArrow_data.csv', base=ba_data)

    if not ma.valid:
        print('Invalid readings in MagArrow file.')
        continue_anyways = input('Continue anyways?')

        if continue_anyways == 'y' or continue_anyways == 'Y' or continue_anyways == 'yes' or continue_anyways == 'Yes':
            ma.valid = True
        else:
            exit()

    ma.start_after_check()

    ma.save_data(new_file='MagArrow_data_prepared.csv')
