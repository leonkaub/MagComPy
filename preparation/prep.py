"""
This file processes .csv files from one of the two mag sensors: Geometrics MagArrow or Sensys MagDrone
using functions from prep_functions.py and GUI interface from pre_res.py

This file is part of MagComPy.
Copyright (C) 2021  Leon Kaub <lkaub@geophysik.uni-muenchen.de>
"""

import wx
import sys
from preparation.prep_res import MyPrepFrame
from preparation import prep_functions, flightlog_mag_fusion as fmf
import magcompy
import configparser
import pandas as pd
import numpy as np
from datetime import datetime

config = configparser.ConfigParser(interpolation=None)
config.read('magcompy.ini')


class MyFrame(MyPrepFrame):
    def __init__(self, *args, **kwds):
        MyPrepFrame.__init__(self, *args, **kwds)
        self.filename = None
        self.filename_newfile = None
        self.base_filename = None
        self.flightlog_filename = None

        # set stdout to log. should be changed for proper logging
        sys.stdout = self.log_text

    def on_exit(self, event):
        self.Destroy()
        event.Skip()
        # set stout back to standard
        sys.stdout = sys.__stdout__

    def on_about(self, event):
        magcompy.on_about()

    def on_base_file_btn(self, event):
        if not self.base_checkbox.GetValue():
            self.log_text.AppendText(f'{datetime.now()}\tNo basestation data used. Check box first\n')
            return

        dlg = wx.FileDialog(self, "Choose a file", "", "", "*.*", wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.base_filename = dlg.GetPath()
        if self.base_filename is None:
            self.base_fname.SetValue('')
        else:
            self.base_fname.SetValue(self.base_filename)
        dlg.Destroy()

    def on_base_plot_btn(self, event):
        if not self.base_checkbox.GetValue():
            self.log_text.AppendText(f'{datetime.now()}\tNo basestation data used. Check box first\n')
            return

        if self.base_fname.GetValue() == '':
            self.log_text.AppendText(f'{datetime.now()}\tPlease select a basestation file first\n')
            return

        base = prep_functions.Base(self.base_filename, utc=int(config['Base']['utc_offset']))
        base_data = base.read_file()

        base.plot_base(base_data[0], base_data[1])

    def on_file_btn(self, event):
        dlg = wx.FileDialog(self, "Choose a file", "", "", "*.*", wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetPath()

        if self.filename is None:
            self.file_fname.SetValue('')
        else:
            self.file_fname.SetValue(self.filename)
        dlg.Destroy()

    def on_flightlog_btn(self, event):
        dlg = wx.FileDialog(self, "Choose a file", "", "", "*.*", wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.flightlog_filename = dlg.GetPath()

        self.flightlog_file.SetValue(self.flightlog_filename)
        dlg.Destroy()

    def on_start_btn(self, event):
        if self.file_fname.GetValue() == '':
            self.log_text.AppendText(f'{datetime.now()}\tPlease select a file to process first\n')
            return

        if self.base_checkbox.GetValue() and self.base_fname.GetValue() == '':
            self.log_text.AppendText(f'{datetime.now()}\tPlease select a basestation file first\n')
            return

        # BASE STATION DATA
        self.log_text.AppendText(f'{datetime.now()}:\tBase station file: {self.base_filename}\n')
        base = prep_functions.Base(self.base_filename, utc=int(config['Base']['utc_offset']))
        base_data = base.read_file()

        # MAG DATA
        sensor = self.sensor_radio_box.GetStringSelection()

        if sensor == 'MagArrow':
            self.log_text.AppendText(f'{datetime.now()}:\tStarting to process MagArrow file: {self.filename}\n')

            ma = prep_functions.MagArrow(filename=self.filename, base=base_data)

            # check file and if all readings are valid
            try:
                if ma.check_mag() != 0:
                    ma.valid = False
            except IOError:
                self.log_text.AppendText('Wrong file. Please check filepath.\n')
                return

            # deal with file if some readings are invalid
            if not ma.valid:
                msg = 'Invalid readings in MagArrow file. Do you want to continue?'
                warn_dlg = wx.MessageBox(message=msg, caption="Warning", style=wx.YES_NO)

                if warn_dlg == wx.YES:
                    ma.valid = True
                else:
                    return

            # start processing file
            ma.start_after_check()

            # SAVE DATA
            self.log_text.AppendText(f'{datetime.now()}:\tData processing finished\n')
            # get path
            dlg = wx.FileDialog(self, message='Save MagArrow file', defaultDir="", defaultFile="",
                                wildcard="csv files (*.csv)|*.csv", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
            if dlg.ShowModal() == wx.ID_OK:
                self.filename_newfile = dlg.GetPath()
            else:
                self.log_text.AppendText(f'{datetime.now()}:\tUser stopped saving data.\n')
            # save
            if ma.save_data(new_file=self.filename_newfile):
                self.log_text.AppendText(f'{datetime.now()}:\tData saved as {self.filename_newfile}\n\n')
            else:
                self.log_text.AppendText(f'{datetime.now()}:\tCan not save data. Check filepath\n')

        elif sensor == 'MagDrone':
            try:
                with open(self.filename) as f:
                    file_header = f.readline()
            except IOError:
                self.log_text.AppendText('Wrong file. Please check filepath.\n')
                return

            self.log_text.AppendText(f'{datetime.now()}:\tStarting to process MagDrone file: {self.filename} '
                                     f'with header: {file_header}')
            md = prep_functions.MagDrone(filename=self.filename, base=base_data)

            # SAVE DATA
            self.log_text.AppendText(f'{datetime.now()}:\tFinished processing MagDrone file. Save file..\n')

            # get path
            dlg = wx.FileDialog(self, message='Save MagDrone file', defaultDir="", defaultFile="",
                                wildcard="csv files (*.csv)|*.csv", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
            if dlg.ShowModal() == wx.ID_OK:
                self.filename_newfile = dlg.GetPath()
            else:
                self.log_text.AppendText(f'{datetime.now()}:\tUser stopped saving data.\n')
                return

            # save
            if md.save_data(new_file=self.filename_newfile):
                self.log_text.AppendText(f'{datetime.now()}:\tData saved as {self.filename_newfile}\n\n')
            else:
                self.log_text.AppendText(f'{datetime.now()}:\tSomething wrong with path to new file.\n')

        # FLIGHTLOG MERGE
        if self.flightlog_radio_box.GetStringSelection() == 'Yes':
            self.flightlog_merge()

    def flightlog_merge(self):
        """
        merge data from mag sensor with GPS locations and timestamps from UAS
        designed for DJI .DAT files. need to convert .DAT files to csv using DatCon
        (https://datfile.net/DatCon/intro.html)
        """
        self.log_text.AppendText(f'{datetime.now()}\tStarting flightlog merge\n')

        # read flightlog. take only data from sensors that are available. settings read from .ini file
        gps = np.array([config.getboolean('flightlog', 'GPS1_working'),
                        config.getboolean('flightlog', 'GPS2_working'),
                        config.getboolean('flightlog', 'GPS3_working')])
        sensors_working = np.where(gps == True)[0]

        if len(sensors_working) == 3:  # take all GNSS sensors
            fl_columns = ['offsetTime',
                          'GPS(0):Time',
                          'GPS(1):Time',
                          'GPS(2):Time',
                          'GPS(0):Lat[degrees]',
                          'GPS(1):Lat[degrees]',
                          'GPS(2):Lat[degrees]',
                          'GPS(0):Long[degrees]',
                          'GPS(1):Long[degrees]',
                          'GPS(2):Long[degrees]',
                          'IMU_ATTI(0):Latitude[degrees [-180;180]]',
                          'IMU_ATTI(1):Latitude[degrees [-180;180]]',
                          'IMU_ATTI(2):Latitude[degrees [-180;180]]',
                          'IMU_ATTI(0):Longitude[degrees [-180;180]]',
                          'IMU_ATTI(1):Longitude[degrees [-180;180]]',
                          'IMU_ATTI(2):Longitude[degrees [-180;180]]']  # all columns needed from flightlog data

        elif len(sensors_working) == 2:  # take only 2 GNSS sensors
            fl_columns = ['offsetTime',
                          f'GPS({sensors_working[0]}):Time',
                          f'GPS({sensors_working[1]}):Time',
                          f'GPS({sensors_working[0]}):Lat[degrees]',
                          f'GPS({sensors_working[1]}):Lat[degrees]',
                          f'GPS({sensors_working[0]}):Long[degrees]',
                          f'GPS({sensors_working[1]}):Long[degrees]',
                          f'IMU_ATTI({sensors_working[0]}):Latitude[degrees [-180;180]]',
                          f'IMU_ATTI({sensors_working[0]}):Longitude[degrees [-180;180]]',
                          f'IMU_ATTI({sensors_working[1]}):Latitude[degrees [-180;180]]',
                          f'IMU_ATTI({sensors_working[1]}):Longitude[degrees [-180;180]]']

        else:  # take only 1 GNSS sensor
            fl_columns = ['offsetTime',
                          f'GPS({sensors_working[0]}):Time',
                          f'GPS({sensors_working[0]}):Lat[degrees]',
                          f'GPS({sensors_working[0]}):Long[degrees]',
                          f'IMU_ATTI({sensors_working[0]}):Latitude[degrees [-180;180]]',
                          f'IMU_ATTI({sensors_working[0]}):Longitude[degrees [-180;180]]']

        fl_data = pd.read_csv(self.flightlog_filename, usecols=fl_columns)
        # remove 1st row if flightlog converter added NaN to beginning
        if np.isnan(fl_data[f'GPS({sensors_working[0]}):Time'][0]):
            fl_data = fl_data.drop(0)
            fl_data = fl_data.reset_index(drop=True)
        self.log_text.AppendText(f'{datetime.now()}\tFlightlog data loaded\n')

        # create avg gps
        fl_data = fmf.avg_gps_timing(fl_data, sensors_available=sensors_working)

        self.log_text.AppendText(f'{datetime.now()}\tFlightlog data processed\n')

        # read mag data
        mag_data = pd.read_csv(self.filename_newfile)

        self.log_text.AppendText(f'{datetime.now()}\tMag data loaded\n')

        # set timestamps
        mag_data = fmf.set_timestamp_mag(mag_data)
        fl_data = fmf.set_timestamp_flightlog(fl_data, mag_data)

        # down-sample to 100 Hz
        if self.sensor_radio_box.GetStringSelection() == 'MagArrow':
            mag_dec = fmf.decimate_magarrow(mag_data, 10, '1L')  # get down to 100 Hz
            crs = int(config['MA']['crs_epsg'])
        elif self.sensor_radio_box.GetStringSelection() == 'MagDrone':
            mag_dec = fmf.decimate_magdrone(mag_data, 2, '5L')  # get down to 100 Hz
            crs = int(config['MD']['crs_epsg'])
        else:
            return

        # create new dataframe with merged data
        new_df = fmf.merge(fl_data, mag_dec, round_per='10L')

        # average locations of the GPS sensors
        new_df = fmf.avg_gps_loc(new_df, sensors=sensors_working)

        new_df = new_df.interpolate(method='linear', axis=0)

        # correct time format
        new_df = fmf.correct_time_format(new_df)

        # project coordinates
        new_df = fmf.project_coordinates(new_df, crs, int(config['general']['proj_crs_epsg']))
        self.log_text.AppendText(f'{datetime.now()}:\tFlightlog coordinates projected to '
                                 f'{config["general"]["proj_crs"]}.\n')

        # no need to save all data from FL. take only columns specified in fl_cols
        fl_cols = np.array(['GPS(Fuse):Lat[degrees]', 'GPS(Fuse):Long[degrees]', 'GPS(Fuse):Time',
                            'X_flightlog_m', 'Y_flightlog_m'])
        cols = np.concatenate((mag_data.columns.values, fl_cols), axis=None)
        new_df = new_df[cols]

        self.log_text.AppendText(f'{datetime.now()}\tData merged\n')

        new_df_nparray = new_df.to_numpy()

        # get correct save format
        sensor = self.sensor_radio_box.GetStringSelection()
        if sensor == 'MagArrow':
            fl_format = config['MA']['save_col_format'] + ',%.8f,%.8f,%s,%.3f,%.3f'
        elif sensor == 'MagDrone':
            fl_format = config['MD']['save_col_format'] + ',%.8f,%.8f,%s,%.3f,%.3f'
        else:
            self.log_text.AppendText(f'{datetime.now()}:\tWrong sensor selection.')
            return

        # save data
        with wx.FileDialog(self, message='Save merged data to file', defaultDir="", defaultFile="",
                           wildcard="csv files (*.csv)|*.csv", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL:
                self.log_text.AppendText(f'{datetime.now()}\tUser stopped flightlog merge\n')
                return

            merged_path = dlg.GetPath()
            # new_df.to_csv(merged_path, index=False)
            np.savetxt(fname=merged_path, X=new_df_nparray, fmt=fl_format,
                       delimiter=',', newline='\n', header=','.join([i for i in list(new_df.columns)]))

            self.log_text.AppendText(f'{datetime.now()}\tMerged data saved to {merged_path}\n\n')


class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, wx.ID_ANY, "")
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return True


if __name__ == "__main__":
    app = MyApp(0)
    app.MainLoop()
