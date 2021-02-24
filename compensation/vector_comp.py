"""
Magnetic compensation of a vector magnetometer. Calibration data is used to remove magnetic fields originating from an
aircraft. Algorithm finds best fitting parameters that are applied to survey data based on
Olsen, N., Risbo, T., Brauer, P., Merayo, J., Primdahl, F., and Sabaka, T.: In-flight compensation methods used for the
Ørsted mission, Technical University of Denmark, unpublished, (2001).
(https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.41.3567&rep=rep1&type=pdf)

This file uses functions defined in vector_comp_functions.py in a GUI interface created by vector_comp_res.py.

This file is part of MagComPy.
Copyright (C) 2021  Leon Kaub <lkaub@geophysik.uni-muenchen.de>
"""

import numpy as np
from numpy.lib.recfunctions import append_fields
import matplotlib.pyplot as plt
import wx
from datetime import datetime
import configparser
import os

from compensation.vector_comp_res import MyVectorFrame
from compensation import vector_comp_functions as vc
from compensation import scalar_comp
import magcompy
from crossings import crossings
from preparation import prep

# configuration file
config = configparser.ConfigParser(interpolation=None)
config.read('magcompy.ini')


class VectorComp(MyVectorFrame):
    def __init__(self, *args, **kwds):
        MyVectorFrame.__init__(self, *args, **kwds)
        self.find_data, self.apply_data = None, None
        self.find_F, self.apply_F = None, None
        self.scalar = None
        self.coefficients = None
        self.b_calib = None

    def on_exit(self, event):
        self.Destroy()
        event.Skip()
        exit()

    def on_scalar(self, event):
        self.Destroy()
        event.Skip()
        scalar_app = scalar_comp.MyApp(0)
        scalar_app.MainLoop()

    def on_start(self, event):
        self.Destroy()
        event.Skip()
        dlg = magcompy.MyApp(0)
        dlg.MainLoop()

    def on_crossings(self, event):
        self.Destroy()
        event.Skip()
        crossings_app = crossings.MyApp(0)
        crossings_app.MainLoop()

    def on_prep(self, event):
        prep_app = prep.MyApp(0)
        prep_app.MainLoop()

    def on_about(self, event):
        magcompy.on_about()

    def on_find_browse_btn(self, event):
        dlg = wx.FileDialog(self, "Choose a file", "", "", "*.*", wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetPath()
        else:
            filename = ""

        self.find_file.SetValue(filename)
        dlg.Destroy()

    def on_find_load_btn(self, event):
        find_filename = self.find_file.GetValue()

        if find_filename == '':
            self.log_text.AppendText(f'{datetime.now()}: Please select a file to process first\n')
            return

        try:
            self.log_text.AppendText(f'{datetime.now()}: Loading data to find parameters. Please wait.\n')
            # noinspection PyTypeChecker
            self.find_data = np.genfromtxt(find_filename, dtype=None, names=True, delimiter=',', usemask=False,
                                           encoding='ascii')
        except IOError:
            self.log_text.AppendText(f'{datetime.now()}: Something wrong with the file. Please check file path.\n')

        # get fluxgate data
        for choice in [self.find_choice_x, self.find_choice_y, self.find_choice_z]:
            choice.Clear()
            for col in self.find_data.dtype.names:
                choice.Append(col)
        self.log_text.AppendText(f'{datetime.now()}: Data to find parameters loaded from file: '
                                 f'{os.path.basename(find_filename)}\n')

    def on_find_btn(self, event):
        try:
            x_col = self.find_choice_x.GetString(self.find_choice_x.GetSelection())
            y_col = self.find_choice_y.GetString(self.find_choice_y.GetSelection())
            z_col = self.find_choice_z.GetString(self.find_choice_z.GetSelection())
            self.find_F = np.array([self.find_data[x_col], self.find_data[y_col], self.find_data[z_col]]).T
        except ValueError:
            self.log_text.AppendText(f'{datetime.now()}: Please select the correct columns\n')
            return

        self.log_text.AppendText(f'{datetime.now()}: Using XYZ data from columns: X={x_col}, Y={y_col}, Z={z_col}\n')

        # create array of constant from gui
        self.scalar = np.ones(len(self.find_data[x_col])) * self.find_scalar.GetValue()
        self.log_text.AppendText(f'{datetime.now()}: Scalar value = {self.scalar[0]}\n')

        # compute parameters
        self.coefficients, rep = vc.compute_parameters(self.find_F, self.scalar)
        self.coef.SetValue(str(self.coefficients))
        self.log_text.AppendText(f'{datetime.now()}: Coefficients calculated:\n')
        self.log_text.AppendText(rep)
        self.log_text.AppendText('\n')

        # plot result
        find_calib = vc.apply_cof(self.find_F, self.coefficients)
        time = np.arange(len(find_calib)) / 200

        fig, ax = plt.subplots()
        ax.plot(time, np.linalg.norm(self.find_F, axis=1), label='B of selected sensor')
        ax.plot(time, find_calib, label='B calibrated')
        ax.set(xlabel='time in seconds', ylabel='B in nT',
               title=f'{self.find_file.GetValue().rsplit("/")[-1]}\nB_scalar={self.scalar[0]}')
        plt.legend()
        plt.show()

    def on_load_coef_btn(self, event):
        self.log_text.AppendText('coming soon\n')

    def on_save_coef_btn(self, event):
        if self.coefficients is None:
            self.log_text.AppendText(f'{datetime.now()}: Please compute coefficients first\n')

        dlg = wx.FileDialog(self, message='Save parameters', defaultDir="", defaultFile="",
                            wildcard="csv files (*.csv)|*.csv", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.log_text.AppendText(f'{datetime.now()}: Saving parameters in {path}\n')
            self.save_par(fname=path)
        dlg.Destroy()

    def on_apply_browse_btn(self, event):
        dlg = wx.FileDialog(self, "Choose a file", "", "", "*.*", wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetPath()
        else:
            filename = ""

        self.apply_load.SetValue(filename)
        dlg.Destroy()

    def on_apply_load_btn(self, event):
        apply_filename = self.apply_load.GetValue()

        if apply_filename == '':
            self.log_text.AppendText(f'{datetime.now()}: Please select a file to process first\n')
            return

        try:
            self.log_text.AppendText(f'{datetime.now()}: Loading data to apply parameters. Please wait.\n')
            # noinspection PyTypeChecker
            self.apply_data = np.genfromtxt(apply_filename, dtype=None, names=True, delimiter=',', usemask=False,
                                            encoding='ascii')
        except IOError:
            self.log_text.AppendText(f'{datetime.now()}: Something wrong with the file. Please check file path.\n')

        # get fluxgate data
        for choice in [self.apply_choice_x, self.apply_choice_y, self.apply_choice_z]:
            choice.Clear()
            for col in self.apply_data.dtype.names:
                choice.Append(col)
        self.log_text.AppendText(f'{datetime.now()}: Data to apply parameters loaded from file: '
                                 f'{os.path.basename(apply_filename)}\n')

    def on_apply_btn(self, event):
        if self.coefficients is None:
            self.log_text.AppendText('Please calculate or load coefficients from file first\n')
            return

        try:
            x_col = self.apply_choice_x.GetString(self.apply_choice_x.GetSelection())
            y_col = self.apply_choice_y.GetString(self.apply_choice_y.GetSelection())
            z_col = self.apply_choice_z.GetString(self.apply_choice_z.GetSelection())
            self.apply_F = np.array([self.apply_data[x_col], self.apply_data[y_col], self.apply_data[z_col]]).T
        except ValueError:
            self.log_text.AppendText(f'{datetime.now()}: Please select the correct columns\n')
            return
        self.log_text.AppendText(f'{datetime.now()}: Using XYZ data from columns: X={x_col}, Y={y_col}, Z={z_col}\n')

        self.b_calib = vc.apply_cof(self.apply_F, self.coefficients)
        self.log_text.AppendText(f'{datetime.now()}: Parameters applied to '
                                 f'{os.path.basename(self.apply_load.GetValue())}\n')

        # plotting
        time = np.arange(len(self.b_calib)) / 200

        fig, ax = plt.subplots()
        ax.plot(time, np.linalg.norm(self.apply_F, axis=1), label='B of selected sensor')
        ax.plot(time, self.b_calib, label='B calibrated')
        ax.set(xlabel='time in seconds', ylabel='B in nT', title=f'{self.apply_load.GetValue()}')
        plt.legend()
        plt.show()

    def on_save_btn(self, event):
        col_name = self.new_column.GetValue()
        box_dlg = wx.FileDialog(self, message='Save data', defaultDir="", defaultFile="",
                                wildcard="csv files (*.csv)|*.csv", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)

        if box_dlg.ShowModal() == wx.ID_OK:
            path = box_dlg.GetPath()
            self.log_text.AppendText(f'{datetime.now()}: Saving data from '
                                     f'{os.path.basename(self.apply_load.GetValue())} as {path}\n')

            self.apply_data = append_fields(base=self.apply_data, names=col_name, data=self.b_calib, usemask=False)
            np.savetxt(fname=path, X=self.apply_data, fmt=config['vector compensation']['save_col_format'],
                       delimiter=',', newline='\n', header=', '.join([i for i in self.apply_data.dtype.names]))

        box_dlg.Destroy()
        self.log_text.AppendText(f'{datetime.now()}: Data saved.\n')

    def on_save_log_btn(self, event):
        log = self.log_text.GetValue()
        dlg = wx.FileDialog(self, message='Save log to file', defaultDir="", defaultFile="",
                            wildcard="txt files (*.txt)|*.txt", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            log_filename = dlg.GetPath()
            if not log_filename.endswith('.txt'):
                log_filename += '.txt'
            with open(log_filename, 'w') as f:
                f.write(log)
            self.log_text.AppendText(f'{datetime.now()}: Log saved to {log_filename}\n')
        dlg.Destroy()

    def save_par(self, fname):
        if not fname.endswith('.csv'):
            fname += '.csv'

        comment = f'coefficients for vector compensation computed with MagComPy\n' \
                  f'compensation file: {self.find_file.GetValue()}\n' \
                  f'scalar value: {self.scalar[0]}'

        try:
            np.savetxt(fname=fname, X=self.coefficients, fmt='%.6f', delimiter=',', newline='\n', header=comment)
        except (IOError, AttributeError):
            self.log_text.AppendText(f'{datetime.now()}:\tCan not save parameters. Check filepath')


class MyApp(wx.App):
    def OnInit(self):
        # noinspection
        self.frame = VectorComp(None, wx.ID_ANY, "")
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return True


if __name__ == "__main__":
    app = MyApp(0)
    app.MainLoop()
