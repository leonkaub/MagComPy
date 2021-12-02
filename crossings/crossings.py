"""
Computing cross-over differences of magnetic data.

This file is part of MagComPy.
Copyright (C) 2021  Leon Kaub <lkaub@geophysik.uni-muenchen.de>
"""

import wx
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import decimate

from crossings.crossings_res import MyCrossFrame
from crossings import crossings_functions as cross
from shared import filter
from preparation import prep
from compensation import scalar_comp
from compensation import vector_comp
import magcompy


class MyFrame(MyCrossFrame):
    def __init__(self, *args, **kwds):
        MyCrossFrame.__init__(self, *args, **kwds)
        self.filename = self.load_text.GetValue()
        self.data = None
        self.x, self.y = None, None

        self.index_table = None
        self.lines = None
        self.line_channel = None
        self.f_lines, self.t_lines = None, None

        self.crossing_idx = None
        self.cross_x, self.cross_y = None, None
        self.diff = None
        self.q = 1
        self.lp = False

    def on_exit(self, event):
        self.Destroy()
        event.Skip()

    def on_prepare(self, event):
        prepare_app = prep.MyApp(0)
        prepare_app.MainLoop()

    def on_scalar(self, event):
        self.Destroy()
        event.Skip()
        scalar_app = scalar_comp.MyApp(0)
        scalar_app.MainLoop()

    def on_vector(self, event):
        self.Destroy()
        event.Skip()
        vector_app = vector_comp.MyApp(0)
        vector_app.MainLoop()

    def on_start(self, event):
        self.Destroy()
        event.Skip()
        main_app = magcompy.MyApp(0)
        main_app.MainLoop()

    def on_about(self, event):
        magcompy.on_about()

    def on_browse_btn(self, event):
        dlg = wx.FileDialog(self, "Choose a file", "", "", "*.*", wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.filename = dlg.GetPath()

        self.load_text.SetValue(self.filename)
        dlg.Destroy()
        if self.filename == '':
            self.log_text.AppendText(f'{datetime.now()}: No file selected\n')
        else:
            self.log_text.AppendText(f'{datetime.now()}: You selected {self.filename}\n')

    def on_load_btn(self, event):
        if self.filename == '':
            self.log_text.AppendText(f'{datetime.now()}: Please select a file to process first\n')
            return

        try:
            self.filename = self.load_text.GetValue()
            self.log_text.AppendText(f'{datetime.now()}: Loading data from file {self.filename}.\nPlease wait.\n')
            # noinspection PyTypeChecker
            self.data = np.genfromtxt(self.filename, dtype=None, names=True, delimiter=',', usemask=False,
                                      encoding='ascii')

            # fill x,y choices with column names
            for choice in [self.choice_x, self.choice_y]:
                choice.Clear()
                for col in self.data.dtype.names:
                    choice.Append(col)
            # fill stats choice with column names
            self.cross_stats_select.Clear()
            for col in self.data.dtype.names:
                self.cross_stats_select.Append(col)

            self.log_text.AppendText(f'{datetime.now()}: Data loaded\n')
        except IOError:
            self.log_text.AppendText(f'{datetime.now()}: Something wrong with the file. Please check file path.\n')

    def on_turns_btn(self, event):
        if self.data is None:
            self.log_text.AppendText(f'{datetime.now()}: Load data first\n')
            return

        # fill x,y data
        par_x = self.choice_x.GetString(self.choice_x.GetSelection())
        par_y = self.choice_y.GetString(self.choice_y.GetSelection())
        if par_x is '' or par_y is '':
            self.log_text.AppendText(f'{datetime.now()}: Select a column for X and Y coordinates\n')
            return
        self.x, self.y = self.data[par_x], self.data[par_y]

        # down-sampling
        self.q = int(self.decimate_factor.GetValue())
        if self.q != 1:
            self.x, self.y = self.x[::self.q], self.y[::self.q]  # for coordinates, take every q-th element

        self.log_text.AppendText(f'{datetime.now()}: Searching for turning points with '
                                 f'epsilon = {self.turns_eps.GetValue()} and '
                                 f'min angle = {self.turns_minangle.GetValue()}\n')

        simple_x, simple_y, self.index_table = cross.get_turns(self.x, self.y,
                                                               float(self.turns_eps.GetValue()),
                                                               float(self.turns_minangle.GetValue()),
                                                               q=10)

        self.log_text.AppendText(f'{datetime.now()}: Found {len(self.index_table)} turning points\n')

        turns_x, turns_y = cross.get_turn_crds(self.index_table, self.x, self.y)

        fig, ax = plt.subplots()
        ax.plot(self.x, self.y, 'b', label='original path')
        ax.plot(simple_x, simple_y, 'y--', label='simplified path')
        ax.plot(turns_x, turns_y, 'rx', label='turning points')
        ax.set_title('Paths & turning points')
        plt.legend()
        plt.show()

    def on_lines_btn(self, event):
        if self.data is None:
            self.log_text.AppendText(f'{datetime.now()}: Load data first\n')
            return

        if self.index_table is None:
            self.log_text.AppendText(f'{datetime.now()}: Find turns first\n')
            return

        self.lines, self.f_lines, self.t_lines, il, it = cross.assign_line_type(self.x, self.y,
                                                                                self.index_table,
                                                                                line_az=float(
                                                                                    self.lines_fl_az.GetValue()),
                                                                                tie_az=float(
                                                                                    self.lines_tl_az.GetValue()),
                                                                                az_tol=float(
                                                                                    self.lines_az_tol.GetValue()),
                                                                                min_l=float(
                                                                                    self.lines_minlength.GetValue()))

        self.log_text.AppendText(f'{datetime.now()}: Found {len(self.f_lines) + len(self.t_lines)} lines: '
                                 f'{len(self.f_lines)} flight lines and {len(self.t_lines)} tie lines.\n')

        self.line_channel = cross.create_line_channel(self.lines, len(self.x))

        self.plot_lines()

    def plot_lines(self):
        flight_crds, tie_crds = cross.get_line_crds(self.x, self.y, self.line_channel)
        flight_x, flight_y = flight_crds.T
        tie_x, tie_y = tie_crds.T

        fig, ax = plt.subplots()
        ax.scatter(flight_x, flight_y, s=1, label='flight lines')
        ax.scatter(tie_x, tie_y, s=1, label='tie lines')
        plt.legend()
        plt.show()

    def on_cut_btn(self, event):
        if self.data is None:
            self.log_text.AppendText(f'{datetime.now()}: Load data first\n')
            return

        if self.index_table is None:
            self.log_text.AppendText(f'{datetime.now()}: Find turns first\n')
            return

        if self.f_lines is None:
            self.log_text.AppendText(f'{datetime.now()}: Split into lines first\n')
            return

        f_idx, t_idx = int(self.cut_fl.GetValue()), int(self.cut_tl.GetValue())
        self.lines, self.f_lines, self.t_lines = cross.change_line_length(self.lines, self.f_lines, self.t_lines,
                                                                          f_idx, t_idx)

        self.log_text.AppendText(f'{datetime.now()}: Cut flight lines by {f_idx} and tie lines by {t_idx} readings.\n')

        self.line_channel = cross.create_line_channel(self.lines, len(self.x))

        self.plot_lines()

    def on_cross_btn(self, event):
        if self.data is None:
            self.log_text.AppendText(f'{datetime.now()}: Load data first\n')
            return

        if self.index_table is None:
            self.log_text.AppendText(f'{datetime.now()}: Find turns first\n')
            return

        if self.f_lines is None:
            self.log_text.AppendText(f'{datetime.now()}: Split into lines first\n')
            return

        self.log_text.AppendText(f'{datetime.now()}: Calculating crossing points. Please wait.\n')
        self.crossing_idx, self.cross_x, self.cross_y = cross.calculate_crossings(self.x, self.y,
                                                                                  self.f_lines, self.t_lines)
        self.log_text.AppendText(f'{datetime.now()}: Found {len(self.crossing_idx)} crossing points.\n')

        self.on_cross_plot_btn(self)

    def on_cross_lp_btn(self, event):
        p = self.cross_stats_select.GetString(self.cross_stats_select.GetSelection())
        if p is '':
            self.log_text.AppendText(f'{datetime.now()}: Nothing selected\n')
            return

        try:
            float(self.data[p][0])
        except ValueError or TypeError:
            self.log_text.AppendText(f'{datetime.now()}: Selected parameter does not contain numbers\n')
            return

        cut_freq = float(self.cross_lp_freq.GetValue())
        sampling_rate = float(self.cross_sampling.GetValue())

        lp_filtered = filter.filter_lp_no_detrend(data=self.data[p], freq=cut_freq, tap_p=0.2, df=sampling_rate)

        t = np.arange(len(self.data[p])) / sampling_rate
        fig, ax = plt.subplots()
        ax.plot(t, self.data[p], label='raw')
        ax.plot(t, lp_filtered, label='LP filtered')
        ax.set(title='LP filter', xlabel='time in s', ylabel='B in nT')
        plt.legend()
        plt.show()

    def on_cross_stats_btn(self, event):
        if self.data is None:
            self.log_text.AppendText(f'{datetime.now()}: Load data first\n')
            return

        if self.index_table is None:
            self.log_text.AppendText(f'{datetime.now()}: Find turns first\n')
            return

        if self.f_lines is None:
            self.log_text.AppendText(f'{datetime.now()}: Split into lines first\n')
            return

        if self.crossing_idx is None:
            self.log_text.AppendText(f'{datetime.now()}: Calculate crossings first\n')
            return

        p = self.cross_stats_select.GetString(self.cross_stats_select.GetSelection())
        if p is '':
            self.log_text.AppendText(f'{datetime.now()}: Nothing selected\n')
            return

        try:
            float(self.data[p][0])
        except ValueError:
            self.log_text.AppendText(f'{datetime.now()}: Selected parameter does not contain numbers\n')
            return

        # down-sampling
        if self.q != 1:
            signal = decimate(self.data[p], q=self.q)
        else:
            signal = self.data[p]

        # apply LP filter if values for lp_freq and sampling_rate are set
        self.lp = False
        lp_cut = float(self.cross_lp_freq.GetValue())
        sampling_rate = int(self.cross_sampling.GetValue())
        if lp_cut == 0 and sampling_rate == 0:
            pass
        elif lp_cut == 0 or sampling_rate == 0:
            self.log_text.AppendText(f'{datetime.now()}: If you want to apply a LP filter, cutting frequency and'
                                     f'sampling rate must be non-zero!\n')
            return
        else:
            signal = filter.filter_lp_no_detrend(data=signal, freq=lp_cut, tap_p=0.2, df=sampling_rate)
            self.lp = True

        # compute differences at crossings
        self.diff = cross.crossing_diffs(signal, self.crossing_idx)

        # compute rms and std of crossing differences
        rms = np.sqrt(np.mean(np.square(self.diff)))
        std = np.std(abs(self.diff))

        if self.lp:
            self.log_text.AppendText(f'{datetime.now()}: Stats for parameter {p} after LP filter: '
                                     f'rms = {rms:.4f}, std = {std:.4f}\n')
        else:
            self.log_text.AppendText(f'{datetime.now()}: Stats for parameter {p}: rms = {rms:.4f}, std = {std:.4f}\n')

    def on_cross_plot_btn(self, event):
        self.on_cross_stats_btn(self)

        p = self.cross_stats_select.GetString(self.cross_stats_select.GetSelection())

        # turns_x, turns_y = get_turn_crds(self.index_table, self.x, self.y)
        flight_crds, tie_crds = cross.get_line_crds(self.x, self.y, self.line_channel)
        flight_x, flight_y = flight_crds.T
        tie_x, tie_y = tie_crds.T

        min_x, min_y = min(self.x), min(self.y)

        fig, ax = plt.subplots(1, 1)
        ax.plot(self.x - min_x, self.y - min_y, c='gray', linestyle='--', label='full path', alpha=0.5)
        ax.scatter(flight_x - min_x, flight_y - min_y, s=1, c='gray')
        ax.scatter(tie_x - min_x, tie_y - min_y, s=1, c='gray')
        if p is '':
            ax.plot(self.cross_x - min_x, self.cross_y - min_y, 'rx', markersize=12)
            title = 'location of crossings'
        else:
            im = ax.scatter(self.cross_x - min_x, self.cross_y - min_y, c=abs(self.diff), s=60, cmap='jet')
            ax.ticklabel_format(useOffset=False, style='plain')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('nT')
            title = 'tie-line cross-over differences'
        ax.set_title(title)
        ax.set_xlabel('X in m')
        ax.set_ylabel('Y in m')
        plt.show()

    def on_cross_save_btn(self, event):
        """
        save tie-line cross-over differences in csv file
        file includes XY coordinate of crossings and cross-over differences of selected parameters
        """
        self.log_text.AppendText(f'{datetime.now()}: save cross-over differences to .csv file..\n')

        if self.data is None:
            self.log_text.AppendText(f'{datetime.now()}: Load data first\n')
            return

        if self.index_table is None:
            self.log_text.AppendText(f'{datetime.now()}: Find turns first\n')
            return

        if self.f_lines is None:
            self.log_text.AppendText(f'{datetime.now()}: Split into lines first\n')
            return

        if self.crossing_idx is None:
            self.log_text.AppendText(f'{datetime.now()}: Calculate crossings first\n')
            return

        p = self.cross_stats_select.GetString(self.cross_stats_select.GetSelection())
        if p is '':
            self.log_text.AppendText(f'{datetime.now()}: Nothing selected\n')
            return

        try:
            float(self.data[p][0])
        except ValueError:
            self.log_text.AppendText(f'{datetime.now()}: Selected parameter does not contain numbers\n')
            return

        self.on_cross_stats_btn(self)

        # get parameters for cross-over differences from a multi choice dialog
        choices = self.data.dtype.names

        with wx.MultiChoiceDialog(None, 'Select variables', 'Cross-differences', choices) as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return  # user changed their mind
            selection = dlg.GetSelections()

        # iterate through selected parameters
        differences, names = [], []
        for var in selection:
            signal_name = self.data.dtype.names[var]

            # check if all signals are valid
            try:
                float(self.data[signal_name][0])
            except ValueError:
                self.log_text.AppendText(f'{datetime.now()}: Parameter {signal_name} does not contain numbers\n')
                return

            # decimate signal if q != 1
            if self.q != 1:
                signal = decimate(self.data[signal_name], q=self.q)
            else:
                signal = self.data[signal_name]

            # apply LP filter if values for lp_freq and sampling_rate are set
            self.lp = False
            lp_cut = float(self.cross_lp_freq.GetValue())
            sampling_rate = int(self.cross_sampling.GetValue())
            if lp_cut == 0 and sampling_rate == 0:
                pass
            elif lp_cut == 0 or sampling_rate == 0:
                self.log_text.AppendText(
                    f'{datetime.now()}: If you want to apply a LP filter, cutting frequency and'
                    f'sampling rate must be non-zero!\n')
                return
            else:
                signal = filter.filter_lp_no_detrend(data=signal, freq=lp_cut, tap_p=0.2, df=sampling_rate)
                self.lp = True

            # compute cross-over differences
            diff = cross.crossing_diffs(signal, self.crossing_idx)

            names.append(signal_name)
            differences.append(diff)

        self.log_text.AppendText(f'{datetime.now()}: selected variables: {",".join(names)}\n')

        differences = np.array(differences)

        # create header
        header = f'TIE-LINE CROSS-DIFFERENCES\nfile: {self.filename}\nvariables: {",".join(names)}\n{datetime.now()}'
        if self.q != 1:
            header += f'\ndecimated with factor={self.q}'
        if self.lp:
            header += f'\nlow-pass filtered with frequency={self.cross_lp_freq.GetValue()}, ' \
                      f'sampling rate={self.cross_sampling.GetValue()}'
        header += '\nX,Y,' + ','.join(names)

        # create data array
        xy = np.column_stack((self.cross_x, self.cross_y))
        data_to_save = np.concatenate((xy, differences.T), axis=1)

        # save data to csv file
        with wx.FileDialog(self, message='Save cross-over differences to file', defaultDir="", defaultFile="",
                           wildcard="csv files (*.csv)|*.csv", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return  # user changed their mind

            cross_diff_fname = dlg.GetPath()
            if not cross_diff_fname.endswith('.csv'):
                cross_diff_fname += '.csv'

            try:
                np.savetxt(cross_diff_fname, data_to_save, fmt='%.5f', delimiter=',', newline='\n', header=header)
                self.log_text.AppendText(f'{datetime.now()}: Saved tie-line cross-over differences of {",".join(names)}'
                                         f'to {cross_diff_fname}\n')
            except IOError:
                self.log_text.AppendText(f'{datetime.now()}: Cannot save cross-over differences in file '
                                         f'{cross_diff_fname}\n')

    def on_log_save_btn(self, event):
        """
        save log text to text file
        """
        log = self.log_text.GetValue()
        with wx.FileDialog(self, message='Save log to file', defaultDir="", defaultFile="",
                           wildcard="txt files (*.txt)|*.txt", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return  # user changed their mind

            log_filename = dlg.GetPath()
            if not log_filename.endswith('.txt'):
                log_filename += '.txt'
            try:
                with open(log_filename, 'w') as f:
                    f.write(log)
                self.log_text.AppendText(f'{datetime.now()}: Log saved in file {log_filename}\n')
            except IOError:
                self.log_text.AppendText(f'{datetime.now()}: Cannot save log in file {log_filename}\n')


class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, wx.ID_ANY, "")
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return True


if __name__ == "__main__":
    app = MyApp(0)
    app.MainLoop()
