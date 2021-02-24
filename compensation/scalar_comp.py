"""
Magnetic compensation of a scalar magnetometer. Calibration data is used to remove magnetic fields originating from an
aircraft. Ridge regression algorithm finds best fitting parameters that are applied to survey data. Method is based on
Leliak, P. (1961). Identification and evaluation of magnetic-field sources of magnetic airborne detector equipped
aircraft. IRE Transactions on Aerospace and Navigational Electronics, (3), 95-105.

This file uses functions defined in scalar_comp_functions.py in a GUI interface created by scalar_comp_res.py.

This file is part of MagComPy.
Copyright (C) 2021  Leon Kaub <lkaub@geophysik.uni-muenchen.de>
"""

import wx
from datetime import datetime

from compensation.scalar_comp_res import MyScalarFrame
from compensation.scalar_comp_functions import ScalarComp
from compensation import vector_comp
from preparation import prep
from crossings import crossings
import magcompy


class MyFrame(MyScalarFrame):
    def __init__(self, *args, **kwds):
        MyScalarFrame.__init__(self, *args, **kwds)
        self.cal_filename = ''
        self.sur_filename = ''
        self.mc = None

    def on_exit(self, event):
        self.Destroy()
        event.Skip()
        exit()

    def on_prep(self, event):
        prep_app = prep.MyApp(0)
        prep_app.MainLoop()

    def on_vector(self, event):
        self.Destroy()
        event.Skip()
        vector_app = vector_comp.MyApp(0)
        vector_app.MainLoop()

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

    def on_about(self, event):
        magcompy.on_about()

    def on_calibfile_btn(self, event):
        dlg = wx.FileDialog(self, "Choose a file", "", "", "*.*", wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.cal_filename = dlg.GetPath()
        self.log_text.AppendText(f'{datetime.now()}: You selected {self.cal_filename} as compensation file\n')
        self.calibfile_text.SetValue(self.cal_filename)
        dlg.Destroy()

    def on_surfile_btn(self, event):
        dlg = wx.FileDialog(self, "Choose a file", "", "", "*.*", wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.sur_filename = dlg.GetPath()
        self.log_text.AppendText(f'{datetime.now()}: You selected {self.sur_filename} as survey file\n')
        self.surfile_text.SetValue(self.sur_filename)
        dlg.Destroy()

    def on_load_btn(self, event):
        self.log_text.AppendText(f'{datetime.now()}: Started loading data\n')
        self.mc = ScalarComp(self.calibfile_text.GetValue(), self.surfile_text.GetValue())
        if self.mc.calib_data is None:
            self.log_text.AppendText(f'{datetime.now()}: Something is wrong with at least one of the selected files. '
                                     f'Please check\n')
        else:
            self.log_text.AppendText(f'{datetime.now()}: Data loaded\n')

    def on_comp_go_btn(self, event):
        if self.mc is None:
            self.log_text.AppendText(f'{datetime.now()}: Load data first!\n')
            return

        lp_cut, hp_cut = self.comp_lp.GetValue(), self.comp_hp.GetValue()
        if lp_cut <= 0 or hp_cut <= 0:
            self.log_text.AppendText(f'{datetime.now()}: Both cutting frequencies must be >0 !\n')
            return

        self.log_text.AppendText(f'{datetime.now()}: Started computing compensation\n')
        self.mc.compensation(lp=lp_cut, hp=hp_cut)
        self.log_text.AppendText(f'{datetime.now()}: Compensation done\n')

        self.results_text.AppendText(f'\n-------\nCompensation using lp={lp_cut} and hp={hp_cut}\n\nParameters:\n')
        for i, par in zip(range(1, len(self.mc.theta) + 1), self.mc.theta):
            if i < len(self.mc.theta):
                self.results_text.AppendText(f'{i}: {par:.3e}, ')
            else:
                self.results_text.AppendText(f'{i}: {par:.3e}\n')
        self.results_text.AppendText(f'\nIR compensation:\t{self.mc.ir_calib:.3f}\nIR survey:\t{self.mc.ir_sur:.3f}\n')

    def on_lpsweep_go_btn(self, event):
        if self.mc is None:
            self.log_text.AppendText(f'{datetime.now()}: Load data first!\n')
            return

        hp_cut = self.lpsweep_hp.GetValue()
        lp_freqs = self.lpsweep_lps.GetValue().split(',')
        try:
            lp_freqs = [float(f) for f in lp_freqs]
        except ValueError:
            self.log_text.AppendText(f'{datetime.now()}: Something wrong with the LP values. Please check')
            return
        self.log_text.AppendText(f'{datetime.now()}: LP sweep using hp={hp_cut} and lps={lp_freqs}\n')

        irb, irs = self.mc.start_sweep(cut=hp_cut, freqs=lp_freqs, mode='lp')

        self.results_text.AppendText('improvement ratios:\nfreq\tcompensation\tsurvey\n')
        for hp, b, s in zip(lp_freqs, irb, irs):
            self.results_text.AppendText(f'{hp}\t{b:.4f}\t{s:.4f}\n')
        self.results_text.AppendText('----------\n')

    def on_hpsweep_go_btn(self, event):
        if self.mc is None:
            self.log_text.AppendText(f'{datetime.now()}: Load data first!\n')
            return

        lp_cut = self.hpsweep_lp.GetValue()
        hp_freqs = self.hpsweep_hps.GetValue().split(',')
        try:
            hp_freqs = [float(f) for f in hp_freqs]
        except ValueError:
            self.log_text.AppendText(f'{datetime.now()}: Something wrong with the HP values. Please check')
            return
        self.log_text.AppendText(f'{datetime.now()}: HP sweep using lp={lp_cut} and hps={hp_freqs}\n')

        irb, irs = self.mc.start_sweep(cut=lp_cut, freqs=hp_freqs, mode='hp')

        self.results_text.AppendText('improvement ratios:\nfreq\tcompensation\tsurvey\n')
        for hp, b, s in zip(hp_freqs, irb, irs):
            self.results_text.AppendText(f'{hp}\t{b:.4f}\t{s:.4f}\n')
        self.results_text.AppendText('----------\n')

    def on_save_btn(self, event):
        cal_dlg = wx.FileDialog(self, message='Save compensation data', defaultDir="", defaultFile="",
                                wildcard="csv files (*.csv)|*.csv", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if cal_dlg.ShowModal() == wx.ID_OK:
            path = cal_dlg.GetPath()
            self.log_text.AppendText(f'{datetime.now()}: Saving compensation data as {path}\n')
            self.mc.save(fname=path, rec_type='compensation')
        cal_dlg.Destroy()
        sur_dlg = wx.FileDialog(self, message='Save survey data', defaultDir="", defaultFile="",
                                wildcard="csv files (*.csv)|*.csv", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if sur_dlg.ShowModal() == wx.ID_OK:
            path = sur_dlg.GetPath()
            self.log_text.AppendText(f'{datetime.now()}: Saving survey data as {path}\n')
            self.mc.save(fname=path, rec_type='survey')
        sur_dlg.Destroy()
        self.log_text.AppendText(f'{datetime.now()}: Data saved\n')

    def on_plot_calib_spectrum_btn(self, event):
        if self.mc is None:
            self.log_text.AppendText(f'{datetime.now()}: Load data first!\n')
            return

        self.mc.spectrum_plot(rec_type='compensation')

    def on_plot_calib_time_btn(self, event):
        if self.mc is None:
            self.log_text.AppendText(f'{datetime.now()}: Load data first!\n')
            return

        self.mc.time_plot(rec_type='compensation')

    def on_plot_calib_scatter_btn(self, event):
        if self.mc is None:
            self.log_text.AppendText(f'{datetime.now()}: Load data first!\n')
            return

        self.mc.scatter_plot(rec_type='compensation')

    def on_plot_sur_spectrum_btn(self, event):
        if self.mc is None:
            self.log_text.AppendText(f'{datetime.now()}: Load data first!\n')
            return

        self.mc.spectrum_plot(rec_type='survey')

    def on_plot_sur_time_btn(self, event):
        if self.mc is None:
            self.log_text.AppendText(f'{datetime.now()}: Load data first!\n')
            return

        self.mc.time_plot(rec_type='survey')

    def on_plot_sur_scatter_btn(self, event):
        if self.mc is None:
            self.log_text.AppendText(f'{datetime.now()}: Load data first!\n')
            return

        self.mc.scatter_plot(rec_type='survey')

    def on_show_par_btn(self, event):
        if self.mc.theta is None:
            self.log_text.AppendText(f'{datetime.now()}: Load data and compute compensation first!\n')
            return

        self.mc.show_par()

    def on_save_par_btn(self, event):
        if self.mc.theta is None:
            self.log_text.AppendText(f'{datetime.now()}: Load data and compute compensation first!\n')
            return

        dlg = wx.FileDialog(self, message='Save parameters', defaultDir="", defaultFile="",
                            wildcard="csv files (*.csv)|*.csv", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.log_text.AppendText(f'{datetime.now()}: Saving parameters in {path}\n')
            self.mc.save_par(filename=path)
        dlg.Destroy()

    def on_log_save_btn(self, event):
        log = self.log_text.GetValue()
        dlg = wx.FileDialog(self, message='Save log to file', defaultDir="", defaultFile="",
                            wildcard="txt files (*.txt)|*.txt", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        if dlg.ShowModal() == wx.ID_OK:
            log_filename = dlg.GetPath()
            if not log_filename.endswith('.txt'):
                log_filename += '.txt'
            with open(log_filename, 'w') as f:
                f.write(log)
        dlg.Destroy()


class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, wx.ID_ANY, "")
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return True


if __name__ == "__main__":
    app = MyApp(0)
    app.MainLoop()
