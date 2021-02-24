"""
MagComPy
Copyright (C) 2021  Leon Kaub <lkaub@geophysik.uni-muenchen.de>

MagComPy (=Magnetic Compensation with Python) is a tool to work with aeromagnetic data collected with
Unmanned Aerial Systems (UAS):
- removal of undesired magnetic signals originating from the UAS for both scalar and vector magnetic data.
- analysis of cross-over differences for quality control of magnetic data.
- file preparation for data collected with Geometrics MagArrow (https://www.geometrics.com/product/magarrow/) or
SenSys MagDrone R3 (https://sensysmagnetometer.com/products/systems/aerial-survey-systems/magdrone-r3-magnetometer-kit/)
- merging with GNSS data from UAS flightlog

developed using Python 3.7.6
graphical interfaces designed using wxGlade 0.9.6 with wxPython 4.0.7.post2
"""

import wx
from wx.adv import AboutDialogInfo, AboutBox
from preparation import prep
from compensation import scalar_comp
from compensation import vector_comp
from crossings import crossings
from shared.magcompy_res import MyDialog

about_text = 'MagComPy processes magnetic data from surveys with Unmanned Aerial Systems.' \
             '\n\nThis project is licensed under the GNU AGPLv3 (full text can be found in the LICENSE file)\n'
version_string = 'v1.0.0'


def on_about():
    about_info = AboutDialogInfo()
    about_info.SetName('MagComPy')
    about_info.SetDescription(about_text)
    about_info.SetCopyright('Copyright (C) 2021 Leon Kaub')
    about_info.SetVersion(version_string)
    about_info.SetWebSite('https://github.com/leonkaub/MagComPy')

    AboutBox(about_info)


class MyDlg(MyDialog):
    def __init__(self, *args, **kwds):
        MyDialog.__init__(self, *args, **kwds)

    def on_exit(self, event):
        self.Destroy()
        event.Skip()
        exit()

    def on_about(self, event):
        on_about()

    def on_prep_btn(self, event):
        prep_app = prep.MyApp(0)
        prep_app.MainLoop()

    def on_scalar_btn(self, event):
        self.Destroy()
        event.Skip()
        scalar_app = scalar_comp.MyApp(0)
        scalar_app.MainLoop()

    def on_vector_btn(self, event):
        self.Destroy()
        event.Skip()
        vector_app = vector_comp.MyApp(0)
        vector_app.MainLoop()

    def on_crossings_btn(self, event):
        self.Destroy()
        event.Skip()
        crossings_app = crossings.MyApp(0)
        crossings_app.MainLoop()


class MyApp(wx.App):
    def OnInit(self):
        self.dlg = MyDlg(None, wx.ID_ANY, "")
        self.SetTopWindow(self.dlg)
        self.dlg.Show()
        return True


if __name__ == "__main__":
    app = MyApp(0)
    app.MainLoop()
