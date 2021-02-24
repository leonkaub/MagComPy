# MagComPy

This is software to work with aeromag data from sUAS (small Unmanned Aerial Systems) flights. It is mainly designed for Magnetic Compensation that is possible with both scalar and vector magnetic data. Two sensor systems are currently supported: Geometrics MagArrow (scalar) and SenSys MagDrone (vector). Other sensor systems or platforms can easily be adapted.

Main features are a tool to prepare files from raw sensor format to a more usable csv format, tools for magnetic compensation for both scalar and vector data, and a tool to calculate tie-line cross-differences for quality control.

## Installation

You'll need the following python packages to run this software:

- numpy
- matplotlib
- scipy
- wxpython
- pyproj
- rdp
- pandas

We recommend using [Anaconda](https://docs.conda.io/projects/conda/en/latest/) as a Python package manager.

With Anaconda installed, start Anaconda Prompt and run the command below in your magcompy repository. It will create a virtual environment with all necessary packages installed. You can change the name of the environment in the environment.yml file.

`conda env create -f environment.yml`

First activate the environment (replace 'magcompy' if you changed the name of the environment):

`conda activate magcompy`

Then start the software:

`python magcompy.py`

On Mac, run with `pythonw magcompy.py` to alleviate error message:
"This program needs access to the screen. Please run with a Framework build of python, and only when you are logged in on the main display of your Mac."

## Contact

lkaub@geophysik.uni-muenchen.de

Main developer: Leon Kaub

Many thanks to: Gordon Keller, Claire Bouligand, Grégory More, Jonathan Glen

## License

This project is licensed under [GNU AGPL v3](https://www.gnu.org/licenses/agpl-3.0.en.html). 

If you use this software, please cite:

Kaub, L., Keller, G., Bouligand, C., & Glen, J.M.G. (2021). Magnetic surveys with Unmanned Aerial Systems: software for assessing and comparing the accuracy of different sensor systems, suspension designs and compensation methods. Submitted to Geochemistry, Geophysics, Geosystems on 24 Feb 2021.
