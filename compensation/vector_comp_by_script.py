import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.recfunctions import append_fields

from compensation import vector_comp_functions as vc


if __name__ == '__main__':
    # INPUT PARAMETERS
    directory = 'D:\\Projects\\Santa Cruz_Sep2020\\data\\3_processed\\'
    calibration_file = directory + 'MD_frame_calib_v2.csv'
    list_of_survey_files = ['MD_frame_survey_normal_v3.csv',
                            'MD_frame_survey_reversed_v2.csv'
                            ]
    sensor = 1
    save_file_extension = f'B{sensor}calib.csv'
    # END OF INPUT PARAMETERS

    x_col = f'B{sensor}x_nT'
    y_col = f'B{sensor}y_nT'
    z_col = f'B{sensor}z_nT'

    print(f'start calibration of vector data\ncalibration file={calibration_file}\n'
          f'survey files={list_of_survey_files}\nsensor={sensor}\n')

    # add directory to each filename
    list_of_survey_files = [directory + file for file in list_of_survey_files]

    # read in compensation data
    calib_data = np.genfromtxt(calibration_file, dtype=None, names=True, delimiter=',', usemask=False, encoding='ascii')
    calib_F = np.array([calib_data[x_col], calib_data[y_col], calib_data[z_col]]).T
    print(f'loaded calibration data from file {calibration_file}')

    for file in list_of_survey_files:
        # read in survey data
        apply_data = np.genfromtxt(file, dtype=None, names=True, delimiter=',', usemask=False, encoding='ascii')
        apply_F = np.array([apply_data[x_col], apply_data[y_col], apply_data[z_col]]).T
        print(f'loaded data from file {file}')

        # create array of scalar value
        scalar = np.ones(len(calib_data[x_col])) * np.mean(apply_data['Btavg_nT'])
        print(f'scalar value = {scalar[0]}')

        # compute coefficients
        coefficients, _report = vc.compute_parameters(calib_F, scalar)
        print(f'coefficients={coefficients}')

        # plot compensation
        find_calib = vc.apply_cof(calib_F, coefficients)
        time = np.arange(len(find_calib)) / 200

        fig1, ax = plt.subplots()
        ax.plot(time, np.linalg.norm(calib_F, axis=1), label=f'B of sensor {sensor}')
        ax.plot(time, find_calib, label='B calibrated')
        ax.set(xlabel='time in seconds', ylabel='B in nT', title=f'{calibration_file}\nB_scalar={scalar[0]:.3f}')
        plt.legend()
        plt.show()

        # apply coefficients to survey
        b_calib = vc.apply_cof(apply_F, coefficients)

        # plot survey
        time = np.arange(len(b_calib)) / 200

        fig2, ax = plt.subplots()
        ax.plot(time, np.linalg.norm(apply_F, axis=1), label=f'B of sensor {sensor}')
        ax.plot(time, b_calib, label='B calibrated')
        ax.set(xlabel='time in seconds', ylabel='B in nT', title=f'{file}')
        plt.legend()
        plt.show()

        # save data
        apply_data = append_fields(base=apply_data, names=f'B{sensor}_calib', data=b_calib, usemask=False)

        save_format = '%i,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%s,%s,%.3f,%.3f,%.1f,%.3f,%.3f,%.3f,%.3f,%.3f'
        save_file = file[:-4] + save_file_extension

        np.savetxt(fname=save_file, X=apply_data, fmt=save_format,
                   delimiter=',', newline='\n', header=', '.join([i for i in apply_data.dtype.names]))
        print(f'saved data as {save_file}')
