"""
Functions for computing cross-over differences of magnetic data.

This file is part of MagComPy.
Copyright (C) 2021  Leon Kaub <lkaub@geophysik.uni-muenchen.de>
"""

import numpy as np
import matplotlib.pyplot as plt
from rdp import rdp  # Ramer-Douglas-Peucker algorithm


def angle(direction):
    """
    Returns the angles between vectors.

    0: vectors point in the same direction
    pi/2: vectors are orthogonal
    pi: vectors point in opposite directions

    :param direction: 2D-array of shape (N,M) representing N vectors in M-dimensional space
    :return: 1D-array of shape (N-1,), with each value between 0 and pi
    """
    dir1 = direction[:-1]
    dir2 = direction[1:]
    return np.arccos((dir1 * dir2).sum(axis=1) / (np.sqrt((dir1 ** 2).sum(axis=1) * (dir2 ** 2).sum(axis=1))))


def get_turns(x, y, tolerance, min_angle, q=10):
    """
    comuptes coordinates of turns in flight path
    Ramer-Douglas-Peucker algorithm used to simplify path
    http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
    Python implementation: https://github.com/sebleier/RDP/ Copyright (c) 2014 Fabian Hirschmann

    :param x: X coordinate of full flight
    :param y: Y coordinate of full flight
    :param tolerance: used in rdp algorithm
    :param min_angle: used in rdp algorithm
    :param q: decimation factor, needs to be an integer >= 1
    :return: X,Y coordinates of simplified path and index table for crossing locations
    """
    min_angle = np.deg2rad(min_angle)

    # simplify path
    x_red = x[::q]  # take every q-th element to decrease processing time of rdp algorithm
    y_red = y[::q]
    points = np.vstack((x_red, y_red)).T  # rdp algorithm needs a single array of x, y tuples
    # RDP
    simplified = np.array(rdp(points.tolist(), tolerance))
    simp_x, simp_y = simplified.T  # simp_x and simp_y are simplified x and y after the RDP algorithm

    # directions and angles
    directions = np.diff(simplified, axis=0)  # compute direction vectors of the simplified curve
    theta = angle(directions)  # angles for each point in simplified
    # indices of theta where theta > min_angle
    idx = np.where(theta > min_angle)[0] + 1

    # transferring indices from theta to indices of x and y
    # number of turns is incremented with idx during the process, so the lengh of idx is the number of turns
    nbturn = len(idx)
    idx_table = np.zeros((nbturn, 1), dtype=int)  # create empty array to stock indices
    # loop used to fill the index table
    for i in range(nbturn):
        coordx, coordy = simplified[idx[i]]  # take coordinates from simplified curve where theta > min_angle
        idx_table[i] = np.where((x == coordx) & (y == coordy))[0][0]  # search for these coordinates in x, y

    return simp_x, simp_y, idx_table


def get_turn_crds(idx_table, x, y):
    """
    returns coordinates of turns based on index table
    :param idx_table: indices of crossing locations as returned by get_turns()
    :param x: X coordinate of full flight
    :param y: Y coordinate of full flight
    :return: X,Y coordinates of crossing locations
    """
    nbturn = len(idx_table)

    turns_x = np.zeros((nbturn, 1))
    turns_y = np.zeros((nbturn, 1))

    # Loop used to fill the arrays freshly created with the coord X and Y of the turnings points we have by their index
    for i in range(nbturn):
        ind = idx_table[i]
        turns_x[i] = x[ind]
        turns_y[i] = y[ind]

    return turns_x, turns_y


def assign_line_type(x, y, indices, line_az, tie_az, az_tol, min_l):
    """
    splits path into lines, assigns type to each line
    lines table columns: index of beginning of a line, index of end of a line, line type
    line type = 0: short line
    line type = 1: tie-line
    line type = 2: flight-line

    :param x: X coordinate of full flight
    :param y: X coordinate of full flight
    :param indices: indices of crossing locations as returned by get_turns()
    :param line_az: flight-line azimuth in degrees
    :param tie_az: tie-line azimuth in degrees
    :param az_tol: tolerance in degrees
    :param min_l: minimum line length
    :return: all lines, flight lines, tie lines, flight line indices, tie-line indices
    """
    nb_turn = len(indices)
    lines = np.zeros(((nb_turn + 1), 3), dtype=int)

    for i in range(nb_turn + 1):
        lines[i, 0] = indices[i - 1]
        lines[i - 1, 1] = indices[i - 1] - 1

    lines[0, 0] = 0
    lines[-1, 1] = len(x) - 1

    diffx = x[lines[:, 1]] - x[lines[:, 0]]
    diffy = y[lines[:, 1]] - y[lines[:, 0]]

    line_azimuth = np.arctan2(diffx, diffy) * 180 / np.pi
    line_length = np.sqrt(diffx ** 2 + diffy ** 2)

    ilong = np.asarray(np.where(line_length > min_l))[0]
    # np where gives array of tuples but only 1 index in the 1st dim:
    # np asarray converts to array (removes the tuple layer) and [0] removes the useless 1st dim

    azimuth_diff = (line_azimuth[ilong]) % 180
    # indices where azimuth conditions fit FOR LINES
    # laz: line_azimuth, taz: tie_line_azimuth, daz: azimuth_tolerance
    idx_l = np.where((azimuth_diff < line_az + az_tol) & (azimuth_diff > line_az - az_tol) |
                     (azimuth_diff < line_az + az_tol + 180) & (azimuth_diff > line_az - az_tol + 180))

    # indices where azimuth conditions fit FOR TIELINES
    idx_t = np.where((azimuth_diff < tie_az + az_tol) & (azimuth_diff > tie_az - az_tol) |
                     (azimuth_diff < tie_az + az_tol + 180) & (azimuth_diff > tie_az - az_tol + 180))

    lines[ilong[idx_l[0]], 2] = 1
    lines[ilong[idx_t[0]], 2] = 2

    flight_lines = np.zeros(((len(idx_l[0])), 3), dtype=int)
    tie_lines = np.zeros(((len(idx_t[0])), 3), dtype=int)

    incr1, incr2 = 0, 0
    for line in lines:
        if line[2] == 1:
            flight_lines[incr1] = line
            incr1 += 1

        elif line[2] == 2:
            tie_lines[incr2] = line
            incr2 += 1

    return lines, flight_lines, tie_lines, idx_l, idx_t


def create_line_channel(lines_table, survey_length):
    """
    channel with same length as full survey encoded with 0,1,2:
    0: short line
    1: tie-line
    2: flight-line

    :param lines_table: lines as returned by assign_line_type()
    :param survey_length: length of full survey
    :return: 1D np array, containing line type
    """
    channel = np.zeros((survey_length,), dtype=int)

    for i in lines_table:
        if i[2] == 1:
            channel[i[0]:i[1] + 1] = 1
        elif i[2] == 2:
            channel[i[0]:i[1] + 1] = 2

    return channel


def get_line_crds(x, y, line_channel):
    """
    returns coordinates of lines

    :param x: X coordinate of full flight
    :param y: Y coordinate of full flight
    :param line_channel: line channel containing line type as returned by create_line_type()
    :return: flight-line and tie-line coordinates
    """
    flight_line_crds = np.array([[x_, y_] for x_, y_, typ in zip(x, y, line_channel) if typ == 1])
    tie_line_crds = np.array([[x_, y_] for x_, y_, typ in zip(x, y, line_channel) if typ == 2])

    return flight_line_crds, tie_line_crds


def change_line_length(lines, flight_lines, tie_lines, no_fl_idx, no_tl_idx, min_length=None):
    """
    optional: reduce line length before computing crossings. currently only cutting a certain number of indices possible
    :param lines: lines as returned by assign_line_type()
    :param flight_lines: flight-lines as returned by assign_line_type()
    :param tie_lines: tie-lines as returned by assign_line_type()
    :param no_fl_idx: number of indices flight-lines get cut
    :param no_tl_idx: number of indices flight-lines get cut
    :param min_length: applies to a reduction in meters
    :return: updated lines, flight-lines, tie-lines
    """
    mode = 'idx_redct'
    new_lines = np.zeros((len(lines), 3), dtype=int)

    if mode == 'idx_redct':

        for i in range(len(lines)):
            if lines[i][2] == 1:
                new_lines[i] = [lines[i][0] + no_fl_idx, lines[i][1] - no_fl_idx, lines[i][2]]
            elif lines[i][2] == 2:
                new_lines[i] = [lines[i][0] + no_tl_idx, lines[i][1] - no_tl_idx, lines[i][2]]
            elif lines[i][2] == 0:
                new_lines[i] = lines[i]

        flight_lines[:, 0] += no_fl_idx
        flight_lines[:, 1] -= no_fl_idx

        tie_lines[:, 0] += no_tl_idx
        tie_lines[:, 1] -= no_tl_idx

    elif mode == 'perc_redct':
        percentage = 10  # how much is getting removed in percent

        flight_lines[:, 0] += ((flight_lines[:, 1] - flight_lines[:, 0]) * percentage / 200)
        flight_lines[:, 1] -= ((flight_lines[:, 1] - flight_lines[:, 0]) * percentage / 200)

        tie_lines[:, 0] += ((tie_lines[:, 1] - tie_lines[:, 0]) * percentage / 200)
        tie_lines[:, 1] -= ((tie_lines[:, 1] - tie_lines[:, 0]) * percentage / 200)

    elif mode == 'meter_redct':
        redct_meter = 10
        line_length = []

        line_length = line_length[~(line_length[:] < min_length)]
        nb_array_permeter = ((sum(tie_lines[:, 1] - tie_lines[:, 0])) / len(tie_lines)) / (
                sum(line_length[:]) / len(line_length))
        nb_array_permeter = nb_array_permeter.astype(int)

        flight_lines[:, 0] += redct_meter * nb_array_permeter
        flight_lines[:, 1] -= redct_meter * nb_array_permeter

        tie_lines[:, 0] += redct_meter * nb_array_permeter
        tie_lines[:, 1] -= redct_meter * nb_array_permeter

    return new_lines, flight_lines, tie_lines


def _rect_inter_inner(x1, x2):
    """
    python implementation from: https://github.com/sukhbinder/intersection Copyright (c) 2017 Sukhbinder Singh
    based on matlab script:
    Douglas Schwarz (2020). Fast and Robust Curve Intersections
    (https://www.mathworks.com/matlabcentral/fileexchange/11837-fast-and-robust-curve-intersections),
    MATLAB Central File Exchange. Retrieved June 1, 2020.
    """
    n1 = x1.shape[0] - 1
    n2 = x2.shape[0] - 1
    x_1 = np.c_[x1[:-1], x1[1:]]
    x_2 = np.c_[x2[:-1], x2[1:]]
    s1 = np.tile(x_1.min(axis=1), (n2, 1)).T
    s2 = np.tile(x_2.max(axis=1), (n1, 1))
    s3 = np.tile(x_1.max(axis=1), (n2, 1)).T
    s4 = np.tile(x_2.min(axis=1), (n1, 1))
    return s1, s2, s3, s4


def _rectangle_intersection_(x1, y1, x2, y2):
    """
    python implementation from: https://github.com/sukhbinder/intersection Copyright (c) 2017 Sukhbinder Singh
    based on matlab script:
    Douglas Schwarz (2020). Fast and Robust Curve Intersections
    (https://www.mathworks.com/matlabcentral/fileexchange/11837-fast-and-robust-curve-intersections),
    MATLAB Central File Exchange. Retrieved June 1, 2020.
    """
    s1, s2, s3, s4 = _rect_inter_inner(x1, x2)
    s5, s6, s7, s8 = _rect_inter_inner(y1, y2)

    c1 = np.less_equal(s1, s2)
    c2 = np.greater_equal(s3, s4)
    c3 = np.less_equal(s5, s6)
    c4 = np.greater_equal(s7, s8)

    ii, jj = np.nonzero(c1 & c2 & c3 & c4)
    return ii, jj


def intersection(x1, y1, x2, y2):
    """
    intersections of curves

    python implementation from: https://github.com/sukhbinder/intersection Copyright (c) 2017 Sukhbinder Singh
    based on matlab script:
    Douglas Schwarz (2020). Fast and Robust Curve Intersections
    (https://www.mathworks.com/matlabcentral/fileexchange/11837-fast-and-robust-curve-intersections),
    MATLAB Central File Exchange. Retrieved June 1, 2020.

    Computes the (x,y) locations where two curves intersect. The curves
    can be broken with NaNs or have vertical segments.

    usage:
    x, y, index1, index2 = intersection(x1, y1, x2, y2)
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    t = np.zeros((4, n))
    aa = np.zeros((4, 4, n))
    aa[0:2, 2, :] = -1
    aa[2:4, 3, :] = -1
    aa[0::2, 0, :] = dxy1[ii, :].T
    aa[1::2, 1, :] = dxy2[jj, :].T

    bb = np.zeros((4, n))
    bb[0, :] = -x1[ii].ravel()
    bb[1, :] = -x2[jj].ravel()
    bb[2, :] = -y1[ii].ravel()
    bb[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            t[:, i] = np.linalg.solve(aa[:, :, i], bb[:, i])
        except np.linalg.LinAlgError:
            t[:, i] = np.Inf

    in_range = (t[0, :] >= 0) & (t[1, :] >= 0) & (
            t[0, :] <= 1) & (t[1, :] <= 1)

    xy0 = t[2:, in_range]
    xy0 = xy0.T

    ind1 = ii[in_range] + t[0, in_range]
    ind2 = jj[in_range] + t[1, in_range]

    return xy0[:, 0], xy0[:, 1], ind1, ind2


def calculate_crossings(x, y, flight_lines, tie_lines):
    """
    returns a 2 columns array giving the float index of every crossing point from both line index and tieline index
    E.g.: "[2307.72615384 3259.15384615]" means there is a crossing point between index 2307 and 2308 and this same
    crossing point between index 3259 and 3260. Points following are from left to right, then from up to down.

    :param x: X coordinate of full flight
    :param y: Y coordinate of full flight
    :param flight_lines: flight-lines as returned by assign_line_type()
    :param tie_lines: tie-lines as returned by assign_line_type()
    :return: indices and X,Y coordinates of crossings
    """

    nb_pot_cros = len(flight_lines) * len(tie_lines)

    crossing_idx = np.zeros((nb_pot_cros, 2), dtype=float)
    xb = np.zeros((nb_pot_cros, 1))
    yb = np.zeros((nb_pot_cros, 1))
    idx_incr = nocross_count = fill_incr = 0

    for fl in flight_lines:  # iterate through flight lines
        x1 = x[fl[0]:fl[1]]
        y1 = y[fl[0]:fl[1]]

        for tl in tie_lines:  # iterate through tie lines
            x2 = x[tl[0]:tl[1]]
            y2 = y[tl[0]:tl[1]]

            x_cros, y_cros, ind1, ind2 = intersection(x1, y1, x2, y2)  # find intersections

            if len(ind1) != 0 and len(ind2) != 0:  # if there are crossings
                """
                # check if crossing is close to flight line. this is to avoid problems with multiple merged files
                fl_start = np.array([x1[0], y1[0]])
                fl_end = np.array([x1[-1], y1[-1]])
                tl_start = np.array([x2[0], y2[0]])
                tl_end = np.array([x2[-1], y2[-1]])
                crossing = np.array([x_cros[0], y_cros[0]])

                if not check_distance_point_to_line(fl_start, fl_end, crossing, threshold=10) \
                        and not check_distance_point_to_line(tl_start, tl_end, crossing, threshold=10):
                    print(crossing)
                    nocross_count += 1
                    continue
                """

                # fill crossing index table
                crossing_idx[idx_incr, 0] = ind1[0] + fl[0]
                crossing_idx[idx_incr, 1] = ind2[0] + tl[0]

                xb[fill_incr] = np.interp(ind1[0], range(len(x1)), x1)
                yb[fill_incr] = np.interp(ind1[0], range(len(y1)), y1)

                fill_incr += 1
                idx_incr += 1
            else:
                nocross_count += 1

    if nocross_count != 0:
        crossing_idx = crossing_idx[:-nocross_count]
        xb = xb[:-nocross_count]
        yb = yb[:-nocross_count]

    return crossing_idx, xb, yb


def crossing_diffs(mag, cross_idx):
    """
    returns crossing differences. At each crossing point the magnetic value from one line is subtracted from the other.
    Since crossings are not exactly at the mag readings, the mag values are interpolated from readings before and
    after crossing.

    :param mag: magnetic field recordings (can also be a different signal, e.g., altitude)
    :param cross_idx: 2d array containing indices of crossings. Computed by calculate crossings.
    """
    cross_idx_int = cross_idx.astype(int)
    cross_idx_dec = cross_idx - cross_idx_int

    cross_mag_before = mag[cross_idx_int]  # mag reading before crossing
    cross_mag_after = mag[cross_idx_int + 1]  # mag reading after crossing

    cross_mag = ((cross_mag_after - cross_mag_before) * cross_idx_dec) + cross_mag_before  # interpolation

    return cross_mag[:, 1] - cross_mag[:, 0]


def check_distance_point_to_line(p1, p2, p3, threshold):
    """
    calculate the distance between point p3 and line p2-p1

    :param p1: numpy array with x,y of point 1
    :param p2: numpy array with x,y of point 2
    :param p3: numpy array with x,y of point 3
    :param threshold: distance that is acceptable
    :return: boolean
    """
    a = p2 - p1
    b = p1 - p3
    distance = (a[0] * b[1] - a[1] * b[0]) / np.linalg.norm(a)
    if distance < threshold:
        return True
    else:
        return False


if __name__ == '__main__':
    # VARIABLES
    # path simplification
    rdp_tolerance = 0.5  # tolerance for RDP alg
    rdp_min_angle = 35  # in degree

    # Line treatment
    laz = 90  # lines azimuth
    taz = 0  # tie lines azimuth
    daz = 30  # tolerance for lines & tie lines azimuth
    minimum_length = 30  # in the same unit than the survey unit

    # Length reduction
    meter_redct = 10  # take a meter value, that will be retired from the lines extremity
    idx_redct_tline = 400
    idx_redct_fline = 500

    # DATA
    filename = 'E:/Scripts/test_data/MD_survey_processed.csv'
    # noinspection PyTypeChecker
    survey_data = np.genfromtxt(filename, dtype=None, names=True, delimiter=',', usemask=False, encoding='ascii')
    sx, sy = survey_data['X_NAD83UTM10N_m'], survey_data['Y_NAD83UTM10N_m']
    st1, st2, std = survey_data['diurnal_1_nT'], survey_data['diurnal_2_nT'], survey_data['diurnal_avg_nT']
    Bt_avg, Bt_1, Bt_2 = survey_data['Btavg_nT'], survey_data['Bt1_nT'], survey_data['Bt2_nT']

    # RDP
    simple_x, simple_y, index_table = get_turns(sx, sy, rdp_tolerance, rdp_min_angle, q=10)
    turns_crds_x, turns_crds_y = get_turn_crds(index_table, sx, sy)

    # LINE TYPE
    all_lines, f_lines, t_lines, il, it = assign_line_type(sx, sy, index_table, line_az=laz, tie_az=taz,
                                                           az_tol=daz, min_l=minimum_length)
    # LINE CHANNEL
    lines_channel = create_line_channel(all_lines, len(sx))

    # LINE REDUCTION
    l_red, fl_red, tl_red = change_line_length(all_lines, f_lines, t_lines, no_fl_idx=idx_redct_fline,
                                               no_tl_idx=idx_redct_tline, min_length=minimum_length)

    # PLOTTING
    fig, ax = plt.subplots()
    ax.plot(sx, sy, 'b', label='original path')
    ax.plot(simple_x, simple_y, 'y--', label='simplified path')
    ax.plot(turns_crds_x, turns_crds_y, 'rx', label='turning points')
    fig.suptitle('Paths & turning points')
    plt.legend()
    plt.show()

    # CROSSINGS
    crossing_index, x_array, y_array = calculate_crossings(sx, sy, fl_red, tl_red)

    # PLOTTING
    fig, ax = plt.subplots(1, 1)
    ax.plot(sx, sy, c='k', label='full path')
    ax.plot(turns_crds_x, turns_crds_y, 'bo', markersize=4, label='turning points')
    ax.plot(x_array, y_array, 'yx', markersize=10)
    ax.set_title('Lines & tielines intersections')
    plt.show()

    BTavg_diff = crossing_diffs(Bt_avg, crossing_index)
    BT1_diff = crossing_diffs(Bt_1, crossing_index)
    BT2_diff = crossing_diffs(Bt_2, crossing_index)

    DIUavg_diff = crossing_diffs(std, crossing_index)
    DIU1avg_diff = crossing_diffs(st1, crossing_index)
    DIU2avg_diff = crossing_diffs(st2, crossing_index)
