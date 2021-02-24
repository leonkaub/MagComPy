"""
Functions for magnetic compensation of a vector magnetometer. Translated to Python from original Matlab script:
Olsen, N., Risbo, T., Brauer, P., Merayo, J., Primdahl, F., and Sabaka, T.: In-flight compensation methods used for the
Ørsted mission, Technical University of Denmark, unpublished, (2001).
(https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.41.3567&rep=rep1&type=pdf)

This file is part of MagComPy.
Copyright (C) 2021  Leon Kaub <lkaub@geophysik.uni-muenchen.de>
"""

import numpy as np


def f_scalar(m, f, s1, p):
    p1, p1_x, p1_y, p1_z = p[0], p[1], p[2], p[3]

    f_tmp = np.zeros((len(f), 3))
    f_tmp[:, 0] = f[:, 0] - m[0]
    f_tmp[:, 1] = f[:, 1] - m[1]
    f_tmp[:, 2] = f[:, 2] - m[2]

    r = s1 @ p1.conj().T @ p1 @ s1

    b = np.sqrt(f_tmp[:, 0] * (r[0, 0] * f_tmp[:, 0] + r[0, 1] * f_tmp[:, 1] + r[0, 2] * f_tmp[:, 2]) +
                f_tmp[:, 1] * (r[1, 0] * f_tmp[:, 0] + r[1, 1] * f_tmp[:, 1] + r[1, 2] * f_tmp[:, 2]) +
                f_tmp[:, 2] * (r[2, 0] * f_tmp[:, 0] + r[2, 1] * f_tmp[:, 1] + r[2, 2] * f_tmp[:, 2]))

    return b, f_tmp


def non_ortho(u1, u2, u3):
    s1, c1 = np.sin(u1), np.cos(u1)
    s2, c2 = np.sin(u2), np.cos(u2)
    s3, c3 = np.sin(u3), np.cos(u3)

    tmp = np.sqrt(1 - s2 ** 2 - s3 ** 2)

    p1 = np.array([[1, 0, 0],
                   [s1 / c1, 1 / c1, 0],
                   [-(s1 * s3 + c1 * s2) / (c1 * tmp), -s3 / c1 / tmp, 1 / tmp]])

    # derivative of P vrt. u1
    p1_1 = np.array([[0, 0, 0],
                     [1 + (s1 / c1) ** 2, s1 / c1 ** 2, 0],
                     [-s3 / (c1 ** 2 * tmp), -s3 * s1 / (c1 ** 2 * tmp), 0]])

    # derivative of P vrt. u2
    p1_2 = np.array([[0, 0, 0],
                     [0, 0, 0],
                     [-c2 * (c1 * c3 ** 2 + s2 * s1 * s3) / (c1 * tmp ** 3), -s3 * s2 * c2 / (c1 * tmp ** 3),
                      s2 * c2 / tmp ** 3]])

    # derivative of P vrt. u3
    p1_3 = np.array([[0, 0, 0],
                     [0, 0, 0],
                     [-c3 * (s1 * c2 ** 2 + s3 * c1 * s2) / (c1 * tmp ** 3), -c3 * c2 ** 2 / (c1 * tmp),
                      s3 * c3 / tmp ** 3]])

    return [p1, p1_1, p1_2, p1_3]


def gradf_scalar(m, s1, p, f_tmp, b_scalar):
    p1, p1_x, p1_y, p1_z = p[0], p[1], p[2], p[3]

    r = s1 @ p1.conj().T @ p1 @ s1

    d_offsets_1 = -2 * (r[0, 0] * f_tmp[:, 0] + r[1, 0] * f_tmp[:, 1] + r[2, 0] * f_tmp[:, 2])
    d_offsets_2 = -2 * (r[0, 1] * f_tmp[:, 0] + r[1, 1] * f_tmp[:, 1] + r[2, 1] * f_tmp[:, 2])
    d_offsets_3 = -2 * (r[0, 2] * f_tmp[:, 0] + r[1, 2] * f_tmp[:, 1] + r[2, 2] * f_tmp[:, 2])

    d_scale_1 = -2 / m[3] * f_tmp[:, 0] * (r[0, 0] * f_tmp[:, 0] + r[1, 0] * f_tmp[:, 1] + r[2, 0] * f_tmp[:, 2])
    d_scale_2 = -2 / m[4] * f_tmp[:, 1] * (r[0, 1] * f_tmp[:, 0] + r[1, 1] * f_tmp[:, 1] + r[2, 1] * f_tmp[:, 2])
    d_scale_3 = -2 / m[5] * f_tmp[:, 2] * (r[0, 2] * f_tmp[:, 0] + r[1, 2] * f_tmp[:, 1] + r[2, 2] * f_tmp[:, 2])

    r = s1 @ (p1_x.conj().T @ p1 + p1_x @ p1.conj().T) @ s1

    d_non_or_1 = (f_tmp[:, 0] * (r[0, 0] * f_tmp[:, 0] + r[0, 1] * f_tmp[:, 1] + r[0, 2] * f_tmp[:, 2]) +
                  f_tmp[:, 1] * (r[1, 0] * f_tmp[:, 0] + r[1, 1] * f_tmp[:, 1] + r[1, 2] * f_tmp[:, 2]) +
                  f_tmp[:, 2] * (r[2, 0] * f_tmp[:, 0] + r[2, 1] * f_tmp[:, 1] + r[2, 2] * f_tmp[:, 2]))

    r = s1 @ (p1_y.conj().T @ p1 + p1_y @ p1.conj().T) @ s1
    d_non_or_2 = (f_tmp[:, 0] * (r[0, 0] * f_tmp[:, 0] + r[0, 1] * f_tmp[:, 1] + r[0, 2] * f_tmp[:, 2]) +
                  f_tmp[:, 1] * (r[1, 0] * f_tmp[:, 0] + r[1, 1] * f_tmp[:, 1] + r[1, 2] * f_tmp[:, 2]) +
                  f_tmp[:, 2] * (r[2, 0] * f_tmp[:, 0] + r[2, 1] * f_tmp[:, 1] + r[2, 2] * f_tmp[:, 2]))

    r = s1 @ (p1_z.conj().T @ p1 + p1_z @ p1.conj().T) @ s1
    d_non_or_3 = (f_tmp[:, 0] * (r[0, 0] * f_tmp[:, 0] + r[0, 1] * f_tmp[:, 1] + r[0, 2] * f_tmp[:, 2]) +
                  f_tmp[:, 1] * (r[1, 0] * f_tmp[:, 0] + r[1, 1] * f_tmp[:, 1] + r[1, 2] * f_tmp[:, 2]) +
                  f_tmp[:, 2] * (r[2, 0] * f_tmp[:, 0] + r[2, 1] * f_tmp[:, 1] + r[2, 2] * f_tmp[:, 2]))

    dg_dm = np.array([d_offsets_1, d_offsets_2, d_offsets_3,
                      d_scale_1, d_scale_2, d_scale_3,
                      d_non_or_1, d_non_or_2, d_non_or_3]).T

    for p in range(9):
        dg_dm[:, p] = dg_dm[:, p] / b_scalar / 2

    return dg_dm


def huber_w(residuals):
    # Robust weights after Huber
    c = 1.5

    w = c / abs(residuals)

    w[np.where(w >= 1)] = 1

    return w


def compute_parameters(f, b):
    max_iter = 5
    step = 1
    rad = np.pi / 180
    g = None
    report = ''

    m_i = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])
    m_prior = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])
    m_weights = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    ic_prior = np.diag(m_weights)
    w = np.ones(len(b))

    for i in range(max_iter):
        s_1 = np.diag(1 / m_i[3:6])
        p = non_ortho(m_i[6], m_i[7], m_i[8])
        b_csc, f_tmp = f_scalar(m_i, f, s_1, p)

        delta_d = b - b_csc  # residual |B_scalar| - |B_CSC|

        rms = np.sqrt(((w.conj().T * delta_d.conj().T).dot((w * delta_d)) / w.sum(axis=0)))  # achieved rms misfit
        # print(f'Iteration: {i + 1}, rms (nT): {rms:.4f}')
        report += f'Iteration: {i + 1}, rms (nT): {rms:.4f}\n'

        w = huber_w(delta_d / rms)  # robust weights
        g = gradf_scalar(m_i, s_1, p, f_tmp, b_csc)  # kernel matrix

        wg = np.zeros(g.shape)
        for n in range(g.shape[1]):  # Multiply matrix G with weights w
            wg[:, n] = w * g[:, n]
        # correction to model vector
        delta_m = np.linalg.lstsq((g.conj().T.dot(wg) + ic_prior),
                                  (wg.conj().T.dot(delta_d) + ic_prior.dot(m_prior - m_i)),
                                  rcond=None)[0]

        m_i = m_i + step * delta_m  # apply correction

    cov_m = np.linalg.inv(g.conj().T.dot(g) + ic_prior)  # model covariance matrix
    d_m = np.sqrt(np.diag(cov_m))  # standard deviation of model parameters
    """
    # print model parameters and their standard deviation
    print('\n')
    with np.printoptions(precision=4):
        print(f'Offsets:\t\t{m_i[:3]} | {d_m[:3]}')
        print(f'Scale values:\t{m_i[3:6]} | {d_m[3:6]}')
        print(f'Non-orthogon:\t{m_i[6:] / rad * 3600} | {d_m[6:] / rad * 3600}')
    print('\n')
    """
    report += f'\nOffsets:\t\t{m_i[:3]} | {d_m[:3]}\n'
    report += f'Scale values:\t{m_i[3:6]} | {d_m[3:6]}\n'
    report += f'Non-orthogon:\t{m_i[6:] / rad * 3600} | {d_m[6:] / rad * 3600}\n\n'

    for n in range(9):  # Calculate and print model correlation matrix
        for k in range(9):
            cov_m[n, k] = cov_m[n, k] / (d_m[n] * d_m[k])
    """
    print('Model covariance matrix:')
    with np.printoptions(precision=3, suppress=True):
        print(cov_m)
    print('\n')

    print(f'Data used [%%]: {w.sum(axis=0) / len(w) * 100:.2f}\n')

    print(f'Resulting model parameters:\n{m_i}')

    report += 'Model covariance matrix:\n'
    report += f'{cov_m}\n\n'
    """
    report += f'Data used [%%]: {w.sum(axis=0) / len(w) * 100:.2f}\n\n'
    report += f'Resulting model parameters:\n{m_i}'

    return m_i, report


def apply_cof(flux, m):
    """
    flux: Fluxgate data in 2d matrix format with x,y,z as columns. F = (Fx, Fy, Fz)
    m: coefficient vector from MATLAB script
    """
    # offsets
    offsets = np.array([m[0], m[1], m[2]])

    # sensitivities
    # s = np.array([1 / m[3], 1 / m[4], 1 / m[5]])
    # S_i = np.diag(s)
    s_i = np.diag(1 / m[3:6])

    # non-orthogonalities
    u1 = m[6]  # * np.pi/(180 * 3600)
    u2 = m[7]  # * np.pi/(180 * 3600)
    u3 = m[8]  # * np.pi/(180 * 3600)
    s1, c1 = np.sin(u1), np.cos(u1)
    s2 = np.sin(u2)
    s3 = np.sin(u3)
    tmp = np.sqrt(1 - s2 * s2 - s3 * s3)
    p_i = np.array([[1, 0, 0],
                    [s1 / c1, 1 / c1, 0],
                    [-(s1 * s3 + c1 * s2) / (c1 * tmp), -s3 / (c1 * tmp), 1 / tmp]])

    # calculate
    ft = []
    for i in range(len(flux)):
        b = p_i @ s_i @ (flux[i] - offsets)
        ft.append(np.linalg.norm(b))
    return np.array(ft)
