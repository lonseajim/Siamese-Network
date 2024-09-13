"""
preprocessing
"""

import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve


# spike removal
def despiking(y, filter_size=5, dynamic_factor=4.5):
    len_y = len(y)
    crr_dot = []
    for i in range(filter_size, len_y - filter_size - 1):
        # maxY = y[i - filterSize:i + filterSize + 1].max()
        min_y = y[i - filter_size:i + filter_size + 1].min()
        median_y = np.median(y[i - filter_size:i + filter_size + 1])
        if (y[i] - min_y) > (median_y - min_y) * dynamic_factor:
            crr_dot.append(i)
    n2 = len(crr_dot)
    if n2 == 0:
        return y
    else:
        for j in range(n2):
            count = 0
            while ((j + count + 1) < n2) and ((crr_dot[j + count + 1] - crr_dot[j + count]) == 1):
                count = count + 1
            x1 = crr_dot[j] - 1
            x2 = crr_dot[j + count] + 1
            y1 = y[x1]
            y2 = y[x2]
            y[crr_dot[j]] = (y2 - y1) * (crr_dot[j] - x1) / (x2 - x1) + y1
        return y


def whittaker_smooth(x, w, lambda_):
    matrix_x = np.matrix(x)
    m = matrix_x.size
    # i = np.arange(0, m)
    E = eye(m, format='csc')
    D = E[1:] - E[:-1]
    # numpy.diff() does not work with sparse matrix. This is a workaround.
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * D.T * D))
    B = csc_matrix(W * matrix_x.T)
    background = spsolve(A, B)
    return np.array(background)


def airPLS(x, lambda_=100, itermax=15):
    x = np.array(x)
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = whittaker_smooth(x, w, lambda_)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if dssn < 0.001 * (abs(x)).sum() or i == itermax:
            if i == itermax:
                print('WARING max iteration reached!')
            break
        w[d >= 0] = 0
        # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]
    z = d
    return z.tolist()


def max_min_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
