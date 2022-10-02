import numpy as np

def unnormed_cosine(x,y):
    return x[0]*y[0]*np.cos(x[1]-y[1])

def reverse_polar_coords(x,y):
    x0 = x[0]*np.sin(x[1])
    x1 = x[0]*np.cos(x[1])
    y0 = y[0] * np.sin(y[1])
    y1 = y[0] * np.cos(y[1])

    x0_ = 0.5 *x0 + x1
    x1_ = 1*x0 - x1

    y0_ = 0.5 *y0 + y1
    y1_ = 1 * y0 - y1

    return x0_ * y0_ + x1_ * y1_