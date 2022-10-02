import numpy as np

def absx_sign(x):
    original_shape = list(x.shape)
    x = x.reshape(-1, x.shape[-1])
    abs_x = np.abs(x)
    sign_x = np.sign(x)
    result = np.concatenate((abs_x, sign_x), 1)
    original_shape[-1] *= 2
    return result.reshape(original_shape)

def x2_sign(x):
    original_shape = list(x.shape)
    x = x.reshape(-1, x.shape[-1])
    abs_x = np.power(x,2)
    sign_x = np.sign(x)
    result = np.concatenate((abs_x, sign_x), 1)
    original_shape[-1] *= 2
    return result.reshape(original_shape)

def pm_sqrt(x):
    return np.sign(x)*np.sqrt(np.abs(x))