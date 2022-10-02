import numpy as np
import torch
import os
import datetime


def linear_activation(tensor):
    return tensor


def kernel_mat(kernel_fun, x, y=None):
    n = x.shape[0]
    if y is None:
        K = np.empty((n, n))

        for i in range(n):
            for j in range(n):
                K[i, j] = kernel_fun(x[i], x[j])

    else:
        K = np.empty((n, y.shape[0]))
        for i in range(n):
            for j in range(y.shape[0]):
                K[i, j] = kernel_fun(x[i], y[j])
    return K


def get_current_device(module):
    return next(module.parameters()).device


def get_gpu_or_cpu():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def clone_and_detach_state_dict(model):
    state_dict = model.state_dict()
    detached_state_dict = {}
    for key in state_dict:
        detached_state_dict[key] = state_dict[key].clone().detach()
    return detached_state_dict


def save_state_dict_to_file(model, path):
    path = os.path.join(path, 'state_dicts')
    states = model.state_dict()
    time = datetime.datetime.now(datetime.timezone.utc)
    filename = time.strftime('%Y_%m_%d_%H_%M_%S')
    os.makedirs(path, exist_ok=True)
    return save_file(states, path, filename)


def save_model(model, path):
    path = os.path.join(path, 'models')
    time = datetime.datetime.now(datetime.timezone.utc)
    filename = time.strftime('%Y_%m_%d_%H_%M_%S')
    os.makedirs(path, exist_ok=True)
    return save_file(model, path, filename)


def save_file(obj, folder, name_without_p):
    counter = 0
    filepath = os.path.join(folder, name_without_p + '.p')
    while True:
        if os.path.exists(filepath):
            filepath = os.path.join(folder, name_without_p + '_' + str(counter) + '.p')
            counter += 1
        else:
            torch.save(obj, filepath)
            break
    return filepath
