import numpy as np
from data import clock_LGS
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
import matplotlib as mpl


def SVCCA(X, Y, use_pca=True):
    true_features = Y.shape[-1]
    estimated_features = X.shape[-1]
    if use_pca:
        if estimated_features < true_features:
            zeros = np.zeros((X.shape[0], true_features))
            zeros[:, :estimated_features] = X
            X = zeros

        estimated_features = X.shape[-1]

        if estimated_features > true_features:
            k = true_features
            reduced = False
            while not reduced:
                pca = PCA(k)
                Xr = pca.fit_transform(X)
                if np.sum(pca.explained_variance_ratio_) > 0.99 or k == estimated_features:
                    reduced = True
                    X = Xr
                else:
                    k += 1
        elif estimated_features == true_features:
            pass
        else:
            raise Exception('Expected more dimensions in predictions!')

    cca = CCA(true_features)
    x, y = cca.fit_transform(X, Y)
    assert x.shape == y.shape, 'Unexpected shape mismatch'
    corrs = [np.corrcoef(x[:, i], y[:, i])[0, 1] for i in range(x.shape[1])]

    x /= np.std(x, 0)
    y /= np.std(y, 0)
    lr = LinearRegression()
    lr.fit(y, Y)
    y = lr.predict(y)
    x = lr.predict(x)

    return np.mean(corrs), (x, y)


# PARAMETERS FOR PLOTTING
USE_PCA = True  # SWITCHES BETWEEN CCA AND SVCCA
name = 'latent_recovery_colors_2500'
width = 6
height = 6
margin = 2
alpha = .3
howmany = 150
dot_size = 10

#######################################################################################################################
# SPECIFY THE PRETRAINED MODEL FILES BELOW AND MAKE SURE TO TEST THEM ON THE SAME SYSTEM SEED AS DURING TRAINING
#######################################################################################################################

# 2500 Samples Classic Rep
model_path_1 = 'models/2022_05_13_00_35_32.p'
state_dict_path_1 = 'state_dicts/2022_05_13_00_35_32.p'
title1 = "Classic Rep."

# 2500 Samples CRL
state_dict_path_2 = "state_dicts/2022_05_13_01_22_00.p"
model_path_2 = 'models/2022_05_13_01_22_00.p'
title2 = 'CRL'

#######################################################################################################################
#######################################################################################################################

pos_z1 = np.linspace(-width / 2, width / 2, howmany)
pos_z2 = np.linspace(-height / 2, height / 2, howmany)
all_pos_z1, all_pos_z2 = np.meshgrid(pos_z1, pos_z2)
color_linspace = np.linspace(0, 1, howmany)
color_meshgrid_two = np.meshgrid(color_linspace, color_linspace)
color3 = (color_meshgrid_two[0], color_meshgrid_two[1], np.ones(color_meshgrid_two[0].shape) * 0.5)
colors = np.stack(color3, -1)

original_image = (all_pos_z1, all_pos_z2, colors.reshape(-1, 3))

system = clock_LGS(1, 1000, 1, seed=0)
true_z = np.stack((all_pos_z1.reshape(-1, 1), all_pos_z2.reshape(-1, 1)), -1)

observations = system.transform_z_to_x(true_z).reshape(-1, 1, 28 ** 2)
model = torch.load(model_path_1, map_location=torch.device('cpu'))
model.torch_model.load_state_dict(torch.load(state_dict_path_1, map_location=torch.device('cpu')))
z_hat = model.predict_latent(np.tile(observations, (1, 5, 1)))[:, 0]
coefs, (z_hat, z_good) = SVCCA(z_hat, true_z.reshape(-1, true_z.shape[-1]), USE_PCA)

fontsize = 21

mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.set_title('Data Generating Process', fontsize=fontsize)
ax1.scatter(z_good[:, 0], z_good[:, 1], marker='o', c=original_image[2], s=dot_size)
ax1.set_xlabel('Latent Component 1', fontsize=fontsize)
ax1.set_ylabel('Latent Component 2', fontsize=fontsize)
ax1.set(facecolor='#000000')
ax1.set_xlim(-width / 2 - margin, width / 2 + margin)
ax1.set_ylim(-height / 2 - margin, height / 2 + margin)

ax2.set_title(title1, fontsize=fontsize)
ax2.scatter(z_hat[:, 0], z_hat[:, 1], marker='o', c=original_image[2], s=dot_size, alpha=alpha)
ax2.set_xlabel('Latent Component 1', fontsize=fontsize)
ax2.set_ylabel('Latent Component 2', fontsize=fontsize)
ax2.set(facecolor='#000000')
ax2.set_xlim(-width / 2 - margin, width / 2 + margin)
ax2.set_ylim(-height / 2 - margin, height / 2 + margin)

model = torch.load(model_path_2, map_location=torch.device('cpu'))
model.torch_model.load_state_dict(torch.load(state_dict_path_2, map_location=torch.device('cpu')))
z_hat = model.predict_latent(np.tile(observations, (1, 5, 1)))[:, 0]
coefs, (z_hat, z_good) = SVCCA(z_hat, true_z.reshape(-1, true_z.shape[-1]), USE_PCA)

ax3.set_title(title2, fontsize=fontsize)
ax3.scatter(z_hat[:, 0], z_hat[:, 1], marker='o', c=original_image[2], s=dot_size, alpha=alpha)
ax3.set_xlabel('Latent Component 1', fontsize=fontsize)
ax3.set_ylabel('Latent Component 2', fontsize=fontsize)
ax3.set(facecolor='#000000')
ax3.set_xlim(-width / 2 - margin, width / 2 + margin)
ax3.set_ylim(-height / 2 - margin, height / 2 + margin)

plt.tight_layout()
plt.savefig(name + '.png', format='png', dpi=300, facecolor='#ffffff')
plt.show()
