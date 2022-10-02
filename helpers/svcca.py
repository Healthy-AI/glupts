from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
import numpy as np
import warnings

epsilon = 1e-7
# https://arxiv.org/pdf/1706.05806.pdf
# SVCCA
def SVCCA(X, Y, use_PCA = True):
    true_features = Y.shape[-1]
    estimated_features = X.shape[-1]
    if true_features > estimated_features:
        warnings.warn('Expected more dimensions in predictions!')
        zeros = np.zeros((X.shape[0], true_features))
        zeros[:, :estimated_features] = X
        X = zeros

    if use_PCA:
        estimated_features = X.shape[-1]

        if estimated_features > true_features:
            k = true_features
            reduced = False
            while not reduced:
                pca = PCA(k)
                Xr = pca.fit_transform(X)
                if np.sum(pca.explained_variance_ratio_)>0.99 or k == estimated_features:
                    reduced = True
                    X = Xr
                else:
                    k += 1

    cca = CCA(true_features)
    x, y = cca.fit_transform(X, Y)
    assert x.shape == y.shape, 'Unexpected shape mismatch'
    corrs = [np.corrcoef(x[:, i], y[:, i])[0, 1] for i in range(x.shape[1])]
    x_std = np.std(x, 0)
    y_std = np.std(y,0)
    if np.all(x_std>epsilon) and np.all(y_std>epsilon):
        x /= x_std
        y /= y_std
    else:
        warnings.warn('Could not rescale features after SVCCA.')
    return np.mean(corrs), (x, y)
