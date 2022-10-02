from data.linear_gaussian import linear_gaussian_system
import numpy as np

MAX_SINGULAR_VAL = 10


class latent_linear_gaussian(linear_gaussian_system):
    def __init__(self, latent_features, visible_features, y_features=1, test_size=1000, z_to_x='linear', seed=0):
        super(latent_linear_gaussian, self).__init__(latent_features, y_features, test_size, seed)
        if z_to_x == 'linear':
            self.linear = True
            self.name = f'Latent Linear-Gaussian L{latent_features}  V{visible_features} Y{y_features} linear'

        else:
            self.linear = False
            self.name = f'Latent Linear-Gaussian L{latent_features} Y{y_features} ' + z_to_x.__name__

        self.transform_z_to_x = z_to_x
        self.latent_features = latent_features
        self.visible_features = visible_features
        self.y_features = y_features

    def get_train_test_data(self, seq_length, seq_step, test_ratio, sample_size, seed, copy_instead_of_split=False,
                            trans_noise=1):
        (z_tr, y_tr), (z_te, y_te) = super(latent_linear_gaussian, self).get_train_test_data(seq_length, seq_step,
                                                                                             test_ratio, sample_size,
                                                                                             seed,
                                                                                             copy_instead_of_split,
                                                                                             noise_std=trans_noise)
        if self.linear:
            # linear map from hidden to visible state
            np.random.seed(self.system_seed)
            self.phi = np.random.randn(self.visible_features, self.latent_features)
            np.fill_diagonal(self.phi, 1)

            u, s, vt = np.linalg.svd(self.phi)
            max_s = np.max(s)
            if max_s > MAX_SINGULAR_VAL:
                s = s / max_s * MAX_SINGULAR_VAL
            s_mat = np.zeros(self.phi.shape)
            np.fill_diagonal(s_mat, s)
            self.phi = u @ s_mat @ vt

            # apply map
            x_tr = z_tr @ self.phi.T
            x_te = z_te @ self.phi.T

        else:
            x_tr = self.transform_z_to_x(z_tr)
            x_te = self.transform_z_to_x(z_te)

            # correct number of visible features if needed
            self.visible_features = x_te.shape[-1]

        self.latent_variables = (z_tr, z_te)

        return (x_tr, y_tr), (x_te, y_te)

    def get_info(self):
        return {'Latent Features': self.latent_features, 'Visible Features': self.visible_features,
                **super(latent_linear_gaussian, self).get_info()}

    def get_latent_vars(self):
        return self.latent_variables

if __name__ == '__main__':
    def trans_x2_sign(x):
        original_shape = list(x.shape)
        x = x.reshape(-1, x.shape[-1])
        abs_x = np.abs(x)
        sign_x = np.sign(x)
        result = np.concatenate((abs_x, sign_x), 1)
        original_shape[-1] *= 2
        return result.reshape(original_shape)


    dataset = latent_linear_gaussian(5, 10, 1,1000, trans_x2_sign)
    (train_x, train_y), (test_x, test_y) = dataset.get_train_test_data(5, 1, 0.2, 300, 2)
    print(train_x.shape)
    print(dataset.coefficients)
    dataset = latent_linear_gaussian(5, 10, 1, 1000, trans_x2_sign)
    (train_x, train_y), (test_x, test_y) = dataset.get_train_test_data(5, 1, 0.2, 300, 2)
    print(dataset.coefficients)