from data.dataset import dataset
import numpy as np

class linear_gaussian_system(dataset):
    def __init__(self, no_features, output_features=1, test_size=1000, seed=0):
        self.no_features = no_features
        self.output_features = output_features
        self.name = f'Linear Gaussian System V{no_features} Y{output_features}'
        self.test_size = test_size
        self.target_cols = ['']
        self.system_seed = seed



    def prepare_for_experiment(self, seq_lengths, seq_steps, seq_gap, *args):
        pass

    def get_train_test_data(self, seq_length, seq_step, test_ratio, sample_size, seed, copy_instead_of_split=False,
                            noise_std=1, noise_at_zero = 5, a_mean=0.0, a_std = 0.2, beta_std=0.2, spectral_rad = 1.3):

        self.noise_std = noise_std
        self.noise_at_zero = noise_at_zero
        self.a_mean = a_mean
        self.a_std = a_std
        self.spectral_rad = spectral_rad
        self.beta_std = beta_std

        T = seq_length
        sample_size = int(sample_size)
        total_N = int(sample_size + self.test_size)
        no_features = int(self.no_features)

        self.generate_transition_matrices(no_features, T)

        # CREATING STARTING VALUES t = 0
        X = np.empty((total_N, T - 1, no_features))
        np.random.seed(seed)
        X[:, 0, :] = np.random.normal(0, noise_at_zero, (total_N, no_features))

        for t in range(1, T - 1):
            X[:, t, :] = np.einsum('oi,ni -> no', self.transitions[t - 1], X[:, t - 1, :])
            noise = np.random.normal(0, noise_std, (X.shape[0], X.shape[-1]))
            X[:, t, :] += noise

        Y = np.einsum('oi,ni -> no', self.transitions[-1], X[:, -1, :])
        noise = np.random.normal(0, noise_std, (Y.shape[0], Y.shape[-1]))
        Y += noise
        Y = Y.reshape(-1, 1, Y.shape[-1])

        A0 = self.transitions[0]
        for A in self.transitions[1:]:
            A0 = A @ A0
        self.coefficients = A0

        self.latent_variables = (X[:sample_size], X[sample_size:])

        return (X[:sample_size], Y[:sample_size]), (X[sample_size:], Y[sample_size:])

    def generate_transition_matrices(self, no_features, T):
        if self.system_seed is not None:
            np.random.seed(self.system_seed)
        self.transitions = []
        for t in range(T - 2):
            new_A = np.random.normal(self.a_mean, self.a_std, (no_features, no_features))
            np.fill_diagonal(new_A, 1)
            lambda_, U = np.linalg.eig(new_A)
            spectral_rad = np.max(np.abs(lambda_))
            lambda_ = lambda_ / spectral_rad * self.spectral_rad
            A = np.real(U @ np.diag(lambda_) @ np.linalg.inv(U))
            self.transitions.append(A)

        beta = np.random.normal(self.a_mean, self.beta_std, (self.output_features, no_features))
        self.transitions.append(beta)

    def get_info(self):
        return {'Features in PI Steps': self.no_features, 'System_Seed': self.system_seed}


if __name__ == '__main__':
    dataset = linear_gaussian_system(3)
    train, test = dataset.get_train_test_data(4, 1, 0.2, 10000, 5000)

