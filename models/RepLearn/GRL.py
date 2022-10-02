from models.lenet import LeNet5
import torch.nn as nn
import torch.nn.functional as F
from models.base import model
import wandb
from helpers.helpers import *
from models.mlp import multi_layer_perceptron
from helpers.hyperparameters import hyperparameter
from helpers.helpers import clone_and_detach_state_dict

LINECOUNT_PRINTING = 1
HYPERPARAM_TRIALS = 5

class GRL(model):
    def __init__(self, scaling=True, hidden_layers=3, epochs=1500, batch_size=30, lr=0.0001,
                 color='#23A2CF', verbose=False, early_stopping_range=100, latent_size=None, LeNet=False):
        super(GRL, self).__init__()
        self.scaling = scaling
        self.latent_size = latent_size
        self.name = 'GRL'
        self.wandb = False
        self.mlp_layers = hidden_layers
        assert hidden_layers >= 1
        self.color = color
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.early_stopping_range = early_stopping_range
        self.color = color
        self.hyperparams = ['lambda_']
        self.lambda_ = hyperparameter(0.5, (0, 1), tuning=True)
        self.hyperparam_trials = HYPERPARAM_TRIALS
        self.use_LeNet = bool(LeNet)
        self.model_path = None
        self.state_dict_path = None

    def fit(self, X, y, savepath=None, *args, **kwargs):
        X_, y_ = self.scaler_fit_transform(X, y, self.scaling)
        input_features = X_.shape[-1]
        pi_steps = X_.shape[1] - 1
        output_features = y.shape[-1]
        self.torch_model = GRL_torch_model(input_features, self.latent_size, output_features,
                                           self.mlp_layers, pi_steps, self.early_stopping_range, self.lambda_.get(),
                                           self.use_LeNet)
        if self.verbose:
            print('Training ', self.name)

        self.torch_model.train_loop(self.epochs, self.batch_size, X_, y_, torch.optim.Adam, self.lr, self.verbose,
                                    use_wandb=self.wandb)
        # save state dict to file
        if savepath:
            self.state_dict_path = save_state_dict_to_file(self.torch_model, savepath)
            self.model_path = save_model(self, savepath)

    def predict(self, X, *args, **kwargs):
        self.torch_model.eval()
        X_ = self.scaler_transform_X(X, self.scaling)
        if not hasattr(self, 'torch_model'):
            raise Exception('Neural network does not exist before training.')
        else:
            X_ = torch.tensor(X_[:, 0, :], device=get_gpu_or_cpu(), dtype=torch.float32)
            with torch.no_grad():
                predictions = self.torch_model(X_)
        predictions = predictions.detach().cpu().numpy()
        predictions = predictions.reshape(-1, 1, predictions.shape[-1])
        return self.scaler_inverse_transform_Y(predictions, self.scaling)

    def predict_latent(self, X):
        self.torch_model.eval()
        with torch.no_grad():
            X_ = self.scaler_transform_X(X, self.scaling)
            if not hasattr(self, 'torch_model'):
                raise Exception('Neural network does not exist before training.')
            else:
                X_ = torch.tensor(X_, device=get_gpu_or_cpu(), dtype=torch.float32)
                with torch.no_grad():
                    predictions = self.torch_model.encoder(X_)
                return predictions.detach().cpu().numpy()

    def get_info(self):
        info = super(GRL, self).get_info()
        return {'Model_Path': self.model_path, 'State_Dict': self.state_dict_path, **info}


class GRL_torch_model(nn.Module):
    def __init__(self, input_features, latent_size, output_features, mlp_layers, pi_steps,
                 early_stopping_range=100, lambda_=0.5, use_LeNet=False):
        super(GRL_torch_model, self).__init__()

        if latent_size is None:
            self.width_latent_state = input_features
        else:
            self.width_latent_state = latent_size

        self.name = 'GRL'
        self.mlp_layers = mlp_layers
        self.pi_steps = pi_steps
        self.early_stopping_range = early_stopping_range
        self.width = self.width_latent_state
        self.input_features = input_features
        self.output_features = output_features
        self.device = get_gpu_or_cpu()
        self.mse = nn.MSELoss()
        self.lambda_ = float(lambda_)

        linear_layers = []
        for p in range(pi_steps + 1):
            layer = nn.Linear(self.width_latent_state, output_features, bias=False, device=self.device)
            linear_layers.append(layer)

        self.thetas = nn.ModuleList(linear_layers)

        neurons_mlp = [input_features, self.width_latent_state] + [self.width_latent_state] * self.mlp_layers

        if use_LeNet:
            self.encoder = LeNet5(latent_size)
        else:
            self.encoder = multi_layer_perceptron(neurons_mlp, device=self.device, activation_last_step=False,
                                                  activation_func=F.leaky_relu)

    def forward(self, X):
        if len(X.shape) > 2:
            x = X[:, 0]
        else:
            x = X
        z0 = self.encoder(x)
        y_hat = self.thetas[0](z0)
        return y_hat

    def train_loop(self, epochs, batch_size, X, y, optim, lr, verbose, validation_ratio=0.2, use_wandb=False):
        if use_wandb:
            config = {
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': lr,
                'validation_ratio': validation_ratio,
                'model_name': self.name,
                'pi_steps': self.pi_steps,
                'lambda': self.lambda_
            }
            self.wandb_run = wandb.init(config=config)
            wandb.watch(self)
            self.wandb_run.name = self.name + '_' + self.wandb_run.name

        # Put Model in Training Mode
        self.train()

        # Get Device
        device = self.device
        self.to(device)

        # Turn training data into tensors
        X_all = torch.tensor(X, device=device, dtype=torch.float32)
        y_all = torch.tensor(y[:, 0, :], device=device, dtype=torch.float32)
        assert X_all.shape[0] == y_all.shape[0]

        # Shuffle the data according to some random permutation
        # Split into validation and training sets
        permutation = torch.randperm(X_all.shape[0])
        val_split = int(X_all.shape[0] * validation_ratio)
        X_val = X_all[permutation[:val_split]]
        X_train = X_all[permutation[val_split:]]
        y_val = y_all[permutation[:val_split]]
        y_train = y_all[permutation[val_split:]]

        # pass params to optimizer
        optimizer = optim(self.parameters(), lr=lr)

        # Early Stopping
        if self.early_stopping_range is not None:
            earlyStopping = True
        else:
            earlyStopping = False
        best_val_loss = np.inf
        best_state_dict = None
        waiting_time = 0

        # Training Loop
        for e in range(1, 1 + epochs):

            # Get a random permutation for the training batches
            permutation = torch.randperm(X_train.shape[0])

            # Set Loss to zero
            acc_loss = 0.0

            # Iterate over Batches
            for start_idx in range(0, X_train.shape[0], batch_size):
                # Index for next batch
                indices = permutation[start_idx:start_idx + batch_size]

                # Get next batch
                x = X_train[indices]
                y = y_train[indices]

                # Zero Gradients
                optimizer.zero_grad()

                # compute some kind of forward pass and its loss
                loss = self.compute_loss(x, y)

                # backward pass
                loss.backward()

                # SGD step
                optimizer.step()

                # Add to loss sum
                acc_loss += float(loss)

            # Validation
            val_loss = self.perform_validation(X_val, y_val)

            # Printing or WandB
            if (verbose and e % LINECOUNT_PRINTING == 0) or use_wandb:
                r2_score = self.get_true_r2_score(X_val, y_val)
                if verbose:
                    print(f'Epoch {e}, Train Loss: ', acc_loss / X_train.shape[0], 'Val_Loss: ',
                          float(val_loss) / X_val.shape[0], 'R2_Val: ', r2_score)

                if use_wandb:
                    wandb.log({'train/loss': loss, 'val/loss': val_loss, 'epoch': e, 'val/r2': r2_score})

            # Early Stopping
            if earlyStopping:
                if val_loss < best_val_loss:
                    best_val_loss = float(val_loss)
                    best_state_dict = clone_and_detach_state_dict(self)
                    waiting_time = 0
                else:
                    waiting_time += 1
                if waiting_time > self.early_stopping_range:
                    break

            # Load best state dict in case of early stopping
        if earlyStopping:
            self.load_state_dict(best_state_dict)

        if use_wandb:
            self.wandb_run.finish()

    def compute_loss(self, x, y):
        z_hat = self.encoder(x)

        first_loss = self.mse(self.thetas[0](z_hat[:, 0]), y)
        pi_loss = 0.0
        howmany = len(self.thetas) - 1
        for i, layer in enumerate(self.thetas[1:]):
            t = i + 1
            y_hat = layer(z_hat[:, t])
            pi_loss = pi_loss + 1 / howmany * self.mse(y_hat, y)
        loss = self.lambda_ * pi_loss + (1 - self.lambda_) * first_loss
        return loss

    def perform_validation(self, X_val, y_val):
        self.eval()
        with torch.no_grad():
            loss = self.compute_loss(X_val, y_val)
        self.train()
        return loss

    def get_true_r2_score(self, x, y):
        self.eval()
        with torch.no_grad():
            y_hat = self(x[:, 0, :])
        self.train()
        return float(r2_loss(y_hat, y))
