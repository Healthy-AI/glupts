from helpers.hyperparameters import hyperparameter
import helpers.helpers
import torch.nn as nn
import torch.nn.functional as F
from models.base import model
import wandb
from helpers.helpers import *
from models.mlp import multi_layer_perceptron
from models.lenet import LeNet5
from models.RepLearn.baseline_net import baseline_net_torch_model as baseline_model

LINECOUNT_PRINTING = 1
HYPERPARAM_TRIALS = 5


class generalized_distillation_net(model):
    def __init__(self, latent_size, scaling=True, hidden_layers=3, hidden_layer_teacher=5, width_teacher=100, epochs=1500,
                 batch_size=30, lr=0.0001, color='#23CE8D', verbose=False, early_stopping_range=100,
                 LeNet=False):
        super(generalized_distillation_net, self).__init__()
        self.scaling = scaling
        self.latent_size = latent_size
        self.name = 'Generalized Distillation'
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
        self.model_path = None
        self.state_dict_path = None
        self.hidden_layers_teacher = hidden_layer_teacher
        self.width_teacher = width_teacher
        self.hyperparams = ['lambda_']
        self.lambda_ = hyperparameter(0.5, (0, 1), tuning=True)
        self.hyperparam_trials = HYPERPARAM_TRIALS
        self.useLeNet = LeNet

    def fit(self, X, y, savepath=None, *args, **kwargs):
        X_, y_ = self.scaler_fit_transform(X, y, self.scaling)
        input_features = X_.shape[-1]
        time_steps = X.shape[1]
        output_features = y.shape[-1]

        # SETUP TEACHER AND STUDENT
        if self.useLeNet:
            self.teacher_model = teacher_LeNet(time_steps, output_features, self.width_teacher,
                                               self.early_stopping_range)
        else:
            self.teacher_model = teacher_MLP(input_features * time_steps, self.width_teacher, output_features,
                                             self.hidden_layers_teacher, self.early_stopping_range)
        self.student_model = baseline_model(input_features, self.latent_size, output_features, self.mlp_layers,
                                            self.early_stopping_range, self.useLeNet)

        # TRAIN THE TEACHER
        if self.verbose:
            print('Training Teacher', self.name)
        teacher_train_loop(self.teacher_model, self.epochs, self.batch_size, X_, y_, torch.optim.Adam, self.lr,
                           self.verbose,
                           use_wandb=self.wandb)

        # PREDICT SOFT TARGETS
        self.teacher_model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_, device=helpers.helpers.get_gpu_or_cpu(), dtype=torch.float32)
            soft_targets = self.teacher_model(x_tensor)

        # TRAIN STUDENT MODEL
        if self.verbose:
            print('Training Student', self.name)
        student_train_loop(self.student_model, loss_student, self.epochs, self.batch_size, X_, y_, soft_targets,
                           torch.optim.Adam, self.lr, self.verbose, self.lambda_.get(), use_wandb=self.wandb)
        # save state dict to file
        if savepath:
            self.state_dict_path = save_state_dict_to_file(self.student_model, savepath)
            self.model_path = save_model(self, savepath)

    def predict(self, X, *args, **kwargs):
        X_ = self.scaler_transform_X(X, self.scaling)
        if not hasattr(self, 'student_model'):
            raise Exception('Neural network does not exist before training.')
        else:
            self.student_model.eval()
            X_ = torch.tensor(X_[:, 0, :], device=get_gpu_or_cpu(), dtype=torch.float32)
            with torch.no_grad():
                predictions = self.student_model(X_)
        predictions = predictions.detach().cpu().numpy()
        predictions = predictions.reshape(-1, 1, predictions.shape[-1])
        return self.scaler_inverse_transform_Y(predictions, self.scaling)

    def predict_latent(self, X):
        with torch.no_grad():
            X_ = self.scaler_transform_X(X, self.scaling)
            if not hasattr(self, 'student_model'):
                raise Exception('Neural network does not exist before training.')
            else:
                self.student_model.eval()
                X_ = torch.tensor(X_, device=get_gpu_or_cpu(), dtype=torch.float32)
                with torch.no_grad():
                    predictions = self.student_model.encoder(X_)
                return predictions.detach().cpu().numpy()

    def get_info(self):
        info = super(generalized_distillation_net, self).get_info()
        return {'Model_Path': self.model_path, 'State_Dict': self.state_dict_path, **info}


class teacher_MLP(nn.Module):
    def __init__(self, input_features, width, output_features, mlp_layers,
                 early_stopping_range=100):
        super(teacher_MLP, self).__init__()
        self.name = 'Teacher MLP'
        self.mlp_layers = mlp_layers
        self.early_stopping_range = early_stopping_range
        self.input_features = input_features
        self.output_features = output_features
        self.width = width
        self.device = get_gpu_or_cpu()
        neurons_mlp = [input_features] + [self.width] * self.mlp_layers + [output_features]
        self.mlp = multi_layer_perceptron(neurons_mlp, device=self.device, activation_last_step=False,
                                          activation_func=F.leaky_relu)
        self.been_fit = False

    def forward(self, X):
        if len(X.shape) > 2:
            x = X.view(X.shape[0], -1)
        return self.mlp(x)


class teacher_LeNet(nn.Module):
    def __init__(self, input_images, output_dim, width=25, early_stopping_range=100):
        super(teacher_LeNet, self).__init__()
        self.encoder = LeNet5(width)
        self.input_image_count = input_images
        self.output_features = output_dim
        neuron_list = [width * input_images, width, output_dim]
        self.mlp = multi_layer_perceptron(neuron_list, F.leaky_relu, device=get_gpu_or_cpu(), activation_last_step=False)
        self.name = 'Teacher LeNet'
        self.early_stopping_range = early_stopping_range
        self.width = width
        self.device = get_gpu_or_cpu()
        self.been_fit = False

    def forward(self, x):
        embedding = self.encoder(x).view(x.shape[0], -1)
        return self.mlp(embedding)


def student_train_loop(model, loss_fun, epochs, batch_size, X, y, y_soft, optim, lr, verbose, lambda_,
                       validation_ratio=0.2,
                       use_wandb=False):
    if use_wandb:
        config = {
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': lr,
            'validation_ratio': validation_ratio,
            'model_name': model.name,
            'lambda': model.lambda_
        }
        model.wandb_run = wandb.init(config=config)
        wandb.watch(model)
        model.wandb_run.name = model.name + '_' + model.wandb_run.name

    # Put Model in Training Mode
    model.train()

    # Get Device
    device = model.device
    model.to(device)

    # Turn training data into tensors
    X_all = torch.tensor(X, device=device, dtype=torch.float32)
    y_all = torch.tensor(y[:, 0, :], device=device, dtype=torch.float32)
    if not torch.is_tensor(y_soft):
        y_soft = torch.tensor(y_soft, device=device)

    assert X_all.shape[0] == y_all.shape[0]

    # Shuffle the data according to some random permutation
    # Split into validation and training sets
    permutation = torch.randperm(X_all.shape[0])
    val_split = int(X_all.shape[0] * validation_ratio)
    X_val = X_all[permutation[:val_split]]
    X_train = X_all[permutation[val_split:]]
    y_val = y_all[permutation[:val_split]]
    y_train = y_all[permutation[val_split:]]
    y_soft_val = y_soft[permutation[:val_split]]
    y_soft_train = y_soft[permutation[val_split:]]

    # pass params to optimizer
    optimizer = optim(model.parameters(), lr=lr)

    # Early Stopping
    if model.early_stopping_range is not None:
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
            y_s = y_soft_train[indices]

            # Zero Gradients
            optimizer.zero_grad()

            # compute some kind of forward pass and its loss
            y_hat = model(x)
            loss = loss_fun(y_hat, y, y_s, lambda_)

            # backward pass
            loss.backward()

            # SGD step
            optimizer.step()

            # Add to loss sum
            acc_loss += float(loss)

        # Validation
        val_loss = student_validation(loss_fun, model, X_val, y_val, y_soft_val, lambda_)

        # Printing or WandB
        if (verbose and e % LINECOUNT_PRINTING == 0) or use_wandb:
            r2_score = get_true_r2_score(model, X_val, y_val)
            if verbose:
                print(f'Epoch {e}, Train Loss: ', acc_loss / X_train.shape[0], 'Val_Loss: ',
                      float(val_loss) / X_val.shape[0], 'R2_Val: ', r2_score)

            if use_wandb:
                wandb.log({'train/loss': loss, 'val/loss': val_loss, 'epoch': e, 'val/r2': r2_score})

        # Early Stopping
        if earlyStopping:
            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                best_state_dict = clone_and_detach_state_dict(model)
                waiting_time = 0
            else:
                waiting_time += 1
            if waiting_time > model.early_stopping_range:
                break

    # Load best state dict in case of early stopping
    if earlyStopping:
        model.load_state_dict(best_state_dict)

    if use_wandb:
        model.wandb_run.finish()


def teacher_train_loop(model, epochs, batch_size, X, y, optim, lr, verbose, validation_ratio=0.2, use_wandb=False):
    if use_wandb:
        config = {
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': lr,
            'validation_ratio': validation_ratio,
            'model_name': model.name
        }
        model.wandb_run = wandb.init(config=config)
        wandb.watch(model)
        model.wandb_run.name = model.name + '_' + model.wandb_run.name

    # Put Model in Training Mode
    model.train()

    # Get Device
    device = model.device
    model.to(device)

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
    optimizer = optim(model.parameters(), lr=lr)

    # Early Stopping
    if model.early_stopping_range is not None:
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
            y_hat = model(x)
            loss = F.mse_loss(y_hat, y)

            # backward pass
            loss.backward()

            # SGD step
            optimizer.step()

            # Add to loss sum
            acc_loss += float(loss)

        # Validation
        with torch.no_grad():
            model.eval()
            val_loss = F.mse_loss(model(X_val), y_val)
            model.train()

        # Printing or WandB
        if (verbose and e % LINECOUNT_PRINTING == 0) or use_wandb:
            r2_score = get_true_r2_score(model, X_val, y_val)
            if verbose:
                print(f'Epoch {e}, Train Loss: ', acc_loss / X_train.shape[0], 'Val_Loss: ',
                      float(val_loss) / X_val.shape[0], 'R2_Val: ', r2_score)

            if use_wandb:
                wandb.log({'train/loss': loss, 'val/loss': val_loss, 'epoch': e, 'val/r2': r2_score})

        # Early Stopping
        if earlyStopping:
            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                best_state_dict = clone_and_detach_state_dict(model)
                waiting_time = 0
            else:
                waiting_time += 1
            if waiting_time > model.early_stopping_range:
                break

    # Load best state dict in case of early stopping
    if earlyStopping:
        model.load_state_dict(best_state_dict)

    if use_wandb:
        model.wandb_run.finish()

    model.been_fit = True


def student_validation(loss_fun, model, X_val, y_val, y_soft_val, lambda_):
    model.eval()
    with torch.no_grad():
        y_hat = model(X_val)
        val_loss = loss_fun(y_hat, y_val, y_soft_val, lambda_)
    model.train()
    return val_loss


def get_true_r2_score(model, x, y):
    model.eval()
    with torch.no_grad():
        y_hat = model(x)
    model.train()
    return float(r2_loss(y_hat, y))


def loss_student(y_hat, y, y_soft, lambda_):
    hard_loss = F.mse_loss(y, y_hat)
    soft_loss = F.mse_loss(y, y_soft)
    return hard_loss * lambda_ + (1 - lambda_) * soft_loss
