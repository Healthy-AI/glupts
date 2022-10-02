import torch.nn as nn

class multi_layer_perceptron(nn.Module):
    def __init__(self, neuron_list, activation_func, device, bias=True, activation_last_step=True):
        super(multi_layer_perceptron, self).__init__()
        assert (len(neuron_list) >= 2)
        layer_list = []
        for i, n in enumerate(neuron_list[:-1]):
            layer_list.append(nn.Linear(n, neuron_list[i + 1], bias=bias, device=device))
        self.layers = nn.ModuleList(layer_list)
        self.activation = activation_func
        self.activation_last_step = activation_last_step

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        x = self.layers[-1](x)
        if self.activation_last_step:
            return self.activation(x)
        else:
            return x