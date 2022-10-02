import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, output_dim):
        super(LeNet5, self).__init__()
        self.output_dim = output_dim
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2))

        self.linear_stack = nn.Sequential(
            nn.Linear(400, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, output_dim))

    def forward(self, x):
        x = self.check_dimensions(x)
        x = self.conv_stack(x)
        N = x.shape[0]
        x = x.view(N,-1)
        x = self.linear_stack(x)
        x = self.rearrange_dims(x)
        return x

    def check_dimensions(self, x):
        if x.shape[-3:] != (1,28,28):
            shape = list(x.shape)
            assert shape[-1] == 28**2, 'Expected a square image 28 x 28'
            self.original_first_dims = shape[:-1]
            shape = (-1,1,28,28)
            x = x.view(shape)
        return x

    def rearrange_dims(self,x):
        assert hasattr(self,'original_first_dims')
        shape = tuple(list(self.original_first_dims) + [self.output_dim])
        return x.view(shape)

if __name__ == '__main__':
    lenet = LeNet5(1)
    data = torch.randn(100, 1, 28 * 28)
    out = lenet(data)
    print(out)