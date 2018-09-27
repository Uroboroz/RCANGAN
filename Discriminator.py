import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, size, padding=1, pool_layer=nn.MaxPool2d(2, stride=2),
                 bn=False, dropout=False, activation_fn=nn.ReLU(), stride=1, device=torch.device('cpu')):
        super(ConvLayer, self).__init__()
        layers = [nn.Conv2d(size[0], size[1], size[2], padding=padding, stride=stride)]
        if pool_layer is not None:
            layers.append(pool_layer)
        if bn:
            layers.append(nn.BatchNorm2d(size[1]))
        if dropout:
            layers.append(nn.Dropout2d())
        layers.append(activation_fn)

        self.model = nn.Sequential(*layers)
        self.to(device)

    def forward(self, x):
        return self.model(x)


class FullyConnected(nn.Module):
    def __init__(self, sizes, dropout=False, activation_fn=nn.Tanh(), flatten=False, last_fn=None,
                 device=torch.device('cpu')):
        super(FullyConnected, self).__init__()
        layers = []
        self.flatten = flatten
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(activation_fn)
        else:
            layers.append(nn.Linear(sizes[-2], sizes[-1]))
        if last_fn is not None:
            layers.append(last_fn)
        self.model = nn.Sequential(*layers)
        self.to(device)

    def forward(self, x):
        if self.flatten:
            x = x.view(x.shape[0], -1)
        return self.model(x)


class DiscGAN(nn.Module):
    def __init__(self, latent_size=1000, ff=66, batchnorm=False, device=torch.device('cpu')):
        super(DiscGAN, self).__init__()

        self.latent_size = latent_size
        self._conv1 = ConvLayer([3, 3, 3], padding=0, bn=batchnorm, stride=2, pool_layer=None,
                                activation_fn=nn.LeakyReLU(), device=device)
        self._conv2 = ConvLayer([3, 3, 3], padding=0, bn=batchnorm, stride=2, pool_layer=None,
                                activation_fn=nn.LeakyReLU(), device=device)
        self._conv3 = ConvLayer([3, 3, 3], padding=0, bn=batchnorm, stride=2, pool_layer=None,
                                activation_fn=nn.LeakyReLU(), device=device)
        self._conv4 = ConvLayer([3, 3, 3], padding=0, bn=batchnorm, stride=2, pool_layer=None,
                                activation_fn=nn.LeakyReLU(), device=device)

        self.fce1 = FullyConnected([3 * ff * ff, latent_size], activation_fn=nn.LeakyReLU(), flatten=True,
                                   device=device)
        self.fce2 = FullyConnected([latent_size, 1], activation_fn=nn.Sigmoid, device=device)
        self.to(device)

    def forward(self, x):
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._conv4(x)
        return self.fce2(self.fce1(x))
