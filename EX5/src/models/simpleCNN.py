from abc import ABC

import torch


class SimpleCNN(torch.nn.Module, ABC):

    def __init__(self, n_hidden: int, n_in_chn: int, n_kernels: int, kernel_size: int,
                 activation: str):
        """
        Creates a simple CNN specified by parameters.

        Parameters
        ----------
        n_hidden : int
            number of hidden layers
        n_in_chn : int
            number of input channels
        n_kernels : int
            number of kernels
        kernel_size : int
            kernel size
        activation : str
            activation function as string ('relu' and 'selu' supported)
        """
        super().__init__()
        self._activation = activation.lower()
        self._hidden_layers = self._create_hidden_layers(n_hidden, n_in_chn, n_kernels, kernel_size)
        self._output_layer = torch.nn.Conv2d(n_kernels, out_channels=1, kernel_size=kernel_size,
                                             padding=kernel_size // 2)

    def _create_hidden_layers(self, n_hidden: int, n_in_chn: int, n_kernels: int, kernel_size: int):
        layers = []
        for i in range(n_hidden):
            layers.append(torch.nn.Conv2d(
                n_in_chn,
                out_channels=n_kernels,
                kernel_size=kernel_size,
                bias=True,
                padding=kernel_size // 2)
            )
            layers.append(self._get_activation())
            n_in_chn = n_kernels
        return torch.nn.Sequential(*layers)

    def _get_activation(self):
        if self._activation == 'relu':
            return torch.nn.ReLU()
        elif self._activation == 'selu':
            return torch.nn.SELU()
        else:
            raise NotImplemented(f"Activation {self._activation} is not supported!")

    def forward(self, x):
        mask = x[:, 1].clone().to(dtype=torch.bool)
        x = self._hidden_layers(x)
        # use selu for output_layer
        x = torch.nn.functional.selu(self._output_layer(x))
        return x, mask
