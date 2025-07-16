import torch
import torch.nn as nn


class Exp(nn.Module):
    def forward(self, x):
        return torch.exp(x)


class Unsqueeze(torch.nn.Module):
    """Unsqueeze for sequential models"""

    def __init__(self, dim_unsqueeze: int = -1):
        self.dim_unsqueeze = dim_unsqueeze
        super(Unsqueeze, self).__init__()

    def forward(self, x):
        return x.unsqueeze(self.dim_unsqueeze)


class Squeeze(torch.nn.Module):
    """Squeeze for sequential models"""

    def __init__(self, dim_squeeze: int = -1):
        self.dim_squeeze = dim_squeeze
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze(self.dim_squeeze)


def make_explainn(
    num_cnns: int = 20,
    input_length: int = 1000,
    filter_size: int = 11,
    pool_size: int = 7,
    pool_stride: int = 7,
    dropout: float = 0.3,
    num_classes: int = 1,
    channels: int = 4,
    channels_mid: int = 100,
    # # the architecture we used is the same as the one in the paper for num_fc=2
    # num_fc: int = 2,
):
    """
    The ExplaiNN model (PMID: 37370113)

    Requires in the forward pass (due to the grouped convolutions):
    x = x.repeat(1, num_cnns, 1)
    """
    return nn.Sequential(
        nn.Conv1d(
            in_channels=channels * num_cnns,
            out_channels=1 * num_cnns,
            kernel_size=filter_size,
            groups=num_cnns,
        ),
        nn.BatchNorm1d(num_cnns),
        Exp(),  # exponential activation
        nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride),
        # ##############################################################################
        # Alternative MaxPool layer in case some downstream analysis does not support
        # well the MaxPool1d layer (e.g. kundajelab/shap repo).
        # Unsqueeze(dim_unsqueeze=2),
        # nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride)),
        # Squeeze(dim_squeeze=2),
        # ##############################################################################
        nn.Flatten(),
        Unsqueeze(dim_unsqueeze=-1),
        nn.Conv1d(
            in_channels=int(
                ((input_length - (filter_size - 1)) - (pool_size - 1)) / pool_stride + 1
            )
            * num_cnns,
            out_channels=channels_mid * num_cnns,
            kernel_size=1,  # Effectively a linear layer
            groups=num_cnns,
        ),
        nn.BatchNorm1d(channels_mid * num_cnns),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Conv1d(
            in_channels=channels_mid * num_cnns,
            out_channels=1 * num_cnns,
            kernel_size=1,  # Effectively a linear layer
            groups=num_cnns,
        ),
        nn.BatchNorm1d(1 * num_cnns),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(num_cnns, num_classes),
    )
