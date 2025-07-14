import torch
import torch.nn as nn


class ExpActivation(nn.Module):
    """
    Exponential activation function from Koo & Ploenzke, 2021 (PMID: 34322657)
    """

    def __init__(self):
        super(ExpActivation, self).__init__()

    def forward(self, x):
        return torch.exp(x)


class Unsqueeze(torch.nn.Module):
    """
    Unsqueeze for sequential models
    """

    def forward(self, x):
        return x.unsqueeze(-1)


class ExplaiNN(nn.Module):
    """
    The ExplaiNN model (PMID: 37370113)
    """

    def __init__(
        self,
        num_cnns: int = 30,
        input_length: int = 1000,
        num_classes: int = 1,
        filter_size: int = 19,
        num_fc: int = 2,
        pool_size: int = 7,
        pool_stride: int = 7,
        dropout: float = 0.3,
        channels: int = 4,
        channels_mid: int = 100,
    ):
        """
        :param num_cnns: number of independent cnn units
        :param input_length: input sequence length
        :param num_classes: number of outputs
        :param filter_size: size of the unit's filter
        :param num_fc: number of FC layers in the unit
        :param pool_size: size of the unit's maxpooling layer
        :param pool_stride: stride of the unit's maxpooling layer
        :param dropout: dropout rate
        :param channels: number of channels in the input sequence
        """
        super().__init__()

        self.num_cnns = num_cnns
        self.input_length = input_length
        self.num_classes = num_classes
        self.filter_size = filter_size
        self.num_fc = num_fc
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.dropout = dropout
        self.channels = channels
        self.channels_mid = channels_mid

        if num_fc == 0:
            self.linears = nn.Sequential(
                nn.Conv1d(
                    in_channels=channels * num_cnns,
                    out_channels=1 * num_cnns,
                    kernel_size=filter_size,
                    groups=num_cnns,
                ),
                nn.BatchNorm1d(1 * num_cnns),
                ExpActivation(),
                nn.MaxPool1d(input_length - (filter_size - 1)),
                nn.Flatten(),
            )
        elif num_fc == 1:
            self.linears = nn.Sequential(
                nn.Conv1d(
                    in_channels=channels * num_cnns,
                    out_channels=1 * num_cnns,
                    kernel_size=filter_size,
                    groups=num_cnns,
                ),
                nn.BatchNorm1d(num_cnns),
                ExpActivation(),
                nn.MaxPool1d(pool_size, pool_stride),
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(
                    in_channels=int(
                        ((input_length - (filter_size - 1)) - (pool_size - 1) - 1)
                        / pool_stride
                        + 1
                    )
                    * num_cnns,
                    out_channels=1 * num_cnns,
                    kernel_size=1,
                    groups=num_cnns,
                ),
                nn.BatchNorm1d(1 * num_cnns),
                nn.ReLU(),
                nn.Flatten(),
            )
        elif num_fc == 2:
            self.linears = nn.Sequential(
                nn.Conv1d(
                    in_channels=channels * num_cnns,
                    out_channels=1 * num_cnns,
                    kernel_size=filter_size,
                    groups=num_cnns,
                ),
                nn.BatchNorm1d(num_cnns),
                ExpActivation(),
                nn.MaxPool1d(pool_size, pool_stride),
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(
                    in_channels=int(
                        ((input_length - (filter_size - 1)) - (pool_size - 1) - 1)
                        / pool_stride
                        + 1
                    )
                    * num_cnns,
                    out_channels=channels_mid * num_cnns,
                    kernel_size=1,
                    groups=num_cnns,
                ),
                nn.BatchNorm1d(channels_mid * num_cnns),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(
                    in_channels=channels_mid * num_cnns,
                    out_channels=1 * num_cnns,
                    kernel_size=1,
                    groups=num_cnns,
                ),
                nn.BatchNorm1d(1 * num_cnns),
                nn.ReLU(),
                nn.Flatten(),
            )
        else:
            self.linears = nn.Sequential(
                nn.Conv1d(
                    in_channels=channels * num_cnns,
                    out_channels=1 * num_cnns,
                    kernel_size=filter_size,
                    groups=num_cnns,
                ),
                nn.BatchNorm1d(1 * num_cnns),
                ExpActivation(),
                nn.MaxPool1d(pool_size, pool_stride),
                nn.Flatten(),
                Unsqueeze(),
                nn.Conv1d(
                    in_channels=int(
                        ((input_length - (filter_size - 1)) - (pool_size - 1) - 1)
                        / pool_stride
                        + 1
                    )
                    * num_cnns,
                    out_channels=channels_mid * num_cnns,
                    kernel_size=1,
                    groups=num_cnns,
                ),
                nn.BatchNorm1d(channels_mid * num_cnns),
                nn.ReLU(),
            )

            self.linears_bg = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Conv1d(
                            in_channels=channels_mid * num_cnns,
                            out_channels=channels_mid * num_cnns,
                            kernel_size=1,
                            groups=num_cnns,
                        ),
                        nn.BatchNorm1d(channels_mid * num_cnns),
                        nn.ReLU(),
                    )
                    for _ in range(num_fc - 2)
                ]
            )

            self.last_linear = nn.Sequential(
                nn.Dropout(dropout),
                nn.Conv1d(
                    in_channels=channels_mid * num_cnns,
                    out_channels=1 * num_cnns,
                    kernel_size=1,
                    groups=num_cnns,
                ),
                nn.BatchNorm1d(1 * num_cnns),
                nn.ReLU(),
                nn.Flatten(),
            )

        self.final = nn.Linear(num_cnns, num_classes)

    def forward(self, x):
        x = x.repeat(1, self.num_cnns, 1)  # NB copies data --> uses more memory

        if self.num_fc <= 2:
            outs = self.linears(x)
        else:
            outs = self.linears(x)
            for i in range(len(self.linears_bg)):
                outs = self.linears_bg[i](outs)
            outs = self.last_linear(outs)
        out = self.final(outs)
        return out
