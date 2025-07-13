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
        num_cnns: int,
        input_length: int,
        num_classes: int,
        filter_size: int = 19,
        num_fc: int = 2,
        pool_size: int = 7,
        pool_stride: int = 7,
    ):
        """
        :param num_cnns: int, number of independent cnn units
        :param input_length: int, input sequence length
        :param num_classes: int, number of outputs
        :param filter_size: int, size of the unit's filter, default=19
        :param num_fc: int, number of FC layers in the unit, default=2
        :param pool_size: int, size of the unit's maxpooling layer, default=7
        :param pool_stride: int, stride of the unit's maxpooling layer, default=7
        :param weight_path: string, path to the file with model weights
        """
        super(ExplaiNN, self).__init__()

        self._options = {
            "num_cnns": num_cnns,
            "input_length": input_length,
            "num_classes": num_classes,
            "filter_size": filter_size,
            "num_fc": num_fc,
            "pool_size": pool_size,
            "pool_stride": pool_stride,
        }

        if num_fc == 0:
            self.linears = nn.Sequential(
                nn.Conv1d(
                    in_channels=4 * num_cnns,
                    out_channels=1 * num_cnns,
                    kernel_size=filter_size,
                    groups=num_cnns,
                ),
                nn.BatchNorm1d(num_cnns),
                ExpActivation(),
                nn.MaxPool1d(input_length - (filter_size - 1)),
                nn.Flatten(),
            )
        elif num_fc == 1:
            self.linears = nn.Sequential(
                nn.Conv1d(
                    in_channels=4 * num_cnns,
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
                nn.BatchNorm1d(1 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU(),
                nn.Flatten(),
            )
        elif num_fc == 2:
            self.linears = nn.Sequential(
                nn.Conv1d(
                    in_channels=4 * num_cnns,
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
                    out_channels=100 * num_cnns,
                    kernel_size=1,
                    groups=num_cnns,
                ),
                nn.BatchNorm1d(100 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(
                    in_channels=100 * num_cnns,
                    out_channels=1 * num_cnns,
                    kernel_size=1,
                    groups=num_cnns,
                ),
                nn.BatchNorm1d(1 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU(),
                nn.Flatten(),
            )
        else:
            self.linears = nn.Sequential(
                nn.Conv1d(
                    in_channels=4 * num_cnns,
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
                    out_channels=100 * num_cnns,
                    kernel_size=1,
                    groups=num_cnns,
                ),
                nn.BatchNorm1d(100 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU(),
            )

            self.linears_bg = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Conv1d(
                            in_channels=100 * num_cnns,
                            out_channels=100 * num_cnns,
                            kernel_size=1,
                            groups=num_cnns,
                        ),
                        nn.BatchNorm1d(100 * num_cnns, 1e-05, 0.1, True),
                        nn.ReLU(),
                    )
                    for i in range(num_fc - 2)
                ]
            )

            self.last_linear = nn.Sequential(
                nn.Dropout(0.3),
                nn.Conv1d(
                    in_channels=100 * num_cnns,
                    out_channels=1 * num_cnns,
                    kernel_size=1,
                    groups=num_cnns,
                ),
                nn.BatchNorm1d(1 * num_cnns, 1e-05, 0.1, True),
                nn.ReLU(),
                nn.Flatten(),
            )

        self.final = nn.Linear(num_cnns, num_classes)

    def forward(self, x):
        x = x.repeat(1, self._options["num_cnns"], 1)
        if self._options["num_fc"] <= 2:
            outs = self.linears(x)
        else:
            outs = self.linears(x)
            for i in range(len(self.linears_bg)):
                outs = self.linears_bg[i](outs)
            outs = self.last_linear(outs)
        out = self.final(outs)
        return out
