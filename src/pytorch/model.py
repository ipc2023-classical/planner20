import torch
import torch.nn as nn
from math import sqrt


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.resblock = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.resblock(x)
        out += identity
        out = self.relu(out)
        return out


# H(euristic) Neural Network
class HNN(nn.Module):
    def __init__(
        self,
        input_units: int,
        hidden_units,
        output_units: int,
        hidden_layers: int,
        activation: str,
        output_layer: str,
        dropout_rate: float,
        linear_output: bool,
        use_bias: bool,
        use_bias_output: bool,
        weights_method: str,
        model: str,
    ):
        super(HNN, self).__init__()
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.output_layer = output_layer
        self.linear_output = linear_output
        self.use_bias = use_bias
        self.use_bias_output = use_bias_output
        self.model = model

        if model == "resnet":
            self.flatten = nn.Flatten()

        hu = self.set_hidden_units()
        self.hid = self.set_hidden_layers(hu)

        if model == "resnet":
            self.resblock = ResidualBlock(hidden_units[0])

        # If `use_bias` is set to False, `bias_output` is set to False regardless
        # of the value in `use_bias_output`.
        bias_output = False if self.use_bias is False else use_bias_output
        self.opt = nn.Linear(hu[-1], output_units, bias=bias_output)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=False)

        self.activation = self.set_activation(activation)
        self.output_activation = self.set_output_activation(activation)

        # In PyTorch 1.9, the default initialization used is an adapted Kaiming,
        # "Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features))"
        # See:
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L92
        # https://github.com/pytorch/pytorch/issues/57109
        if weights_method != "default":
            self.initialize_weights(weights_method)

    def set_hidden_units(self) -> list:
        hu = [self.input_units]

        if len(self.hidden_units) == 0:  # Layers with scalable number of units.
            unit_diff = self.input_units - self.output_units
            step = int(unit_diff / (self.hidden_layers + 1))
            for i in range(self.hidden_layers):
                hu.append(self.input_units - (i + 1) * step)
        elif len(self.hidden_units) == 1:  # All layers with the same number of units.
            hu += self.hidden_units * self.hidden_layers
        else:
            hu += self.hidden_units

        return hu

    def set_hidden_layers(self, hu: list) -> nn.ModuleList():
        hid = nn.ModuleList()
        for i in range(self.hidden_layers):
            hid.append(nn.Linear(hu[i], hu[i + 1], bias=self.use_bias))
        return hid

    def set_activation(self, activation: str):
        if activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "leakyrelu":
            return nn.LeakyReLU()
        else:
            raise NotImplementedError(f"{activation} function not implemented!")

    def set_output_activation(self, activation: str):
        if self.output_layer == "regression":
            return nn.ReLU() if activation != "leakyrelu" else nn.LeakyReLU()
        elif self.output_layer == "prefix":
            return nn.Sigmoid()
        elif self.output_layer == "one-hot":
            return nn.Softmax()
        else:
            raise NotImplementedError(
                f"{self.output_layer} not implemented for output layer!"
            )

    def initialize_weights(self, method):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if "kaiming" in method:
                    self.set_kaiming_init(m, method)
                elif method == "sqrt_k":
                    k = 1.0 / m.in_features
                    self.set_uniform_init(m, a=-sqrt(k), b=sqrt(k))
                elif method == "1":
                    self.set_uniform_init(m, a=-1.0, b=1.0)
                elif method == "01":
                    self.set_uniform_init(m, a=0.0, b=1.0)
                elif "xavier" in method:
                    self.set_xavier_init(m, method)
                else:
                    raise NotImplementedError(
                        f"Weights method {method} not implemented!"
                    )

    def set_uniform_init(self, m: nn.modules.linear.Linear, a: float, b: float):
        torch.nn.init.uniform_(m.weight, a, b)
        if self.use_bias:
            torch.nn.init.uniform_(m.bias, a, b)

    def set_kaiming_init(
        self, m: nn.modules.linear.Linear, method: str, zero_bias: bool = True
    ):
        if "uniform" in method:
            torch.nn.init.kaiming_uniform_(m.weight)
        elif "normal" in method:
            torch.nn.init.kaiming_normal_(m.weight)
        if zero_bias and self.use_bias:
            torch.nn.init.zeros_(m.bias)

    def set_xavier_init(
        self,
        m: nn.modules.linear.Linear,
        method: str,
        zero_bias: bool = True,
        gain_: float = 1.0,
    ):
        if "uniform" in method:
            torch.nn.init.xavier_uniform_(m.weight, gain=gain_)
        elif "normal" in method:
            torch.nn.init.xavier_normal_(m.weight, gain=gain_)
        if zero_bias and self.use_bias:
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.model == "resnet":
            if len(x.size()) > 1:
                x = self.flatten(x)
            for h in self.hid:
                x = self.activation(h(x))
                if self.dropout_rate > 0:
                    x = self.dropout(x)
            x = self.resblock(x)
            out = self.opt(x)
            return self.output_activation(out)

        else:
            for h in self.hid:
                x = self.activation(h(x))
                if self.dropout_rate > 0:
                    x = self.dropout(x)

            if self.linear_output:
                return self.opt(x)
            return self.output_activation(self.opt(x))
