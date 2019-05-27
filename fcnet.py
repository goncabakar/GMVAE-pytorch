import torch.nn as nn
import torch.nn.functional as F

class FCNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_layers, act_out = "relu"):
        super(FCNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.layers = nn.ModuleList()
        self.act = nn.ReLU(True)
        self.flag_act_out = True

        if act_out == "tanh":
            self.act_out = nn.Tanh()
        elif act_out == "relu":
            self.act_out = nn.ReLU(True)
        else:
            self.flag_act_out = False

        if hidden_layers == 0:
            self.layers.append(nn.Linear(input_dim, output_dim))

        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for i in range(1, hidden_layers):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if i == self.hidden_layers:
                if self.flag_act_out:
                    x = self.act_out(x)
                else:
                    return x
            else:
                x = self.act(x)
        return x
