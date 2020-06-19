import torch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions
import numpy as np
# import matplotlib.animation as animation
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SubgraphNet_Layer(nn.Module):
    def __init__(self, input_channels=128, hidden_channels=64):
        super().__init__()
        self.fc = nn.Linear(input_channels, hidden_channels) #single fully connected network
        nn.init.kaiming_normal_(self.fc.weight)
        # self.layer_norm = nn.LayerNorm(hidden_channels)  # layer norm

    def forward(self, input):
        hidden = self.fc(input).unsqueeze(0)
        encode_data = F.relu(F.layer_norm(hidden, hidden.size()[1:]))  # layer norm
        kernel_size = encode_data.size()[1]
        maxpool = nn.MaxPool1d(kernel_size)  # max pool
        polyline_feature = maxpool(encode_data.transpose(1,2)).squeeze()
        polyline_feature = polyline_feature.repeat(kernel_size, 1)
        output = torch.cat([encode_data.squeeze(), polyline_feature], 1)
        return output

class SubgraphNet(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.sublayer1 = SubgraphNet_Layer(input_channels)
        self.sublayer2 = SubgraphNet_Layer()
        self.sublayer3 = SubgraphNet_Layer() #output = 128

    def forward(self, input):
        out1 = self.sublayer1(input)
        out2 = self.sublayer2(out1)
        out3 = self.sublayer3(out2)
        kernel_size = out3.size()[0]
        maxpool = nn.MaxPool1d(kernel_size)
        polyline_feature = maxpool(out3.unsqueeze(1).transpose(0,2)).squeeze()
        return polyline_feature
