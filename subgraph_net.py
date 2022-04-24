import torch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions
import numpy as np
# import matplotlib.animation as animation
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

'''输入feature维的单polyline tensor，输出feature+global\ feature维的单polyline tensor'''
# 声明一个layer，包含一个全连接层
class SubgraphNet_Layer(nn.Module):
    def __init__(self, input_channels=128, hidden_channels=64):
        super().__init__()
        self.fc = nn.Linear(input_channels, hidden_channels) #single fully connected network
        nn.init.kaiming_normal_(self.fc.weight) # 权重初始化
        # self.layer_norm = nn.LayerNorm(hidden_channels)  # layer norm

    def forward(self, input):
        hidden = self.fc(input).unsqueeze(0)                           # 一个全连接层,unsqueeze增加一维 torch.Size([r, c]) -> torch.Size([1, r, c])
        encode_data = F.relu(F.layer_norm(hidden, hidden.size()[1:]))  # layer norm and relu
        kernel_size = encode_data.size()[1]                            # 30
        maxpool = nn.MaxPool1d(kernel_size)                            # 最大值池化
        polyline_feature = maxpool(encode_data.transpose(1,2)).squeeze()
        polyline_feature = polyline_feature.repeat(kernel_size, 1)
        output = torch.cat([encode_data.squeeze(), polyline_feature], 1) # 拼接relu结果和池化结果，输出shape(r,2*c)
        return output

class SubgraphNet(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.sublayer1 = SubgraphNet_Layer(input_channels)
        self.sublayer2 = SubgraphNet_Layer()
        self.sublayer3 = SubgraphNet_Layer() #output = 128

    def forward(self, input):           # input.shape -> torch.Size([30, 6]) torch.Size([18, 8]),
        out1 = self.sublayer1(input)    # 调用SubgraphNet_Layer.forward(input)，  out -> (30, 128)
        out2 = self.sublayer2(out1)     # out2 -> (128, 128)
        out3 = self.sublayer3(out2)     # out3 -> (128, 128)
        kernel_size = out3.size()[0]    # 128
        maxpool = nn.MaxPool1d(kernel_size)
        polyline_feature = maxpool(out3.unsqueeze(1).transpose(0,2)).squeeze()  # polyline_feature.shape -> torch.Size([128])
        return polyline_feature         # torch.Size([128])
