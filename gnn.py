import torch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions
# import dgl
# import networkx as nx
import numpy as np
# import matplotlib.animation as animation
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Because it is a fully-connected graph, so there is no necessity to build a graph
class GraphAttentionNet(nn.Module):
    def __init__(self, in_dim=128, key_dim=64, value_dim=64):
        super().__init__()
        self.queryFC = nn.Linear(in_dim, key_dim)
        nn.init.kaiming_normal_(self.queryFC.weight)
        self.keyFC = nn.Linear(in_dim, key_dim)
        nn.init.kaiming_normal_(self.keyFC.weight)
        self.valueFC = nn.Linear(in_dim, value_dim)
        nn.init.kaiming_normal_(self.valueFC.weight)

    def forward(self, polyline_feature):
        p_query = F.relu(self.queryFC(polyline_feature))
        p_key = F.relu(self.keyFC(polyline_feature))
        p_value = F.relu(self.valueFC(polyline_feature))
        query_result = p_query.mm(p_key.t())    # 矩阵乘
        query_result = query_result / (p_key.shape[1] ** 0.5)
        attention = F.softmax(query_result, dim=1)
        output = attention.mm(p_value)
        return output + p_query
