import torch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions
import numpy as np
# import matplotlib.animation as animation
import matplotlib.pyplot as plt
from subgraph_net import SubgraphNet
from gnn import GraphAttentionNet

import warnings
warnings.filterwarnings('ignore')


class VectorNet(nn.Module):
    def __init__(self, traj_features=6, map_features=8, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = dict(device=torch.device('cpu'))
        self.cfg = cfg
        self.traj_subgraphnet = SubgraphNet(traj_features)
        self.map_subgraphnet = SubgraphNet(map_features)
        self.graphnet = GraphAttentionNet()
        # decoder
        prediction_step = 2*(49 - self.cfg['last_observe'])  # TODO
        self.fc = nn.Linear(64, 64)
        nn.init.kaiming_normal_(self.fc.weight)
        self.layer_norm = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, prediction_step)
        nn.init.kaiming_normal_(self.fc2.weight)

        self.loss_fn = nn.MSELoss(size_average=False, reduce=True) # MSEloss 就是 Gaussian NLL

    def _forward_train(self, trajectory_batch, vectormap_batch):
        PITmap_list = vectormap_batch['PIT']
        MIAmap_list = vectormap_batch['MIA']
        city_name_list = vectormap_batch['city_name']
        batch_size = trajectory_batch.size()[0]   # trajectory_batch.size() -> torch.Size([2, 49, 6]), [batch_size, trajectory_vector_size, feature_size]

        label = trajectory_batch[:, self.cfg['last_observe']:, 2:4] # label.size() -> torch.Size([2, 19, 2]), last 19 trajectory_vector, [x1, y1]

        predict_list = []
        for i in range(batch_size):
            polyline_list = []
            if city_name_list[i] == 'PIT':
                vectormap_list = PITmap_list[i]
            else:
                vectormap_list = MIAmap_list[i]

            polyline_list.append(self.traj_subgraphnet(trajectory_batch[i, :self.cfg['last_observe']]).unsqueeze(0))
            for vec_map in vectormap_list:
                vec_map = vec_map.to(device=self.cfg['device'], dtype=torch.float)
                map_feature = self.map_subgraphnet(vec_map)
                polyline_list.append(map_feature.unsqueeze(0))
            polyline_feature = F.normalize(torch.cat(polyline_list, dim=0), p=2, dim=1)  # L2 Normalize
            out = self.graphnet(polyline_feature)
            decoded_data_perstep = self.fc2(F.relu(self.layer_norm(self.fc(out[0].unsqueeze(0))))).view(1, -1, 2)  # corresponding one
            decoded_data = torch.cumsum(decoded_data_perstep, dim=0)  # parameterized as per-step coordinate offsets
            predict_list.append(decoded_data)
        predict_batch = torch.cat(predict_list, dim=0)
        loss = self.loss_fn(predict_batch, label)
        if np.isnan(loss.item()):
            print(trajectory_batch[:, 0, -1])
            raise Exception("Loss ERROR!")
        return loss

    def _forward_test(self, trajectory_batch):
        batch_size = trajectory_batch.size()[0]

        traj_label = trajectory_batch[:, self.cfg['last_observe']:, 2:4]
        # predict_list = []
        result, label = dict(), dict()
        for i in range(batch_size):
            polyline_feature = self.traj_subgraphnet(trajectory_batch[i, :self.cfg['last_observe']]).unsqueeze(0)
            polyline_feature = F.normalize(polyline_feature, p=2, dim=1)  # L2 Normalize
            out = self.graphnet(polyline_feature)
            decoded_data_perstep = self.fc2(F.relu(self.layer_norm(self.fc(out)))).view(-1, 2)  # corresponding one
            decoded_data = torch.cumsum(decoded_data_perstep, dim=0)  # parameterized as per-step coordinate offsets
            key = str(trajectory_batch[i, 0, -1].int().item())
            predict_step = self.cfg['predict_step']
            result.update({key: decoded_data[:predict_step]})
            label.update({key: traj_label[i, :predict_step]})
        return result, label

    def forward(self, trajectory, vectormap):
        if self.training:
            return self._forward_train(trajectory, vectormap)
        else:
            return self._forward_test(trajectory)
