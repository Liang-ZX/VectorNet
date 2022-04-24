import torch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions
import numpy as np
# import matplotlib.animation as animation
import matplotlib.pyplot as plt
from subgraph_net import SubgraphNet
from gnn import GraphAttentionNet
import sys

import warnings
warnings.filterwarnings('ignore')


class VectorNet(nn.Module):
    def __init__(self, traj_features=4, map_features=8, cfg=None):
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

        self.loss_fn = nn.MSELoss(size_average=False, reduce=True) # MSEloss 就是 Gaussian NLL，均方损失函数，reduce为True则loss返回标量（各元素均方差之和），size_average=False不求均值

    def _forward_train(self, trajectory_batch, vectormap_batch):
        ''' 分别把前3秒traj和vectormap放入traj_subgraphnet和map_subgraphnet，得到的结果放入polyline_list（后2秒traj作为label）
            对polyline_list再做normalize得到polyline_feature放入graphnet
            再经过relu和fc2得到后2秒的traj与label计算loss并返回'''
        # PITmap_list = vectormap_batch['PIT']
        # MIAmap_list = vectormap_batch['MIA']
        # city_name_list = vectormap_batch['city_name']

        # vectormap_batch.size() -> torch.Size([2, n, 18, 8])
        # trajectory_batch.size() -> torch.Size([2, 49, 6]), [batch_size, trajectory_vector_size, feature_size]
        batch_size = trajectory_batch.size()[0]

        # traj_out = trajectory_batch[0].cpu().numpy()
        # print(trajectory_batch.size())
        # mf = []

        # print(trajectory_batch.size())
        # print(len(vectormap_batch))
                                                  
        label = trajectory_batch[:, self.cfg['last_observe']:, 2:4] # label.size() -> torch.Size([2, 19, 2]), last 19 trajectory_vector, [x1, y1]

        predict_list = []
        for i in range(batch_size):
            polyline_list = []
            # if city_name_list[i] == 'PIT':
            #     vectormap_list = PITmap_list[i]
            # else:
            #     vectormap_list = MIAmap_list[i]
            polyline_list.append(self.traj_subgraphnet(trajectory_batch[i, :self.cfg['last_observe']]).unsqueeze(0)) # 将轨迹前last_observe个点数据(30,6)放入traj_subgraphnet

            # 每个batch里有多个vec_map（轨迹点周围的地图），每个vec_map都是(18,8)
            for vec_map in vectormap_batch:
                # mf.append(vec_map[0].numpy())
                vec_map = vec_map.to(device=self.cfg['device'], dtype=torch.float)
                map_feature = self.map_subgraphnet(vec_map.squeeze())
                polyline_list.append(map_feature.unsqueeze(0))

            # mapfeature = np.vstack(mf)

            # for vec_map in vectormap_list:
            #     vec_map = vec_map.to(device=self.cfg['device'], dtype=torch.float)
            #     map_feature = self.map_subgraphnet(vec_map)  # vector map 放入 map_subgraphnet
            #     polyline_list.append(map_feature.unsqueeze(0))
            polyline_feature = F.normalize(torch.cat(polyline_list, dim=0), p=2, dim=1)  # L2 Normalize, torch.Size([1+n, 128])
            out = self.graphnet(polyline_feature)   # [1+n, 64]
            decoded_data_perstep = self.fc2(F.relu(self.layer_norm(self.fc(out[0].unsqueeze(0))))).view(1, -1, 2)  # corresponding one, [1,19,2]
            decoded_data = torch.cumsum(decoded_data_perstep, dim=0)  # parameterized as per-step coordinate offsets
            predict_list.append(decoded_data)
        predict_batch = torch.cat(predict_list, dim=0)      # 纵向拼接，torch.Size([1, 19, 2])
        loss = self.loss_fn(predict_batch, label)
        if np.isnan(loss.item()):
            raise Exception("Loss ERROR!")
        # if (loss.item() < 0.00005):
        #     np.savetxt('pred.txt', predict_batch.cpu().detach().numpy()[0],fmt='%0.8f')
        #     np.savetxt('map.txt',mapfeature,fmt='%0.8f')
        #     np.savetxt('traj.txt',traj_out,fmt='%0.8f')
        #     sys.exit()
        return loss

    def _forward_test(self, trajectory_batch):
        batch_size = trajectory_batch.size()[0]

        traj_label = trajectory_batch[:, self.cfg['last_observe']:, 2:4]   # 2nd and 3rd columes as traj label
        # predict_list = []
        result, label = dict(), dict()
        for i in range(batch_size):
            polyline_feature = self.traj_subgraphnet(trajectory_batch[i, :self.cfg['last_observe']]).unsqueeze(0)
            # print(polyline_feature)
            polyline_feature = F.normalize(polyline_feature, p=2, dim=1)  # L2 Normalize
            # print(polyline_feature)
            out = self.graphnet(polyline_feature)
            decoded_data_perstep = self.fc2(F.relu(self.layer_norm(self.fc(out)))).view(-1, 2)  # corresponding one
            decoded_data = torch.cumsum(decoded_data_perstep, dim=0)  # parameterized as per-step coordinate offsets
            # print(decoded_data.shape)
            key = str(trajectory_batch[i, 0, -1].int().item())
            # print(trajectory_batch[i, 0, -1])
            predict_step = self.cfg['predict_step']
            result.update({key: decoded_data[:predict_step]})
            label.update({key: traj_label[i, :predict_step]})
        return result, label

    def forward(self, trajectory, vectormap):
        if self.training: # 继承自nn.Module
            return self._forward_train(trajectory, vectormap)
        else:
            return self._forward_test(trajectory)
