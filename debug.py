import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
import networkx as nx
# import matplotlib.animation as animation
import matplotlib.pyplot as plt
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.vector_map_loader import load_lane_segments_from_xml
from ArgoverseDataset import ArgoverseForecastDataset
import warnings
warnings.filterwarnings('ignore')

argo_dst = ArgoverseForecastDataset()
train_loader = DataLoader(dataset= argo_dst, batch_size= 2, shuffle=True, num_workers=0)
# my_map = dst.generate_vector_map()
USE_GPU = False
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

for i, trajectory in enumerate(train_loader):
    b = 2
# map_fpath = "./data/map_files/pruned_argoverse_PIT_10314_vector_map.xml"
# tmp = load_lane_segments_from_xml(map_fpath)
print(device)
a = 1
