#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import os
from ArgoverseDataset import ArgoverseForecastDataset
from vectornet import VectorNet
import logging

def main():
    USE_GPU = True
    RUN_PARALLEL = True
    device_ids = [0, 1]
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        if torch.cuda.device_count() <= 1:
            RUN_PARALLEL = False
            pass
    else:
        device = torch.device('cpu')
        RUN_PARALLEL = False

    learning_rate = 1e-3
    learning_rate_decay = 0.3
    cfg = dict(device=device, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay,
               last_observe=10, epochs=10, print_every=10, save_every=2, batch_size=2,
               data_locate="/home/tangx2/storage/projects/git/argoverse-api/train/data_10", save_path="./model_ckpt/", # /workspace/argoverse-api/train/data 
               log_file="./log.txt", tensorboard_path="runs/train_visualization")
    
    argo_dst = ArgoverseForecastDataset(cfg)
    train_loader = DataLoader(dataset=argo_dst, batch_size=cfg['batch_size'], shuffle=True, num_workers=0, drop_last=True)

    for i, (traj_batch, map_batch) in enumerate(train_loader):
        trajectory_batch = traj_batch
        batch_size = trajectory_batch.size()[0]
        # print(trajectory_batch)
        # print(trajectory_batch.size())
        # print(batch_size)
        break

if __name__ == "__main__":
    main()