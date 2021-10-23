#!/usr/bin/python
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
import warnings
warnings.filterwarnings('ignore')
from common import *
import pprint
import time
import numpy as np

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
               last_observe=30, epochs=12, print_every=100, save_every=2, batch_size=1,
               data_locate="/home/tangx2/storage/projects/git/argoverse-api/train/data_5000", save_path="./model_ckpt/", # /workspace/argoverse-api/train/data 
               log_file="./log.txt", tensorboard_path="runs/train_visualization")

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)
    print('RUN_PARALLEL = ' + str(RUN_PARALLEL))
    print()
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if not os.path.isdir(cfg['save_path']):
        os.mkdir(cfg['save_path'])
    writer = SummaryWriter(cfg['tensorboard_path'])
    logger = init_logger(cfg)

    argo_dst = ArgoverseForecastDataset(cfg)
    train_loader = DataLoader(dataset=argo_dst, batch_size=cfg['batch_size'], shuffle=True, num_workers=0, drop_last=True)

    model = VectorNet(traj_features=4, map_features=8, cfg=cfg)
    model.to(device)
    model.train() # Sets the module in training mode.
    if RUN_PARALLEL:
        model = nn.DataParallel(model, device_ids=device_ids)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.Adadelta(model.parameters(), rho=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=learning_rate_decay)  # MultiStepLR TODO

    logger.info("Start Training...")
    do_train(model, cfg, train_loader, optimizer, scheduler=scheduler, writer=writer, logger=logger)

def do_train(model, cfg, train_loader, optimizer, scheduler, writer, logger):
    start_time = time.strftime('%Y-%m-%d %X',time.localtime(time.time()))
    device = cfg['device']
    print_every = cfg['print_every']
    for e in range(cfg['epochs']):
        # print('map_batch.shape:')
        for i, (traj_batch, map_batch) in enumerate(train_loader):
            # print(len(map_batch))
            # print(i)
            # print(traj_batch.size())
            # print(map_batch.size())
            # print('---')
            traj_batch = traj_batch.to(device=device, dtype=torch.float)  # move to device, e.g. GPU
            loss = model(traj_batch, map_batch).sum()   # 调用VectorNet.forward()函数
            # save loss

            optimizer.zero_grad()   # 先将梯度归零
            loss.backward()         # 反向传播计算得到每个参数的梯度值
            optimizer.step()        # 通过梯度下降执行一步参数更新

            if i % print_every == 0:
                logger.info('Epoch %d/%d: Iteration %d, loss = %f' % (e+1, cfg['epochs'], i, loss.item()))
                writer.add_scalar('training_loss', loss.item(), e)
        scheduler.step()

        if (e+1) % cfg['save_every'] == 0:
            file_path = cfg['save_path'] + "model_epoch" + str(e+1) + ".pth"
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss
            }, file_path)
            logger.info("Save model "+file_path)
    torch.save(model.state_dict(), cfg['save_path']+"model_final.pth")
    logger.info("Save final model "+cfg['save_path']+"model_final.pth")
    logger.info("Finish Training")
    end_time = time.strftime('%Y-%m-%d %X',time.localtime(time.time()))
    print('start time -> ' + start_time)
    print('end time -> ' + end_time)


def init_logger(cfg):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    log_file = cfg['log_file']
    handler = logging.FileHandler(log_file, mode='w')
    handler.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


if __name__ == "__main__":
    main()