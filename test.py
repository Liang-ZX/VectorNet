#!/usr/bin/python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from ArgoverseDataset import ArgoverseForecastDataset
from vectornet import VectorNet
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import pprint
import time
import sys
import matplotlib.pyplot as plt

def render_traj(traj_batch):
    print(traj_batch)
    traj = traj_batch[0].cpu().numpy()
    rows = np.size(traj, 0)
    print(rows)
    for i in range(rows):
	    plt.annotate('', xy=(traj[:,2][i],traj[:,3][i]),xytext=(traj[:,0][i],traj[:,1][i]),arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
    plt.show()

def show_result(traj_batch, map_batch, single_result=dict()):
    print(traj_batch)
    # print(map_batch)
    # print(single_result)
    # sys.exit()
    traj = traj_batch[0].cpu().numpy()
    mv = []
    for vec_map in map_batch:
        vec_map = vec_map[0].cpu().numpy()
        vec_map = np.reshape(vec_map[:,0:4],(-1,8))
        mv.append(vec_map)
    for key in single_result:
        pred = single_result[key].cpu().numpy() / 10
    map_vec = np.vstack(mv)
    rows = np.size(map_vec, 0)
    map_count = rows // 9
    print(map_count)
    for i in range(map_count):
        plt.plot(map_vec[i*9:(i+1)*9,0], map_vec[i*9:(i+1)*9,1])
        plt.plot(map_vec[i*9:(i+1)*9,2], map_vec[i*9:(i+1)*9,3])
        plt.plot(map_vec[i*9:(i+1)*9,4], map_vec[i*9:(i+1)*9,5])
        plt.plot(map_vec[i*9:(i+1)*9,6], map_vec[i*9:(i+1)*9,7])
    
    rows = np.size(traj, 0)
    print(rows)
    for i in range(rows):
	    plt.annotate('', xy=(traj[:,2][i],traj[:,3][i]),xytext=(traj[:,0][i],traj[:,1][i]),arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))

    rows = np.size(pred, 0)
    for i in range(rows-1):
        length = np.sqrt(np.sum(np.square(pred[i+1] - pred[i])))
        plt.arrow(pred[i][0],pred[i][1],pred[i+1][0]-pred[i][0],pred[i+1][1]-pred[i][1],
                    length_includes_head=True, # 增加的长度包含箭头部分
                    head_width = length*0.125,
                    head_length = length*0.25,
                    width = length*0.03,
                    fc='r',
                    ec='b')
    plt.axis('equal')
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()

def main():
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    cfg = dict(device=device, last_observe=30, batch_size=1, predict_step=19,
               data_locate="/home/tangx2/storage/projects/git/argoverse-api/train/data_5000", save_path="./model_ckpt/inference/",
               model_path="./model_ckpt/model_final.pth")

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)
    print()

    if not os.path.isdir(cfg['save_path']):
        os.mkdir(cfg['save_path'])

    argo_dst = ArgoverseForecastDataset(cfg)
    val_loader = DataLoader(dataset=argo_dst, batch_size=cfg['batch_size'], shuffle=True, num_workers=0, drop_last=True)

    model = VectorNet(traj_features=4, map_features=8, cfg=cfg)
    model.to(device)

    # load from checkpoint
    # checkpoint = torch.load("./model_ckpt2/model_epoch10.pth")
    # model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
    # model.load_state_dict(checkpoint['model_state_dict'])

    # load from model_final
    # model.load_state_dict(torch.load(cfg['model_path']))
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(cfg['model_path']).items()})
    model.eval() # Sets training as false.

    start_time = time.strftime('%Y-%m-%d %X',time.localtime(time.time()))
    inference(model, cfg, val_loader)
    end_time = time.strftime('%Y-%m-%d %X',time.localtime(time.time()))
    print('start time -> ' + start_time)
    print('end time -> ' + end_time)


def inference(model, cfg, val_loader):
    device = cfg['device']
    result, label = dict(), dict()
    file_path = cfg['save_path'] + "inference.txt"
    file_handler = open(file_path, mode='w')
    pbar = tqdm(total=len(os.listdir(cfg['data_locate']))//2*2)
    pbar.set_description("Calculate Average Displacement Loss on Test Set")
    with torch.no_grad():
        for i, (traj_batch, map_batch) in enumerate(val_loader):
            traj_batch = traj_batch.to(device=device, dtype=torch.float)  # move to device, e.g. GPU
            single_result, single_label = model(traj_batch, map_batch)
            result.update(single_result)
            label.update(single_label)
            pbar.update(2)
            show_result(traj_batch, map_batch, single_result)
            print(result)
            print(label)
            # break
        pbar.close()
        print('length of result : ' + str(len(result)))
        print('length of label : ' + str(len(label)))
        predictions, loss = evaluate(val_loader.dataset, result, label)
        for (k, v) in predictions.items():
            file_handler.write("%06d: " % int(k))
            file_handler.writelines("[%.2f, %.2f], " % (i[0], i[1]) for i in v.tolist())
            file_handler.write("\n")
    print("-------------------TEST RESULT----------------------")
    print("ADE=", loss)


def evaluate(dataset, predictions, labels):
    loss_list = []
    pred_coordinate = dict()
    for key in predictions:
        city_name = dataset.city_name[key]
        max_coordinate = dataset.axis_range[city_name]['max']
        min_coordinate = dataset.axis_range[city_name]['min']
        rotate_matrix = dataset.rotate_matrix[key]
        center_xy = dataset.center_xy[key]
        tmp_prediction = predictions[key].cpu().numpy()*(max_coordinate-min_coordinate)/10
        tmp_label = labels[key].cpu().numpy()*(max_coordinate-min_coordinate)/10
        tmp_prediction = tmp_prediction.dot(rotate_matrix)
        tmp_label = tmp_label.dot(rotate_matrix)
        pred_coordinate.update({key: tmp_prediction+center_xy})

        loss_list.append(np.mean(np.sqrt(np.sum(np.square(tmp_prediction - tmp_label), axis=1))))
    loss = np.mean(np.array(loss_list))
    return pred_coordinate, loss


if __name__ == "__main__":
    main()