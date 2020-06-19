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


def main():
    USE_GPU = True
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    cfg = dict(device=device, last_observe=30, batch_size=2, predict_step=5,
               data_locate="./data/forecasting_dataset/val/", save_path="./model_ckpt/inference/",
               model_path="./model_ckpt/model_final.pth")

    if not os.path.isdir(cfg['save_path']):
        os.mkdir(cfg['save_path'])

    argo_dst = ArgoverseForecastDataset(cfg)
    val_loader = DataLoader(dataset=argo_dst, batch_size=cfg['batch_size'], shuffle=True, num_workers=0, drop_last=True)

    model = VectorNet(traj_features=6, map_features=8, cfg=cfg)
    model.to(device)

    # load from checkpoint
    # checkpoint = torch.load("./model_ckpt2/model_epoch10.pth")
    # model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
    # model.load_state_dict(checkpoint['model_state_dict'])

    # load from model_final
    # model.load_state_dict(torch.load(cfg['model_path']))
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(cfg['model_path']).items()})
    model.eval()

    inference(model, cfg, val_loader)


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
        pbar.close()
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