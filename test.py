#!/usr/bin/env python3
import argparse
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
import os
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import model
from detection_layers.modules import MultiBoxLoss
from test_dataset import TestSet
from dataset import DeepfakeDataset

from lib.util import load_config, update_learning_rate, my_collate, get_video_auc


def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='The path to the config.', default='./configs/caddm_test.cfg')
    args = parser.parse_args()
    return args


def load_checkpoint(ckpt, net, device):
    checkpoint = torch.load(ckpt)

    gpu_state_dict = OrderedDict()
    for k, v in checkpoint['network'] .items():
        name = "module." + k  # add `module.` prefix
        gpu_state_dict[name] = v.to(device)
    net.load_state_dict(gpu_state_dict)
    return net


def test():
    args = args_func()

    # load conifigs
    cfg = load_config(args.cfg)

    # init model.
    net = model.get(backbone=cfg['model']['backbone'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net = nn.DataParallel(net)
    net.eval()
    if cfg['model']['ckpt']:
        net = load_checkpoint(cfg['model']['ckpt'], net, device)

    # get testing data
    print(f"Load deepfake dataset from {cfg['dataset']['img_path']}..")

    test_dataset = DeepfakeDataset('test', cfg)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg['test']['batch_size'],
                             shuffle=True, num_workers=0,
                             )
    
    '''test_dataset = TestSet('./dataset.yaml')
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg['test']['batch_size'],
                             num_workers=1)'''

    # start testing.
    frame_pred_list = list()
    frame_label_list = list()
    video_name_list = list()

    for batch_data, batch_labels in test_loader:

        labels, video_name = batch_labels
        labels = labels.long()
        
        outputs = net(batch_data)
        outputs = outputs[:, 1]
        frame_pred_list.extend(outputs.detach().cpu().numpy().tolist())
        frame_label_list.extend(labels.detach().cpu().numpy().tolist())
        video_name_list.extend(list(video_name))

    #f_auc = roc_auc_score(frame_label_list, frame_pred_list)
    write_to_csv(f'TestSet', video_name_list, frame_pred_list, frame_label_list)
    #video_names, video_preds, video_labels = get_video_data(image_name_list, frame_pred_list, frame_label_list)
    #write_to_csv(f'TestSet_Unnormalized_video', video_names, video_preds, video_labels)
    #v_auc = get_video_auc(frame_label_list, video_name_list, frame_pred_list)
    #print(f"Frame-AUC of {cfg['dataset']['name']} is {f_auc:.4f}")
    #print(f"Video-AUC of {cfg['dataset']['name']} is {v_auc:.4f}")

def write_to_csv(name, img_names, y_pred, y_true):
    csv_name = f'{name}_results.csv'
    with open(csv_name, 'w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(['file', 'predicted', 'label'])
        
        for img_name, pred, label in zip(img_names, y_pred, y_true):
            writer.writerow([img_name, pred, label]) 

def get_video_data(image, pred, label):
    result_dict = {}
    new_label = []
    new_pred = []
    new_names = []

    for item in np.transpose(np.stack((image, pred, label)), (1, 0)):
        s = item[0]
        if '\\' in s:
            parts = s.split('\\')
        else:
            parts = s.split('/')
        a = parts[-2]
        b = parts[-1]

        if a not in result_dict:
            result_dict[a] = []

        result_dict[a].append(item)
    image_arr = list(result_dict.items())

    for video_name, video in image_arr:
        pred_sum = 0
        label_sum = 0
        leng = 0
        for frame in video:
            pred_sum += float(frame[1])
            label_sum += int(frame[2])
            leng += 1
        new_names.append(video_name)
        new_pred.append(pred_sum / leng)
        new_label.append(int(label_sum / leng))
    return (new_names, new_pred, new_label)

if __name__ == "__main__":
    test()

# vim: ts=4 sw=4 sts=4 expandtab
