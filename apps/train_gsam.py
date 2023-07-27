import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import json
import cv2 as cv
import numpy as np
from tqdm import tqdm
import pickle
import argparse
import yacs
import random
import torch.distributed as dist
import torch.multiprocessing as mp


from models.manolayer import ManoLayer

from utils.config import load_cfg
# from core.gcn_trainer import train_gcn
from core.gsam_trainer import train_gcn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='utils/defaults.yaml')
    parser.add_argument('--gpu', type=str, default='0')
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

    gpu_list = opt.gpu.split(',')
    num_gpus = len(gpu_list)
    dist_training = (num_gpus > 1)

    cfg = load_cfg(opt.cfg)

    if not os.path.isdir(cfg.SAVE.SAVE_DIR):
        os.makedirs(cfg.SAVE.SAVE_DIR, exist_ok=True)
    if not os.path.isdir(cfg.TB.SAVE_DIR):
        os.makedirs(cfg.TB.SAVE_DIR, exist_ok=True)
    with open(os.path.join(cfg.SAVE.SAVE_DIR, 'config.yaml'), 'w') as file:
        file.write(cfg.dump())

    if not dist_training:
        train_gcn(cfg=cfg)
    else:
        mp.spawn(train_gcn,
                 args=(num_gpus, cfg, True),
                 nprocs=num_gpus,
                 join=True)
