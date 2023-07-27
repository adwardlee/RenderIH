import json
import os.path as osp
from tqdm import tqdm
import cv2 as cv
import numpy as np
import torch
import pickle
import random
from glob import glob

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from models.manolayer import ManoLayer, rodrigues_batch
from dataset.dataset_utils import IMG_SIZE, HAND_BBOX_RATIO, HEATMAP_SIGMA, HEATMAP_SIZE, cut_img
from dataset.heatmap import HeatmapGenerator
# from utils.vis_utils import mano_two_hands_renderer
from utils.manoutils import get_mano_path


def fix_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1

class InterHand_mixsubset():
    def __init__(self, data_path, syns_path, split):
        assert split in ['train', 'test', 'val']
        self.split = split
        mano_path = get_mano_path()
        self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                           'left': ManoLayer(mano_path['left'], center_idx=None)}
        fix_shape(self.mano_layer)

        self.data_path = data_path
        self.size = len(glob(osp.join(data_path, split, 'anno', '*.pkl')))
        self.syns_path = syns_path
        name = 'train_subset5.pkl'### 1,10,20,30,40,50,60,70,80,90
        self.sample_idx = pickle.load(open(name, 'rb'))
        self.real_size = len(self.sample_idx)
        self.extra_syns = True

    def __len__(self):
        if self.extra_syns and self.split == 'train':
            return self.real_size + 1020000
        return self.real_size

    def __getitem__(self, idx):
        if self.extra_syns and self.split == 'train' and idx >= self.real_size:
            idx = idx - self.real_size + 366358
            img = cv.imread(osp.join(self.syns_path, self.split, 'color_img', '{}.jpg'.format(idx)))

            hand_dict = np.load(os.path.join(self.syns_path, self.split, 'ori_handdict', '{}.npy'.format(idx)),
                                allow_pickle=True)
        else:
            idx = self.sample_idx[idx]
            img = cv.imread(osp.join(self.data_path, self.split, 'img', '{}.jpg'.format(idx)))

            hand_dict = np.load(os.path.join(self.data_path, self.split, 'ori_handdict', '{}.npy'.format(idx)),
                                allow_pickle=True)
        hand_dict = hand_dict[()]
        return img, hand_dict
        # return img, mask, dense, hand_dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    opt = parser.parse_args()

