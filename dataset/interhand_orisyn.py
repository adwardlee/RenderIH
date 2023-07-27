import json
import os.path as osp
from tqdm import tqdm
import cv2 as cv
import numpy as np
import torch
import pickle
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

class InterHand_orisyn():
    def __init__(self, data_path, syn_path, split):
        assert split in ['train', 'test', 'val']
        self.split = split
        mano_path = get_mano_path()
        self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                           'left': ManoLayer(mano_path['left'], center_idx=None)}
        fix_shape(self.mano_layer)

        self.data_path = data_path
        self.size = len(glob(osp.join(data_path, split, 'anno', '*.pkl')))
        self.syns_path = syn_path
        self.extra_syns = False
        self.imgdir = 'color_img' ### color_img
        self.anno = 'color_anno' #### color_anno

    def __len__(self):
        if self.extra_syns and self.split == 'train':
            return self.size + 366358
        return self.size

    def __getitem__(self, idx):
        if self.extra_syns and self.split == 'train' and idx >= self.size:
            idx = idx - self.size
            img = cv.imread(osp.join(self.syns_path, self.split, self.imgdir, '{}.jpg'.format(idx)))
            with open(os.path.join(self.syns_path, self.split, self.anno, '{}.pkl'.format(idx)), 'rb') as file:
                data = pickle.load(file)
            # alldict = np.load(os.path.join(self.syns_path, self.split, 'black_all', '{}.npy'.format(idx)), allow_pickle=True)
            # all_dict = alldict[()]
            # hand_dict = {}
            # hand_dict['left'] = all_dict['left']
            # hand_dict['right'] = all_dict['right']
            # return img, hand_dict
        else:
            img = cv.imread(osp.join(self.data_path, self.split, 'img', '{}.jpg'.format(idx)))
        # mask = cv.imread(osp.join(self.data_path, self.split, 'mask', '{}.jpg'.format(idx)))
        # dense = cv.imread(osp.join(self.data_path, self.split, 'dense', '{}.jpg'.format(idx)))

            with open(os.path.join(self.data_path, self.split, 'anno', '{}.pkl'.format(idx)), 'rb') as file:
                data = pickle.load(file)

        R = data['camera']['R']
        T = data['camera']['t']
        camera = data['camera']['camera']

        hand_dict = {}
        for hand_type in ['left', 'right']:
            # hms = []
            # for hIdx in range(7):
            #     hm = cv.imread(os.path.join(self.data_path, self.split, 'hms', '{}_{}_{}.jpg'.format(idx, hIdx, hand_type)))
            #     hm = cv.resize(hm, (img.shape[1], img.shape[0]))
            #     hms.append(hm)

            params = data['mano_params'][hand_type]
            handV, handJ = self.mano_layer[hand_type](torch.from_numpy(params['R']).float(),
                                                      torch.from_numpy(params['pose']).float(),
                                                      torch.from_numpy(params['shape']).float(),
                                                      trans=torch.from_numpy(params['trans']).float())
            handV = handV[0].numpy()
            handJ = handJ[0].numpy()
            handV = handV @ R.T + T
            handJ = handJ @ R.T + T

            handV2d = handV @ camera.T
            handV2d = handV2d[:, :2] / handV2d[:, 2:]
            handJ2d = handJ @ camera.T
            handJ2d = handJ2d[:, :2] / handJ2d[:, 2:]

            hand_dict[hand_type] = {'hms': [],
                                    'verts3d': handV, 'joints3d': handJ,
                                    'verts2d': handV2d, 'joints2d': handJ2d,
                                    'R': R @ params['R'][0],
                                    'pose': params['pose'][0],
                                    'shape': params['shape'][0],
                                    'camera': camera
                                    }
        return img, hand_dict
        # return img, mask, dense, hand_dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    opt = parser.parse_args()

    for split in ['train', 'test', 'val']:
        select_data(opt.data_path, opt.save_path, split=split)

    for split in ['train', 'test', 'val']:
        render_data(opt.save_path, split)
