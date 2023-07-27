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
from utils.manoutils import get_mano_path


def fix_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1


class InterHand_other():
    def __init__(self, data_path, cfg, split):
        assert split in ['train', 'test', 'val']
        self.split = split
        mano_path = get_mano_path()
        self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                           'left': ManoLayer(mano_path['left'], center_idx=None)}
        fix_shape(self.mano_layer)

        self.data_path = data_path
        self.size = len(glob(osp.join(data_path, split, 'anno', '*.pkl')))
        self.black_path = cfg.BLACK_PATH
        self.syns_path = cfg.SYN_PATH
        self.ego3d_annot_path = cfg.EGO3D_PATH + 'annot/'#'/mnt/user/E-shenfei.llj-356552/workgroup/lijun/hand_dataset/ego3dhands/annot/'
        self.ego3d_img_path = cfg.EGO3D_PATH + 'img/'#'/mnt/user/E-shenfei.llj-356552/workgroup/lijun/hand_dataset/ego3dhands/img/'
        self.h2o3d_img_path = cfg.H2O3D_PATH + 'img/' #'/mnt/user/E-shenfei.llj-356552/workgroup/lijun/hand_dataset/h2o3d/refine/img/'
        self.h2o3d_annot_path = cfg.H2O3D_PATH + 'anno/' #'/mnt/user/E-shenfei.llj-356552/workgroup/lijun/hand_dataset/h2o3d/refine/anno/'
        self.single_path = cfg.SINGLE_PATH #'/mnt/user/E-shenfei.llj-356552/data/dataset/interhand_5fps/singlehand/'
        self.syns_img_path = ''
        self.syns_annot_path = ''
        self.sample_idx = random.sample(range(366358, 1020000 + 366358), 1020000)
        self.syns = False
        self.h2o3d = False
        self.ego3d = False
        self.black = False
        self.single = True

    def __len__(self):###58666 #### 39903
        if self.split == 'train':
            if self.h2o3d:
                self.size = 58666
                return 58666
            elif self.ego3d:
                self.size = 39003
                return 39903
            elif self.black:
                self.size = 40000
                return self.size
            elif self.single:
                self.size = 340000
                return self.size
            return 1020000
        return self.size

    def __getitem__(self, idx):
        if self.syns:
            cur_idx = self.sample_idx[idx - self.size]
            img = cv.imread(osp.join(self.syns_path, self.split, 'color_img', '{}.jpg'.format(cur_idx)))
            hand_dict = np.load(os.path.join(self.syns_path, self.split, 'ori_handdict', '{}.npy'.format(cur_idx)),
                                allow_pickle=True)
            hand_dict = hand_dict[()]
        elif self.h2o3d:
            img = cv.imread(self.h2o3d_img_path + '{}.jpg'.format(idx))
            hand_dict = pickle.load(open(self.h2o3d_annot_path + '{}.pkl'.format(idx), 'rb'))
            hand_dict['left']['verts2d'] = hand_dict['left']['verts2d'][:, :2]
            hand_dict['right']['verts2d'] = hand_dict['right']['verts2d'][:, :2]
        elif self.ego3d:
            img = cv.imread(self.ego3d_img_path + '{}.jpg'.format(idx))
            hand_dict = np.load(self.ego3d_annot_path + '{}.npy'.format(idx), allow_pickle=True)
            hand_dict = hand_dict[()]
            hand_dict['left']['verts2d'] = hand_dict['left']['joints2d'][:, :2]
            hand_dict['right']['verts2d'] = hand_dict['right']['joints2d'][:, :2]
            # img = cv.imread(osp.join(self.syns_path, self.split, 'color_img', '{}.jpg'.format(idx)))
            # hand_dict = np.load(os.path.join(self.data_path, self.split, 'ori_handdict', '{}.npy'.format(idx)), allow_pickle=True)
            # hand_dict = hand_dict[()]
        elif self.black:
            img = cv.imread(self.black_path.replace('black_all', 'black_img') + '{}.jpg'.format(idx))
            hand_dict = np.load(self.black_path + '{}.npy'.format(idx), allow_pickle=True)
            hand_dict = hand_dict[()]
            hand_dict['left']['verts2d'] = (hand_dict['left']['verts2d'][:, :2] + 1) / 2 * 256
            hand_dict['right']['verts2d'] = (hand_dict['right']['verts2d'][:, :2] + 1) / 2 * 256
            hand_dict['left']['joints2d'][:, :2] = (hand_dict['left']['joints2d'][:, :2] + 1 ) / 2 * 256
            hand_dict['right']['joints2d'][:, :2] = (hand_dict['right']['joints2d'][:, :2] + 1) /2 * 256
            hand_dict['left']['pose'] = hand_dict['left']['mano_pose']
            hand_dict['left']['shape'] = hand_dict['left']['mano_shape']
            hand_dict['right']['pose'] = hand_dict['right']['mano_pose']
            hand_dict['right']['shape'] = hand_dict['right']['mano_shape']
        elif self.single:
            cur_idx = idx
            img = cv.imread(osp.join(self.single_path, self.split, 'img', '{}.jpg'.format(cur_idx)))
            hand_dict = np.load(os.path.join(self.single_path, self.split, 'ori_handdict', '{}.npy'.format(cur_idx)),
                                allow_pickle=True)
            hand_dict = hand_dict[()]
        hand_dict['left']['joints2d'] = hand_dict['left']['joints2d'][:, :2]
        hand_dict['right']['joints2d'] = hand_dict['right']['joints2d'][:, :2]

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
