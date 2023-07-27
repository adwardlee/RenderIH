import json
import os.path as osp
from tqdm import tqdm
import cv2 as cv
import numpy as np
import torch
import pickle
from glob import glob
import cv2
from matplotlib import pyplot as plt

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from models.manolayer import ManoLayer, rodrigues_batch
from dataset.dataset_utils import IMG_SIZE, HAND_BBOX_RATIO, HEATMAP_SIGMA, HEATMAP_SIZE, cut_img
from dataset.heatmap import HeatmapGenerator
from utils.vis_utils import mano_two_hands_renderer
from utils.manoutils import get_mano_path


def fix_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1

color_hand_joints = [[1.0, 0.0, 0.0],
                     [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                     [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                     [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                     [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                     [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little

def plot_2d_hand(axis, coords_hw, vis=None, color_fixed=None, linewidth='1', order='hw', draw_kp=True):
    """ Plots a hand stick figure into a matplotlib figure. """
    if order == 'uv':
        coords_hw = coords_hw[:, ::-1]

    colors = np.array(color_hand_joints)
    # define connections and colors of the bones
    bones = [((0, 1), colors[1, :]),
             ((1, 2), colors[2, :]),
             ((2, 3), colors[3, :]),
             ((3, 4), colors[4, :]),

             ((0, 5), colors[5, :]),
             ((5, 6), colors[6, :]),
             ((6, 7), colors[7, :]),
             ((7, 8), colors[8, :]),

             ((0, 9), colors[9, :]),
             ((9, 10), colors[10, :]),
             ((10, 11), colors[11, :]),
             ((11, 12), colors[12, :]),

             ((0, 13), colors[13, :]),
             ((13, 14), colors[14, :]),
             ((14, 15), colors[15, :]),
             ((15, 16), colors[16, :]),

             ((0, 17), colors[17, :]),
             ((17, 18), colors[18, :]),
             ((18, 19), colors[19, :]),
             ((19, 20), colors[20, :])]

    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0

    for connection, color in bones:
        if (vis[connection[0]] == False) or (vis[connection[1]] == False):
            continue

        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 1], coords[:, 0], color_fixed, linewidth=linewidth)

    if not draw_kp:
        return

    for i in range(21):
        if vis[i] > 0.5:
            axis.plot(coords_hw[i, 1], coords_hw[i, 0], 'o', color=colors[i, :], markersize=3)
            axis.text(coords_hw[i, 1], coords_hw[i, 0], '{}'.format(i), fontsize=5, color='white')

class InterHandLoader():
    def __init__(self, data_path, split='train', mano_path=None):
        assert split in ['train', 'test', 'val']

        self.root_path = data_path
        # self.img_root_path = '/mnt/workspace/workgroup/lijun/hand_dataset/color_test/'
        # self.img_root_path = '/mnt/workspace/workgroup/lijun/hand_dataset/order_xinchuan_200wimg/' ####### change ###############
        # self.img_root_path = '/mnt/workspace/workgroup/lijun/hand_dataset/render_color/'#os.path.join(self.root_path, 'images')
        self.img_root_path = '/mnt/workspace/workgroup/lijun/hand_dataset/synthesis/sdf_xinchuan/render_img/'
        self.annot_root_path = '/mnt/workspace/workgroup/lijun/hand_dataset/synthesis/sdf_xinchuan/nodup_annot/'#os.path.join(self.root_path, 'annotations')

        self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                           'left': ManoLayer(mano_path['left'], center_idx=None)}
        fix_shape(self.mano_layer)

        self.split = split

        # self.size = len(glob(osp.join(data_path, split, 'xinchuan_200w/order_oriannot', '*.pkl'))) ##### change #@@@@@@@@@@@@@@@
        self.size = len(glob(os.path.join('/mnt/workspace/workgroup/lijun/hand_dataset/synthesis/sdf_xinchuan/nodup_annot/', '*.pkl')))

        # self.data_size = len(self.data_info['images'])

    def __len__(self):
        return self.size

    def load_img(self, idx):
        img = cv.imread(osp.join(self.img_root_path, '0_{}.png'.format(idx)))
        # img = cv.imread(osp.join(self.img_root_path, '{}.jpg'.format(idx)))
        return img

    def mano_trans(self, mano_params):
        out_params = mano_params.copy()
        for hand_type in ['left', 'right']:
            mano_param = mano_params[hand_type]
            mano_pose = torch.FloatTensor(mano_param['pose']).view(-1, 3)
            root_pose = mano_pose[0].view(1, 3)
            hand_pose = mano_pose[1:, :].view(1, -1)
            root_pose = rodrigues_batch(root_pose)

            mano = self.mano_layer[hand_type]
            mean_pose = mano.hands_mean
            hand_pose = mano.axis2pca(hand_pose + mean_pose)
            out_params[hand_type]['oripose'] = mano_param['pose']#.numpy()
            out_params[hand_type]['pose'] = hand_pose.numpy()
            out_params[hand_type]['R'] = root_pose.numpy()
        return out_params


def cut_inter_img(loader, save_path, split, data_path):
    os.makedirs(osp.join(save_path, split, 'color_img/'), exist_ok=True) ##### change #@@@@@@@@@@@@@@@
    os.makedirs(osp.join(save_path, split, 'color_annot/'), exist_ok=True)##### change #@@@@@@@@@@@@@@@

    #idx = 0
    print('start {} - end {}'.format(1536358, 1586358), flush=True) #### 1386358 ~ 1606358
    for idx in tqdm(range(1536358, 1586358)): ### 366358# len(loader) ###660000 690000  ### 1015851
        with open(loader.annot_root_path + '{}.pkl'.format(idx), 'rb') as file:
        # with open(os.path.join(data_path, split, 'xinchuan_sdf/order_oriannot/', '{}.pkl'.format(idx)), 'rb') as file:##### change #@@@@@@@@@@@@@@@
        # with open(os.path.join(data_path, split, 'anno', '{}.pkl'.format(idx)), 'rb') as file:
            annot = pickle.load(file)
        mano_param = annot['mano_params']
        out_mano_param = loader.mano_trans(mano_param)
        left_param = out_mano_param['left']
        right_param = out_mano_param['right']
        cam_R = annot['camera']['R'].reshape(3, 3)
        cam_t = annot['camera']['t'].reshape(1, 3)
        cameraIn = annot['camera']['camera']
        img = loader.load_img(idx)
        if img is None:
            print('no img id ', idx, flush=True)
            continue

        left_vert, left_j3d = loader.mano_layer['left'](root_rotation=torch.from_numpy(left_param['R']),
                                        pose=torch.from_numpy(left_param['pose']),
                                        shape=torch.from_numpy(left_param['shape']),
                                        trans=torch.from_numpy(left_param['trans']))
        right_vert, right_j3d = loader.mano_layer['right'](root_rotation=torch.from_numpy(right_param['R']),
                                           pose=torch.from_numpy(right_param['pose']),
                                           shape=torch.from_numpy(right_param['shape']),
                                           trans=torch.from_numpy(right_param['trans']))

        left = left_j3d[0].numpy() @ cam_R.T + cam_t
        left2d = left @ cameraIn.T
        left2d = left2d[:, :2] / left2d[:, 2:]
        right = right_j3d[0].numpy() @ cam_R.T + cam_t
        right2d = right @ cameraIn.T
        right2d = right2d[:, :2] / right2d[:, 2:]

        ##### ori annot ####
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax1.imshow(img[:,:,::-1])
        # plot_2d_hand(ax1, left2d[:, :2], order='uv')
        # # fig.savefig(os.path.join('./tmp', 'ori_{}_{}.jpg'.format(idx, 'left')), )
        # plot_2d_hand(ax1, right2d[:, :2], order='uv')
        # fig.savefig(os.path.join('./tmp', 'ori_{}_{}.jpg'.format(idx, 'right')), )
        # plt.close()
        ################


        [img], _, cameraIn = \
            cut_img([img], [left2d, right2d], camera=cameraIn, radio=HAND_BBOX_RATIO, img_size=IMG_SIZE, bbox_gen=False)
        cv.imwrite(osp.join(save_path, split, 'color_img', '{}.jpg'.format(idx)), img) ##### change #@@@@@@@@@@@@@@@
        # cv.imwrite(osp.join('./tmp', '{}.jpg'.format(idx)), img)

        ##### after annot #####
        # left2d = left @ cameraIn.T
        # left2d = left2d[:, :2] / left2d[:, 2:]
        # right2d = right @ cameraIn.T
        # right2d = right2d[:, :2] / right2d[:, 2:]
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax1.imshow(img[:,:,::-1])
        # plot_2d_hand(ax1, left2d[:, :2], order='uv')
        # # fig.savefig(os.path.join('./tmp', 'after_{}_{}.jpg'.format(idx, 'left')), )
        # plot_2d_hand(ax1, right2d[:, :2], order='uv')
        # fig.savefig(os.path.join('./tmp', 'after_{}_{}.jpg'.format(idx, 'right')), )
        # plt.close()
        ############################

        data_info = {}
        data_info['inter_idx'] = idx
        data_info['image'] = annot['image']
        data_info['annotation'] = annot['annotation']
        data_info['mano_params'] = out_mano_param
        data_info['camera'] = {'R': cam_R, 't': cam_t, 'camera': cameraIn}
        with open(osp.join(save_path, split, 'color_annot/', '{}.pkl'.format(idx)), 'wb') as file: ##### change #@@@@@@@@@@@@@@@
            pickle.dump(data_info, file)
        # with open(osp.join(save_path, split, 'bbox', '{}.pkl'.format(idx)), 'wb') as file:
        #     pickle.dump(bbox_info, file)
        #idx = idx + 1


def select_data(DATA_PATH, save_path, split):
    loader = InterHandLoader(DATA_PATH, split=split, mano_path=get_mano_path())
    print(' loader length ', len(loader), flush=True)
    cut_inter_img(loader, save_path, split, DATA_PATH)



class InterHand_dataset():
    def __init__(self, data_path, split):
        assert split in ['train', 'test', 'val']
        self.split = split
        mano_path = get_mano_path()
        self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                           'left': ManoLayer(mano_path['left'], center_idx=None)}
        fix_shape(self.mano_layer)

        self.data_path = data_path
        self.size = len(glob(osp.join(data_path, split, 'anno', '*.pkl')))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = cv.imread(osp.join(self.data_path, self.split, 'img', '{}.jpg'.format(idx)))
        mask = cv.imread(osp.join(self.data_path, self.split, 'mask', '{}.jpg'.format(idx)))
        dense = cv.imread(osp.join(self.data_path, self.split, 'dense', '{}.jpg'.format(idx)))

        with open(os.path.join(self.data_path, self.split, 'anno', '{}.pkl'.format(idx)), 'rb') as file:
            data = pickle.load(file)

        R = data['camera']['R']
        T = data['camera']['t']
        camera = data['camera']['camera']

        hand_dict = {}
        for hand_type in ['left', 'right']:
            hms = []
            for hIdx in range(7):
                hm = cv.imread(os.path.join(self.data_path, self.split, 'hms', '{}_{}_{}.jpg'.format(idx, hIdx, hand_type)))
                hm = cv.resize(hm, (img.shape[1], img.shape[0]))
                hms.append(hm)

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

            hand_dict[hand_type] = {'hms': hms,
                                    'verts3d': handV, 'joints3d': handJ,
                                    'verts2d': handV2d, 'joints2d': handJ2d,
                                    'R': R @ params['R'][0],
                                    'pose': params['pose'][0],
                                    'shape': params['shape'][0],
                                    'camera': camera
                                    }

        return img, mask, dense, hand_dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/mnt/workspace/workgroup/lijun/hand_dataset/synthesis/sdf_xinchuan/')
    parser.add_argument("--save_path", type=str, default='/mnt/workspace/workgroup/lijun/hand_dataset/synthesis/sdf_xinchuan/')
    opt = parser.parse_args()

    for split in ['train']:#, 'test', 'val']:
        select_data(opt.data_path, opt.save_path, split=split)


