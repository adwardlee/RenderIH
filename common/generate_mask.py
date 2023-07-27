import json
import os.path as osp
from tqdm import tqdm
import cv2 as cv
import numpy as np
import torch
import pickle
from glob import glob
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from matplotlib import pyplot as plt
import cv2
from common.utils.mano import MANO


color_hand_joints = [[1.0, 0.0, 0.0],
                     [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                     [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                     [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                     [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                     [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little

def fix_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1

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

def render_data(save_path, data_path, split):
    mano_path = get_mano_path()
    os.makedirs(osp.join(save_path, split, 'hms'), exist_ok=True)

    size = len(glob(osp.join(data_path, split, 'anno', '*.pkl')))
    mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                  'left': ManoLayer(mano_path['left'], center_idx=None)}
    fix_shape(mano_layer)

    for idx in tqdm(range(size)):
        img = cv.imread(osp.join(data_path, split, 'img', '{}.jpg'.format(idx)))
        img = cv2.resize(img, (224, 224))
        with open(osp.join(data_path, split, 'anno', '{}.pkl'.format(idx)), 'rb') as file:
            data = pickle.load(file)

        R = data['camera']['R']
        T = data['camera']['t']
        camera = data['camera']['camera']
        camera[0, 0] = camera[0, 0] * 224 / 256
        camera[1, 1] = camera[1, 1] * 224 / 256
        camera[0, 2] = camera[0, 2] * 224 / 256
        camera[1, 2] = camera[1, 2] * 224 / 256

        for hand_type in ['left', 'right']:
            params = data['mano_params'][hand_type]
            handV, handJ = mano_layer[hand_type](torch.from_numpy(params['R']).float(),
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

            # verts.append(torch.from_numpy(handV).float().cuda().unsqueeze(0))
            # hms = np.split(hmg(handJ2d * HEATMAP_SIZE / IMG_SIZE)[0], 7)  # 21 x h x w
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.imshow(img)
            plot_2d_hand(ax1, handJ2d[:, :2], order='uv')
            fig.savefig(os.path.join(save_path, split, 'hms', '{}_{}.jpg'.format(idx, hand_type)),)




class InterHand_dataset(Dataset):
    def __init__(self, data_path, split, img_size, transforms=None):
        assert split in ['train', 'test', 'val']
        self.split = split
        print('load dataset {} '.format(split), flush=True)
        mano_left = MANO(hand_type='left')
        mano_right = MANO(hand_type='right')
        self.mano_layer = {'right': mano_right.layer,
                           'left': mano_left.layer}

        self.data_path = data_path
        self.size = len(glob(osp.join(data_path, split, 'anno', '*.pkl')))
        self.mean = [0.485, 0.456, 0.406]
        self.std= [0.229, 0.224, 0.225]

        self.transforms = transforms
        self.img_size = img_size[0]

    def mask_processing(self, mask):
        ### 64,64, 3
        outleft = mask[:, :, 2] / 255.0
        outright = mask[:, :, 1] / 255.0
        return outleft, outright

    def __len__(self):
        return self.size

    # def __getitem__(self, idx):
    #     thearray = np.load(osp.join(self.data_path, self.split, '{}.npy'.format(idx)))
    #     out = {}
    #     out['img'] = thearray[()]['img']

    def __getitem__(self, idx):
        img = cv.imread(osp.join(self.data_path, self.split, 'img', '{}.jpg'.format(idx)))
        mask = cv.imread(osp.join(self.data_path, self.split, 'mask', '{}.jpg'.format(idx)))
        dense = cv.imread(osp.join(self.data_path, self.split, 'dense', '{}.jpg'.format(idx)))
        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size // 4, self.img_size // 4))
        left_mask = mask[:,:,2:3] / 255.0
        right_mask = mask[:,:,1:2] / 255.0
        mask = {}
        mask['left'] = left_mask.transpose(2,0,1)
        mask['right'] = right_mask.transpose(2, 0, 1)
        dense = cv2.resize(dense, (self.img_size // 4, self.img_size // 4)) / 255.0
        dense = dense.transpose(2, 0, 1)
        img = cv2.cvtColor(img, cv.COLOR_BGR2RGB)
        assert img.shape[0] == self.img_size
        # mask = cv.imread(osp.join(self.data_path, self.split, 'mask', '{}.jpg'.format(idx)))
        # dense = cv.imread(osp.join(self.data_path, self.split, 'dense', '{}.jpg'.format(idx)))

        with open(os.path.join(self.data_path, self.split, 'anno', '{}.pkl'.format(idx)), 'rb') as file:
            data = pickle.load(file)

        R = data['camera']['R']
        T = data['camera']['t']
        camera = data['camera']['camera']
        camera[0, 0] = camera[0, 0] * self.img_size / 256
        camera[1, 1] = camera[1, 1] * self.img_size / 256
        camera[0, 2] = camera[0, 2] * self.img_size / 256
        camera[1, 2] = camera[1, 2] * self.img_size / 256

        joint_valid = np.array(data['annotation']['joint_valid']).reshape(2, 21)
        filename = data['image']['file_name']
        # left_valid = joint_valid[1]
        # right_valid = joint_valid[0]

        hand_dict = {}
        inputs = {}
        hms = {'right': [], 'left': []}
        for hand_idx, hand_type in enumerate(['right', 'left']):

            for hIdx in range(7):
                hm = cv.imread(os.path.join(self.data_path, self.split, 'hms', '{}_{}_{}.jpg'.format(idx, hIdx, hand_type)))
                hm = cv.resize(hm, (img.shape[1] // 4, img.shape[0] // 4)).transpose(2, 0, 1) / 255.0
                hms[hand_type].append(hm)
            hms[hand_type] = np.concatenate(hms[hand_type])
            params = data['mano_params'][hand_type]
            root_pose = cv.Rodrigues(params['R'][0])[0].reshape(-1)
            all_pose = np.concatenate([root_pose, params['pose'][0]], axis=0).reshape(1, 48).astype(np.float32)
            handV, handJ = self.mano_layer[hand_type](torch.from_numpy(params['R']).float(),
                                                     torch.from_numpy(params['pose']).float(),
                                                      torch.from_numpy(params['shape']).float(),
                                                      trans=torch.from_numpy(params['trans']).float())
            handV = handV / 1000
            handJ = handJ / 1000
            handV = handV[0].float().cpu().numpy()
            handJ = handJ[0].float().cpu().numpy()
            handV = handV @ R.T + T
            handJ = handJ @ R.T + T

            handV2d = handV @ camera.T
            handV2d = handV2d[:, :2] / handV2d[:, 2:]
            handV2d = 2 * handV2d / self.img_size - 1.0
            handJ2d = handJ @ camera.T
            handJ2d = handJ2d[:, :2] / handJ2d[:, 2:]
            handJ2d = 2 * handJ2d / self.img_size - 1.0

            params['trans'][:] = 0
            orihandV, orihandJ = self.mano_layer[hand_type](torch.from_numpy(params['R']).float(),
                                                     torch.from_numpy(params['pose']).float(),
                                                      torch.from_numpy(params['shape']).float(),
                                                      trans=torch.from_numpy(params['trans']).float())
            orihandV = orihandV[0].cpu().numpy() / 1000
            orihandJ = orihandJ[0].cpu().numpy() / 1000
            hand_dict[hand_type] = {'hms': hms[hand_type].astype(np.float32),
                                    'verts3d': orihandV, 'joints3d': orihandJ,
                                    'verts2d': handV2d, 'joints2d': handJ2d,
                                    'joints_img': ((handJ2d + 1) / 2),
                                    'R': params['R'][0],
                                    'root_rotmat': params['R'][0], ### 3
                                    'mano_pose': all_pose[0],
                                    'mano_shape': params['shape'][0].astype(np.float32),
                                    'camera': camera,
                                    'joint_valid': joint_valid[hand_idx],
                                    'w_smpl': 1,
                                    'mask': mask[hand_type].astype(np.float32),
                                    'dense': dense.astype(np.float32),
                                    # 'cam': camera.astype(np.float32),
                                    # 'cam_R': R.astype(np.float32),
                                    # 'cam_T': T.astype(np.float32),
                                    # 'trans': params['trans'][0].astype(np.float32),
                                    }

        img = torch.tensor(img) / 255
        img = img.permute(2, 0, 1)
        img = F.normalize(img, self.mean, self.std)
        inputs['img'] = img
        hand_dict['filename'] = filename
        if self.transforms:
            hand_dict = self.transforms(hand_dict)
        hand_dict['img'] = img
        np.save(osp.join(self.data_path, self.split, 'all', '{}'.format(idx)), hand_dict)
        return inputs, hand_dict
        # return img, mask, dense, hand_dict


if __name__ == '__main__':
    import argparse
    from main.config import cfg
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    opt = parser.parse_args()

    # for split in ['train', 'test', 'val']:
    #     select_data(opt.data_path, opt.save_path, split=split)

    # for split in ['train', 'test', 'val']:
    #     render_data(opt.save_path, opt.data_path, split)
    dataset = InterHand_dataset(cfg.data_dir, split='train', img_size=cfg.input_img_shape)
    batch_generator = DataLoader(dataset, batch_size=cfg.train_batch_size, shuffle=False,
                                      num_workers=24, pin_memory=True)
    for itr, (inputs, targets) in enumerate(batch_generator):
        print('{} iter'.format(itr), flush=True)
