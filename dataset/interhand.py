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


class InterHandLoader():
    def __init__(self, data_path, split='train', mano_path=None):
        assert split in ['train', 'test', 'val']

        self.root_path = data_path
        self.img_root_path = os.path.join(self.root_path, 'images')
        self.annot_root_path = os.path.join(self.root_path, 'annotations')

        self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                           'left': ManoLayer(mano_path['left'], center_idx=None)}
        fix_shape(self.mano_layer)

        self.split = split

        with open(osp.join(self.annot_root_path, self.split,
                           'InterHand2.6M_' + self.split + '_data.json')) as f:
            self.data_info = json.load(f)
        with open(osp.join(self.annot_root_path, self.split,
                           'InterHand2.6M_' + self.split + '_camera.json')) as f:
            self.cam_params = json.load(f)
        with open(osp.join(self.annot_root_path, self.split,
                           'InterHand2.6M_' + self.split + '_joint_3d.json')) as f:
            self.joints = json.load(f)
        with open(osp.join(self.annot_root_path, self.split,
                           'InterHand2.6M_' + self.split + '_MANO_NeuralAnnot.json')) as f:
            self.mano_params = json.load(f)

        self.data_size = len(self.data_info['images'])

    def __len__(self):
        return self.data_size

    def show_data(self, idx):
        for k in self.data_info['images'][idx].keys():
            print(k, self.data_info['images'][idx][k])
        for k in self.data_info['annotations'][idx].keys():
            print(k, self.data_info['annotations'][idx][k])

    def load_camera(self, idx):
        img_info = self.data_info['images'][idx]
        capture_idx = img_info['capture']
        cam_idx = img_info['camera']

        capture_idx = str(capture_idx)
        cam_idx = str(cam_idx)
        cam_param = self.cam_params[str(capture_idx)]
        cam_t = np.array(cam_param['campos'][cam_idx], dtype=np.float32).reshape(3)
        cam_R = np.array(cam_param['camrot'][cam_idx], dtype=np.float32).reshape(3, 3)
        cam_t = -np.dot(cam_R, cam_t.reshape(3, 1)).reshape(3) / 1000  # -Rt -> t

        # add camera intrinsics
        focal = np.array(cam_param['focal'][cam_idx], dtype=np.float32).reshape(2)
        princpt = np.array(cam_param['princpt'][cam_idx], dtype=np.float32).reshape(2)
        cameraIn = np.array([[focal[0], 0, princpt[0]],
                             [0, focal[1], princpt[1]],
                             [0, 0, 1]])
        return cam_R, cam_t, cameraIn

    def load_mano(self, idx):
        img_info = self.data_info['images'][idx]
        capture_idx = img_info['capture']
        frame_idx = img_info['frame_idx']

        capture_idx = str(capture_idx)
        frame_idx = str(frame_idx)
        mano_dict = {}
        coord_dict = {}
        for hand_type in ['left', 'right']:
            try:
                mano_param = self.mano_params[capture_idx][frame_idx][hand_type]
                mano_pose = torch.FloatTensor(mano_param['pose']).view(-1, 3)
                root_pose = mano_pose[0].view(1, 3)
                hand_pose = mano_pose[1:, :].view(1, -1)
                # hand_pose = hand_pose.view(1, -1, 3)
                mano = self.mano_layer[hand_type]
                mean_pose = mano.hands_mean
                hand_pose = mano.axis2pca(hand_pose + mean_pose)
                shape = torch.FloatTensor(mano_param['shape']).view(1, -1)
                trans = torch.FloatTensor(mano_param['trans']).view(1, 3)
                root_pose = rodrigues_batch(root_pose)

                handV, handJ = self.mano_layer[hand_type](root_pose, hand_pose, shape, trans=trans)
                mano_dict[hand_type] = {'R': root_pose.numpy(), 'pose': hand_pose.numpy(), 'shape': shape.numpy(), 'trans': trans.numpy()}
                coord_dict[hand_type] = {'verts': handV, 'joints': handJ}
            except:
                mano_dict[hand_type] = None
                coord_dict[hand_type] = None

        return mano_dict, coord_dict

    def load_img(self, idx):
        img_info = self.data_info['images'][idx]
        img = cv.imread(osp.join(self.img_root_path, self.split, img_info['file_name']))
        return img


def cut_inter_img(loader, save_path, split):
    os.makedirs(osp.join(save_path, split, 'img'), exist_ok=True)
    os.makedirs(osp.join(save_path, split, 'anno'), exist_ok=True)

    idx = 0
    for i in tqdm(range(len(loader))):
        annotation = loader.data_info['annotations'][i]
        images_info = loader.data_info['images'][i]
        hand_type = annotation['hand_type']
        hand_type_valid = annotation['hand_type_valid']

        if hand_type == 'interacting' and hand_type_valid:
            mano_dict, coord_dict = loader.load_mano(i)
            if coord_dict['left'] is not None and coord_dict['right'] is not None:
                left = coord_dict['left']['verts'][0].detach().numpy()
                right = coord_dict['right']['verts'][0].detach().numpy()
                dist = np.linalg.norm(left - right, ord=2, axis=-1).min()
                if dist < 9999999:
                    img = loader.load_img(i)
                    if img.mean() < 10:
                        continue

                    cam_R, cam_t, cameraIn = loader.load_camera(i)
                    left = left @ cam_R.T + cam_t
                    left2d = left @ cameraIn.T
                    left2d = left2d[:, :2] / left2d[:, 2:]
                    right = right @ cam_R.T + cam_t
                    right2d = right @ cameraIn.T
                    right2d = right2d[:, :2] / right2d[:, 2:]

                    [img], _, cameraIn = \
                        cut_img([img], [left2d, right2d], camera=cameraIn, radio=HAND_BBOX_RATIO, img_size=IMG_SIZE)
                    cv.imwrite(osp.join(save_path, split, 'img', '{}.jpg'.format(idx)), img)

                    data_info = {}
                    data_info['inter_idx'] = idx
                    data_info['image'] = images_info
                    data_info['annotation'] = annotation
                    data_info['mano_params'] = mano_dict
                    data_info['camera'] = {'R': cam_R, 't': cam_t, 'camera': cameraIn}
                    with open(osp.join(save_path, split, 'anno', '{}.pkl'.format(idx)), 'wb') as file:
                        pickle.dump(data_info, file)

                    idx = idx + 1


def select_data(DATA_PATH, save_path, split):
    loader = InterHandLoader(DATA_PATH, split=split, mano_path=get_mano_path())
    cut_inter_img(loader, save_path, split)


def render_data(save_path, split):
    mano_path = get_mano_path()
    os.makedirs(osp.join(save_path, split, 'mask'), exist_ok=True)
    os.makedirs(osp.join(save_path, split, 'dense'), exist_ok=True)
    os.makedirs(osp.join(save_path, split, 'hms'), exist_ok=True)

    size = len(glob(osp.join(save_path, split, 'anno', '*.pkl')))
    mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                  'left': ManoLayer(mano_path['left'], center_idx=None)}
    fix_shape(mano_layer)
    renderer = mano_two_hands_renderer(img_size=IMG_SIZE, device='cuda')
    hmg = HeatmapGenerator(HEATMAP_SIZE, HEATMAP_SIGMA)

    for idx in tqdm(range(size)):
        with open(osp.join(save_path, split, 'anno', '{}.pkl'.format(idx)), 'rb') as file:
            data = pickle.load(file)

        R = data['camera']['R']
        T = data['camera']['t']
        camera = data['camera']['camera']

        verts = []
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

            verts.append(torch.from_numpy(handV).float().cuda().unsqueeze(0))
            hms = np.split(hmg(handJ2d * HEATMAP_SIZE / IMG_SIZE)[0], 7)  # 21 x h x w
            for hIdx in range(len(hms)):
                cv.imwrite(os.path.join(save_path, split, 'hms', '{}_{}_{}.jpg'.format(idx, hIdx, hand_type)),
                           hms[hIdx].transpose(1, 2, 0) * 255)

        img_mask = renderer.render_mask(cameras=torch.from_numpy(camera).float().cuda().unsqueeze(0),
                                        v3d_left=verts[0], v3d_right=verts[1])
        img_mask = img_mask.detach().cpu().numpy()[0] * 255
        cv.imwrite(osp.join(save_path, split, 'mask', '{}.jpg'.format(idx)), img_mask)

        img_dense, _ = renderer.render_densepose(cameras=torch.from_numpy(camera).float().cuda().unsqueeze(0),
                                                 v3d_left=verts[0], v3d_right=verts[1])
        img_dense = img_dense.detach().cpu().numpy()[0] * 255
        cv.imwrite(osp.join(save_path, split, 'dense', '{}.jpg'.format(idx)), img_dense)


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
        self.syns_path = '/mnt/user/E-shenfei.llj-356552/workgroup/lijun/hand_dataset/synthesis/sdf_xinchuan/'
        self.extra_syns = False

    def __len__(self):
        if self.extra_syns and self.split == 'train':
            return self.size + 1220000
        return self.size

    def __getitem__(self, idx):
        if self.extra_syns and self.split == 'train' and idx >= self.size:
            img = cv.imread(osp.join(self.syns_path, self.split, 'color_img', '{}.jpg'.format(idx)))
            hand_dict = np.load(os.path.join(self.syns_path, self.split, 'ori_handdict', '{}.npy'.format(idx)), allow_pickle=True)


            # return img, hand_dict
        else:
            img = cv.imread(osp.join(self.data_path, self.split, 'img', '{}.jpg'.format(idx)))
            hand_dict = np.load(os.path.join(self.data_path, self.split, 'ori_handdict', '{}.npy'.format(idx)),
                                allow_pickle=True)
        hand_dict = hand_dict[()]
        root_pose = cv.Rodrigues(hand_dict['left']['R'])[0].reshape(-1)
        hand_dict['left']['pose'] = \
            np.concatenate([root_pose, hand_dict['left']['pose']], axis=0).reshape(48).astype(np.float32)
        root_pose = cv.Rodrigues(hand_dict['right']['R'])[0].reshape(-1)
        hand_dict['right']['pose'] = \
            np.concatenate([root_pose, hand_dict['right']['pose']], axis=0).reshape(48).astype(np.float32)

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
