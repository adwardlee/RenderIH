import sys
import os
import random
import torch
import cv2 as cv
import pickle
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import imgaug.augmenters as iaa
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.dataset_utils import BONE_LENGTH
from dataset.interhand import InterHand_dataset
from dataset.interhand_withother import InterHand_other
from dataset.interhand_orisyn import InterHand_orisyn
from dataset.interhand_subset import InterHand_subset
from dataset.interhand_fullsyn_realsubset import InterHand_mixsubset
from dataset.interhand_realsubset import InterHand_realsubset
from utils.manoutils import imgUtils, get_mano_path
from models.manolayer import ManoLayer, rodrigues_batch
# from common.utils.vis import save_obj

def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(rot)), -np.sin(np.deg2rad(rot)), 0],
                  [np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv.Rodrigues(np.dot(R,per_rdg))
    aa = (resrot.T)[0]
    return aa

def flip_pose(pose):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    pose = pose
    # we also negate the second and the third dimension of the axis-angle
    pose[1::3] = -pose[1::3]
    pose[2::3] = -pose[2::3]
    # pose[1] = -pose[1]
    # pose[2] = -pose[2]
    return pose

class handDataset2(Dataset):
    """mix different hand datasets"""

    def __init__(self, cfg, mano_path=None,
                 interPath=None,
                 synPath=None,
                 theta=[-90, 90], scale=[0.75, 1.25], uv=[-10, 10],
                 flip=True,
                 train=True,
                 aux_size=64,
                 bone_length=BONE_LENGTH,
                 noise=0.0,
                 data_type=0):
        if mano_path is None:
            mano_path = get_mano_path()
        self.dataset = {}
        self.dataName = []
        self.sizeList = []
        self.theta = theta
        self.scale = scale
        self.uv = uv
        self.noise = noise
        self.flip = flip

        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

        self.train = train
        self.aux_size = aux_size
        self.bone_length = bone_length

        if interPath is not None and os.path.exists(str(interPath)):
            if self.train:
                split = 'train'
            else:
                split = 'test'
            if data_type == 2:
                self.dataset['inter'] = InterHand_other(interPath, split)
            elif data_type == 3:
                self.dataset['inter'] = InterHand_orisyn(interPath, split)
            elif data_type == 4:
                self.dataset['inter'] = InterHand_subset(interPath, split)
            elif data_type == 5:
                self.dataset['inter'] = InterHand_realsubset(interPath, split)
            elif data_type == 6:
                self.dataset['inter'] = InterHand_mixsubset(interPath, split)
            else:
                self.dataset['inter'] = InterHand_dataset(interPath, split)
            self.dataName.append('inter')
            self.sizeList.append(len(self.dataset['inter']))
            print('load interhand2.6m dataset, size: {}'.format(len(self.dataset['inter'])))

        self.size = 0
        for s in self.sizeList:
            self.size += s

        for i in range(1, len(self.sizeList)):
            self.sizeList[i] += self.sizeList[i - 1]

        self.seq = iaa.Sequential([
                        iaa.Dropout(p=(0, 0.2)),
                        iaa.AverageBlur(),
                        iaa.MotionBlur(k=(3,7)),
                        iaa.GaussianBlur(sigma=(0, 2.0)),
                        iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)
                    ])
        self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                           'left': ManoLayer(mano_path['left'], center_idx=None)}
        self.mano_layer['left'].shapedirs[:, 0, :] *= -1
        self.leftface = self.mano_layer['left'].faces
        self.rightface = self.mano_layer['right'].faces

    def __len__(self):
        return self.size

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose = pose.astype('float32')
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def augm_params(self):
        theta = random.random() * (self.theta[1] - self.theta[0]) + self.theta[0]
        scale = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        u = random.random() * (self.uv[1] - self.uv[0]) + self.uv[0]
        v = random.random() * (self.uv[1] - self.uv[0]) + self.uv[0]
        flip = random.random() > 0.5 if self.flip else False
        return theta, scale, u, v, flip

    # def process_data(self, img, mask, dense, hand_dict):
    def process_data(self, img, hand_dict, idx):
        label2d_list = [hand_dict['left']['verts2d'],### 0 - 256
                        hand_dict['left']['joints2d'],
                        hand_dict['right']['verts2d'],
                        hand_dict['right']['joints2d']]
        label3d_list = [hand_dict['left']['verts3d'],
                        hand_dict['left']['joints3d'],
                        hand_dict['right']['verts3d'],
                        hand_dict['right']['joints3d']]
        left_pose = hand_dict['left']['pose']
        left_shape = hand_dict['left']['shape']
        right_pose = hand_dict['right']['pose']
        right_shape = hand_dict['right']['shape']

        if self.train:
            # random sacle and translation
            # hms_left = hand_dict['left']['hms']
            # hms_right = hand_dict['right']['hms']

            theta, scale, u, v, flip = self.augm_params()
            imgList, label2d_list, label3d_list, _ \
                = imgUtils.data_augmentation(theta, scale, u, v,
                                             # img_list=[img, mask, dense] + hms_left + hms_right,
                                             img_list=[img],
                                             label2d_list=label2d_list,
                                             label3d_list=label3d_list,
                                             img_size=img.shape[0])
            if flip:
                left_pose = torch.from_numpy(left_pose)
                right_pose = torch.from_numpy(right_pose)
                left_axisangle = self.mano_layer['left'].pca2axis(left_pose[3:].reshape(1, 45)) - self.mano_layer['left'].hands_mean
                right_axisangle = self.mano_layer['right'].pca2axis(right_pose[3:].reshape(1, 45))- self.mano_layer['right'].hands_mean
                newleft_axisangle = torch.cat((left_pose[:3], left_axisangle[0]))
                newright_axisangle = torch.cat((right_pose[:3], right_axisangle[0]))
                left_pose = self.pose_processing(newleft_axisangle.numpy(), theta, flip)
                right_pose = self.pose_processing(newright_axisangle.numpy(), theta, flip)
                mean_pose = self.mano_layer['right'].hands_mean
                lefthand_pose = self.mano_layer['right'].axis2pca(torch.from_numpy(left_pose[3:]).reshape(1, 45) + mean_pose)[0]
                mean_pose = self.mano_layer['left'].hands_mean
                righthand_pose = self.mano_layer['left'].axis2pca(torch.from_numpy(right_pose[3:]).reshape(1, 45) + mean_pose)[0]
                left_pose[3:] = lefthand_pose.numpy()
                right_pose[3:] = righthand_pose.numpy()
            else:
                left_pose = self.pose_processing(left_pose, theta, flip)
                right_pose = self.pose_processing(right_pose, theta, flip)


            img = imgList[0]

            ############ check pose ################
            # handV, handJ = self.mano_layer['right'](torch.from_numpy(cv.Rodrigues(left_pose[:3])[0].reshape(1, 3, 3)).float(),
            #                                         torch.from_numpy(left_pose[None, 3:]).float(),
            #                                         torch.from_numpy(left_shape).reshape(1, 10).float(),)
            # handV = handV - handJ[:, 0:1]
            # save_obj(handV[0], self.rightface, file_name='tmp/rightmano_{}.obj'.format(idx))
            # handV, handJ = self.mano_layer['left'](
            #     torch.from_numpy(cv.Rodrigues(right_pose[:3])[0].reshape(1, 3, 3)).float(),
            #     torch.from_numpy(right_pose[None, 3:]).float(),
            #     torch.from_numpy(right_shape).reshape(1, 10).float(), )
            # handV = handV - handJ[:, 0:1]
            # save_obj(handV[0], self.leftface, file_name='tmp/leftmano_{}.obj'.format(idx))
            #                                         trans=torch.from_numpy(params['trans']).float())
            #######################################


            # mask = imgList[1]
            # dense_map = imgList[2]
            # hms = imgList[3:]

            # add img noise
            img = self.seq(image=img)
            # img = imgUtils.add_noise(img.astype(np.float32),
            #                          noise=self.noise,
            #                          scale=255.0,
            #                          alpha=0.3, beta=0.05).astype(np.uint8)
        else:
            flip = False

        if flip:
            img = cv.flip(img, 1)
        #     mask = cv.flip(mask, 1)
        #     dense_map = cv.flip(dense_map, 1)
        #     for i in range(len(hms)):
        #         hms[i] = cv.flip(hms[i], 1)
        #
        # # to torch tensor
        # dense_map = cv.resize(dense_map, (self.aux_size, self.aux_size))
        # dense_map = torch.tensor(dense_map, dtype=torch.float32) / 255
        # dense_map = dense_map.permute(2, 0, 1)

        # mask = cv.resize(mask, (self.aux_size, self.aux_size))
        # ret, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
        # mask = mask.astype(np.float) / 255
        # mask = mask[..., 1:]
        # if flip:
        #     mask = mask[..., [1, 0]]
        # mask = torch.tensor(mask, dtype=torch.float32)
        # mask = mask.permute(2, 0, 1)
        #
        # for i in range(len(hms)):
        #     hms[i] = cv.resize(hms[i], (self.aux_size, self.aux_size))
        # hms = np.concatenate(hms, axis=-1)
        # if flip:
        #     idx = [i + 21 for i in range(21)] + [i for i in range(21)]
        #     hms = hms[..., idx]
        # hms = torch.tensor(hms, dtype=torch.float32) / 255
        # hms = hms.permute(2, 0, 1)

        ori_img = torch.tensor(img, dtype=torch.float32) / 255
        ori_img = ori_img.permute(2, 0, 1)
        imgTensor = torch.tensor(cv.cvtColor(img, cv.COLOR_BGR2RGB), dtype=torch.float32) / 255
        imgTensor = imgTensor.permute(2, 0, 1)
        imgTensor = self.normalize_img(imgTensor)

        root_left = label3d_list[1][0]
        root_right = label3d_list[3][0]
        root_rel = root_right - root_left
        label3d_list[0] = label3d_list[0] - root_left
        label3d_list[1] = label3d_list[1] - root_left
        label3d_list[2] = label3d_list[2] - root_right
        label3d_list[3] = label3d_list[3] - root_right

        if self.bone_length is not None:
            length_left = np.linalg.norm(label3d_list[1][9] - label3d_list[1][0])
            length_right =  np.linalg.norm(label3d_list[3][9] - label3d_list[3][0])
            length = (length_left + length_right)/ 2
            scale = self.bone_length / length
            root_rel = root_rel * scale
            for i in range(2):
                label3d_list[i] = label3d_list[i] * self.bone_length / length_left
            for i in range(2, 4):
                label3d_list[i] = label3d_list[i] * self.bone_length / length_right

        root_rel = torch.tensor(root_rel, dtype=torch.float32)
        for i in range(4):
            label2d_list[i] = torch.tensor(label2d_list[i], dtype=torch.float32)
            label3d_list[i] = torch.tensor(label3d_list[i], dtype=torch.float32)

        if flip:
            root_rel[1:] = -root_rel[1:]
            for i in range(4):
                label2d_list[i][:, 0] = img.shape[0] - label2d_list[i][:, 0]
                label3d_list[i][:, 0] = -label3d_list[i][:, 0]

            [v2d_r, j2d_r, v2d_l, j2d_l] = label2d_list
            [v3d_r, j3d_r, v3d_l, j3d_l] = label3d_list
            lp, ls, rp, rs = right_pose, right_shape, left_pose, left_shape
        else:
            [v2d_l, j2d_l, v2d_r, j2d_r] = label2d_list
            [v3d_l, j3d_l, v3d_r, j3d_r] = label3d_list
            lp, ls, rp, rs = left_pose, left_shape, right_pose, right_shape

        # save_obj(v3d_r.reshape(778, 3), self.rightface,
        #          file_name='tmp/rightreal_{}.obj'.format(idx))
        # save_obj(v3d_l.reshape(778, 3), self.leftface,
        #          file_name='tmp/leftreal_{}.obj'.format(idx))
        ####mask, dense_map, hms, \
        return ori_img, \
            imgTensor, \
            v2d_l, j2d_l, v2d_r, j2d_r,\
            v3d_l, j3d_l, v3d_r, j3d_r, \
            root_rel, lp, ls, rp, rs

    def __getitem__(self, idx):
        # for i in range(len(self.sizeList)):
        #     if idx < self.sizeList[i]:
        #         idx2 = idx - (0 if i == 0 else self.sizeList[i - 1])
        #         name = self.dataName[i]
        #         if name == 'inter':
        #             img, mask, dense, hand_dict = self.dataset[name][idx2]
        #         break

        #img, mask, dense, hand_dict = self.dataset['inter'][idx]
        img, hand_dict = self.dataset['inter'][idx]

        # return self.process_data(img, mask, dense, hand_dict)

        return self.process_data(img, hand_dict, idx)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = handDataset2(interPath="/mnt/user/E-shenfei.llj-356552/data/dataset/interhand_5fps/interhand_data/")
    loader = DataLoader(dataset, batch_size=2)
    verts_left, _ = dataset.mano_layer['left'](torch.from_numpy(cv.Rodrigues(np.zeros((3)))[0].reshape(1, 3, 3)).float(),torch.zeros((1, 45)), torch.zeros((1, 10)))
    verts_left[:,:, 0] = -verts_left[:,:, 0]
    left_face = dataset.leftface
    right_face = dataset.rightface
    save_obj(verts_left[0].numpy(), right_face, 'tmp/right1.obj')
    verts_right, _ =dataset.mano_layer['right'](torch.from_numpy(cv.Rodrigues(np.zeros((3)))[0].reshape(1, 3, 3)).float(),torch.zeros((1, 45)), torch.zeros((1, 10)))

    save_obj(verts_right[0].numpy(), right_face, 'tmp/right.obj')
    for one in loader:
        print('1')