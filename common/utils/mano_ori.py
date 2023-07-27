import numpy as np
import torch
import torch.nn as nn
import os.path as osp
import json
from main.config import cfg

import sys
sys.path.insert(0, cfg.mano_path)
import manopth
# from common.utils.manolayer import ManoLayer
from manopth.manolayer import ManoLayer

class MANO(nn.Module):
    def __init__(self, hand_type='right'):
        super(MANO, self).__init__()
        self.hand_type = hand_type
        self.layer = self.get_layer()
        self.vertex_num = 778
        self.face = self.layer.th_faces.numpy()
        self.joint_regressor = self.layer.th_J_regressor.numpy()

        self.joint_num = 21
        self.joints_name = ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4')
        self.skeleton = ( (0,1), (0,5), (0,9), (0,13), (0,17), (1,2), (2,3), (3,4), (5,6), (6,7), (7,8), (9,10), (10,11), (11,12), (13,14), (14,15), (15,16), (17,18), (18,19), (19,20) )
        self.root_joint_idx = self.joints_name.index('Wrist')

        # add fingertips to joint_regressor
        self.fingertip_vertex_idx = [728, 353, 442, 576, 694]  # mesh vertex idx

        thumbtip_onehot = np.array([1 if i == 728 else 0 for i in range(self.joint_regressor.shape[1])],
                                   dtype=np.float32).reshape(1, -1)
        indextip_onehot = np.array([1 if i == 353 else 0 for i in range(self.joint_regressor.shape[1])],
                                   dtype=np.float32).reshape(1, -1)
        middletip_onehot = np.array([1 if i == 442 else 0 for i in range(self.joint_regressor.shape[1])],
                                    dtype=np.float32).reshape(1, -1)
        ringtip_onehot = np.array([1 if i == 576 else 0 for i in range(self.joint_regressor.shape[1])],
                                  dtype=np.float32).reshape(1, -1)
        pinkytip_onehot = np.array([1 if i == 694 else 0 for i in range(self.joint_regressor.shape[1])],
                                   dtype=np.float32).reshape(1, -1)

        self.joint_regressor = np.concatenate(
            (self.joint_regressor, thumbtip_onehot, indextip_onehot, middletip_onehot, ringtip_onehot, pinkytip_onehot))
        self.joint_regressor = self.joint_regressor[
                               [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20], :]

        # self.fingertip_vertex_idx = [745, 317, 444, 556, 673]  # mesh vertex idx (right hand)
        # thumbtip_onehot = np.array([1 if i == 745 else 0 for i in range(self.joint_regressor.shape[1])],
        #                            dtype=np.float32).reshape(1, -1)
        # indextip_onehot = np.array([1 if i == 317 else 0 for i in range(self.joint_regressor.shape[1])],
        #                            dtype=np.float32).reshape(1, -1)
        # middletip_onehot = np.array([1 if i == 445 else 0 for i in range(self.joint_regressor.shape[1])],
        #                             dtype=np.float32).reshape(1, -1)
        # ringtip_onehot = np.array([1 if i == 556 else 0 for i in range(self.joint_regressor.shape[1])],
        #                           dtype=np.float32).reshape(1, -1)
        # pinkytip_onehot = np.array([1 if i == 673 else 0 for i in range(self.joint_regressor.shape[1])],
        #                            dtype=np.float32).reshape(1, -1)
        #
        # self.joint_regressor = np.concatenate((self.joint_regressor, thumbtip_onehot, indextip_onehot, middletip_onehot, ringtip_onehot, pinkytip_onehot))
        # self.joint_regressor = self.joint_regressor[[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],:]
        joint_regressor_torch = torch.from_numpy(self.joint_regressor).float()
        self.register_buffer('joint_regressor_torch', joint_regressor_torch)

    def get_layer(self, hand_type='right'):
        return ManoLayer(mano_root=osp.join(cfg.mano_path), flat_hand_mean=False, use_pca=False, side=hand_type)# load right hand MANO model

    def get_3d_joints(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 778, 3)
        Output:
            3D joints: size = (B, 21, 3)
        """
        joints = torch.einsum('bik,ji->bjk', [vertices, self.joint_regressor_torch])
        return joints