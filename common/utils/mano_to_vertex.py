import cv2 as cv
import numpy as np
import torch
from main.config import cfg
from common.nets.mano_head import batch_rodrigues

def mano_convert(params, mano_layer):
        #root_pose = batch_rodrigues(params['R'])
        #all_pose = np.concatenate([root_pose, params['pose'][0]], axis=0).reshape(1, 48).astype(np.float32)
        handV, handJ = mano_layer(params['R'],
                                  params['mano_pose'][:, 3:],
                                  params['mano_shape'],
                                  trans=params['trans'])
        # handV = handV / 1000
        handJ = handJ / 1000
        # handV = torch.bmm(handV, params['cam_R'].permute(0, 2, 1)) + params['cam_T'].unsqueeze(1)
        handJ = torch.bmm(handJ, params['cam_R'].permute(0, 2, 1)) + params['cam_T'].unsqueeze(1)

        # handV2d = torch.bmm(handV, params['cam'].permute(0, 2, 1))
        # handV2d = handV2d[:, :2] / handV2d[:, 2:]
        # handV2d = 2 * handV2d / cfg.input_img_shape[0] - 1.0
        handJ2d = torch.bmm(handJ, params['cam'].permute(0, 2, 1))
        handJ2d = handJ2d[:, :, :2] / handJ2d[:, :, 2:]
        # handJ2d = 2 * handJ2d / cfg.input_img_shape[0] - 1.0
        handJ2d = handJ2d / cfg.input_img_shape[0]


        params['trans'][:] = 0
        orihandV, orihandJ = mano_layer(params['R'],
                                        params['mano_pose'][:, 3:],
                                        params['mano_shape'],
                                        trans=params['trans'])
        orihandV = orihandV / 1000
        orihandJ = orihandJ / 1000
        return orihandV, orihandJ, handJ2d