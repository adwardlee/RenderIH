import numpy as np
import random
import math
import cv2 as cv
import pickle
import torch

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from utils.config import get_cfg_defaults
from models.model_zoo import build_graph


def projection(scale, trans2d, label3d, img_size=256):
    scale = scale * img_size
    trans2d = trans2d * img_size / 2 + img_size / 2
    trans2d = trans2d

    label2d = scale * label3d[:, :2] + trans2d
    return label2d


def projection_batch(scale, trans2d, label3d, img_size=256):
    """orthodox projection
    Input:
        scale: (B)
        trans2d: (B, 2)
        label3d: (B x N x 3)
    Returns:
        (B, N, 2)
    """
    scale = scale * img_size  # bs
    if scale.dim() == 1:
        scale = scale.unsqueeze(-1).unsqueeze(-1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(-1)
    trans2d = trans2d * img_size / 2 + img_size / 2  # bs x 2
    trans2d = trans2d.unsqueeze(1)

    label2d = scale * label3d[..., :2] + trans2d
    return label2d


def projection_batch_np(scale, trans2d, label3d, img_size=256):
    """orthodox projection
    Input:
        scale: (B)
        trans2d: (B, 2)
        label3d: (B x N x 3)
    Returns:
        (B, N, 2)
    """
    scale = scale * img_size  # bs
    if scale.dim() == 1:
        scale = scale[..., np.newaxis, np.newaxis]
    if scale.dim() == 2:
        scale = scale[..., np.newaxis]
    trans2d = trans2d * img_size / 2 + img_size / 2  # bs x 2
    trans2d = trans2d[:, np.newaxis, :]

    label2d = scale * label3d[..., :2] + trans2d
    return label2d


def get_mano_path():
    cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path = os.path.join(abspath, cfg.MISC.MANO_PATH)
    mano_path = {'left': os.path.join(path, 'MANO_LEFT.pkl'),
                 'right': os.path.join(path, 'MANO_RIGHT.pkl')}
    return mano_path


def get_graph_dict_path():
    cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    graph_path = {'left': os.path.join(abspath, cfg.MISC.GRAPH_LEFT_DICT_PATH),
                  'right': os.path.join(abspath, cfg.MISC.GRAPH_RIGHT_DICT_PATH)}
    return graph_path


def get_dense_color_path():
    cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dense_path = os.path.join(abspath, cfg.MISC.DENSE_COLOR)
    return dense_path


def get_mano_seg_path():
    cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    seg_path = os.path.join(abspath, cfg.MISC.MANO_SEG_PATH)
    return seg_path


def get_upsample_path():
    cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    upsample_path = os.path.join(abspath, cfg.MISC.UPSAMPLE_PATH)
    return upsample_path


def build_mano_graph():
    graph_path = get_graph_dict_path()
    mano_path = get_mano_path()
    for hand_type in ['left', 'right']:
        if not os.path.exists(graph_path[hand_type]):
            manoData = pickle.load(open(mano_path[hand_type], 'rb'), encoding='latin1')
            faces = manoData['f']
            graph_dict = build_graph(faces, coarsening_levels=4)
            with open(graph_path[hand_type], 'wb') as file:
                pickle.dump(graph_dict, file)


class imgUtils():
    @ staticmethod
    def pad2squre(img, color=None):
        if img.shape[0] > img.shape[1]:
            W = img.shape[0] - img.shape[1]
        else:
            W = img.shape[1] - img.shape[0]
        W1 = int(W / 2)
        W2 = W - W1
        if color is None:
            if img.shape[2] == 3:
                color = (0, 0, 0)
            else:
                color = 0
        if img.shape[0] > img.shape[1]:
            return cv.copyMakeBorder(img, 0, 0, W1, W2, cv.BORDER_CONSTANT, value=color)
        else:
            return cv.copyMakeBorder(img, W1, W2, 0, 0, cv.BORDER_CONSTANT, value=color)

    @ staticmethod
    def cut2squre(img):
        if img.shape[0] > img.shape[1]:
            s = int((img.shape[0] - img.shape[1]) / 2)
            return img[s:(s + img.shape[1])]
        else:
            s = int((img.shape[1] - img.shape[0]) / 2)
            return img[:, s:(s + img.shape[0])]

    @ staticmethod
    def get_scale_mat(center, scale=1.0):
        scaleMat = np.zeros((3, 3), dtype='float32')
        scaleMat[0, 0] = scale
        scaleMat[1, 1] = scale
        scaleMat[2, 2] = 1.0
        t = np.matmul((np.identity(3, dtype='float32') - scaleMat), center)
        scaleMat[0, 2] = t[0]
        scaleMat[1, 2] = t[1]
        return scaleMat

    @ staticmethod
    def get_rotation_mat(center, theta=0):
        t = theta * (3.14159 / 180)
        rotationMat = np.zeros((3, 3), dtype='float32')
        rotationMat[0, 0] = math.cos(t)
        rotationMat[0, 1] = -math.sin(t)
        rotationMat[1, 0] = math.sin(t)
        rotationMat[1, 1] = math.cos(t)
        rotationMat[2, 2] = 1.0
        t = np.matmul((np.identity(3, dtype='float32') - rotationMat), center)
        rotationMat[0, 2] = t[0]
        rotationMat[1, 2] = t[1]
        return rotationMat

    @ staticmethod
    def get_rotation_mat3d(theta=0):
        t = theta * (3.14159 / 180)
        rotationMat = np.zeros((3, 3), dtype='float32')
        rotationMat[0, 0] = math.cos(t)
        rotationMat[0, 1] = -math.sin(t)
        rotationMat[1, 0] = math.sin(t)
        rotationMat[1, 1] = math.cos(t)
        rotationMat[2, 2] = 1.0
        return rotationMat

    @ staticmethod
    def get_affine_mat(theta=0, scale=1.0,
                       u=0, v=0,
                       height=480, width=640):
        center = np.array([width / 2, height / 2, 1], dtype='float32')
        rotationMat = imgUtils.get_rotation_mat(center, theta)
        scaleMat = imgUtils.get_scale_mat(center, scale)
        trans = np.identity(3, dtype='float32')
        trans[0, 2] = u
        trans[1, 2] = v
        affineMat = np.matmul(scaleMat, rotationMat)
        affineMat = np.matmul(trans, affineMat)
        return affineMat

    @staticmethod
    def img_trans(theta, scale, u, v, img):
        size = img.shape[0]
        u = int(u * size / 2)
        v = int(v * size / 2)
        affineMat = imgUtils.get_affine_mat(theta=theta, scale=scale,
                                            u=u, v=v,
                                            height=256, width=256)
        return cv.warpAffine(src=img,
                             M=affineMat[0:2, :],
                             dsize=(256, 256),
                             dst=img,
                             flags=cv.INTER_LINEAR,
                             borderMode=cv.BORDER_REPLICATE,
                             borderValue=(0, 0, 0)
                             )

    @staticmethod
    def data_augmentation(theta, scale, u, v,
                          img_list=None, label2d_list=None, label3d_list=None,
                          R=None,
                          img_size=224):
        affineMat = imgUtils.get_affine_mat(theta=theta, scale=scale,
                                            u=u, v=v,
                                            height=img_size, width=img_size)
        if img_list is not None:
            img_list_out = []
            for img in img_list:
                img_list_out.append(cv.warpAffine(src=img,
                                                  M=affineMat[0:2, :],
                                                  dsize=(img_size, img_size)))
        else:
            img_list_out = None

        if label2d_list is not None:
            label2d_list_out = []
            for label2d in label2d_list:
                label2d_list_out.append(np.matmul(label2d, affineMat[0:2, 0:2].T) + affineMat[0:2, 2:3].T)
        else:
            label2d_list_out = None

        if label3d_list is not None:
            label3d_list_out = []
            R_delta = imgUtils.get_rotation_mat3d(theta)
            for label3d in label3d_list:
                label3d_list_out.append(np.matmul(label3d, R_delta.T))
        else:
            label3d_list_out = None

        if R is not None:
            R_delta = imgUtils.get_rotation_mat3d(theta)
            R = np.matmul(R_delta, R)
        else:
            R = None

        return img_list_out, label2d_list_out, label3d_list_out, R

    @ staticmethod
    def add_noise(img, noise=0.00, scale=255.0, alpha=0.3, beta=0.05):
        # add brightness noise & add random gaussian noise
        a = np.random.uniform(1 - alpha, 1 + alpha, 3)
        b = scale * beta * (2 * random.random() - 1)
        img = a * img + b + scale * np.random.normal(loc=0.0, scale=noise, size=img.shape)
        img = np.clip(img, 0, scale)
        return img
