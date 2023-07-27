import torch
import torch.nn as nn
from torch.nn import functional as F
import os

from common.utils.rotation_transform import matrix_to_axis_angle
from common.nets.backbone import FPN
from common.nets.transformer import Transformer
from common.nets.regressor import Regressor
from common.utils.transforms import orthographic_projection
from common.utils.mano import MANO
import math
from dataset.interhand import fix_shape
from main.config import cfg

def rodrigues_batch(axis):
    # axis : bs * 3
    # return: bs * 3 * 3
    bs = axis.shape[0]
    Imat = torch.eye(3, dtype=axis.dtype, device=axis.device).repeat(bs, 1, 1)  # bs * 3 * 3
    angle = torch.norm(axis, p=2, dim=1, keepdim=True) + 1e-8  # bs * 1
    axes = axis / angle  # bs * 3
    sin = torch.sin(angle).unsqueeze(2)  # bs * 1 * 1
    cos = torch.cos(angle).unsqueeze(2)  # bs * 1 * 1
    L = torch.zeros((bs, 3, 3), dtype=axis.dtype, device=axis.device)
    L[:, 2, 1] = axes[:, 0]
    L[:, 1, 2] = -axes[:, 0]
    L[:, 0, 2] = axes[:, 1]
    L[:, 2, 0] = -axes[:, 1]
    L[:, 1, 0] = axes[:, 2]
    L[:, 0, 1] = -axes[:, 2]
    return Imat + sin * L + (1 - cos) * L.bmm(L)

def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.Hardswish(inplace=True))

    return nn.Sequential(*layers)

class ParamRegressor(nn.Module):
    def __init__(self, joint_num=778, test=False):
        super(ParamRegressor, self).__init__()
        self.joint_num = joint_num
        self.fc_left = make_linear_layers([self.joint_num * 3, 1024, 512], use_bn=False)
        self.fc_pose_left = make_linear_layers([512, 128, 16 * 6], relu_final=False)  # hand joint orientation
        self.fc_shape_left = make_linear_layers([512, 128, 10], relu_final=False)  # shape parameter

        self.fc_right = make_linear_layers([self.joint_num * 3, 1024, 512], use_bn=False)
        self.fc_pose_right = make_linear_layers([512, 128, 16 * 6], relu_final=False)  # hand joint orientation
        self.fc_shape_right = make_linear_layers([512, 128, 10], relu_final=False)  # shape parameter
        # self.fc_translate = make_linear_layers([512, 128, 3], relu_final=False)  # shape parameter
        self.mano_left = MANO('left')
        self.mano_right = MANO('right')
        self.mano = {'left': self.mano_left.layer, 'right': self.mano_right.layer}
        fix_shape(self.mano)
        print('fix regressor shapedirs', flush=True)
        self.test = test

    def rot6d_to_rotmat(self, x):
        x = x.view(-1, 3, 2)
        a1 = x[:, :, 0]
        a2 = x[:, :, 1]
        b1 = F.normalize(a1)
        b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
        b3 = torch.cross(b1, b2)
        return torch.stack((b1, b2, b3), dim=-1)

    def forward(self, targets, mode):
        if not self.test:
            joint3d_left = targets['left']['joints3d'] - targets['left']['joints3d'][:, 0:1]
            joint3d_right = targets['right']['joints3d'] - targets['right']['joints3d'][:, 0:1]
            vert3d_left = targets['left']['verts3d'] - targets['left']['joints3d'][:, 0:1]
            vert3d_right = targets['right']['verts3d'] - targets['right']['joints3d'][:, 0:1]
        else:
            joint3d_left = self.mano_left.get_3d_joints(targets['left']['verts3d'])
            joint3d_right = self.mano_right.get_3d_joints(targets['right']['verts3d'])
            vert3d_left = targets['left']['verts3d'] - joint3d_left[:, 0:1]
            vert3d_right = targets['right']['verts3d'] - joint3d_right[:, 0:1]

        batch_size = vert3d_left.shape[0]

        vert3d_left = vert3d_left.reshape(batch_size, self.joint_num * 3)
        feat_left = self.fc_left(vert3d_left)

        pose_left = self.fc_pose_left(feat_left)
        out_rotmat_left = self.rot6d_to_rotmat(pose_left)
        pose_left = out_rotmat_left#torch.cat([out_rotmat_left, torch.zeros((out_rotmat_left.shape[0], 3, 1)).cuda().float()], 2)
        pose_left = matrix_to_axis_angle(pose_left).reshape(batch_size, -1)
        out_rotmat_left = out_rotmat_left.reshape(batch_size, -1, 3, 3)
        shape_left = self.fc_shape_left(feat_left)
        shape_left = F.tanh(shape_left) * 3

        vert3d_right = vert3d_right.reshape(batch_size, self.joint_num * 3)
        feat_right = self.fc_right(vert3d_right)

        pose_right = self.fc_pose_right(feat_right)
        out_rotmat_right = self.rot6d_to_rotmat(pose_right)
        pose_right = out_rotmat_right# torch.cat([out_rotmat_right, torch.zeros((out_rotmat_right.shape[0], 3, 1)).cuda().float()], 2)
        pose_right = matrix_to_axis_angle(pose_right).reshape(batch_size, -1)
        out_rotmat_right = out_rotmat_right.reshape(batch_size, -1, 3, 3)
        shape_right = self.fc_shape_right(feat_right)
        shape_right = F.tanh(shape_right) * 3
        pred_vert_left, pred_joint_left = self.mano['left'](out_rotmat_left[:, 0], pose_left[:, 3:], shape_left)
        pred_vert_right, pred_joint_right = self.mano['right'](out_rotmat_right[:, 0], pose_right[:, 3:], shape_right)
        pred_vert_left /= 1000
        pred_vert_right /= 1000
        pred_joint_left /= 1000
        pred_joint_right /= 1000
        pred_vert_left = pred_vert_left - pred_joint_left[:, 0:1]
        pred_vert_right = pred_vert_right - pred_joint_right[:, 0:1]
        pred_joint_left = pred_joint_left - pred_joint_left[:, 0:1]
        pred_joint_right = pred_joint_right - pred_joint_right[:, 0:1]
        if mode == 'train':
            # loss functions
            loss = {}
            rotmat_left = rodrigues_batch(targets['left']['mano_pose'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)
            rotmat_right = rodrigues_batch(targets['right']['mano_pose'].reshape(-1, 3)).reshape(batch_size, -1, 3, 3)
            loss['pose_left'] = cfg.lambda_mano_pose * F.l1_loss(out_rotmat_left, rotmat_left)
            loss['shape_left'] = cfg.lambda_mano_shape * F.l1_loss(shape_left, targets['left']['mano_shape'])
            loss['pose_right'] = cfg.lambda_mano_pose * F.l1_loss(out_rotmat_right, rotmat_right)
            loss['shape_right'] = cfg.lambda_mano_shape * F.l1_loss(shape_right, targets['right']['mano_shape'])
            loss['vert_left'] = cfg.lambda_mano_verts * F.l1_loss(pred_vert_left, vert3d_left.reshape(batch_size, 778, 3))
            loss['vert_right'] = cfg.lambda_mano_verts * F.l1_loss(pred_vert_right, vert3d_right.reshape(batch_size, 778, 3))
            loss['joint_left'] = cfg.lambda_mano_joints * F.mse_loss(pred_joint_left, joint3d_left)
            loss['joint_right'] = cfg.lambda_mano_joints * F.mse_loss(pred_joint_right, joint3d_right)

            return loss

        else:
            # test output
            out = {}
            out['pose_left'] = pose_left
            out['shape_left'] = shape_left
            out['pose_right'] = pose_right
            out['shape_right'] = shape_right
            out['mesh_coord_cam_left'] = pred_vert_left
            out['mesh_coord_cam_right'] = pred_vert_right
            out['joints_coord_cam_left'] = pred_joint_left
            out['joints_coord_cam_right'] = pred_joint_right
            # else:
            #     out['cam_param'] = torch.zeros((3))
            return out

def load_param_regressor():
    model = ParamRegressor(test=True)
    model_path = 'results/regressor_limit3_b128/model_dump/snapshot_40.pth.tar'#'results/regressor_b128/model_dump/snapshot_45.pth.tar'#'results/regressor/model_dump/snapshot_80.pth.tar'###'models/regress_80.pth.tar'
    assert os.path.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))
    ckpt = torch.load(model_path)
    print(model.load_state_dict(ckpt['network'], strict=False))
    return model