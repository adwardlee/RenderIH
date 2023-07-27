import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.manoutils import get_upsample_path

MANO_PARENT = [-1, 0, 1, 2, 3,
               0, 5, 6, 7,
               0, 9, 10, 11,
               0, 13, 14, 15,
               0, 17, 18, 19]


class GraphLoss():
    def __init__(self, J_regressor, faces, level=4,
                 device='cuda'):
        # loss function
        self.L1Loss = nn.SmoothL1Loss() #nn.L1Loss()### change llj
        self.L2Loss = nn.MSELoss()
        self.smoothL1Loss = nn.SmoothL1Loss(beta=0.05)

        self.device = device

        self.level = level + 1
        self.process_J_regressor(J_regressor)
        self.faces = torch.from_numpy(faces.astype(np.int64)).to(self.device)

        with open(get_upsample_path(), 'rb') as file:
            upsample_weight = pickle.load(file)
        self.upsample_weight = torch.from_numpy(upsample_weight).to(self.device)

    def process_J_regressor(self, J_regressor):
        J_regressor = J_regressor.clone().detach()
        tip_regressor = torch.zeros_like(J_regressor[:5])
        tip_regressor[0, 745] = 1.0
        tip_regressor[1, 317] = 1.0
        tip_regressor[2, 444] = 1.0
        tip_regressor[3, 556] = 1.0
        tip_regressor[4, 673] = 1.0
        J_regressor = torch.cat([J_regressor, tip_regressor], dim=0)
        new_order = [0,
                     13, 14, 15, 16,
                     1, 2, 3, 17,
                     4, 5, 6, 18,
                     10, 11, 12, 19,
                     7, 8, 9, 20]
        self.J_regressor = J_regressor[new_order].contiguous().to(self.device)

    def mesh_downsample(self, feat, p=2):
        # feat: bs x N x f
        feat = feat.permute(0, 2, 1).contiguous()  # x = bs x f x N
        feat = nn.AvgPool1d(p)(feat)  # bs x f x N/p
        feat = feat.permute(0, 2, 1).contiguous()  # x = bs x N/p x f
        return feat

    def mesh_upsample(self, x, p=2):
        x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
        x = nn.Upsample(scale_factor=p)(x)  # B x F x (V*p)
        x = x.permute(0, 2, 1).contiguous()  # x = B x (V*p) x F
        return x

    def norm_loss(self, verts_pred, verts_gt):
        edge_gt = verts_gt[:, self.faces]
        edge_gt = torch.stack([edge_gt[:, :, 0] - edge_gt[:, :, 1],
                               edge_gt[:, :, 1] - edge_gt[:, :, 2],
                               edge_gt[:, :, 2] - edge_gt[:, :, 0],
                               ], dim=2)  # B x F x 3 x 3
        edge_pred = verts_pred[:, self.faces]
        edge_pred = torch.stack([edge_pred[:, :, 0] - edge_pred[:, :, 1],
                                 edge_pred[:, :, 1] - edge_pred[:, :, 2],
                                 edge_pred[:, :, 2] - edge_pred[:, :, 0],
                                 ], dim=2)  # B x F x 3 x 3

        # norm loss
        face_norm_gt = torch.cross(edge_gt[:, :, 0], edge_gt[:, :, 1], dim=-1)
        face_norm_gt = F.normalize(face_norm_gt, dim=-1)
        face_norm_gt = face_norm_gt.unsqueeze(2)  # B x F x 1 x 3
        edge_pred_normed = F.normalize(edge_pred, dim=-1)
        temp = torch.sum(edge_pred_normed * face_norm_gt, dim=-1)  # B x F x 3
        return self.L1Loss(temp, torch.zeros_like(temp))

    def edge_loss(self, verts_pred, verts_gt):
        edge_gt = verts_gt[:, self.faces]
        edge_gt = torch.stack([edge_gt[:, :, 0] - edge_gt[:, :, 1],
                               edge_gt[:, :, 1] - edge_gt[:, :, 2],
                               edge_gt[:, :, 2] - edge_gt[:, :, 0],
                               ], dim=2)  # B x F x 3 x 3
        edge_pred = verts_pred[:, self.faces]
        edge_pred = torch.stack([edge_pred[:, :, 0] - edge_pred[:, :, 1],
                                 edge_pred[:, :, 1] - edge_pred[:, :, 2],
                                 edge_pred[:, :, 2] - edge_pred[:, :, 0],
                                 ], dim=2)  # B x F x 3 x 3
        edge_length_gt = torch.linalg.norm(edge_gt, dim=-1)  # B x F x 3
        edge_length_pred = torch.linalg.norm(edge_pred, dim=-1)  # B x F x 3
        edge_length_loss = self.L1Loss(edge_length_pred, edge_length_gt)
        return edge_length_loss

    def calc_mano_loss(self, v3d_pred, v2d_pred, v3d_gt, v2d_gt, img_size):
        J_r_pred = torch.matmul(self.J_regressor, v3d_pred)
        J_r_gt = torch.matmul(self.J_regressor, v3d_gt)

        loss_dict = {}
        loss_dict['vert2d_loss'] = self.L2Loss((v2d_pred / img_size * 2 - 1),
                                               (v2d_gt / img_size * 2 - 1))
        loss_dict['vert3d_loss'] = self.L1Loss(v3d_pred, v3d_gt)
        loss_dict['joint_loss'] = self.L1Loss(J_r_pred, J_r_gt)
        loss_dict['norm_loss'] = self.norm_loss(v3d_pred, v3d_gt)
        loss_dict['edge_loss'] = self.edge_loss(v3d_pred, v3d_gt)
        return loss_dict

    def upsample_weight_loss(self, w):
        x = w - self.upsample_weight
        return self.L1Loss(x, torch.zeros_like(x))

    def rel_loss(self, v1, v2, v1_gt, v2_gt):
        rel_gt = v1.unsqueeze(1) - v2.unsqueeze(2)
        rel_gt = torch.linalg.norm(rel_gt, dim=-1)  # bs x V x V
        rel_pred = v1_gt.unsqueeze(1) - v2_gt.unsqueeze(2)
        rel_pred = torch.linalg.norm(rel_pred, dim=-1)  # bs x 21 x 21
        return self.L1Loss(rel_gt, rel_pred)

    def calc_loss(self, converter,
                  v3d_gt, v2d_gt,
                  v3d_pred, v2d_pred,
                  v3dList, v2dList,
                  img_size):
        assert self.faces.device == v3d_gt.device
        assert self.faces.device == v3d_pred.device
        mano_loss_dict = self.calc_mano_loss(v3d_pred, v2d_pred, v3d_gt, v2d_gt, img_size)

        v3dList_gt = []
        v2dList_gt = []
        v3d_gcn = converter.vert_to_GCN(v3d_gt)
        v2d_gcn = converter.vert_to_GCN(v2d_gt)

        for i in range(self.level):
            v3dList_gt.append(v3d_gcn)
            v2dList_gt.append(v2d_gcn)
            v3d_gcn = self.mesh_downsample(v3d_gcn)
            v2d_gcn = self.mesh_downsample(v2d_gcn)

        v3dList_gt.reverse()
        v2dList_gt.reverse()

        coarsen_loss_dict = {}
        coarsen_loss_dict['v3d_loss'] = []
        coarsen_loss_dict['v2d_loss'] = []
        for i in range(len(v2dList)):
            for j in range(len(v3dList_gt)):
                if v3dList[i].shape[1] == v3dList_gt[j].shape[1]:
                    break

            coarsen_loss_dict['v3d_loss'].append(self.L1Loss(v3dList[i],
                                                             v3dList_gt[j]))
            coarsen_loss_dict['v2d_loss'].append(self.L2Loss((v2dList[i] / img_size * 2 - 1),
                                                             (v2dList_gt[j] / img_size * 2 - 1)))

        return mano_loss_dict, coarsen_loss_dict

    def range_loss(self, label, Min, Max):
        l1 = self._zero_norm_loss(torch.clamp(Min - label, min=0.))
        l2 = self._zero_norm_loss(torch.clamp(label - Max, min=0.))
        return l1 + l2

    def _one_norm_loss(self, p):
        return self.L1Loss(p, torch.ones_like(p))

    def _zero_norm_loss(self, p):
        return self.L1Loss(p, torch.zeros_like(p))




def calc_aux_loss(cfg, hand_loss,
                  dataDict,
                  mask, dense, hms):
    loss_dict = {}
    total_loss = 0
    if 'mask' in dataDict:
        loss_dict['mask_loss'] = hand_loss.smoothL1Loss(dataDict['mask'], mask)
        total_loss = total_loss + loss_dict['mask_loss'] * cfg.LOSS_WEIGHT.AUX.MASK
    if 'dense' in dataDict:
        loss_l = hand_loss.smoothL1Loss(dataDict['dense'][:, :3] * mask[:, :1], dense * mask[:, :1])
        loss_r = hand_loss.smoothL1Loss(dataDict['dense'][:, 3:] * mask[:, 1:], dense * mask[:, 1:])
        loss_dict['dense_loss'] = (loss_l + loss_r) / 2
        total_loss = total_loss + loss_dict['dense_loss'] * cfg.LOSS_WEIGHT.AUX.DENSEPOSE
    if 'hms' in dataDict:
        loss_dict['hms_loss'] = hand_loss.L2Loss(dataDict['hms'], hms)
        total_loss = total_loss + loss_dict['hms_loss'] * cfg.LOSS_WEIGHT.AUX.HMS
    if total_loss > 0:
        loss_dict['total_loss'] = total_loss
    return loss_dict


def calc_loss_GCN(cfg, epoch,
                  graph_loss_left, graph_loss_right,
                  converter_left, converter_right,
                  result, paramsDict, handDictList, otherInfo,
                  mask, dense, hms,
                  v2d_l, j2d_l, v2d_r, j2d_r,
                  v3d_l, j3d_l, v3d_r, j3d_r,
                  root_rel, img_size,
                  upsample_weight=None):
    aux_lost_dict = {}
    aux_lost_dict['total_loss'] = 0#calc_aux_loss(cfg, graph_loss_left,otherInfo,mask, dense, hms)

    v3d_r = v3d_r + root_rel.unsqueeze(1)
    j3d_r = j3d_r + root_rel.unsqueeze(1)

    v2dList = []
    v3dList = []
    for i in range(len(handDictList)):
        v2dList.append(handDictList[i]['verts2d']['left'])
        v3dList.append(handDictList[i]['verts3d']['left'])
    mano_loss_dict_left, coarsen_loss_dict_left \
        = graph_loss_left.calc_loss(converter_left,
                                    v3d_l, v2d_l,
                                    result['verts3d']['left'], result['verts2d']['left'],
                                    v3dList, v2dList,
                                    img_size)

    v2dList = []
    v3dList = []
    for i in range(len(handDictList)):
        v2dList.append(handDictList[i]['verts2d']['right'])
        v3dList.append(handDictList[i]['verts3d']['right'])
    mano_loss_dict_right, coarsen_loss_dict_right \
        = graph_loss_right.calc_loss(converter_right,
                                     v3d_r, v2d_r,
                                     result['verts3d']['right'], result['verts2d']['right'],
                                     v3dList, v2dList,
                                     img_size)

    mano_loss_dict = {}
    for k in mano_loss_dict_left.keys():
        mano_loss_dict[k] = (mano_loss_dict_left[k] + mano_loss_dict_right[k]) / 2

    coarsen_loss_dict = {}
    for k in coarsen_loss_dict_left.keys():
        coarsen_loss_dict[k] = []
        for i in range(len(coarsen_loss_dict_left[k])):
            coarsen_loss_dict[k].append((coarsen_loss_dict_left[k][i] + coarsen_loss_dict_right[k][i]) / 2)

    cfg = cfg.LOSS_WEIGHT
    alpha = 0 if epoch < cfg.GRAPH.NORM.NORM_EPOCH else 1

    if upsample_weight is not None:
        mano_loss_dict['upsample_norm_loss'] = graph_loss_left.upsample_weight_loss(upsample_weight)
    else:
        mano_loss_dict['upsample_norm_loss'] = torch.zeros_like(mano_loss_dict['vert3d_loss'])

    mano_loss = 0 \
        + cfg.DATA.LABEL_3D * mano_loss_dict['vert3d_loss'] \
        + cfg.DATA.LABEL_2D * mano_loss_dict['vert2d_loss'] \
        + cfg.DATA.LABEL_3D * mano_loss_dict['joint_loss'] \
        + cfg.GRAPH.NORM.NORMAL * mano_loss_dict['norm_loss'] \
        + alpha * cfg.GRAPH.NORM.EDGE * mano_loss_dict['edge_loss']

    coarsen_loss = 0
    for i in range(len(coarsen_loss_dict['v3d_loss'])):
        coarsen_loss = coarsen_loss \
            + cfg.DATA.LABEL_3D * coarsen_loss_dict['v3d_loss'][i]  \
            + cfg.DATA.LABEL_2D * coarsen_loss_dict['v2d_loss'][i]


    total_loss = mano_loss + coarsen_loss + cfg.NORM.UPSAMPLE * mano_loss_dict['upsample_norm_loss']

    if 'total_loss' in aux_lost_dict:
        total_loss = total_loss + aux_lost_dict['total_loss']

    return total_loss, aux_lost_dict, mano_loss_dict, coarsen_loss_dict
