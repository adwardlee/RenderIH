import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

from common.myhand.utils.utils import projection_batch, get_dense_color_path, get_graph_dict_path, get_upsample_path
from common.myhand.model_zoo import GCN_vert_convert, graph_upsample, graph_avg_pool
from common.myhand.model_attn import DualGraph
from common.utils.mano import MANO
from common.utils.mano import Jr
from common.myhand.utils.comm import rotation_matrix_to_angle_axis

from main.config import cfg



def weights_init(layer):
    classname = layer.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)


class decoder(nn.Module):
    def __init__(self,
                 global_feature_dim=2048,
                 f_in_Dim=[256, 256, 256, 256],
                 f_out_Dim=[128, 64, 32],
                 gcn_in_dim=[256, 128, 128],
                 gcn_out_dim=[128, 128, 64],
                 graph_k=2,
                 graph_layer_num=4,
                 left_graph_dict={},
                 right_graph_dict={},
                 vertex_num=778,
                 dense_coor=None,
                 num_attn_heads=4,
                 upsample_weight=None,
                 dropout=0.05):
        super(decoder, self).__init__()
        assert len(f_in_Dim) == 4
        f_in_Dim = f_in_Dim[:-1]
        assert len(gcn_in_dim) == 3
        for i in range(len(gcn_out_dim) - 1):
            assert gcn_out_dim[i] == gcn_in_dim[i + 1]

        graph_dict = {'left': left_graph_dict, 'right': right_graph_dict}
        graph_dict['left']['coarsen_graphs_L'].reverse()
        graph_dict['right']['coarsen_graphs_L'].reverse()
        graph_L = {}
        for hand_type in ['left', 'right']:
            graph_L[hand_type] = graph_dict[hand_type]['coarsen_graphs_L']

        self.vNum_in = graph_L['left'][0].shape[0]
        self.vNum_out = graph_L['left'][2].shape[0]
        self.vNum_all = graph_L['left'][-1].shape[0]
        self.vNum_mano = vertex_num
        self.gf_dim = global_feature_dim
        self.gcn_in_dim = gcn_in_dim
        self.gcn_out_dim = gcn_out_dim

        if dense_coor is not None:
            dense_coor = torch.from_numpy(dense_coor).float()
            self.register_buffer('dense_coor', dense_coor)

        self.converter = {}
        for hand_type in ['left', 'right']:
            self.converter[hand_type] = GCN_vert_convert(vertex_num=self.vNum_mano,
                                                         graph_perm_reverse=graph_dict[hand_type]['graph_perm_reverse'],
                                                         graph_perm=graph_dict[hand_type]['graph_perm'])

        self.dual_gcn = DualGraph(verts_in_dim=self.gcn_in_dim,
                                  verts_out_dim=self.gcn_out_dim,
                                  graph_L_Left=graph_L['left'][:3],
                                  graph_L_Right=graph_L['right'][:3],
                                  graph_k=[graph_k, graph_k, graph_k],
                                  graph_layer_num=[graph_layer_num, graph_layer_num, graph_layer_num],
                                  img_size=[8, 16, 32],
                                  img_f_dim=f_in_Dim,
                                  grid_size=[8, 8, 8],
                                  grid_f_dim=f_out_Dim,
                                  n_heads=num_attn_heads,
                                  dropout=dropout)

        self.gf_layer_left = nn.Sequential(*(nn.Linear(self.gf_dim, self.gcn_in_dim[0] - 3),
                                             nn.LayerNorm(self.gcn_in_dim[0] - 3, eps=1e-6)))
        self.gf_layer_right = nn.Sequential(*(nn.Linear(self.gf_dim, self.gcn_in_dim[0] - 3),
                                              nn.LayerNorm(self.gcn_in_dim[0] - 3, eps=1e-6)))
        self.unsample_layer = nn.Linear(self.vNum_out, self.vNum_mano, bias=False)

        self.coord_head = nn.Linear(self.gcn_out_dim[-1], 3)
        self.avg_head = nn.Linear(self.vNum_out, 1)
        self.params_head = nn.Linear(self.gcn_out_dim[-1], 3)

        self.mano_left = MANO('left')
        self.mano_left_layer = self.mano_left.layer
        self.mano_right = MANO('right')
        self.mano_right_layer = self.mano_right.layer

        if torch.sum(torch.abs(self.mano_left_layer.shapedirs[:, 0, :] - self.mano_right_layer.shapedirs[:, 0, :])) < 1:
            print('double decoder Fix shapedirs bug of MANO')
            self.mano_left_layer.shapedirs[:, 0, :] *= -1

        self.J_regressor = {'left': Jr(self.mano_left_layer.J_regressor),
                            'right': Jr(self.mano_right_layer.J_regressor)}

        weights_init(self.gf_layer_left)
        weights_init(self.gf_layer_right)
        weights_init(self.coord_head)
        weights_init(self.avg_head)
        weights_init(self.params_head)

        if upsample_weight is not None:
            state = {'weight': upsample_weight.to(self.unsample_layer.weight.data.device)}
            self.unsample_layer.load_state_dict(state)
        else:
            weights_init(self.unsample_layer)

    def get_upsample_weight(self):
        return self.unsample_layer.weight.data

    def get_converter(self):
        return self.converter

    def get_hand_pe(self, bs, num=None):
        if num is None:
            num = self.vNum_in
        dense_coor = self.dense_coor.repeat(bs, 1, 1) * 2 - 1
        pel = self.converter['left'].vert_to_GCN(dense_coor)
        pel = graph_avg_pool(pel, p=pel.shape[1] // num)
        per = self.converter['right'].vert_to_GCN(dense_coor)
        per = graph_avg_pool(per, p=per.shape[1] // num)
        return pel, per

    def forward(self, x, fmaps):
        assert x.shape[1] == self.gf_dim
        fmaps = fmaps[:-1]
        bs = x.shape[0]

        pel, per = self.get_hand_pe(bs, num=self.vNum_in)
        Lf = torch.cat([self.gf_layer_left(x).unsqueeze(1).repeat(1, self.vNum_in, 1), pel], dim=-1)
        Rf = torch.cat([self.gf_layer_right(x).unsqueeze(1).repeat(1, self.vNum_in, 1), per], dim=-1)

        Lf, Rf = self.dual_gcn(Lf, Rf, fmaps)

        scale = {}
        trans2d = {}
        temp = self.avg_head(Lf.transpose(-1, -2))[..., 0]
        temp = self.params_head(temp)
        scale['left'] = temp[:, 0]
        trans2d['left'] = temp[:, 1:]
        temp = self.avg_head(Rf.transpose(-1, -2))[..., 0]
        temp = self.params_head(temp)
        scale['right'] = temp[:, 0]
        trans2d['right'] = temp[:, 1:]

        verts3d = {'left': self.coord_head(Lf), 'right': self.coord_head(Rf)}
        verts2d = {}
        result = {}
        pred_mano_left = {}
        pred_mano_right = {}
        for hand_type in ['left', 'right']:
            verts2d[hand_type] = projection_batch(scale[hand_type], trans2d[hand_type], verts3d[hand_type],
                                                  img_size=cfg.input_img_shape[0])
            result['v3d' + '_' + hand_type] = self.unsample_layer(verts3d[hand_type].transpose(1, 2)).transpose(1, 2)
            result['verts2d' + '_' + hand_type] = projection_batch(scale[hand_type], trans2d[hand_type],
                                                                   result['v3d' + '_' + hand_type],
                                                                   img_size=cfg.input_img_shape[0])
            result['scale' + '_' + hand_type] = scale[hand_type]
            result['trans2d' + '_' + hand_type] = trans2d[hand_type]
        result['j3d_left'] = self.J_regressor['left'](
            result['v3d_left'])  # self.mano_left.get_3d_joints(result['v3d_left'])
        result['j3d_right'] = self.J_regressor['right'](
            result['v3d_right'])  # self.mano_right.get_3d_joints(result['v3d_right'])
        result['j2d_left'] = projection_batch(scale['left'], trans2d['left'], result['j3d_left'], img_size=1)
        result['j2d_right'] = projection_batch(scale['right'], trans2d['right'], result['j3d_right'],
                                               img_size=1)
        pred_mano_left['verts3d'] = result['v3d_left']
        pred_mano_left['joints3d'] = result['j3d_left']
        pred_mano_right['verts3d'] = result['v3d_right']
        pred_mano_right['joints3d'] = result['j3d_right']
        pred_mano_left['joints_img'] = projection_batch(scale['left'], trans2d['left'], pred_mano_left['joints3d'],
                                                        img_size=1)  # / cfg.input
        pred_mano_right['joints_img'] = projection_batch(scale['right'], trans2d['right'], pred_mano_right['joints3d'],
                                                         img_size=1)
        # pred_mano_right['joints_img'] = result['verts2d' + '_' +'right'] / cfg.input_img_shape[0]

            # test output
        out = {}
        out['j2d_left'] = projection_batch(scale['left'], trans2d['left'], pred_mano_left['joints3d'], img_size=256)
        out['j2d_right'] = projection_batch(scale['right'], trans2d['right'], pred_mano_right['joints3d'],
                                            img_size=256)
        out['j3d_left'] = result['j3d_left']
        out['j3d_right'] = result['j3d_right']
        out['vert3d_left'] = result['v3d_left']
        out['vert3d_right'] = result['v3d_right']
        out['joints_coord_cam_left'] = pred_mano_left['joints3d']
        out['mesh_coord_cam_left'] = pred_mano_left['verts3d']
        out['joints_coord_cam_right'] = pred_mano_right['joints3d']
        out['mesh_coord_cam_right'] = pred_mano_right['verts3d']
        # else:
        #     out['cam_param'] = torch.zeros((3))
        return out


def load_decoder(cfg, encoder_info):
    graph_path = get_graph_dict_path()
    with open(graph_path['left'], 'rb') as file:
        left_graph_dict = pickle.load(file)
    with open(graph_path['right'], 'rb') as file:
        right_graph_dict = pickle.load(file)

    dense_path = get_dense_color_path()
    with open(dense_path, 'rb') as file:
        dense_coor = pickle.load(file)

    upsample_path = get_upsample_path()
    with open(upsample_path, 'rb') as file:
        upsample_weight = pickle.load(file)
    upsample_weight = torch.from_numpy(upsample_weight).float()

    model = decoder(
        global_feature_dim=encoder_info['global_feature_dim'],
        f_in_Dim=encoder_info['fmaps_dim'],
        f_out_Dim=cfg.MODEL.IMG_DIMS,
        gcn_in_dim=cfg.MODEL.GCN_IN_DIM,
        gcn_out_dim=cfg.MODEL.GCN_OUT_DIM,
        graph_k=cfg.MODEL.graph_k,
        graph_layer_num=cfg.MODEL.graph_layer_num,
        vertex_num=778,
        dense_coor=dense_coor,
        left_graph_dict=left_graph_dict,
        right_graph_dict=right_graph_dict,
        num_attn_heads=4,
        upsample_weight=upsample_weight,
        dropout=cfg.TRAIN.dropout
    )

    return model
