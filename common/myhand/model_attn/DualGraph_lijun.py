import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .img_attn import img_ex
from .inter_attn_lijun import inter_attn


def graph_upsample(x, p):
    if p > 1:
        x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
        x = nn.Upsample(scale_factor=p)(x)  # B x F x (V*p)
        x = x.permute(0, 2, 1).contiguous()  # x = B x (V*p) x F
        return x
    else:
        return x

def weights_init(layer):
    classname = layer.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)

class GCN_ResBlock(nn.Module):
    # x______________conv + norm (optianal)_____________ x ____activate
    #  \____conv____activate____norm____conv____norm____/
    def __init__(self, in_dim, out_dim, mid_dim,
                 graph_L, graph_k,
                 drop_out=0.01):
        super(GCN_ResBlock, self).__init__()

        self.graph_k = graph_k
        self.in_dim = in_dim

        self.norm1 = nn.LayerNorm(in_dim, eps=1e-6)
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.norm2 = nn.LayerNorm(out_dim, eps=1e-6)
        self.fc2 = nn.Linear(mid_dim, out_dim)
        self.dropout = nn.Dropout(drop_out)
        self.shortcut = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(out_dim, eps=1e-6)

    def forward(self, x):
        # x : B x V x f
        assert x.shape[-1] == self.in_dim

        x1 = F.relu(self.norm1(x))
        x1 = self.fc1(x1)
        x1 = F.relu(self.norm2(x1))
        x1 = self.fc2(x1)
        x1 = self.dropout(x1)
        x2 = self.shortcut(x)

        return self.norm3(x1 + x2)


class GraphLayer(nn.Module):
    def __init__(self,
                 in_dim=256,
                 out_dim=256,
                 graph_L=None,
                 graph_k=2,
                 graph_layer_num=3,
                 drop_out=0.01):
        super().__init__()
        assert graph_k > 1

        self.GCN_blocks = nn.ModuleList()
        self.GCN_blocks.append(GCN_ResBlock(in_dim, out_dim, out_dim, graph_L, graph_k, drop_out))
        for i in range(graph_layer_num - 1):
            self.GCN_blocks.append(GCN_ResBlock(out_dim, out_dim, out_dim, graph_L, graph_k, drop_out))

        for m in self.modules():
            weights_init(m)

    def forward(self, verts_f):
        for i in range(len(self.GCN_blocks)):
            verts_f = self.GCN_blocks[i](verts_f)
            if i != (len(self.GCN_blocks) - 1):
                verts_f = F.relu(verts_f)

        return verts_f


class DualGraphLayer(nn.Module):
    def __init__(self,
                 verts_in_dim=256,
                 verts_out_dim=256,
                 graph_L_Left=None,
                 graph_L_Right=None,
                 graph_k=2,
                 graph_layer_num=4,
                 img_size=64,
                 img_f_dim=256,
                 grid_size=8,
                 grid_f_dim=128,
                 n_heads=4,
                 dropout=0.01):
        super().__init__()
        self.verts_num = graph_L_Left.shape[0]
        self.verts_in_dim = verts_in_dim
        self.img_size = img_size
        self.img_f_dim = img_f_dim

        self.position_embeddings = nn.Embedding(self.verts_num, self.verts_in_dim)

        self.graph_left = GraphLayer(verts_in_dim, verts_out_dim,
                                     graph_L_Left, graph_k, graph_layer_num,
                                     dropout)
        self.graph_right = GraphLayer(verts_in_dim, verts_out_dim,
                                      graph_L_Right, graph_k, graph_layer_num,
                                      dropout)

        self.img_ex_left = img_ex(img_size, img_f_dim,
                                  grid_size, grid_f_dim,
                                  verts_out_dim,
                                  n_heads=n_heads,
                                  dropout=dropout)
        self.img_ex_right = img_ex(img_size, img_f_dim,
                                   grid_size, grid_f_dim,
                                   verts_out_dim,
                                   n_heads=n_heads,
                                   dropout=dropout)
        self.attn = inter_attn(verts_out_dim, n_heads=n_heads, dropout=dropout)

    def forward(self, Lf, Rf, img_f):
        BS1, V, f = Lf.shape
        assert V == self.verts_num
        assert f == self.verts_in_dim
        BS2, V, f = Rf.shape
        assert V == self.verts_num
        assert f == self.verts_in_dim
        BS3, C, H, W = img_f.shape
        assert C == self.img_f_dim
        assert H == self.img_size
        assert W == self.img_size
        assert BS1 == BS2
        assert BS2 == BS3
        BS = BS1

        position_ids = torch.arange(self.verts_num, dtype=torch.long, device=Lf.device)
        position_ids = position_ids.unsqueeze(0).repeat(BS, 1)
        position_embeddings = self.position_embeddings(position_ids)
        Lf = Lf + position_embeddings
        Rf = Rf + position_embeddings

        Lf = self.graph_left(Lf)
        Rf = self.graph_right(Rf)

        Lf = self.img_ex_left(img_f, Lf)
        Rf = self.img_ex_right(img_f, Rf)

        Lf, Rf = self.attn(Lf, Rf)

        return Lf, Rf


class DualGraph(nn.Module):
    def __init__(self,
                 verts_in_dim=[512, 256, 128],
                 verts_out_dim=[256, 128, 64],
                 graph_L_Left=None,
                 graph_L_Right=None,
                 graph_k=[2, 2, 2],
                 graph_layer_num=[4, 4, 4],
                 img_size=[16, 32, 64],
                 img_f_dim=[256, 256, 256],
                 grid_size=[8, 8, 16],
                 grid_f_dim=[256, 128, 64],
                 n_heads=4,
                 dropout=0.01):
        super().__init__()
        for i in range(len(verts_in_dim) - 1):
            assert verts_out_dim[i] == verts_in_dim[i + 1]
        for i in range(len(verts_in_dim) - 1):
            assert graph_L_Left[i + 1].shape[0] == 2 * graph_L_Left[i].shape[0]
            assert graph_L_Right[i + 1].shape[0] == 2 * graph_L_Right[i].shape[0]

        self.layers = nn.ModuleList()
        for i in range(len(verts_in_dim)):
            self.layers.append(DualGraphLayer(verts_in_dim=verts_in_dim[i],
                                              verts_out_dim=verts_out_dim[i],
                                              graph_L_Left=graph_L_Left[i],
                                              graph_L_Right=graph_L_Right[i],
                                              graph_k=graph_k[i],
                                              graph_layer_num=graph_layer_num[i],
                                              img_size=img_size[i],
                                              img_f_dim=img_f_dim[i],
                                              grid_size=grid_size[i],
                                              grid_f_dim=grid_f_dim[i],
                                              n_heads=n_heads,
                                              dropout=dropout))

    def forward(self, Lf, Rf, img_f_list):
        assert len(img_f_list) == len(self.layers)
        for i in range(len(self.layers)):
            Lf, Rf = self.layers[i](Lf, Rf, img_f_list[i])

            if i != len(self.layers) - 1:
                Lf = graph_upsample(Lf, 2)
                Rf = graph_upsample(Rf, 2)

        return Lf, Rf
