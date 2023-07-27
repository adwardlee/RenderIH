import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def weights_init(layer):
    classname = layer.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)


def sparse_python_to_torch(sp_python):
    L = sp_python.tocoo()
    indices = np.column_stack((L.row, L.col)).T
    indices = indices.astype(np.int64)
    indices = torch.from_numpy(indices)
    indices = indices.type(torch.LongTensor)
    L_data = L.data.astype(np.float32)
    L_data = torch.from_numpy(L_data)
    L_data = L_data.type(torch.FloatTensor)
    L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))
#     if torch.cuda.is_available():
#         L = L.cuda()

    return L


def graph_conv_cheby(x, cl, L, K=3):
    # parameters
    # B = batch size
    # V = nb vertices
    # Fin = nb input features
    # Fout = nb output features
    # K = Chebyshev order & support size
    B, V, Fin = x.size()
    B, V, Fin = int(B), int(V), int(Fin)

    # transform to Chebyshev basis
    x0 = x.permute(1, 2, 0).contiguous()  # V x Fin x B
    x0 = x0.view([V, Fin * B])  # V x Fin*B
    x = x0.unsqueeze(0)  # 1 x V x Fin*B

    def concat(x, x_):
        x_ = x_.unsqueeze(0)  # 1 x V x Fin*B
        return torch.cat((x, x_), 0)  # K x V x Fin*B

    if K > 1:
        x1 = torch.mm(L, x0)  # V x Fin*B
        x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
    for k in range(2, K):
        x2 = 2 * torch.mm(L, x1) - x0
        x = torch.cat((x, x2.unsqueeze(0)), 0)  # M x Fin*B
        x0, x1 = x1, x2

    x = x.view([K, V, Fin, B])  # K x V x Fin x B
    x = x.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
    x = x.view([B * V, Fin * K])  # B*V x Fin*K

    # Compose linearly Fin features to get Fout features
    x = cl(x)  # B*V x Fout
    x = x.view([B, V, -1])  # B x V x Fout

    return x


class GCN_ResBlock(nn.Module):
    # x______________conv + norm (optianal)_____________ x ____activate
    #  \____conv____activate____norm____conv____norm____/
    def __init__(self, in_dim, out_dim, mid_dim,
                 graph_L, graph_k,
                 drop_out=0.01):
        super(GCN_ResBlock, self).__init__()
        if isinstance(graph_L, np.ndarray):
            self.register_buffer('graph_L',
                                 torch.from_numpy(graph_L).float(),
                                 persistent=False)
        else:
            self.register_buffer('graph_L',
                                 sparse_python_to_torch(graph_L).to_dense(),
                                 persistent=False)

        self.graph_k = graph_k
        self.in_dim = in_dim

        self.norm1 = nn.LayerNorm(in_dim, eps=1e-6)
        self.fc1 = nn.Linear(in_dim * graph_k, mid_dim)
        self.norm2 = nn.LayerNorm(out_dim, eps=1e-6)
        self.fc2 = nn.Linear(mid_dim * graph_k, out_dim)
        self.dropout = nn.Dropout(drop_out)
        self.shortcut = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(out_dim, eps=1e-6)

    def forward(self, x):
        # x : B x V x f
        assert x.shape[-1] == self.in_dim

        x1 = F.relu(self.norm1(x))
        x1 = graph_conv_cheby(x, self.fc1, self.graph_L, K=self.graph_k)
        x1 = F.relu(self.norm2(x1))
        x1 = graph_conv_cheby(x1, self.fc2, self.graph_L, K=self.graph_k)
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
