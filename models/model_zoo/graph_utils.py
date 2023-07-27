import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# forked from https://github.com/3d-hand-shape/hand-graph-cnn


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


def graph_max_pool(x, p):
    if p > 1:
        x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
        x = nn.MaxPool1d(p)(x)  # B x F x V/p
        x = x.permute(0, 2, 1).contiguous()  # x = B x V/p x F
        return x
    else:
        return x


def graph_avg_pool(x, p):
    if p > 1:
        x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
        x = nn.AvgPool1d(p)(x)  # B x F x V/p
        x = x.permute(0, 2, 1).contiguous()  # x = B x V/p x F
        return x
    else:
        return x

# Upsampling of size p.


def graph_upsample(x, p):
    if p > 1:
        x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
        x = nn.Upsample(scale_factor=p)(x)  # B x F x (V*p)
        x = x.permute(0, 2, 1).contiguous()  # x = B x (V*p) x F
        return x
    else:
        return x


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


class Graph_CNN_Feat_Mesh(nn.Module):
    def __init__(self, num_input_chan, num_mesh_output_chan, graph_L):
        print('Graph ConvNet: feature to mesh')

        super(Graph_CNN_Feat_Mesh, self).__init__()

        self.num_input_chan = num_input_chan
        self.num_mesh_output_chan = num_mesh_output_chan
        self.graph_L = graph_L

        # parameters
        self.CL_F = [64, 32, num_mesh_output_chan]
        self.CL_K = [3, 3]
        self.layers_per_block = [2, 2]

        self.FC_F = [num_input_chan, 512, self.CL_F[0] * self.graph_L[-1].shape[0]]

        self.fc = nn.Sequential()
        for fc_id in range(len(self.FC_F) - 1):
            if fc_id == 0:
                use_activation = True
            else:
                use_activation = False
            self.fc.add_module('fc_%d' % (fc_id + 1), FCLayer(self.FC_F[fc_id],
                                                              self.FC_F[fc_id + 1], use_dropout=False,
                                                              use_activation=use_activation))

        _cl = []
        _bn = []
        for block_i in range(len(self.CL_F) - 1):
            for layer_i in range(self.layers_per_block[block_i]):
                Fin = self.CL_K[block_i] * self.CL_F[block_i]

                if layer_i is not self.layers_per_block[block_i] - 1:
                    Fout = self.CL_F[block_i]
                else:
                    Fout = self.CL_F[block_i + 1]

                _cl.append(nn.Linear(Fin, Fout))

                scale = np.sqrt(2.0 / (Fin + Fout))
                _cl[-1].weight.data.uniform_(-scale, scale)
                _cl[-1].bias.data.fill_(0.0)

                if block_i == len(self.CL_F) - 2 and layer_i == self.layers_per_block[block_i] - 1:
                    _bn.append(None)
                else:
                    _bn.append(nn.BatchNorm1d(Fout))

        self.cl = nn.ModuleList(_cl)
        self.bn = nn.ModuleList(_bn)

        # convert scipy sparse matric L to pytorch
        for graph_i in range(len(graph_L)):
            self.graph_L[graph_i] = sparse_python_to_torch(self.graph_L[graph_i])

    def init_weights(self, W, Fin, Fout):
        scale = np.sqrt(2.0 / (Fin + Fout))
        W.uniform_(-scale, scale)

        return W

    def graph_conv_cheby(self, x, cl, bn, L, Fout, K):
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
            x1 = my_sparse_mm()(L, x0)  # V x Fin*B
            x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
        for k in range(2, K):
            x2 = 2 * my_sparse_mm()(L, x1) - x0
            x = torch.cat((x, x2.unsqueeze(0)), 0)  # M x Fin*B
            x0, x1 = x1, x2

        x = x.view([K, V, Fin, B])  # K x V x Fin x B
        x = x.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
        x = x.view([B * V, Fin * K])  # B*V x Fin*K

        # Compose linearly Fin features to get Fout features
        x = cl(x)  # B*V x Fout
        if bn is not None:
            x = bn(x)  # B*V x Fout
        x = x.view([B, V, Fout])  # B x V x Fout

        return x

    # Upsampling of size p.
    def graph_upsample(self, x, p):
        if p > 1:
            x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
            x = nn.Upsample(scale_factor=p)(x)  # B x F x (V*p)
            x = x.permute(0, 2, 1).contiguous()  # x = B x (V*p) x F
            return x
        else:
            return x

    def forward(self, x):
        # x: B x num_input_chan
        x = self.fc(x)
        # x: B x (self.CL_F[0] * self.graph_L[-1].shape[0])
        x = x.view(-1, self.graph_L[-1].shape[0], self.CL_F[0])
        # x: B x 80 x 64

        cl_i = 0
        for block_i in range(len(self.CL_F) - 1):
            x = self.graph_upsample(x, 2)
            x = self.graph_upsample(x, 2)

            for layer_i in range(self.layers_per_block[block_i]):
                if layer_i is not self.layers_per_block[block_i] - 1:
                    Fout = self.CL_F[block_i]
                else:
                    Fout = self.CL_F[block_i + 1]

                x = self.graph_conv_cheby(x, self.cl[cl_i], self.bn[cl_i], self.graph_L[-(block_i * 2 + 3)],
                                          # 2 - block_i*2],
                                          Fout, self.CL_K[block_i])
                if block_i is not len(self.CL_F) - 2 or layer_i is not self.layers_per_block[block_i] - 1:
                    x = F.relu(x)

                cl_i = cl_i + 1

        return x  # x: B x 1280 x 3
