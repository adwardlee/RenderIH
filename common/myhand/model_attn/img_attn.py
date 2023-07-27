import torch
import torch.nn as nn
import torch.nn.functional as F

from .self_attn import SelfAttn


def weights_init(layer):
    classname = layer.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)


class MLP_res_block(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, in_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x):
        x = self.fc2(self.dropout1(F.relu(self.fc1(x))))
        return self.dropout2(x)

    def forward(self, x):
        x = x + self._ff_block(self.layer_norm(x))
        return x


class img_feat_to_grid(nn.Module):
    def __init__(self, img_size, img_f_dim, grid_size, grid_f_dim, n_heads=4, dropout=0.01):
        super().__init__()
        self.img_f_dim = img_f_dim
        self.img_size = img_size
        self.grid_f_dim = grid_f_dim
        self.grid_size = grid_size
        self.position_embeddings = nn.Embedding(grid_size * grid_size, grid_f_dim)

        patch_size = img_size // grid_size
        self.proj = nn.Conv2d(img_f_dim, grid_f_dim, kernel_size=patch_size, stride=patch_size)
        self.self_attn = SelfAttn(grid_f_dim, n_heads=n_heads, hid_dim=grid_f_dim, dropout=dropout)

    def forward(self, img):
        bs = img.shape[0]
        assert img.shape[1] == self.img_f_dim
        assert img.shape[2] == self.img_size
        assert img.shape[3] == self.img_size

        position_ids = torch.arange(self.grid_size * self.grid_size, dtype=torch.long, device=img.device)
        position_ids = position_ids.unsqueeze(0).repeat(bs, 1)
        position_embeddings = self.position_embeddings(position_ids)

        grid_feat = F.relu(self.proj(img))
        grid_feat = grid_feat.view(bs, self.grid_f_dim, -1).transpose(-1, -2)
        grid_feat = grid_feat + position_embeddings

        grid_feat = self.self_attn(grid_feat)

        return grid_feat


class img_attn(nn.Module):
    def __init__(self, verts_f_dim, img_f_dim, n_heads=4, d_q=None, d_v=None, dropout=0.1):
        super().__init__()
        self.img_f_dim = img_f_dim
        self.verts_f_dim = verts_f_dim

        self.fc = nn.Linear(img_f_dim, verts_f_dim)
        self.Attn = SelfAttn(verts_f_dim, n_heads=n_heads, hid_dim=verts_f_dim, dropout=dropout)

    def forward(self, verts_f, img_f):
        assert verts_f.shape[2] == self.verts_f_dim
        assert img_f.shape[2] == self.img_f_dim
        assert verts_f.shape[0] == img_f.shape[0]
        V = verts_f.shape[1]

        img_f = self.fc(img_f)

        x = torch.cat([verts_f, img_f], dim=1)
        x = self.Attn(x)

        verts_f = x[:, :V]

        return verts_f


class img_ex(nn.Module):
    def __init__(self, img_size, img_f_dim,
                 grid_size, grid_f_dim,
                 verts_f_dim,
                 n_heads=4,
                 dropout=0.01):
        super().__init__()
        self.verts_f_dim = verts_f_dim
        self.encoder = img_feat_to_grid(img_size, img_f_dim, grid_size, grid_f_dim, n_heads, dropout)
        self.attn = img_attn(verts_f_dim, grid_f_dim, n_heads=n_heads, dropout=dropout)

        for m in self.modules():
            weights_init(m)

    def forward(self, img, verts_f):
        assert verts_f.shape[2] == self.verts_f_dim
        grid_feat = self.encoder(img)
        verts_f = self.attn(verts_f, grid_feat)
        return verts_f
