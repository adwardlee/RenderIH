import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import math
from .self_attn import SelfAttn

class MyBlock(nn.Module):

    def __init__(self, latent_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(latent_dim, 4 * latent_dim),
        )
        self.final_out = nn.Sequential(nn.Dropout(p=dropout),
                                       nn.Linear(4 *latent_dim, latent_dim))

    def forward(self, h):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        h = self.norm(h)
        h = self.out_layers(h)
        h = self.final_out(h)
        return h


class LinearTemporalCrossAttention(nn.Module):

    def __init__(self, latent_dim, num_head=8, dropout=0.1):
        super().__init__()
        self.num_head = num_head
        self.L_self_attn_layer = SelfAttn(latent_dim, n_heads=num_head, hid_dim=4 * latent_dim, dropout=dropout)
        self.R_self_attn_layer = SelfAttn(latent_dim, n_heads=num_head, hid_dim=4 * latent_dim, dropout=dropout)
        self.build_inter_attn(latent_dim, dropout=dropout)

    def build_inter_attn(self, f_dim, n_heads=4, d_q=None, d_v=None, dropout=0.1):
        if d_q is None:
            d_q = f_dim // n_heads
        if d_v is None:
            d_v = f_dim // n_heads

        self.n_heads = n_heads
        self.d_q = d_q
        self.d_v = d_v
        self.norm = d_q ** 0.5
        self.f_dim = f_dim

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.l_qs = nn.Linear(f_dim, 1)
        self.l_ks = nn.Linear(f_dim, n_heads * d_q)
        self.l_vs = nn.Linear(f_dim, n_heads * d_v)

        self.r_qs = nn.Linear(f_dim, 1)
        self.r_ks = nn.Linear(f_dim, n_heads * d_q)
        self.r_vs = nn.Linear(f_dim, n_heads * d_v)
        self.fc = nn.Linear(n_heads * d_v, f_dim)

        self.layer_norm1 = nn.LayerNorm(f_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(f_dim, eps=1e-6)
        self.ffL = MyBlock(f_dim, dropout)
        self.ffR = MyBlock(f_dim, dropout)

    def forward(self, Lf, Rf):
        BS, V, fdim = Lf.shape
        assert fdim == self.f_dim
        BS, V, fdim = Rf.shape
        assert fdim == self.f_dim

        Lf = self.L_self_attn_layer(Lf)
        Rf = self.R_self_attn_layer(Rf)

        Lf2 = self.layer_norm1(Lf)
        Rf2 = self.layer_norm2(Rf)

        Lq = self.l_qs(Lf2).view(BS, V, 1) # BS x v x 1
        Lk = self.l_ks(Lf2) # BS x V x C
        Lv = self.l_vs(Lf2)  # BS x V X C

        Rq = self.r_qs(Rf2).view(BS, V, 1)  # BS x V x 1
        Rk = self.r_ks(Rf2)  # BS x V X C
        Rv = self.r_vs(Rf2)  # BS x V x C


        weight_L = torch.softmax(Lq, dim=1) * Lk ### BS x V x C
        weight_L = torch.sum(weight_L, dim=1,keepdim=True) ### BS x 1 x C
        L_feat = self.ffL(Rv * weight_L + Lf)

        weight_R = torch.softmax(Rq, dim=1) * Rk
        weight_R = torch.sum(weight_R, dim=1, keepdim=True)
        R_feat = self.ffR(Lv * weight_R + Rf)


        return L_feat, R_feat