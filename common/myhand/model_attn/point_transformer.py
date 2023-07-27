import torch
from torch import nn, einsum
from einops import repeat, rearrange

from common.myhand.model_attn.inter_attn_lijun import MLP_res_block, weights_init
from common.myhand.model_attn.self_attn import SelfAttn
# helpers

# classes
class Pointatt(nn.Module):
    def __init__(self, f_dim, vert_dim, n_heads=8, d_q=None, d_v=None, dropout=0.1, num_neighbors=None, mask_percent=0):
        super().__init__()
        self.pos_mlp = nn.Sequential(
            nn.Linear(f_dim, 2 * f_dim),
            nn.ReLU(),
            nn.Linear(2 * f_dim, f_dim)
        )
        self.build_inter_attn(f_dim, n_heads, d_q, d_v, dropout)
        self.num_neighbors = num_neighbors
        self.vert_dim = vert_dim
        for m in self.modules():
            weights_init(m)

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
        inner_dim = f_dim
        attn_inner_dim = 2 * inner_dim

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.left_qs = nn.Linear(f_dim, n_heads * d_q)
        self.left_vs = nn.Linear(f_dim, n_heads * d_v)

        self.right_ks = nn.Linear(f_dim, n_heads * d_q)

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(f_dim, attn_inner_dim, 1, groups=n_heads),
            nn.ReLU(),
            nn.Conv2d(attn_inner_dim, f_dim, 1, groups=n_heads),
        )

        self.ffL = MLP_res_block(f_dim, 2 * f_dim, dropout)

    def forward(self, Lf, Rf, left_pos, right_pos, mask_L2R=None, mask_R2L=None):
        shape = Lf.shape
        h = self.n_heads
        rel_pos = left_pos[:, :, None, :] - right_pos[:, None, :, :]
        n = shape[1]
        rel_pos_emb = self.pos_mlp(rel_pos)
        rel_pos_emb = rearrange(rel_pos_emb, 'b i j (h d) -> b h i j d', h=h)

        BS, V, fdim = Lf.shape
        assert fdim == self.f_dim
        BS, V, fdim = Rf.shape
        assert fdim == self.f_dim

        Lq = self.left_qs(Lf).view(BS, V, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q
        Lv = self.left_vs(Lf).view(BS, V, self.n_heads, self.d_v).transpose(1, 2)  # BS x h x V x v

        Rk = self.right_ks(Rf).view(BS, V, self.n_heads, self.d_q).transpose(1, 2)  # BS x h x V x q

        qk_rel = rearrange(Lq, 'b h i d -> b h i 1 d') - rearrange(Rk, 'b h j d -> b h 1 j d')

        v = repeat(Lv, 'b h j d -> b h i j d', i=n)

        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first

        attn_mlp_input = qk_rel + rel_pos_emb
        attn_mlp_input = rearrange(attn_mlp_input, 'b h i j d -> b (h d) i j')

        sim = self.attn_mlp(attn_mlp_input)
        attn = sim.softmax(dim=-2)

        # aggregate

        v = rearrange(v, 'b h i j d -> b i j (h d)')
        agg_left = einsum('b d i j, b i j d -> b i d', attn, v)

        attn_R2L = self.dropout1(agg_left)
        Lf = self.ffL(Lf + attn_R2L)

        return Lf

class inter_point(nn.Module):
    def __init__(self, f_dim, vert_dim, n_heads=8, d_q=None, d_v=None, dropout=0.1, num_neighbors=None, mask_percent=0):
        super().__init__()

        self.L_self_attn_layer = SelfAttn(f_dim, n_heads=n_heads, hid_dim=f_dim, dropout=dropout)
        self.R_self_attn_layer = SelfAttn(f_dim, n_heads=n_heads, hid_dim=f_dim, dropout=dropout)
        self.pos_mlp = nn.Sequential(
            nn.Linear(f_dim, 2 * f_dim),
            nn.ReLU(),
            nn.Linear(2 * f_dim, f_dim)
        )
        self.left_trans = Pointatt(f_dim, vert_dim, n_heads, d_q, d_v, dropout)
        self.right_trans = Pointatt(f_dim, vert_dim, n_heads, d_q, d_v, dropout)
        self.num_neighbors = num_neighbors
        self.vert_dim = vert_dim
        # position_ids = torch.arange(vert_dim, dtype=torch.float32)
        # position_ids = position_ids.unsqueeze(0).repeat(f_dim, 1).unsqueeze(0)
        self.left_pos = nn.Parameter(torch.zeros(1, vert_dim, f_dim))
        self.right_pos = nn.Parameter(torch.zeros(1, vert_dim, f_dim))
        self.f_dim = f_dim
        for m in self.modules():
            weights_init(m)

    def forward(self, Lf, Rf, mask_L2R=None, mask_R2L=None):
        BS, V, fdim = Lf.shape
        assert fdim == self.f_dim
        BS, V, fdim = Rf.shape
        assert fdim == self.f_dim

        Lf = self.L_self_attn_layer(Lf)
        Rf = self.R_self_attn_layer(Rf)
        Lf = self.left_trans(Lf, Rf, self.left_pos, self.right_pos)
        Rf = self.right_trans(Rf, Lf, self.right_pos, self.left_pos)

        return Lf, Rf
