import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

class Transformer(nn.Module):
    def __init__(self, inp_res=28, dim=256, depth=2, num_heads=4, mlp_ratio=4., injection=True): ### change llj input_res h // 8
        super().__init__()

        self.injection=injection
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, injection=injection))

        if self.injection:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim*2, dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(dim*2, dim, 1, padding=0),
            )

    def forward(self, query, key):
        output = query
        for i, layer in enumerate(self.layers):
            output = layer(query=output, key=key)
        
        if self.injection:
            output = torch.cat([key, output], dim=1)
            output = self.conv1(output) + self.conv2(output)

        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, query, key, value, query2, key2, use_sigmoid):
        B, N, C = query.shape
        query = query.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        key = key.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        value = value.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
            
        if use_sigmoid:
            query2 = query2.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            key2 = key2.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            attn2 = torch.matmul(query2, key2.transpose(-2, -1)) * self.scale
            attn2 = torch.sum(attn2, dim=-1)
            attn2 = self.sigmoid(attn2)
            attn = attn * attn2.unsqueeze(3) 
        
        x = torch.matmul(attn, value).transpose(1, 2).reshape(B, N, C)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm, injection=True):
        super().__init__()

        self.injection = injection

        self.channels = dim

        self.encode_value = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.encode_query = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.encode_key = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)

        if self.injection:
            self.encode_query2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
            self.encode_key2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)

        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.q_embedding = nn.Parameter(torch.randn(1, 256, 32, 32))###28 change llj h // 8
        self.k_embedding = nn.Parameter(torch.randn(1, 256, 32, 32))###28 change llj h // 8

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, query, key, query_embed=None, key_embed=None):
        b, c, h, w = query.shape
        query_embed = repeat(self.q_embedding, '() n c d -> b n c d', b = b)
        key_embed = repeat(self.k_embedding, '() n c d -> b n c d', b = b)

        q_embed = self.with_pos_embed(query, query_embed)
        k_embed = self.with_pos_embed(key, key_embed)

        v = self.encode_value(key).view(b, self.channels, -1)
        v = v.permute(0, 2, 1)

        q = self.encode_query(q_embed).view(b, self.channels, -1)
        q = q.permute(0, 2, 1)

        k = self.encode_key(k_embed).view(b, self.channels, -1)
        k = k.permute(0, 2, 1)
        
        query = query.view(b, self.channels, -1).permute(0, 2, 1)

        if self.injection:
            q2 = self.encode_query2(q_embed).view(b, self.channels, -1)
            q2 = q2.permute(0, 2, 1)

            k2 = self.encode_key2(k_embed).view(b, self.channels, -1)
            k2 = k2.permute(0, 2, 1)

            query = self.attn(query=q, key=k, value=v,query2 = q2, key2 = k2, use_sigmoid=True)
        else:
            q2 = None
            k2 = None

            query = query + self.attn(query=q, key=k, value=v, query2 = q2, key2 = k2, use_sigmoid=False)
 
        query = query + self.mlp(self.norm2(query))
        query = query.permute(0, 2, 1).contiguous().view(b, self.channels, h, w)

        return query
