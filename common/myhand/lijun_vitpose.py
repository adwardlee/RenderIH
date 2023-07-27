import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

from models.vitpose.vitpose import vit_base_patch16_224, PatchEmbed, Myattention, vit_large_patch16_224
from common.myhand.decoder_lijun_newgraph import load_decoder

from common.myhand.config import load_cfg
from main.config import cfg as cfgs

class HandNET_GCN(nn.Module):
    def __init__(self, encoder, decoder, cliff):
        super(HandNET_GCN, self).__init__()
        self.encoder = encoder
        self.patch_embed = PatchEmbed(img_size=256, patch_size=8, embed_dim=encoder.embed_dim)
        self.conv1 = nn.Conv2d(kernel_size=1, in_channels=encoder.embed_dim, out_channels=encoder.embed_dim, stride=1)
        self.downsample = Myattention(encoder.embed_dim, encoder.embed_dim)
        self.decoder = decoder
        self.embedding_dim = encoder.embed_dim
        self.cliff = cliff

    def forward(self, img):
        feature16x16 = self.encoder(img)## 768, 16, 16 ### 2048,8,8; 1024,16,16; 512,32,32; 256, 64, 64; ###128,8,8; 128,16,16; 128,32,32; 128,64,64
        feature32x32, _ = self.patch_embed(img)
        feature32x32 = feature32x32.permute(0, 2, 1).reshape(-1, self.embedding_dim, 32, 32).contiguous()
        feature32x32 =  self.conv1(feature32x32 + F.interpolate(feature16x16, scale_factor=2))
        feature8x8 = self.downsample(feature16x16, 16, 16)
        fmaps = [feature8x8, feature16x16, feature32x32]
        global_feature = F.adaptive_avg_pool2d(feature16x16, 1).reshape(-1, self.embedding_dim)
        result = self.decoder(global_feature, fmaps)
        return result


def load_vit(cfg, cliff=False):
    if isinstance(cfg, str):
        cfg = load_cfg(cfg)
    print('start graph nohms model', flush=True)
    cfg.mano_flag = cfgs.mano_flag
    cfg.render = cfgs.render
    cfg.normal = cfgs.normal
    cfg.edge = cfgs.edge
    cfg.vert2d = cfgs.vert2d
    cfg.dice = cfgs.dice
    cfg.sdf = cfgs.sdf
    cfg.lambda_sdf = cfgs.lambda_sdf
    cfg.lambda_render = cfgs.lambda_render
    cfg.lambda_normal = cfgs.lambda_normal
    cfg.lambda_edge = cfgs.lambda_edge
    cfg.sdf_thresh = cfgs.sdf_thresh
    cfg.data_type = cfgs.data_type
    cfg.reverse = cfgs.reverse
    if cfg.MODEL_NAME == 'vit_large':
        print('loading vit large :')
        encoder = vit_large_patch16_224(pretrained=True)
        cfg.channels = 1024
    else:
        print('loading vit base :')
        encoder = vit_base_patch16_224(pretrained=True)
        cfg.channels = 768

    decoder = load_decoder(cfg, encoder.embed_dim)
    model = HandNET_GCN(encoder, decoder, cliff)

    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path = os.path.join(abspath, str(cfg.MODEL_PARAM.MODEL_PRETRAIN_PATH)) # 'models/wild_demo.pth'#
    #path = 'results/intag_b64_hasmano/model_dump/snapshot_45.pth.tar'
    if os.path.exists(path):
        state = torch.load(path, map_location='cpu')
        print('load model params from {}'.format(path))
        try:
            print(model.load_state_dict(state))#['network']))
        except:
            state2 = {}
            for k, v in state.items():
                state2[k[7:]] = v
            print(model.load_state_dict(state2, strict=False))#['network']))
        print('succcessssully  load model params from {}'.format(path))
    return model
