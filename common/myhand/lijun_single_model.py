import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

from common.myhand.encoder_lijun import load_encoder
from common.myhand.decoder_lijun_single import load_decoder

from common.myhand.config import load_cfg
from main.config import cfg as cfgs

class HandNET_GCN(nn.Module):
    def __init__(self, encoder, mid_model, decoder, cliff):
        super(HandNET_GCN, self).__init__()
        self.encoder = encoder
        self.mid_model = mid_model
        self.decoder = decoder
        self.cliff = cliff

    def forward(self, img, ):
        #img = oriimg['img']
        img_fmaps = self.encoder(img)### 2048,8,8; 1024,16,16; 512,32,32; 256, 64, 64; ###128,8,8; 128,16,16; 128,32,32; 128,64,64
        global_feature, fmaps = self.mid_model(img_fmaps)
        result, paramsDict, handDictList, otherInfo = self.decoder(global_feature, fmaps)

        return result, paramsDict, handDictList, otherInfo


def load_single_model(cfg, cliff=False):
    if isinstance(cfg, str):
        cfg = load_cfg(cfg)
    print('start graph nohms model', flush=True)
    cfg.mano_flag = False
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
    encoder, mid_model = load_encoder(cfg)
    decoder = load_decoder(cfg, mid_model.get_info())
    model = HandNET_GCN(encoder, mid_model, decoder, cliff)

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
