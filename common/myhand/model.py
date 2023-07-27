import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

from common.myhand.encoder import load_encoder
from common.myhand.decoder import load_decoder
from common.myhand.bbox_decoder import load_decoder_cliff

from common.myhand.config import load_cfg
from main.config import cfg as cfgs

class HandNET_GCN(nn.Module):
    def __init__(self, encoder, mid_model, decoder, cliff):
        super(HandNET_GCN, self).__init__()
        self.encoder = encoder
        self.mid_model = mid_model
        self.decoder = decoder
        self.cliff = cliff

    def forward(self, oriimg, target, otherinfo, mode):
        img = oriimg['img']
        hms, mask, dp, img_fmaps, hms_fmaps, dp_fmaps = self.encoder(img)
        global_feature, fmaps = self.mid_model(img_fmaps, hms_fmaps, dp_fmaps)
        if self.cliff:
            bbox_info = oriimg['bbox_info'].cuda()
            result = self.decoder(global_feature, fmaps, target, bbox_info, mode)
        else:
            result = self.decoder(global_feature, fmaps, target, mode)
        if mode == 'train':
            result['hms'] = cfgs.lambda_hms * (F.mse_loss(hms[:,:21], target['left']['hms']) + F.mse_loss(hms[:,21:], target['right']['hms']))
            result['mask'] = cfgs.lambda_mask * (F.mse_loss(mask[:,0:1], target['left']['mask']) + F.mse_loss(mask[:,1:2], target['right']['mask']))
            result['dense'] = cfgs.lambda_dp * (F.l1_loss(dp, target['left']['dense']))

        return result


def load_model(cfg, cliff=False):
    if isinstance(cfg, str):
        cfg = load_cfg(cfg)
    print('start myhand model', flush=True)
    cfg.mano_flag = cfgs.mano_flag
    cfg.render = cfgs.render
    cfg.normal = cfgs.normal
    cfg.edge = cfgs.edge
    cfg.vert2d = cfgs.vert2d
    encoder, mid_model = load_encoder(cfg)
    if cliff:
        print('load cliff decoder ', flush=True)
        decoder = load_decoder_cliff(cfg, mid_model.get_info())
    else:
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
