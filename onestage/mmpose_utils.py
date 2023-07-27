import numpy as np
import cv2
import math
import torch
import os
from torchvision.transforms import Normalize

from main.config import cfg as cfgs

def get_graph_model(model_path, cfg='common/myhand/defaults.yaml'):
    from common.myhand.lijun_model_graph import HandNET_GCN
    from common.myhand.config import load_cfg
    from common.myhand.encoder_lijun import load_encoder
    from common.myhand.decoder_lijun_graph import load_decoder
    if isinstance(cfg, str):
        cfg = load_cfg(cfg)
    print('start graph nohms model', flush=True)
    cfg.mano_flag = True
    cfg.render = False
    cfg.normal = False
    cfg.edge = False
    cfg.vert2d = False
    cfg.dice = False
    cfg.sdf = False
    cfg.lambda_sdf = 0
    cfg.lambda_render = 0
    cfg.lambda_normal = 0
    cfg.lambda_edge = 0
    cfg.sdf_thresh = 0
    cfg.data_type = 'bac'
    encoder, mid_model = load_encoder(cfg)

    decoder = load_decoder(cfg, mid_model.get_info())
    model = HandNET_GCN(encoder, mid_model, decoder, False)

    assert os.path.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))
    ckpt = torch.load(model_path)
    print(model.load_state_dict(ckpt['network'], strict=True))
    model = model.cuda()
    model.eval()
    return model

class Bbox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

class constants:
    FOCAL_LENGTH = 5000.
    IMG_RES = 256

    # Mean and standard deviation for normalizing input image
    IMG_NORM_MEAN = [0.485, 0.456, 0.406]
    IMG_NORM_STD = [0.229, 0.224, 0.225]


def process_image(img):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    img = img[:, :, ::-1].copy()
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

    norm_img = img.astype(np.float32) / 255.
    norm_img = torch.from_numpy(norm_img).permute(2, 0, 1)
    norm_img = normalize_img(norm_img.clone())#[None]
    norm_img = norm_img[None]
    return norm_img

def get_color(two=0, left=0):
    if two:
        v_color = torch.zeros((778 * 2, 3))
        v_color[:778, 0] = 255  # int(1 * 255)
        v_color[:778, 1] = 255  # int(1 * 255)
        v_color[:778, 2] = 255 # int(1 * 255)
        v_color[778:, 0] = 240  # int(0.6 * 255)
        v_color[778:, 1] = 210  # int(0.6 * 255)
        v_color[778:, 2] = 249  # int(0.6 * 255)
    elif left:
        v_color = torch.zeros((778, 3))
        v_color[:778, 0] = 255  # int(1 * 255)
        v_color[:778, 1] = 255  # int(1 * 255)
        v_color[:778, 2] = 255  # int(1 * 255)
    else:
        v_color = torch.zeros((778, 3))
        v_color[:778, 0] = 240  # int(1 * 255)
        v_color[:778, 1] = 210  # int(1 * 255)
        v_color[:778, 2] = 249  # int(1 * 255)
    return v_color

def rendering(render, scale_left, trans2d_left, scale_right, trans2d_right, v3d_left, v3d_right, left_img=None, right_img=None,
              union_img=None, two=0, single=0, left=0, right=0):
    if two:
        v_color = get_color(two=1)
        if single:
            img_left, mask_left = render.render_single(scale=scale_left,
                                                     trans2d=trans2d_left,
                                                     v3d=v3d_left,
                                                     v_color=v_color[:778], left=True)
            img_right, mask_right = render.render_single(scale=scale_right,
                                                     trans2d=trans2d_right,
                                                     v3d=v3d_right,
                                                     v_color=v_color[778:], left=False)
            img_left = img_left[0].detach().cpu().numpy() * 255
            mask_left = mask_left[0].detach().cpu().numpy()[..., np.newaxis]
            left_hand = img_left * mask_left
            img_left = img_left * mask_left + left_img[:,:,::-1] * (1 - mask_left)
            img_left = img_left.astype(np.uint8)
            img_left = cv2.cvtColor(img_left, cv2.COLOR_RGB2BGR)

            img_right = img_right[0].detach().cpu().numpy() * 255
            mask_right = mask_right[0].detach().cpu().numpy()[..., np.newaxis]
            img_right = img_right * mask_right + right_img[:,:,::-1] * (1 - mask_right)
            img_right = img_right.astype(np.uint8)
            img_right = cv2.cvtColor(img_right, cv2.COLOR_RGB2BGR)
            return img_left, img_right, left_hand, mask_left
        else:
            img, mask = render.render_rgb_orth(scale_left=scale_left,
                                                          scale_right=scale_right,
                                                          trans2d_left=trans2d_left,
                                                          trans2d_right=trans2d_right,
                                                          v3d_left=v3d_left,
                                                          v3d_right=v3d_right, v_color=v_color)
            img_out = img[0].detach().cpu().numpy() * 255
            mask_out = mask[0].detach().cpu().numpy()[..., np.newaxis]
            img_out = img_out * mask_out + union_img[:,:,::-1] * (1 - mask_out)
            img_out = img_out.astype(np.uint8)
            img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
            return img_out
    elif left:
        v_color = get_color(two=0, left=1)
        img_left, mask_left = render.render_single(scale=scale_left ,
                                                   trans2d=trans2d_left,
                                                   v3d=v3d_left,
                                                   v_color=v_color, left=True)
        img_out = img_left[0].detach().cpu().numpy() * 255
        mask_out = mask_left[0].detach().cpu().numpy()[..., np.newaxis]
        img_out = img_out * mask_out + left_img[:,:,::-1] * (1 - mask_out)
        img_out = img_out.astype(np.uint8)
        img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        return img_out
    elif right:
        v_color = get_color(two=0, left=0)
        img_right, mask_right = render.render_single(scale=scale_right,
                                                     trans2d=trans2d_right,
                                                     v3d=v3d_right,
                                                     v_color=v_color, left=False)
        img_out = img_right[0].detach().cpu().numpy() * 255
        mask_out = mask_right[0].detach().cpu().numpy()[..., np.newaxis]
        img_out = img_out * mask_out + right_img[:,:,::-1] * (1 - mask_out)
        img_out = img_out.astype(np.uint8)
        img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
        return img_out