import cv2 as cv
import torch
import numpy as np
import torchvision.transforms as transforms
import math

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.myhand.lijun_model_graph import load_graph_model
from common.myhand.lijun_model_newgraph import load_new_model
from models.model import load_model
from utils.config import load_cfg
from utils.manoutils import imgUtils
from utils.vis_utils import mano_two_hands_renderer
from dataset.dataset_utils import IMG_SIZE



class GraphRender():
    def __init__(self,
                 cfg_path,
                 model_path,
                 input_size=IMG_SIZE,
                 render_size=256):
        self.input_size = input_size
        self.render_size = render_size
        self.renderer = mano_two_hands_renderer(img_size=render_size, device='cuda')
        self.left_faces = self.renderer.mano['left'].get_faces()
        self.right_faces = self.renderer.mano['right'].get_faces()

        cfg = load_cfg(cfg_path)
        if 'newgraph' in cfg.MODEL_NAME:
            self.model = load_new_model(cfg)
        elif 'ori' in cfg.MODEL_NAME:
            self.model = load_model(cfg)
        else:
            self.model = load_graph_model(cfg)
        state = torch.load(model_path, map_location='cpu')
        if 'network' in state:
            state = state['network']
        try:
            print(self.model.load_state_dict(state, strict=False), flush=True)
        except:
            state2 = {}
            for k, v in state.items():
                state2[k[7:]] = v
            print(self.model.load_state_dict(state2), flush=True)
        self.model.eval()
        self.model.cuda()

        self.img_processor = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

    def process_img(self, img):
        img = imgUtils.pad2squre(img)
        img = cv.resize(img, (self.input_size, self.input_size))
        imgTensor = torch.tensor(cv.cvtColor(img, cv.COLOR_BGR2RGB), dtype=torch.float32) / 255
        imgTensor = imgTensor.permute(2, 0, 1)
        imgTensor = self.img_processor(imgTensor).cuda().unsqueeze(0)
        return imgTensor

    @staticmethod
    def save_obj(path, verts, faces, color=None):
        with open(path, 'w') as file:
            for i in range(verts.shape[0]):
                if color is None:
                    file.write('v {} {} {}\n'.format(verts[i, 0], verts[i, 1], verts[i, 2]))
                else:
                    file.write('v {} {} {} {} {} {}\n'.format(verts[i, 0], verts[i, 1], verts[i, 2],
                                                              color[0], color[1], color[2]))
            for i in range(faces.shape[0]):
                file.write('f {} {} {}\n'.format(faces[i, 0] + 1, faces[i, 1] + 1, faces[i, 2] + 1))

    @torch.no_grad()
    def run_mymodel(self, imgTensor):
        result, paramsDict, handDictList, otherInfo = self.model(imgTensor)

        params = {}
        params['scale_left'] = paramsDict['scale']['left']
        params['trans2d_left'] = paramsDict['trans2d']['left']
        params['scale_right'] = paramsDict['scale']['right']
        params['trans2d_right'] = paramsDict['trans2d']['right']
        params['v3d_left'] = result['verts3d']['left']
        params['v3d_right'] = result['verts3d']['right']
        params['otherInfo'] = otherInfo
        params['mano_pose_left'] = otherInfo['verts3d_MANO_list']['left']['mano_pose']
        params['mano_pose_right'] = otherInfo['verts3d_MANO_list']['right']['mano_pose']
        params['mano_shape_left'] = otherInfo['verts3d_MANO_list']['left']['mano_shape']
        params['mano_shape_right'] = otherInfo['verts3d_MANO_list']['right']['mano_shape']
        params['scalelength_left'] = paramsDict['scalelength_left'].reshape(-1)
        params['scalelength_right'] = paramsDict['scalelength_right'].reshape(-1)
        params['root_rel'] = paramsDict['root_rel'].reshape(-1, 3)
        return params

    @torch.no_grad()
    def run_model(self, img):
        imgTensor = self.process_img(img)
        result, paramsDict, handDictList, otherInfo = self.model(imgTensor)

        params = {}
        params['scale_left'] = paramsDict['scale']['left']
        params['trans2d_left'] = paramsDict['trans2d']['left']
        params['scale_right'] = paramsDict['scale']['right']
        params['trans2d_right'] = paramsDict['trans2d']['right']
        params['v3d_left'] = result['verts3d']['left']
        params['v3d_right'] = result['verts3d']['right']
        params['otherInfo'] = otherInfo
        return params

    @torch.no_grad()
    def render(self, params, bg_img=None):
        #### for ours ###
        v_color = torch.zeros((778 * 2, 3))
        v_color[:778, 0] = 240  # int(1 * 255)
        v_color[:778, 1] = 210  # int(1 * 255)
        v_color[:778, 2] = 249  # int(1 * 255)
        v_color[778:, 0] = 255  # int(0.6 * 255)
        v_color[778:, 1] = 255  # int(0.6 * 255)
        v_color[778:, 2] = 255  # int(0.6 * 255)
        ### for synthetic ###
        # v_color[:778, 0] = 50  # int(1 * 255)
        # v_color[:778, 1] = 200  # int(1 * 255)
        # v_color[:778, 2] = 249  # int(1 * 255)
        # v_color[778:, 0] = 100  # int(0.6 * 255)
        # v_color[778:, 1] = 50  # int(0.6 * 255)
        # v_color[778:, 2] = 200  # int(0.6 * 255)
        ### for ori myhand ###
        # v_color[:778, 0] = 204
        # v_color[:778, 1] = 153
        # v_color[:778, 2] = 0
        # v_color[778:, 0] = 102
        # v_color[778:, 1] = 102
        # v_color[778:, 2] = 255

        img_out, mask_out = self.renderer.render_rgb_orth(scale_left=params['scale_left'],
                                                          trans2d_left=params['trans2d_left'],
                                                          scale_right=params['scale_right'],
                                                          trans2d_right=params['trans2d_right'],
                                                          v3d_left=params['v3d_left'],
                                                          v3d_right=params['v3d_right'],
                                                          v_color=v_color)
        img_out = img_out[0].detach().cpu().numpy() * 255
        mask_out = mask_out[0].detach().cpu().numpy()[..., np.newaxis]

        if bg_img is None:
            bg_img = np.ones_like(img_out) * 255
        else:
            bg_img = cv.resize(bg_img, (self.render_size, self.render_size))

        img_out = img_out * mask_out + bg_img * (1 - mask_out)
        img_out = img_out.astype(np.uint8)
        return img_out

    @torch.no_grad()
    def render_other_view(self, params, theta=60):
        c = (torch.mean(params['v3d_left'], axis=1) + torch.mean(params['v3d_right'], axis=1)).unsqueeze(1) / 2
        v3d_left = params['v3d_left'] - c
        v3d_right = params['v3d_right'] - c

        theta = 3.14159 / 180 * theta
        R = [[math.cos(theta), 0, math.sin(theta)],
             [0, 1, 0],
             [-math.sin(theta), 0, math.cos(theta)]]
        R = torch.tensor(R).float().cuda()

        v3d_left = torch.matmul(v3d_left, R)
        v3d_right = torch.matmul(v3d_right, R)

        img_out, mask_out = self.renderer.render_rgb_orth(scale_left=torch.ones((1,)).float().cuda() * 3,
                                                          scale_right=torch.ones((1,)).float().cuda() * 3,
                                                          trans2d_left=torch.zeros((1, 2)).float().cuda(),
                                                          trans2d_right=torch.zeros((1, 2)).float().cuda(),
                                                          v3d_left=v3d_left,
                                                          v3d_right=v3d_right)
        img_out = img_out[0].detach().cpu().numpy() * 255
        img = np.ones_like(img_out) * 255
        mask_out = mask_out[0].detach().cpu().numpy()[..., np.newaxis]
        img_out = img_out * mask_out + img * (1 - mask_out)
        img_out = img_out.astype(np.uint8)

        return img_out
