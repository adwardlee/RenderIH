import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import random
import cv2 as cv
from utils.tb_utils import tbUtils


@torch.no_grad()
def tb_vis_train(cfg, writer, idx, renderer, verts2d,
                 imgTensors, mask, dense,
                 paramsDictList, handDictList, otherInfo):
    img_size = imgTensors.shape[-1]
    tbUtils.add_image(writer, '0_input', idx,
                      imgTensors[0, [2, 1, 0]], dataformats='CHW')

    if 'mask' in otherInfo:
        tbUtils.add_image(writer, '1_img_aux/mask_gt', idx,
                          mask[0], dataformats='HW', clamp=True)
        tbUtils.add_image(writer, '1_img_aux/mask_pred', idx,
                          otherInfo['mask'][0], dataformats='HW', clamp=True)
    if 'dense' in otherInfo:
        tbUtils.add_image(writer, '1_img_aux/dense_gt', idx,
                          dense[0], dataformats='CHW', clamp=True)
        tbUtils.add_image(writer, '1_img_aux/dense_pred', idx,
                          otherInfo['dense'][0] * mask[0].unsqueeze(0), dataformats='CHW', clamp=True)

    tbUtils.draw_verts(writer, '2_mano/vert_gt', idx,
                       imgTensors[0], verts2d[0],
                       color=(0, 0, 255),
                       parent=None)

    for itIdx in range(len(paramsDictList)):
        img, mask = renderer.render_rgb(scale=paramsDictList[itIdx]['scale'][:1],
                                        trans2d=paramsDictList[itIdx]['trans2d'][:1],
                                        v3d=handDictList[itIdx]['verts3d'][:1])
        img = img[0] * mask[0].unsqueeze(-1) + imgTensors[0].permute(1, 2, 0) * (1 - mask[0].unsqueeze(-1))
        tbUtils.add_image(writer, '2_mano/vert_out_{}'.format(itIdx), idx,
                          img[..., [2, 1, 0]], dataformats='HWC')

    if 'attnList' in otherInfo:
        v_idx = random.randint(0, otherInfo['v2dList'][itIdx].shape[1] - 1)
        for itIdx in range(3):
            v2d = otherInfo['v2dList'][itIdx][0, v_idx].detach().cpu().numpy()
            attn = torch.sum(otherInfo['attnList'][itIdx][0], dim=0)
            attn = attn[v_idx].detach().cpu().numpy()
            attn = attn / attn.max()
            attn = cv.resize(attn, (img_size, img_size))
            img = torch.clamp(imgTensors[0], 0, 1) * 255
            img = img.detach().cpu().numpy()
            img = img.transpose(1, 2, 0)
            temp = attn[..., np.newaxis] * img
            temp = temp.copy().astype(np.uint8)
            cv.circle(temp, (int(v2d[0]), int(v2d[1])), 2, (0, 0, 255), -1)
            temp = torch.from_numpy(temp).float() / 255
            tbUtils.add_image(writer, '3_attn/{}'.format(itIdx).format(itIdx), idx,
                              temp[..., [2, 1, 0]], dataformats='HWC')


@torch.no_grad()
def tb_vis_train_gcn(cfg, writer, idx, renderer, verts2d_left, verts2d_right,
                     imgTensors, mask, dense,
                     result, paramsDict, handDictList, otherInfo):
    img_size = imgTensors.shape[-1]
    tbUtils.add_image(writer, '0_input', idx,
                      imgTensors[0, [2, 1, 0]], dataformats='CHW')

    if 'mask' in otherInfo:
        tbUtils.add_image(writer, '1_img_aux/mask_gt', idx,
                          mask[0, 0] * 0.5 + mask[0, 1], dataformats='HW', clamp=True)
        tbUtils.add_image(writer, '1_img_aux/mask_pred', idx,
                          otherInfo['mask'][0, 0] * 0.5 + otherInfo['mask'][0, 1], dataformats='HW', clamp=True)
    if 'dense' in otherInfo:
        tbUtils.add_image(writer, '1_img_aux/dense_gt', idx,
                          dense[0], dataformats='CHW', clamp=True)
        tbUtils.add_image(writer, '1_img_aux/dense_pred', idx,
                          otherInfo['dense'][0, :3] * mask[0, :1] + otherInfo['dense'][0, 3:] * mask[0, 1:],
                          dataformats='CHW', clamp=True)

    tbUtils.draw_verts(writer, '2_mano/vert_gt', idx,
                       imgTensors[0], [verts2d_left[0], verts2d_right[0]],
                       color=[(0, 0, 255), (255, 0, 0)])

    img, mask = renderer.render_rgb_orth(scale_left=paramsDict['scale']['left'][:1],
                                         scale_right=paramsDict['scale']['right'][:1],
                                         trans2d_left=paramsDict['trans2d']['left'][:1],
                                         trans2d_right=paramsDict['trans2d']['right'][:1],
                                         v3d_left=result['verts3d']['left'][:1],
                                         v3d_right=result['verts3d']['right'][:1])
    img = img[0] * mask[0].unsqueeze(-1) + imgTensors[0].permute(1, 2, 0) * (1 - mask[0].unsqueeze(-1))
    tbUtils.add_image(writer, '2_mano/vert_out_result', idx,
                      img[..., [2, 1, 0]], dataformats='HWC')

    for itIdx in range(len(handDictList)):
        img, mask = renderer.render_rgb_orth(scale_left=paramsDict['scale']['left'][:1],
                                             scale_right=paramsDict['scale']['right'][:1],
                                             trans2d_left=paramsDict['trans2d']['left'][:1],
                                             trans2d_right=paramsDict['trans2d']['right'][:1],
                                             v3d_left=otherInfo['verts3d_MANO_list']['left'][itIdx][:1],
                                             v3d_right=otherInfo['verts3d_MANO_list']['right'][itIdx][:1]
                                             )
        img = img[0] * mask[0].unsqueeze(-1) + imgTensors[0].permute(1, 2, 0) * (1 - mask[0].unsqueeze(-1))
        tbUtils.add_image(writer, '2_mano/vert_out_{}'.format(itIdx), idx,
                          img[..., [2, 1, 0]], dataformats='HWC')
