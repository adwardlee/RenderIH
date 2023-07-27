import torch
import numpy as np
import cv2 as cv


MANO_PARENT = [-1, 0, 1, 2, 3,
               0, 5, 6, 7,
               0, 9, 10, 11,
               0, 13, 14, 15,
               0, 17, 18, 19]

MANO_COLOR = [[100, 100, 100],
              [100, 0, 0],
              [150, 0, 0],
              [200, 0, 0],
              [255, 0, 0],
              [100, 100, 0],
              [150, 150, 0],
              [200, 200, 0],
              [255, 255, 0],
              [0, 100, 50],
              [0, 150, 75],
              [0, 200, 100],
              [0, 255, 125],
              [0, 50, 100],
              [0, 75, 150],
              [0, 100, 200],
              [0, 125, 255],
              [100, 0, 100],
              [150, 0, 150],
              [200, 0, 200],
              [255, 0, 255]]


def draw_mano_joints(img, joints):
    for i in range(21):
        cv.circle(img, (int(joints[i, 0]), int(joints[i, 1])),
                  2, tuple(MANO_COLOR[i]), -1)
    for i in range(1, 21):
        cv.line(img,
                (int(joints[i, 0]), int(joints[i, 1])),
                (int(joints[MANO_PARENT[i], 0]), int(joints[MANO_PARENT[i], 1])),
                tuple(MANO_COLOR[i]),
                2)
    return img


class tbUtils():
    @ staticmethod
    def draw_verts(writer, name, idx,
                   imgTensor, vertTensor,
                   color=(0, 0, 255),
                   BGR=True, CHW=True):
        with torch.no_grad():
            img = torch.clamp(imgTensor, 0, 1) * 255
            img = img.detach().cpu().numpy().astype(np.uint8)

        if CHW:
            img = img.transpose(1, 2, 0)
        img = img.copy()

        if not isinstance(vertTensor, list):
            vertTensor = [vertTensor]
        if not isinstance(color, list):
            color = [color]

        for j in range(len(vertTensor)):
            verts2d = vertTensor[j].detach().cpu().long().numpy()  # N x 2
            for i in range(verts2d.shape[0]):
                cv.circle(img, (verts2d[i, 0], verts2d[i, 1]), 1, color[j])

        if BGR:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        writer.add_image(name,
                         torch.from_numpy(img).float() / 255,
                         idx,
                         dataformats='HWC')

    def draw_MANO_joints(writer, name, idx,
                         imgTensor, jointsTensor,
                         BGR=True, CHW=True):
        with torch.no_grad():
            img = torch.clamp(imgTensor, 0, 1) * 255
            img = img.detach().cpu().numpy().astype(np.uint8)

        if CHW:
            img = img.transpose(1, 2, 0)

        joints2d = jointsTensor.detach().cpu().long().numpy()  # 21 x 2
        img = img.copy()

        img = draw_mano_joints(img, joints2d)

        if BGR:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        writer.add_image(name,
                         torch.from_numpy(img).float() / 255,
                         idx,
                         dataformats='HWC')

    @ staticmethod
    def add_image(writer, name, idx,
                  imgTensor, dataformats='HW', clamp=False):
        if clamp:
            imgTensor = torch.clamp(imgTensor, 0, 1)
        writer.add_image(name,
                         imgTensor.float(),
                         idx,
                         dataformats=dataformats)
