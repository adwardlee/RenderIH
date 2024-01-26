import numpy as np
import torch

def batch_persp_proj(joint, intr):
    joint_homo = torch.matmul(joint, intr.transpose(1, 2))
    joint2d = joint_homo / joint_homo[:, :, 2:]
    joint2d = joint2d[:, :, :2]
    return joint2d


def batch_proj2d(verts, camintr, camextr=None):
    # Project 3d vertices on image plane
    if camextr is not None:
        verts = camextr.bmm(verts.transpose(1, 2)).transpose(1, 2)
    verts_hom2d = camintr.bmm(verts.transpose(1, 2)).transpose(1, 2)
    proj_verts2d = verts_hom2d[:, :, :2] / verts_hom2d[:, :, 2:]
    return proj_verts2d


def flip_hand_side(target_side, hand_side):
    # Flip if needed
    if target_side == "right" and hand_side == "left":
        flip = True
        hand_side = "right"
    elif target_side == "left" and hand_side == "right":
        flip = True
        hand_side = "left"
    else:
        flip = False
    return hand_side, flip
