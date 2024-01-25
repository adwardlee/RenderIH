import torch
import numpy as np
from manopth.axislayer import AxisLayer
from hocontact.postprocess.geo_loss import HandLoss
from manopth.rodrigues_layer import batch_rodrigues


class AnatomyMetric:
    def __init__(self):
        self.axislayer = AxisLayer()

    @staticmethod
    def joint_b_axis_loss(b_axis, axis):
        b_soft_idx = [0, 3, 9, 6, 14]
        b_thumb_soft_idx = [12, 13]

        b_axis_cos = torch.einsum("bij,bij->bi", b_axis, axis)  # [B, 15]
        restrict_cos = b_axis_cos[:, [i for i in range(15) if i not in b_soft_idx and i not in b_thumb_soft_idx]]
        soft_loss = torch.relu(torch.abs(b_axis_cos[:, b_soft_idx]) - np.cos(np.pi / 2 - np.pi / 36))  # [-5, 5]
        thumb_soft_loss = torch.relu(
            torch.abs(b_axis_cos[:, b_thumb_soft_idx]) - np.cos(np.pi / 2 - np.pi / 3)
        )  # [-60, 60]

        res = (
            torch.mean(torch.pow(restrict_cos, 2))
            + torch.mean(torch.pow(soft_loss, 2))
            + 0.01 * torch.mean(torch.pow(thumb_soft_loss, 2))
        )
        return res

    @staticmethod
    def joint_u_axis_loss(u_axis, axis):
        u_soft_idx = [0, 3, 9, 6, 14]
        u_thumb_soft_idx = [12, 13]

        u_axis_cos = torch.einsum("bij,bij->bi", u_axis, axis)
        restrict_cos = u_axis_cos[:, [i for i in range(15) if i not in u_soft_idx and i not in u_thumb_soft_idx]]
        soft_loss = torch.relu(torch.abs(u_axis_cos[:, u_soft_idx]) - np.cos(np.pi / 2 - np.pi / 18))  # [-10, 10]
        thumb_soft_loss = torch.relu(
            torch.abs(u_axis_cos[:, u_thumb_soft_idx]) - np.cos(np.pi / 2 - np.pi / 3)
        )  # [-60, 60]

        res = (
            torch.mean(torch.pow(restrict_cos, 2))
            + torch.mean(torch.pow(soft_loss, 2))
            + 0.01 * torch.mean(torch.pow(thumb_soft_loss, 2))
        )
        return res

    @staticmethod
    def joint_l_limit_loss(l_axis, axis):
        l_soft_idx = [0, 3, 9, 6, 14]
        l_thumb_soft_idx = [12, 13]
        l_axis_cos = torch.einsum("bij,bij->bi", l_axis, axis)
        restrict_cos = l_axis_cos[:, [i for i in range(15) if i not in l_soft_idx and i not in l_thumb_soft_idx]]
        soft_loss = torch.relu(-l_axis_cos[:, l_soft_idx] + 1 - np.cos(np.pi / 2 - np.pi / 9))  # [-20, 20]
        thumb_soft_loss = torch.relu(-l_axis_cos[:, l_thumb_soft_idx] + 1 - np.cos(np.pi / 2 - np.pi / 3))

        res = (
            torch.mean(torch.pow(restrict_cos - 1, 2))
            + torch.mean(torch.pow(soft_loss, 2))
            + 0.01 * torch.mean(torch.pow(thumb_soft_loss, 2))
        )
        return res

    @staticmethod
    def rotation_angle_loss(angle, limit_angle=np.pi / 2, eps=1e-10):
        angle_new = torch.zeros_like(angle)  # TENSOR[B, 15]
        nonzero_mask = torch.abs(angle) > eps  # TENSOR[B, 15], bool
        angle_new[nonzero_mask] = angle[nonzero_mask]  # if angle is too small, pick them out of backward graph
        angle_over_limit = torch.relu(angle_new - limit_angle)  # < np.pi/2, 0; > np.pi/2, linear | Tensor[16, ]
        angle_over_limit_squared = torch.pow(angle_over_limit, 2)  # TENSOR[15, ]
        res = torch.mean(angle_over_limit_squared)
        return res

    def compute_loss(self, batch_full_pose_aa, batch_hand_joints):
        batch_size = batch_full_pose_aa.shape[0]
        hand_rotmatrix = batch_rodrigues(batch_full_pose_aa.view(-1, 3))
        hand_rotmatrix = hand_rotmatrix.view(batch_size, -1, 3, 3)
        b_axis, u_axis, l_axis = self.axislayer(batch_hand_joints, hand_rotmatrix)  # [B, 15, 3] each
        axis = batch_full_pose_aa.view(batch_size, -1, 3)[:, 1:, :]  # ignore global rot
        angle = torch.norm(axis, dim=-1, keepdim=False)
        angle_loss = AnatomyMetric.rotation_angle_loss(angle)
        joint_b_loss = AnatomyMetric.joint_b_axis_loss(b_axis, axis)
        joint_u_loss = AnatomyMetric.joint_u_axis_loss(u_axis, axis)
        joint_l_loss = AnatomyMetric.joint_l_limit_loss(l_axis, axis)
        return angle_loss + 0.1 * joint_b_loss + 0.1 * joint_u_loss + 0.1 * joint_l_loss
