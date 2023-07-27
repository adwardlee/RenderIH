import argparse
import cv2 as cv
import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms

class Jr():
    def __init__(self, J_regressor,
                 device='cuda'):
        self.device = device
        self.process_J_regressor(J_regressor)

    def process_J_regressor(self, J_regressor):
        J_regressor = J_regressor.clone().detach()
        tip_regressor = torch.zeros_like(J_regressor[:5])
        tip_regressor[0, 745] = 1.0
        tip_regressor[1, 317] = 1.0
        tip_regressor[2, 444] = 1.0
        tip_regressor[3, 556] = 1.0
        tip_regressor[4, 673] = 1.0
        J_regressor = torch.cat([J_regressor, tip_regressor], dim=0)
        new_order = [0, 13, 14, 15, 16,
                     1, 2, 3, 17,
                     4, 5, 6, 18,
                     10, 11, 12, 19,
                     7, 8, 9, 20]
        self.J_regressor = J_regressor[new_order].contiguous().to(self.device)

    def __call__(self, v):
        return torch.matmul(self.J_regressor, v)


def compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # print('X1', X1.shape)

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1 ** 2)

    # print('var', var1.shape)

    # 3. The outer product of X1 and X2.
    K = X1.mm(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)
    # V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[0], device=S1.device)
    Z[-1, -1] *= torch.sign(torch.det(U @ V.T))
    # Construct R.
    R = V.mm(Z.mm(U.T))

    # print('R', X1.shape)

    # 5. Recover scale.
    scale = torch.trace(R.mm(K)) / var1
    # print(R.shape, mu1.shape)
    # 6. Recover translation.
    t = mu2 - scale * (R.mm(mu1))
    # print(t.shape)

    # 7. Error:
    S1_hat = scale * R.mm(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def batch_compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0,2,1)

    return S1_hat

def eval_hand(verts_left_gt, verts_right_gt, joints_left_gt, joints_right_gt, verts_left_pred, verts_right_pred,
         joints_left_pred, joints_right_pred, joints_loss, verts_loss, pajoints_loss, paverts_loss):
    root_left_gt = joints_left_gt[:, 9:10]
    root_right_gt = joints_right_gt[:, 9:10]
    length_left_gt = torch.linalg.norm(joints_left_gt[:, 9] - joints_left_gt[:, 0], dim=-1)
    length_right_gt = torch.linalg.norm(joints_right_gt[:, 9] - joints_right_gt[:, 0], dim=-1)
    joints_left_gt = joints_left_gt - root_left_gt
    verts_left_gt = verts_left_gt - root_left_gt
    joints_right_gt = joints_right_gt - root_right_gt
    verts_right_gt = verts_right_gt - root_right_gt


    root_left_pred = joints_left_pred[:, 9:10]
    root_right_pred = joints_right_pred[:, 9:10]
    length_left_pred = torch.linalg.norm(joints_left_pred[:, 9] - joints_left_pred[:, 0], dim=-1)
    length_right_pred = torch.linalg.norm(joints_right_pred[:, 9] - joints_right_pred[:, 0], dim=-1)
    scale_left = (length_left_gt / length_left_pred).unsqueeze(-1).unsqueeze(-1)
    scale_right = (length_right_gt / length_right_pred).unsqueeze(-1).unsqueeze(-1)

    ori_joints_left_pred = (joints_left_pred - root_left_pred)
    ori_joints_right_pred = (joints_right_pred - root_right_pred)
    ori_verts_left_pred = (verts_left_pred - root_left_pred)
    ori_verts_right_pred = (verts_right_pred - root_right_pred)

    joints_left_pred = (joints_left_pred - root_left_pred) * scale_left
    verts_left_pred = (verts_left_pred - root_left_pred) * scale_left
    joints_right_pred = (joints_right_pred - root_right_pred) * scale_right
    verts_right_pred = (verts_right_pred - root_right_pred) * scale_right

    joint_left_loss = torch.linalg.norm((joints_left_pred - joints_left_gt), ord=2, dim=-1)
    joints_loss['left'].append(joint_left_loss.cpu().numpy())

    joint_right_loss = torch.linalg.norm((joints_right_pred - joints_right_gt), ord=2, dim=-1)
    joints_loss['right'].append(joint_right_loss.cpu().numpy())

    vert_left_loss = torch.linalg.norm((verts_left_pred - verts_left_gt), ord=2, dim=-1)
    verts_loss['left'].append(vert_left_loss.cpu().numpy())

    vert_right_loss = torch.linalg.norm((verts_right_pred - verts_right_gt), ord=2, dim=-1)
    verts_loss['right'].append(vert_right_loss.cpu().numpy())

    pajoints_loss['left'].append(joint_left_loss.cpu().numpy())
    pajoints_loss['right'].append(joint_right_loss.cpu().numpy())
    paverts_loss['left'].append(vert_left_loss.cpu().numpy())
    paverts_loss['right'].append(vert_right_loss.cpu().numpy())

    ##### real pampjpe calculation
    # Absolute error (MPJPE)
    # # errors_left = torch.sqrt(((ori_joints_left_pred - joints_left_gt) ** 2).sum(dim=-1)).mean(dim=-1)
    # S1_hat = batch_compute_similarity_transform_torch(ori_joints_left_pred.detach().cpu(), joints_left_gt.detach().cpu())
    # errors_pa_left = torch.sqrt(((S1_hat - joints_left_gt.detach().cpu()) ** 2).sum(dim=-1)).mean(dim=-1)
    # pajoints_loss['left'].append(errors_pa_left.cpu().numpy())
    #
    # # errors_right = torch.sqrt(((ori_joints_right_pred - joints_right_gt) ** 2).sum(dim=-1)).mean(
    # #     dim=-1)
    # S1_hat = batch_compute_similarity_transform_torch(ori_joints_right_pred.detach().cpu(), joints_right_gt.detach().cpu())
    # errors_pa_right = torch.sqrt(((S1_hat - joints_right_gt.detach().cpu()) ** 2).sum(dim=-1)).mean(dim=-1)
    # pajoints_loss['right'].append(errors_pa_right.cpu().numpy())
    #
    # # errors_vert_left = torch.sqrt(((ori_verts_left_pred - verts_left_gt) ** 2).sum(dim=-1)).mean(
    # #     dim=-1)
    # S1_hat = batch_compute_similarity_transform_torch(ori_verts_left_pred.detach().cpu(), verts_left_gt.detach().cpu())
    # errors_verts_left = torch.sqrt(((S1_hat - verts_left_gt.detach().cpu()) ** 2).sum(dim=-1)).mean(dim=-1)
    # paverts_loss['left'].append(errors_verts_left.cpu().numpy())
    #
    # # errors_vert_right = torch.sqrt(((ori_verts_right_pred - verts_right_gt) ** 2).sum(dim=-1)).mean(
    # #     dim=-1)
    # S1_hat = batch_compute_similarity_transform_torch(ori_verts_right_pred.detach().cpu(), verts_right_gt.detach().cpu())
    # errors_verts_right = torch.sqrt(((S1_hat - verts_right_gt.detach().cpu()) ** 2).sum(dim=-1)).mean(dim=-1)
    # paverts_loss['right'].append(errors_verts_right.cpu().numpy())
    return

def eval_hand2(verts_left_gt, verts_right_gt, joints_left_gt, joints_right_gt, verts_left_pred, verts_right_pred,
               joints_left_pred, joints_right_pred, joints_loss, verts_loss, pajoints_loss, paverts_loss):
    # J_regressor = {'left': Jr(mano_left.joint_regressor_torch),
    #                'right': Jr(mano_right.joint_regressor_torch)}
    # joints_left_gt = J_regressor['left'](verts_left_gt)
    # joints_right_gt = J_regressor['right'](verts_right_gt)

    root_left_gt = joints_left_gt[:, 0:1]
    root_right_gt = joints_right_gt[:, 0:1]
    length_left_gt = torch.linalg.norm(joints_left_gt[:, 9] - joints_left_gt[:, 0], dim=-1)
    length_right_gt = torch.linalg.norm(joints_right_gt[:, 9] - joints_right_gt[:, 0], dim=-1)
    joints_left_gt = joints_left_gt - root_left_gt
    verts_left_gt = verts_left_gt - root_left_gt
    joints_right_gt = joints_right_gt - root_right_gt
    verts_right_gt = verts_right_gt - root_right_gt

    verts_left_pred = verts_left_pred
    verts_right_pred = verts_right_pred
    # joints_left_pred = J_regressor['left'](verts_left_pred)
    # joints_right_pred = J_regressor['right'](verts_right_pred)

    root_left_pred = joints_left_pred[:, 0:1]
    root_right_pred = joints_right_pred[:, 0:1]
    length_left_pred = torch.linalg.norm(joints_left_pred[:, 9] - joints_left_pred[:, 0], dim=-1)
    length_right_pred = torch.linalg.norm(joints_right_pred[:, 9] - joints_right_pred[:, 0], dim=-1)
    scale_left = (length_left_gt / length_left_pred).unsqueeze(-1).unsqueeze(-1)
    scale_right = (length_right_gt / length_right_pred).unsqueeze(-1).unsqueeze(-1)

    joints_left_pred = (joints_left_pred - root_left_pred) * scale_left
    verts_left_pred = (verts_left_pred - root_left_pred) * scale_left
    joints_right_pred = (joints_right_pred - root_right_pred) * scale_right
    verts_right_pred = (verts_right_pred - root_right_pred) * scale_right

    joint_left_loss = torch.linalg.norm((joints_left_pred - joints_left_gt), ord=2, dim=-1)
    joint_left_loss = joint_left_loss.detach().cpu().numpy()
    joints_loss['left'].append(joint_left_loss)

    joint_right_loss = torch.linalg.norm((joints_right_pred - joints_right_gt), ord=2, dim=-1)
    joint_right_loss = joint_right_loss.detach().cpu().numpy()
    joints_loss['right'].append(joint_right_loss)

    vert_left_loss = torch.linalg.norm((verts_left_pred - verts_left_gt), ord=2, dim=-1)
    vert_left_loss = vert_left_loss.detach().cpu().numpy()
    verts_loss['left'].append(vert_left_loss)

    vert_right_loss = torch.linalg.norm((verts_right_pred - verts_right_gt), ord=2, dim=-1)
    vert_right_loss = vert_right_loss.detach().cpu().numpy()
    verts_loss['right'].append(vert_right_loss)

    pajoints_loss['left'].append(joint_left_loss)
    pajoints_loss['right'].append(joint_right_loss)
    paverts_loss['left'].append(vert_left_loss)
    paverts_loss['right'].append(vert_right_loss)
    return