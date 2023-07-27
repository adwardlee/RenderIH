import argparse
import cv2 as cv
import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
import sys
from glob import glob

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import load_model
from models.manolayer import ManoLayer
from utils.config import load_cfg
# from utils.vis_utils import mano_two_hands_renderer
from utils.manoutils import get_mano_path
from dataset.dataset_utils import IMG_SIZE, cut_img
from dataset.interhand import fix_shape, InterHand_dataset

from common.myhand.lijun_model_graph import load_graph_model
from common.myhand.lijun_model_newgraph import load_new_model
from utils.eval_metrics import compute_cdev, nanmean
from common.utils.loss_utils import DiceLoss, SDFLoss, TryLoss

class Tzionas_dataset():
    def __init__(self, data_path):

        mano_path = get_mano_path()
        self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                           'left': ManoLayer(mano_path['left'], center_idx=None)}
        fix_shape(self.mano_layer)

        self.data_path = data_path
        self.size = len(glob(os.path.join(data_path, 'all', '*.npy')))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        thedict = np.load(os.path.join(self.data_path, 'all', '{}.npy'.format(idx)), allow_pickle=True)
        thedict = thedict[()]
        hand_dict = {'left': {}, 'right': {}}
        hand_dict['left'] = thedict['left']
        hand_dict['right'] = thedict['right']
        img = thedict['img']
        left_v3d = thedict['left']['verts3d']
        left_j3d = thedict['left']['joints3d']

        right_v3d = thedict['right']['verts3d']
        right_j3d = thedict['right']['joints3d']
        return img, left_j3d, left_v3d, right_j3d, right_v3d

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

def compute_similarity_transform(S1, S2):
    """Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat

def get_alignMesh(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re, S1_hat, S2

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


class handDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, hand_dict = self.dataset[idx]
        img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
        imgTensor = torch.tensor(cv.cvtColor(img, cv.COLOR_BGR2RGB), dtype=torch.float32) / 255
        imgTensor = imgTensor.permute(2, 0, 1)
        imgTensor = self.normalize_img(imgTensor)

        # maskTensor = torch.tensor(mask, dtype=torch.float32) / 255

        joints_left_gt = torch.from_numpy(hand_dict['left']['joints3d']).float()
        verts_left_gt = torch.from_numpy(hand_dict['left']['verts3d']).float()
        joints_right_gt = torch.from_numpy(hand_dict['right']['joints3d']).float()
        verts_right_gt = torch.from_numpy(hand_dict['right']['verts3d']).float()

        return imgTensor, joints_left_gt, verts_left_gt, joints_right_gt, verts_right_gt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='misc/model/config.yaml')
    parser.add_argument("--model", type=str, default='misc/model/interhand.pth')#'output/ori_res50/model/exp/170.pth')
    parser.add_argument("--data_path", type=str, default='/mnt/user/E-shenfei.llj-356552/data/dataset/interhand_5fps/interhand_data/')
    parser.add_argument("--bs", type=int, default=1024)
    opt = parser.parse_args()

    iou_list = 'tzionas_iou.npy'
    theiou = np.load(iou_list)
    iou00 = (theiou == 0)
    iou02 = np.logical_and(theiou < 0.2, theiou > 0)
    iou04 = np.logical_and(theiou < 0.4, theiou >= 0.2)
    iou1 = (theiou >= 0.4)
    opt.map = False

    # network = load_new_model(opt.cfg)
    network = load_graph_model(opt.cfg)
    # network = load_model(opt.cfg)

    state = torch.load(opt.model, map_location='cpu')
    if 'network' in state:
        state = state['network']
    try:
        print(network.load_state_dict(state, strict=False))
    except:
        state2 = {}
        for k, v in state.items():
            state2[k[7:]] = v
        print(network.load_state_dict(state2))

    network.eval()
    network.cuda()

    mano_path = get_mano_path()
    mano_layer = {'left': ManoLayer(mano_path['left'], center_idx=None),
                  'right': ManoLayer(mano_path['right'], center_idx=None)}
    fix_shape(mano_layer)
    J_regressor = {'left': Jr(mano_layer['left'].J_regressor),
                   'right': Jr(mano_layer['right'].J_regressor)}

    faces_left = mano_layer['left'].get_faces()
    faces_right = mano_layer['right'].get_faces()
    dataset = Tzionas_dataset('/mnt/user/E-shenfei.llj-356552/workgroup/lijun/hand_dataset/tziona/original/')
    # dataset = handDataset(InterHand_dataset(opt.data_path, split='test'))
    dataloader = DataLoader(dataset, batch_size=opt.bs, shuffle=False,
                            num_workers=4, drop_last=False, pin_memory=True)
    joints_loss = {'left': [], 'right': []}
    verts_loss = {'left': [], 'right': []}

    orijoint_loss = {'left': [], 'right': []}
    orivert_loss = {'left': [], 'right': []}

    pajoints_loss = {'left': [], 'right': []}
    paverts_loss = {'left': [], 'right': []}

    double_predvert = []
    double_predjoint = []
    double_gtvert = []
    double_gtjoint = []

    double_joint_loss = []
    double_vert_loss = []

    double_pajoints_loss = []
    double_paverts_loss = []

    ##### new add #####
    pred_trans_list = []
    gt_trans_list = []

    all_pred_left = []
    all_pred_right = []
    all_gt_left = []
    all_gt_right = []

    idx = 0
    total_time = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            imgTensors = data[0].cuda()
            joints_left_gt = data[1].to(torch.float32).cuda()
            verts_left_gt = data[2].to(torch.float32).cuda()
            joints_right_gt = data[3].to(torch.float32).cuda()
            verts_right_gt = data[4].to(torch.float32).cuda()

            joints_left_gt = J_regressor['left'](verts_left_gt)
            joints_right_gt = J_regressor['right'](verts_right_gt)

            start_time = time.time()
            result, paramsDict, handDictList, otherInfo = network(imgTensors)
            end_time = time.time()

            oriverts_leftgt = verts_left_gt.clone()
            oriverts_rightgt = verts_right_gt.clone()
            orijoints_leftgt = joints_left_gt.clone()
            orijoints_rightgt = joints_right_gt.clone()
            pred_vleft = result['verts3d']['left']
            pred_vright = result['verts3d']['right']
            pred_jleft = J_regressor['left'](pred_vleft)
            pred_jright = J_regressor['right'](pred_vright)

            root_left_gt = joints_left_gt[:, 9:10]  # [:, 9:10]
            root_right_gt = joints_right_gt[:, 9:10]  # [:, 9:10]
            ##### new #####
            gt_trans = root_left_gt - root_right_gt
            gt_trans_list.append(gt_trans.cpu().numpy())
            all_pred_left.append(pred_vleft)
            all_pred_right.append(pred_vright)
            all_gt_left.append(verts_left_gt)
            all_gt_right.append(verts_right_gt)

            length_left_gt = torch.linalg.norm(joints_left_gt[:, 9] - joints_left_gt[:, 0], dim=-1)
            length_right_gt = torch.linalg.norm(joints_right_gt[:, 9] - joints_right_gt[:, 0], dim=-1)
            joints_left_gt = joints_left_gt - root_left_gt
            verts_left_gt = verts_left_gt - root_left_gt
            joints_right_gt = joints_right_gt - root_right_gt
            verts_right_gt = verts_right_gt - root_right_gt

            verts_left_pred = result['verts3d']['left']
            verts_right_pred = result['verts3d']['right']
            joints_left_pred = J_regressor['left'](verts_left_pred)
            joints_right_pred = J_regressor['right'](verts_right_pred)

            root_left_pred = joints_left_pred[:, 9:10]  # [:, 9:10]
            root_right_pred = joints_right_pred[:, 9:10]  # [:, 9:10]
            length_left_pred = torch.linalg.norm(joints_left_pred[:, 9] - joints_left_pred[:, 0], dim=-1)
            length_right_pred = torch.linalg.norm(joints_right_pred[:, 9] - joints_right_pred[:, 0], dim=-1)
            scale_left = (length_left_gt / length_left_pred).unsqueeze(-1).unsqueeze(-1)
            scale_right = (length_right_gt / length_right_pred).unsqueeze(-1).unsqueeze(-1)

            ori_joints_left_pred = (joints_left_pred - root_left_pred)
            ori_joints_right_pred = (joints_right_pred - root_right_pred)
            ori_verts_left_pred = (verts_left_pred - root_left_pred)
            ori_verts_right_pred = (verts_right_pred - root_right_pred)

            ### ori mpjpe ###
            joint_left_loss = torch.linalg.norm((ori_joints_left_pred - joints_left_gt), ord=2, dim=-1)
            orijoint_loss['left'].append(joint_left_loss.cpu().numpy())

            joint_right_loss = torch.linalg.norm((ori_joints_right_pred - joints_right_gt), ord=2, dim=-1)
            orijoint_loss['right'].append(joint_right_loss.cpu().numpy())

            vert_left_loss = torch.linalg.norm((ori_verts_left_pred - verts_left_gt), ord=2, dim=-1)
            orivert_loss['left'].append(vert_left_loss.cpu().numpy())

            vert_right_loss = torch.linalg.norm((ori_verts_right_pred - verts_right_gt), ord=2, dim=-1)
            orivert_loss['right'].append(vert_right_loss.cpu().numpy())

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

            # pajoints_loss['left'].append(ori_joints_left_pred)
            # pajoints_loss['right'].append(ori_joints_right_pred)
            # paverts_loss['left'].append(ori_verts_left_pred)
            # paverts_loss['right'].append(ori_verts_left_pred)

            # Absolute error (MPJPE)
            errors_left = torch.sqrt(((ori_joints_left_pred - joints_left_gt) ** 2).sum(dim=-1)).mean(dim=-1)
            S1_hat = batch_compute_similarity_transform_torch(ori_joints_left_pred, joints_left_gt)
            errors_pa_left = torch.sqrt(((S1_hat - joints_left_gt) ** 2).sum(dim=-1)).mean(dim=-1)
            pajoints_loss['left'].append(errors_pa_left.cpu().numpy())

            errors_right = torch.sqrt(((ori_joints_right_pred - joints_right_gt) ** 2).sum(dim=-1)).mean(
                dim=-1)
            S1_hat = batch_compute_similarity_transform_torch(ori_joints_right_pred, joints_right_gt)
            errors_pa_right = torch.sqrt(((S1_hat - joints_right_gt) ** 2).sum(dim=-1)).mean(dim=-1)
            pajoints_loss['right'].append(errors_pa_right.cpu().numpy())

            errors_vert_left = torch.sqrt(((ori_verts_left_pred - verts_left_gt) ** 2).sum(dim=-1)).mean(
                dim=-1)
            S1_hat = batch_compute_similarity_transform_torch(ori_verts_left_pred, verts_left_gt)
            errors_verts_left = torch.sqrt(((S1_hat - verts_left_gt) ** 2).sum(dim=-1)).mean(dim=-1)
            paverts_loss['left'].append(errors_verts_left.cpu().numpy())

            errors_vert_right = torch.sqrt(((ori_verts_right_pred - verts_right_gt) ** 2).sum(dim=-1)).mean(
                dim=-1)
            S1_hat = batch_compute_similarity_transform_torch(ori_verts_right_pred, verts_right_gt)
            errors_verts_right = torch.sqrt(((S1_hat - verts_right_gt) ** 2).sum(dim=-1)).mean(dim=-1)
            paverts_loss['right'].append(errors_verts_right.cpu().numpy())

            ### left relative to right root ###
            length_trans = pred_jleft[:, 9:10] - root_right_pred
            pred_trans_list.append(length_trans.cpu().numpy())
            length_left = pred_jleft[:, 9:10] - root_left_pred
            gt_length_left = orijoints_leftgt[:, 9:10, :] - orijoints_leftgt[:, 0:1, :]
            gt_trans_left = orijoints_leftgt[:, 9:10, :] - orijoints_rightgt[:, 0:1, :]
            double_leftvert = (pred_vleft - pred_jright[:, 0:1, :]) / (length_trans + 1e-8) * length_left
            double_leftjoint = (pred_jleft - pred_jright[:, 0:1, :]) / (length_trans + 1e-8) * length_left
            double_predvert.append(
                np.concatenate((double_leftvert.cpu().numpy(), (pred_vright - pred_jright[:, 0:1, :]).cpu().numpy()),
                               axis=1))
            double_predjoint.append(
                np.concatenate((double_leftjoint.cpu().numpy(), (pred_jright - pred_jright[:, 0:1, :]).cpu().numpy()),
                               axis=1))
            double_gtvert.append(np.concatenate((((oriverts_leftgt - orijoints_rightgt[:, 0:1, :]) / (
                    gt_trans_left + 1e-8) * gt_length_left).cpu().numpy(),
                                                 verts_right_gt.cpu().numpy()), axis=1))
            double_gtjoint.append(np.concatenate((((orijoints_leftgt - orijoints_rightgt[:, 0:1, :]) / (
                    gt_trans_left + 1e-8) * gt_length_left).cpu().numpy(),
                                                  joints_right_gt.cpu().numpy()), axis=1))
            total_time += (end_time - start_time)
            idx += 1
            # print('avg time : ', total_time / idx, flush=True)

    orijoint_loss['left'] = np.concatenate(orijoint_loss['left'], axis=0)
    orijoint_loss['right'] = np.concatenate(orijoint_loss['right'], axis=0)
    orivert_loss['left'] = np.concatenate(orivert_loss['left'], axis=0)
    orivert_loss['right'] = np.concatenate(orivert_loss['right'], axis=0)

    joints_loss['left'] = np.concatenate(joints_loss['left'], axis=0)
    joints_loss['right'] = np.concatenate(joints_loss['right'], axis=0)
    verts_loss['left'] = np.concatenate(verts_loss['left'], axis=0)
    verts_loss['right'] = np.concatenate(verts_loss['right'], axis=0)

    pajoints_loss['left'] = np.concatenate(pajoints_loss['left'], axis=0)
    pajoints_loss['right'] = np.concatenate(pajoints_loss['right'], axis=0)
    paverts_loss['left'] = np.concatenate(paverts_loss['left'], axis=0)
    paverts_loss['right'] = np.concatenate(paverts_loss['right'], axis=0)

    orijoint_left = orijoint_loss['left'].mean() * 1000
    orijoint_right = orijoint_loss['right'].mean() * 1000
    orivert_left = orivert_loss['left'].mean() * 1000
    orivert_right = orivert_loss['right'].mean() * 1000

    joints_mean_loss_left = joints_loss['left'].mean() * 1000
    joints_mean_loss_right = joints_loss['right'].mean() * 1000
    verts_mean_loss_left = verts_loss['left'].mean() * 1000
    verts_mean_loss_right = verts_loss['right'].mean() * 1000

    #### new #####
    pred_trans_list = np.concatenate(pred_trans_list, axis=0)
    gt_trans_list = np.concatenate(gt_trans_list, axis=0)
    mrrpe = np.sqrt(((pred_trans_list - gt_trans_list) ** 2).sum(axis=1))
    print('iou 0.0 mrrpe: {}'.format(mrrpe[iou00].mean()), flush=True)
    print('iou 0.2 mrrpe: {}'.format(mrrpe[iou02].mean()), flush=True)
    print('iou 0.4 mrrpe: {}'.format(mrrpe[iou04].mean()), flush=True)
    print('iou >0.4 mrrpe: {}'.format(mrrpe[iou1].mean()), flush=True)
    print(' mrrpe: shape {}, value {} '.format(mrrpe.shape, mrrpe.mean()), flush=True)

    all_pred_left = torch.concat(all_pred_left, dim=0).cpu()
    all_pred_right = torch.concat(all_pred_right, dim=0).cpu()
    all_gt_left = torch.concat(all_gt_left, dim=0).cpu()
    all_gt_right = torch.concat(all_gt_right, dim=0).cpu()
    error = compute_cdev(all_pred_left, all_pred_right, all_gt_left, all_gt_right)
    print('iou 0.0 cdev: {}'.format(nanmean(error[iou00])), flush=True)
    print('iou 0.2 cdev: {}'.format(nanmean(error[iou02])), flush=True)
    print('iou 0.4 cdev: {}'.format(nanmean(error[iou04])), flush=True)
    print('iou > 0.4 cdev: {}'.format(nanmean(error[iou1])), flush=True)
    print(' cdev: ', nanmean(error), flush=True)

    print('ori joint mpjpe: ')
    print('    left: {} mm, right: {} mm'.format(orijoint_left, orijoint_right))
    print('    all: {} mm'.format((orijoint_left + orijoint_right) / 2))
    print('iou 0.0 ori joint mpjpe: {}'.format(orijoint_loss['left'][iou00].mean() * 500 +
                                               orijoint_loss['right'][iou00].mean() * 500), flush=True)
    print('iou 0.0 ori joint mpjpe std: {}'.format(orijoint_loss['left'][iou00].std() * 500 +
                                               orijoint_loss['right'][iou00].std() * 500), flush=True)
    print('iou 0.2 ori joint mpjpe: {}'.format(orijoint_loss['left'][iou02].mean() * 500 +
                                               orijoint_loss['right'][iou02].mean() * 500), flush=True)
    print('iou 0.2 ori joint mpjpe std: {}'.format(orijoint_loss['left'][iou02].std() * 500 +
                                               orijoint_loss['right'][iou02].std() * 500), flush=True)
    print('iou 0.4 ori joint mpjpe: {}'.format(orijoint_loss['left'][iou04].mean() * 500 +
                                               orijoint_loss['right'][iou04].mean() * 500), flush=True)
    print('iou 0.4 ori joint mpjpe std: {}'.format(orijoint_loss['left'][iou04].std() * 500 +
                                               orijoint_loss['right'][iou04].std() * 500), flush=True)
    print('iou > 0.4 ori joint mpjpe: {}'.format(orijoint_loss['left'][iou1].mean() * 500 +
                                               orijoint_loss['right'][iou1].mean() * 500), flush=True)
    print('iou > 0.4 ori joint mpjpe std: {}'.format(orijoint_loss['left'][iou1].std() * 500 +
                                                 orijoint_loss['right'][iou1].std() * 500), flush=True)

    print('ori vert mean error:')
    print('    left: {} mm, right: {} mm'.format(orivert_left, orivert_right))
    print('    all: {} mm'.format((orivert_left + orivert_right) / 2))

    print('joint mean error:')
    print('    left: {} mm, right: {} mm'.format(joints_mean_loss_left, joints_mean_loss_right))
    print('    all: {} mm'.format((joints_mean_loss_left + joints_mean_loss_right) / 2))
    print('iou 0.0 joint mpjpe: {}'.format(joints_loss['left'][iou00].mean() * 500 +
                                           joints_loss['right'][iou00].mean() * 500), flush=True)
    print('iou 0.0 joint mpjpe std: {}'.format(joints_loss['left'][iou00].std() * 500 +
                                               joints_loss['right'][iou00].std() * 500), flush=True)
    print('iou 0.2 joint mpjpe std: {}'.format(joints_loss['left'][iou02].std() * 500 +
                                               joints_loss['right'][iou02].std() * 500), flush=True)
    print('iou 0.4 joint mpjpe: {}'.format(joints_loss['left'][iou04].mean() * 500 +
                                               joints_loss['right'][iou04].mean() * 500), flush=True)
    print('iou 0.4 joint mpjpe std: {}'.format(joints_loss['left'][iou04].std() * 500 +
                                               joints_loss['right'][iou04].std() * 500), flush=True)
    print('iou > 0.4 joint mpjpe: {}'.format(joints_loss['left'][iou1].mean() * 500 +
                                                 joints_loss['right'][iou1].mean() * 500), flush=True)
    print('iou > 0.4 joint mpjpe std: {}'.format(joints_loss['left'][iou1].std() * 500 +
                                             joints_loss['right'][iou1].std() * 500), flush=True)
    print('vert mean error:')
    print('    left: {} mm, right: {} mm'.format(verts_mean_loss_left, verts_mean_loss_right))
    print('    all: {} mm'.format((verts_mean_loss_left + verts_mean_loss_right) / 2))

    joints_mean_loss_left = pajoints_loss['left'].mean() * 1000
    joints_mean_loss_right = pajoints_loss['right'].mean() * 1000
    verts_mean_loss_left = paverts_loss['left'].mean() * 1000
    verts_mean_loss_right = paverts_loss['right'].mean() * 1000

    print('joint pa _ mean error:')
    print('    left_pa: {} mm, right: {} mm'.format(joints_mean_loss_left, joints_mean_loss_right))
    print('    all_pa: {} mm'.format((joints_mean_loss_left + joints_mean_loss_right) / 2))
    print('vert mean error:')
    print('    left_pa: {} mm, right: {} mm'.format(verts_mean_loss_left, verts_mean_loss_right))
    print('    all_pa: {} mm'.format((verts_mean_loss_left + verts_mean_loss_right) / 2))

    gt_mesh = np.array(np.concatenate(double_gtvert, axis=0))
    gt_3djoint = np.array(np.concatenate(double_gtjoint, axis=0))
    pred_mesh = np.array(np.concatenate(double_predvert, axis=0))
    pred_3djoint = np.array(np.concatenate(double_predjoint, axis=0))
    pa_mesh_error, _, _ = get_alignMesh(pred_mesh, gt_mesh)
    pa_joint_error, _, _ = get_alignMesh(pred_3djoint, gt_3djoint)
    print('double PAMPJPE: {}'.format(1000 * np.mean(pa_joint_error)))
    print('double PAMPVPE: {}'.format(1000 * np.mean(pa_mesh_error)))
    error_mpjpe = np.mean(np.sqrt(((pred_3djoint - gt_3djoint) ** 2).sum(axis=-1)).mean(axis=-1))
    print('double MPJPE: {}'.format(1000 * error_mpjpe))
    error_mpjpe = np.mean(np.sqrt(((pred_mesh - gt_mesh) ** 2).sum(axis=-1)).mean(axis=-1))
    print('double MPVPE: {}'.format(1000 * error_mpjpe))

    # joints_loss = {'left': [], 'right': []}
    # verts_loss = {'left': [], 'right': []}
    #
    # with torch.no_grad():
    #     for data in tqdm(dataloader):

    #         imgTensors = data[0].cuda()
    #         joints_left_gt = data[2].cuda()
    #         verts_left_gt = data[3].cuda()
    #         joints_right_gt = data[4].cuda()
    #         verts_right_gt = data[5].cuda()
    #
    #         joints_left_gt = J_regressor['left'](verts_left_gt)
    #         joints_right_gt = J_regressor['right'](verts_right_gt)
    #
    #         root_left_gt = joints_left_gt[:, 9:10]
    #         root_right_gt = joints_right_gt[:, 9:10]
    #         length_left_gt = torch.linalg.norm(joints_left_gt[:, 9] - joints_left_gt[:, 0], dim=-1)
    #         length_right_gt = torch.linalg.norm(joints_right_gt[:, 9] - joints_right_gt[:, 0], dim=-1)
    #         joints_left_gt = joints_left_gt - root_left_gt
    #         verts_left_gt = verts_left_gt - root_left_gt
    #         joints_right_gt = joints_right_gt - root_right_gt
    #         verts_right_gt = verts_right_gt - root_right_gt
    #
    #         result, paramsDict, handDictList, otherInfo = network(imgTensors)
    #
    #         verts_left_pred = result['verts3d']['left']
    #         verts_right_pred = result['verts3d']['right']
    #         joints_left_pred = J_regressor['left'](verts_left_pred)
    #         joints_right_pred = J_regressor['right'](verts_right_pred)
    #
    #         root_left_pred = joints_left_pred[:, 9:10]
    #         root_right_pred = joints_right_pred[:, 9:10]
    #         length_left_pred = torch.linalg.norm(joints_left_pred[:, 9] - joints_left_pred[:, 0], dim=-1)
    #         length_right_pred = torch.linalg.norm(joints_right_pred[:, 9] - joints_right_pred[:, 0], dim=-1)
    #         scale_left = (length_left_gt / length_left_pred).unsqueeze(-1).unsqueeze(-1)
    #         scale_right = (length_right_gt / length_right_pred).unsqueeze(-1).unsqueeze(-1)
    #
    #         joints_left_pred = (joints_left_pred - root_left_pred) * scale_left
    #         verts_left_pred = (verts_left_pred - root_left_pred) * scale_left
    #         joints_right_pred = (joints_right_pred - root_right_pred) * scale_right
    #         verts_right_pred = (verts_right_pred - root_right_pred) * scale_right
    #
    #         joint_left_loss = torch.linalg.norm((joints_left_pred - joints_left_gt), ord=2, dim=-1)
    #         joint_left_loss = joint_left_loss.detach().cpu().numpy()
    #         joints_loss['left'].append(joint_left_loss)
    #
    #         joint_right_loss = torch.linalg.norm((joints_right_pred - joints_right_gt), ord=2, dim=-1)
    #         joint_right_loss = joint_right_loss.detach().cpu().numpy()
    #         joints_loss['right'].append(joint_right_loss)
    #
    #         vert_left_loss = torch.linalg.norm((verts_left_pred - verts_left_gt), ord=2, dim=-1)
    #         vert_left_loss = vert_left_loss.detach().cpu().numpy()
    #         verts_loss['left'].append(vert_left_loss)
    #
    #         vert_right_loss = torch.linalg.norm((verts_right_pred - verts_right_gt), ord=2, dim=-1)
    #         vert_right_loss = vert_right_loss.detach().cpu().numpy()
    #         verts_loss['right'].append(vert_right_loss)
    #
    # joints_loss['left'] = np.concatenate(joints_loss['left'], axis=0)
    # joints_loss['right'] = np.concatenate(joints_loss['right'], axis=0)
    # verts_loss['left'] = np.concatenate(verts_loss['left'], axis=0)
    # verts_loss['right'] = np.concatenate(verts_loss['right'], axis=0)
    #
    # joints_mean_loss_left = joints_loss['left'].mean() * 1000
    # joints_mean_loss_right = joints_loss['right'].mean() * 1000
    # verts_mean_loss_left = verts_loss['left'].mean() * 1000
    # verts_mean_loss_right = verts_loss['right'].mean() * 1000
    #
    # print('joint mean error:')
    # print('    left: {} mm, right: {} mm'.format(joints_mean_loss_left, joints_mean_loss_right))
    # print('    all: {} mm'.format((joints_mean_loss_left + joints_mean_loss_right) / 2))
    # print('vert mean error:')
    # print('    left: {} mm, right: {} mm'.format(verts_mean_loss_left, verts_mean_loss_right))
    # print('    all: {} mm'.format((verts_mean_loss_left + verts_mean_loss_right) / 2))
