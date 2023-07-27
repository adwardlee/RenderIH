
import torch
import numpy as np
import pickle
import os.path as osp
import cv2
from main.config import cfg
import smplx


def loadPickleData(fName):
    with open(fName, 'rb') as f:
        try:
            pickData = pickle.load(f, encoding='latin1')
        except:
            pickData = pickle.load(f)

    return pickData

mano_layer = {'right': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=True),
              'left': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=False)}
# fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
    # print('Fix shapedirs bug of MANO')
    mano_layer['left'].shapedirs[:,0,:] *= -1

smpl_data_hands = {}
smpl_data_hands['left'] = loadPickleData(osp.join(cfg.smplx_path, 'mano', 'MANO_LEFT.pkl'))
smpl_data_hands['right'] = loadPickleData(osp.join(cfg.smplx_path, 'mano', 'MANO_RIGHT.pkl'))


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return cam_coord

def world2cam(world_coord, R, T):
    cam_coord = np.dot(R, world_coord - T)
    return cam_coord

def multi_meshgrid(*args):
    """
    Creates a meshgrid from possibly many
    elements (instead of only 2).
    Returns a nd tensor with as many dimensions
    as there are arguments
    """
    args = list(args)
    template = [1 for _ in args]
    for i in range(len(args)):
        n = args[i].shape[0]
        template_copy = template.copy()
        template_copy[i] = n
        args[i] = args[i].view(*template_copy)
        # there will be some broadcast magic going on
    return tuple(args)


def flip(tensor, dims):
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    indices = [torch.arange(tensor.shape[dim] - 1, -1, -1,
                            dtype=torch.int64) for dim in dims]
    multi_indices = multi_meshgrid(*indices)
    final_indices = [slice(i) for i in tensor.shape]
    for i, dim in enumerate(dims):
        final_indices[dim] = multi_indices[i]
    flipped = tensor[final_indices]
    assert flipped.device == tensor.device
    assert flipped.requires_grad == tensor.requires_grad
    return flipped


def getHandJointLocs(betas, hand_type):
    smpl_data = smpl_data_hands[hand_type]
    # smpl_data['v_shaped'] = smpl_data['shapedirs'].dot(betas) + smpl_data['v_template']
    smpl_data['v_shaped'] = mano_layer[hand_type].shapedirs.numpy().dot(betas) + smpl_data['v_template']
    v_shaped = smpl_data['v_shaped']
    J_tmpx = smpl_data['J_regressor'].dot(v_shaped[:, 0])
    J_tmpy = smpl_data['J_regressor'].dot(v_shaped[:, 1])
    J_tmpz = smpl_data['J_regressor'].dot(v_shaped[:, 2])
    smpl_data['J'] = np.vstack((J_tmpx, J_tmpy, J_tmpz)).T

    return smpl_data['J']


def transformManoParamsToCam(mano_pose, mano_trans, mano_shape, cam_rot, cam_trans, hand_type):
    J = getHandJointLocs(mano_shape, hand_type)[0:1, :].T

    RAbsMat = cv2.Rodrigues(cam_rot)[0].dot(cv2.Rodrigues(mano_pose[:3])[0])
    RAbsRod = cv2.Rodrigues(RAbsMat)[0][:, 0]

    TAbs = cv2.Rodrigues(cam_rot)[0].dot(J + np.expand_dims(mano_trans, 0).T) + np.expand_dims(
        cam_trans / 1000, 0).T - J

    mano_pose_trans = np.copy(mano_pose)
    mano_pose_trans[:3] = RAbsRod

    return mano_pose_trans, TAbs, mano_shape


def convert_pose_to_opencv(pose, trans):

    coordChangMat = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    newRot = cv2.Rodrigues(coordChangMat.dot(cv2.Rodrigues(pose[:3])[0]))[0][:,0]
    new_trans = trans.copy().dot(coordChangMat.T)
    new_pose = pose.copy()
    new_pose[:3] = newRot

    return new_pose, new_trans

def rot_param_rot_mat(rot):
    '''

    :param rot: N x 6
    :return:
    '''

    e1 = rot[:,:3]
    e2 = rot[:,3:]

    e1d = e1/torch.linalg.norm(e1,dim=1,keepdim=True)
    e3d = torch.cross(e1d, e2, dim=1)/torch.linalg.norm(e2,dim=1,keepdim=True)
    e2d = torch.cross(e3d, e1d)

    R = torch.stack([e1d, e2d, e3d],dim=2) # N x 3 x 3

    return R

def rot_param_rot_mat_np(rot):
    '''

    :param rot: N x 6
    :return:
    '''

    e1 = rot[:,:3]
    e2 = rot[:,3:]

    e1d = e1/np.linalg.norm(e1,axis=1,keepdims=True)
    e3d = np.cross(e1d, e2, axis=1)/np.linalg.norm(e2,axis=1,keepdims=True)
    e2d = np.cross(e3d, e1d)

    R = np.stack([e1d, e2d, e3d],axis=2) # N x 3 x 3

    return R