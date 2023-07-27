import os
import torch
import pickle
import numpy as np
import cv2 as cv
import os
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt

from common.utils.mano import MANO

color_hand_joints = [[1.0, 0.0, 0.0],
                     [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                     [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                     [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                     [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                     [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little

def plot_hand(
    axis: plt.Axes,
    coords_hand: np.array,
    plot_3d: bool = True,
    linewidth: str = "1",
    linestyle: str = "-",
    alpha: float = 1.0,
    ms=1,
):
    """Makes a hand stick figure from the coordinates wither in uv plane or xyz plane on the passed axes object.
    Code adapted from:  https://github.com/lmb-freiburg/freihand/blob/master/utils/eval_util.py
    Args:
        axis (plt.Axes): Matlplotlib axes, for 3D plots pass axes with 3D projection
        coords_hand (np.array): 21 coordinates of hand as numpy array. (21 x 3). Expects AIT format.
        plot_3d (bool, optional): Pass this as true for using the the depth parameter to plot the hand. Defaults to False.
        linewidth (str, optional): Linewidth to be used for drawing connecting bones. Defaults to "1".
        linestyle (str, optional): MAtplotlib linestyle, Defaults to ":"
    """

    colors = np.array(
        color_hand_joints
    )
    # define connections and colors of the bones
    bones = [
        ((i, i + 1), colors[1 + i, :]) if i % 4 != 0 else ((0, i + 1), colors[1 + i, :])
        for i in range(0, 20)
    ]
    # Making connection between the joints.
    for connection, color in bones:
        coord1 = coords_hand[connection[0], :]
        coord2 = coords_hand[connection[1], :]
        coords = np.stack([coord1, coord2])
        if plot_3d:
            axis.plot(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
            )
        else:
            axis.plot(
                coords[:, 0],
                coords[:, 1],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
            )

    # Highlighting the joints
    for i in range(21):
        if plot_3d:
            axis.plot(
                coords_hand[i, 0],
                coords_hand[i, 1],
                coords_hand[i, 2],
                "o",
                color=colors[i, :],
            )
        else:
            axis.plot(
                coords_hand[i, 0],
                coords_hand[i, 1],
                "o",
                color=colors[i, :],
                linestyle=linestyle,
                alpha=alpha,
                ms=ms,
            )

def plot_2d_hand(axis, coords_hw, vis=None, color_fixed=None, linewidth='1', order='hw', draw_kp=True):
    """ Plots a hand stick figure into a matplotlib figure. """
    if order == 'uv':
        coords_hw = coords_hw[:, ::-1]

    colors = np.array(color_hand_joints)
    # define connections and colors of the bones
    bones = [((0, 1), colors[1, :]),
             ((1, 2), colors[2, :]),
             ((2, 3), colors[3, :]),
             ((3, 4), colors[4, :]),

             ((0, 5), colors[5, :]),
             ((5, 6), colors[6, :]),
             ((6, 7), colors[7, :]),
             ((7, 8), colors[8, :]),

             ((0, 9), colors[9, :]),
             ((9, 10), colors[10, :]),
             ((10, 11), colors[11, :]),
             ((11, 12), colors[12, :]),

             ((0, 13), colors[13, :]),
             ((13, 14), colors[14, :]),
             ((14, 15), colors[15, :]),
             ((15, 16), colors[16, :]),

             ((0, 17), colors[17, :]),
             ((17, 18), colors[18, :]),
             ((18, 19), colors[19, :]),
             ((19, 20), colors[20, :])]

    if vis is None:
        vis = np.ones_like(coords_hw[:, 0]) == 1.0

    for connection, color in bones:
        if (vis[connection[0]] == False) or (vis[connection[1]] == False):
            continue

        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 1], coords[:, 0], color_fixed, linewidth=linewidth)

    if not draw_kp:
        return

    for i in range(21):
        if vis[i] > 0.5:
            axis.plot(coords_hw[i, 1], coords_hw[i, 0], 'o', color=colors[i, :], markersize=3)
            axis.text(coords_hw[i, 1], coords_hw[i, 0], '{}'.format(i), fontsize=5, color='white')

def cut_img(img_list, label2d_list, radio=0.8, img_size=256):
    Min = []
    Max = []
    for label2d in label2d_list:
        Min.append(np.min(label2d, axis=0))
        Max.append(np.max(label2d, axis=0))
    Min = np.min(np.array(Min), axis=0)
    Max = np.max(np.array(Max), axis=0)

    mid = (Min + Max) / 2
    L = np.max(Max - Min) / 2 / radio
    M = img_size / 2 / L * np.array([[1, 0, L - mid[0]],
                                     [0, 1, L - mid[1]]])

    img_list_out = []
    for img in img_list:
        img_list_out.append(cv.warpAffine(img, M, dsize=(img_size, img_size)))

    label2d_list_out = []
    for label2d in label2d_list:
        x = np.concatenate([label2d, np.ones_like(label2d[:, :1])], axis=-1)
        x = x @ M.T
        label2d_list_out.append(x)

    return img_list_out, label2d_list_out,

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

def load_2dgt(filename):
    lines = open(filename, 'r').readlines()
    output = []
    for line in lines:
        _, x, y = line.split()
        if x == 0 and y == 0:
            continue
        output.append([int(x), int(y)])
    thearray = np.array(output).reshape(-1, 2)
    bbox = np.array([min(thearray[:, 0]), min(thearray[:, 1]), max(thearray[:, 0]), max(thearray[:, 1])]).reshape(-1, 2) ### xmin, yxmin, xmax, ymax
    return bbox

def fix_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1

def get_cropimg(img, filename):
    bbox = load_2dgt(filename)
    cropimg, _ = cut_img([img], [bbox])
    return cropimg[0]

def main():
    path = '/mnt/workspace/workgroup/lijun/hand_dataset/ego3dhands/combined_pose_train/'
    output_img_path = '/mnt/workspace/workgroup/lijun/hand_dataset/ego3dhands/img/'
    output_annot_path = '/mnt/workspace/workgroup/lijun/hand_dataset/ego3dhands/annot/'
    #output_annot = '/mnt/workspace/workgroup/lijun/hand_dataset/tziona/mano/'
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mano_left = MANO(hand_type='left')
    mano_right = MANO(hand_type='right')
    mano_layer = {'right': mano_right.layer,
                       'left': mano_left.layer}
    fix_shape(mano_layer)
    J_regressor = {'left': Jr(mano_layer['left'].J_regressor),
                   'right': Jr(mano_layer['right'].J_regressor)}
    idx = 0
    files = os.listdir(path)
    imgname = 'color_new.png'
    joint2d = 'location_2d.npy'
    joint3d = 'location_3d_canonical.npy'
    for one in files:
        cur_path = path + one + '/'
        cur_j2d = np.load(cur_path + joint2d)
        cur_j3d = np.load(cur_path + joint3d)
        cur_img = cv.imread(cur_path + imgname)
        cur_j2d = cur_j2d[:, 1:]
        cur_j3d = cur_j3d[:, 1:]
        cur_j3d = cur_j3d * 10
        if np.sum(cur_j2d[0]) == 0 or np.sum(cur_j2d[1]) == 0:
            continue
        hand_dict = {}
        height, width = cur_img.shape[:2]
        ### j2d transform ###
        cur_j2d[:,:, 0] = cur_j2d[:, :, 0] * height
        cur_j2d[:, :, 1] = cur_j2d[:, :, 1] * width
        cur_j2d = cur_j2d[:,:,::-1]
        #### j3d transform ###
        cur_j3d[:,:, 2] *= -1
        cur_j3d[:,:, 0] *= -1
        new_j3d = np.ones(cur_j3d.shape)
        new_j3d[:,:,0] = cur_j3d[:,:,1]
        new_j3d[:, :, 1] = cur_j3d[:, :, 0]
        new_j3d[:, :, 2] = cur_j3d[:, :, 2]
        # plot_j3d = np.ones(cur_j3d.shape)
        # plot_j3d[:, :, 0] = cur_j3d[:, :, 1]
        # plot_j3d[:, :, 1] = cur_j3d[:, :, 2] * -1
        # plot_j3d[:, :, 2] = cur_j3d[:, :, 0]
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # plot_hand(ax, plot_j3d[0])
        # fig.savefig(os.path.join('/mnt/workspace/workgroup/lijun/hand_dataset/ego3dhands/tmp/', '{}.jpg'.format(idx)))
        #### save img ####
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax1.imshow(cur_img)
        # plot_2d_hand(ax1, cur_j2d[0, :, :2], order='uv')
        # fig.savefig(os.path.join('/mnt/workspace/workgroup/lijun/hand_dataset/ego3dhands/tmp/', '{}.jpg'.format(idx)))

        full_j2d = np.concatenate((cur_j2d), axis=0)
        cropimg, crop_annot = cut_img([cur_img], [full_j2d])
        oriimg = cropimg[0]
        crop_annot = crop_annot[0]
        cv.imwrite(output_img_path + '{}.jpg'.format(idx), oriimg)
        cropimg = cv.cvtColor(oriimg, cv.COLOR_BGR2RGB)
        cropimg = torch.tensor(cropimg) / 255
        cropimg = cropimg.permute(2, 0, 1)
        cropimg = F.normalize(cropimg, mean, std)
        hand_dict['img'] = cropimg


        # plot_j3d = np.ones(cur_j3d.shape)
        # plot_j3d[:, :, 0] = cur_j3d[:, :, 1]
        # plot_j3d[:, :, 1] = cur_j3d[:, :, 2] * -1
        # plot_j3d[:, :, 2] = cur_j3d[:, :, 0]
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # plot_hand(ax, plot_j3d[0])
        # fig.savefig(os.path.join('/mnt/workspace/workgroup/lijun/hand_dataset/ego3dhands/tmp/', '{}.jpg'.format(idx)))
        #### save img ####
        # fig = plt.figure()
        # ax1 = fig.add_subplot(111)
        # ax1.imshow(oriimg)
        # plot_2d_hand(ax1, crop_annot[:21, :2], order='uv')
        # plot_2d_hand(ax1, crop_annot[21:, :2], order='uv')
        # fig.savefig(os.path.join('/mnt/workspace/workgroup/lijun/hand_dataset/ego3dhands/tmp/', '{}.jpg'.format(idx)))


        hand_dict['left'] = {'verts3d': new_j3d[0], 'joints3d': new_j3d[0],'joints2d': crop_annot[:21],
                                'mano_pose': new_j3d[0],
                                'mano_shape': new_j3d[0],
                                'trans': new_j3d[0],
                                'mano_j3d': new_j3d[0],
                                'w_smpl': 1,}
        hand_dict['right'] = {'verts3d': new_j3d[1], 'joints3d': new_j3d[1],'joints2d': crop_annot[21:],
                                'mano_pose': new_j3d[1],
                                'mano_shape': new_j3d[1],
                                'trans': new_j3d[1],
                              'mano_j3d': new_j3d[1],
                                'w_smpl': 1,}
        np.save(os.path.join(output_annot_path, '{}'.format(idx)), hand_dict)
        idx += 1







if __name__ == '__main__':
    # tzionas_get_full2d()
    print('finish generate annotation ', flush=True)
    main()
