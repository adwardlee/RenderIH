import torch
import pickle
import numpy as np
import cv2 as cv
import os
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

from common.utils.mano import MANO
from .vis import plot_2d_hand

def cut_img(img_list, label2d_list, radio=0.7, img_size=256):
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

def load_2dgt(img, filename, vis=False):
    lines = open(filename, 'r').readlines()
    output = []
    for line in lines:
        x, y, _, _ = line.split()
        x = float(x)
        y = float(y)
        if int(x) == 0 or int(y) == 0:
            continue
        if int(x) < 0 or int(y) < 0:
            continue
        output.append([int(x), int(y)])
    if vis:
        cur_path = filename.replace('joints_2D_GT', 'plot2d')
        print('cur apth', cur_path, flush=True)
        if not os.path.exists('/'.join(cur_path.split('/')[:-1])):
            os.mkdir('/'.join(cur_path.split('/')[:-1]))
        tmpimg = img.copy()
        for one in output:
            x = one[0]
            y = one[1]
            cv.circle(tmpimg, (int(x), int(y)), 1, [255, 0, 0], 1)
        cv.imwrite(os.path.join(cur_path.replace('.txt', '.jpg')), tmpimg)
    thearray = np.array(output).reshape(-1, 2)
    bbox = np.array([min(thearray[:, 0]), min(thearray[:, 1]), max(thearray[:, 0]), max(thearray[:, 1])]).reshape(-1, 2) ### xmin, yxmin, xmax, ymax
    return bbox

def fix_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1

def get_cropimg(img, filename):
    bbox = load_2dgt(img, filename, vis=True)
    cropimg, _ = cut_img([img], [bbox])
    return cropimg[0]

def get_union(joint1, joint2):
    output = [np.min(joint1[:, 0], joint2[:, 0]), np.min(joint1[:, 1], joint2[:, 1]), np.max(joint1[:, 0], joint2[:, 0]), np.max(joint1[:, 1], joint2[:, 1])]
    return output

def get_total_joint2d(start, end, filename, filenum, filename1):
    try:
        joints2d = load_2dgt(filename)
        joints2d_next = load_2dgt(filename1)
    except:
        return
    # if start < filenum:
    for i in range(start, end):
        cur_joint = (joints2d + (joints2d_next - joints2d) / (filenum - start) * (i + 1)).astype(np.int)
        outname = filename.replace(str(start - 1).zfill(3), str(i).zfill(3))
        with open(outname, 'w') as file1:
            for cur_idx, onetup in enumerate(cur_joint):
                file1.write(str(cur_idx))
                file1.write('\t')
                file1.write(str(onetup[0]))
                file1.write('\t')
                file1.write(str(onetup[1]))
                file1.write('\n')
    # else:
    #     for i in range(start, end):
    #         cur_joint = joints2d + (joints2d_next - joints2d) / (filenum - start) * (i + 1)
    #         outname = filename.replace(str(filenum).zfill(3), str(i).zfill(3))
    #         with open(outname, 'w') as file1:
    #             for cur_idx, onetup in enumerate(cur_joint):
    #                 file1.write(str(cur_idx))
    #                 file1.write('\t')
    #                 file1.write(onetup[0])
    #                 file1.write('\t')
    #                 file1.write(onetup[1])
    #                 file1.write('\n')
    return

def get_frompkl(filename):
    annots = pickle.load(open(filename, 'rb'), encoding='latin1')
    #'pose', 'J_transformed___j15', 'v', 'model_name', 'J_transformed', 'trans', 'ncomps', 'J_transformed___j21', 'betas'
    joint3d = annots['J_transformed___j21']
    vert3d = annots['v']
    pose = annots['pose']
    shape = annots['betas']
    trans = annots['trans']
    return joint3d, vert3d, pose, shape, trans

def tzionas_get_full2d():
    for j in range(3, 8):
        detection_path = '/mnt/workspace/workgroup/lijun/hand_dataset/tziona/{}/1/joints_2D_GT/'.format(str(j).zfill(2))
        rgb_path = '/mnt/workspace/workgroup/lijun/hand_dataset/tziona/{}/1/rgb/'.format(str(j).zfill(2))
        img_length = len(os.listdir(rgb_path))
        file_length = len([x for x in os.listdir(detection_path) if x[-4:] == '.txt'])
        for i in range(0, file_length - 1):
            get_total_joint2d(i * 5 + 1, i * 5 + 5, detection_path + '{}.txt'.format(str(i*5).zfill(3)), i * 5 + 5, detection_path + '{}.txt'.format(str(i * 5 + 5).zfill(3)))
        if img_length > (file_length - 1) * 5:
            i = file_length - 1
            get_total_joint2d( i * 5 + 1, img_length, detection_path + '{}.txt'.format(str(i * 5).zfill(3)), i * 5,
                              detection_path + '{}.txt'.format(str(i * 5).zfill(3)))
    return


def main():
    allnums = [ x for x in list(range(1, 8))]
    mano_annot = '/mnt/user/E-shenfei.llj-356552/workgroup/lijun/hand_dataset/rgb2hands/RGB2HANDS_Benchmark/data/seq04_scratch/'
    detection_path = '/mnt/user/E-shenfei.llj-356552/workgroup/lijun/hand_dataset/rgb2hands/RGB2HANDS_Benchmark/data/seq04_scratch/annotation/annot2D_color/'
    rgb_path = '/mnt/user/E-shenfei.llj-356552/workgroup/lijun/hand_dataset/rgb2hands/RGB2HANDS_Benchmark/data/seq04_scratch/color/'
    plot_path = '/mnt/user/E-shenfei.llj-356552/workgroup/lijun/hand_dataset/tziona/original/plot/'
    output_path = '/mnt/user/E-shenfei.llj-356552/workgroup/lijun/hand_dataset/rgb2hands/RGB2HANDS_Benchmark/data/seq04_scratch/cropimg/'#all/'
    #output_annot = '/mnt/workspace/workgroup/lijun/hand_dataset/tziona/mano/'

    idx = 0
    img_path = rgb_path
    detection_path1 = detection_path#.format(str(i).zfill(2))
    annot_name = [detection_path1 + x for x in os.listdir(detection_path1) if '.txt' == x[-4:]]
    for one_annot in annot_name:
        cur_idx = one_annot.split('/')[-1].split('_color2.5D.txt')[0]
    # for start_idx in range(nums):
        hand_dict = {}
        # annot_name = detection_path.format(str(i).zfill(2)) + str(start_idx).zfill(3) + '.txt'
        if not os.path.exists(img_path + str(cur_idx)+'_color.png'):
            continue
        img = cv.imread(img_path + str(cur_idx)+'_color.png')
        cropimg = get_cropimg(img, one_annot)
        cv.imwrite(output_path + '/{}.jpg'.format(idx), cropimg)

        # hand_dict['img'] = cropimg
        # np.save(os.path.join(output_path, '{}'.format(idx)), hand_dict)
        idx += 1







if __name__ == '__main__':
    # tzionas_get_full2d()
    print('finish generate annotation ', flush=True)
    main()

