import json
import os.path as osp
from tqdm import tqdm
import cv2 as cv
import numpy as np
import torch
import pickle
import random
from glob import glob
import scipy
import os
import sys
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from models.manolayer import ManoLayer, rodrigues_batch
from dataset.dataset_utils import IMG_SIZE, HAND_BBOX_RATIO, HEATMAP_SIGMA, HEATMAP_SIZE, cut_img
from dataset.heatmap import HeatmapGenerator
# from utils.vis_utils import mano_two_hands_renderer
from utils.manoutils import get_mano_path


def fill_hull(image):
    """
    Compute the convex hull of the given binary image and
    return a mask of the filled hull.

    Adapted from:
    https://stackoverflow.com/a/46314485/162094
    This version is slightly (~40%) faster for 3D volumes,
    by being a little more stingy with RAM.
    """
    # (The variable names below assume 3D input,
    # but this would still work in 4D, etc.)

    assert (np.array(image.shape) <= np.iinfo(np.int16).max).all(), \
        f"This function assumes your image is smaller than {2 ** 15} in each dimension"

    points = np.argwhere(image).astype(np.int16)
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])

    # Instead of allocating a giant array for all indices in the volume,
    # just iterate over the slices one at a time.
    idx_2d = np.indices(image.shape[1:], np.int16)
    idx_2d = np.moveaxis(idx_2d, 0, -1)

    idx_3d = np.zeros((*image.shape[1:], image.ndim), np.int16)
    idx_3d[:, :, 1:] = idx_2d

    mask = np.zeros_like(image, dtype=bool)
    for z in range(len(image)):
        idx_3d[:, :, 0] = z
        s = deln.find_simplex(idx_3d)
        mask[z, (s != -1)] = 1

    return mask

def bm0(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 == 1)       # I assume this is faster as mask1 == 1 is a bool array
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero( np.logical_and( mask1, mask2) )
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou

def fix_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1

class InterHand_realsubset():
    def __init__(self, data_path, split):
        assert split in ['train', 'test', 'val']
        self.split = split

        self.data_path = data_path
        self.size = len(glob(osp.join(data_path, split, 'anno', '*.pkl')))
        self.syns_path = '/mnt/user/E-shenfei.llj-356552/workgroup/lijun/hand_dataset/synthesis/sdf_xinchuan/'
        name = 'train_subset1.pkl'### 1,10,20,30,40,50,60,70,80,90
        self.sample_idx = pickle.load(open(name, 'rb'))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img0 = cv.imread(osp.join(self.data_path, self.split, 'img', '{}.jpg'.format(idx)))
        # mask = cv.imread(osp.join(self.data_path, self.split, 'mask', '{}.jpg'.format(idx)))
        # dense = cv.imread(osp.join(self.data_path, self.split, 'dense', '{}.jpg'.format(idx)))

        hand_dict = np.load(os.path.join(self.data_path, self.split, 'ori_handdict', '{}.npy'.format(idx)),
                            allow_pickle=True)
        hand_dict = hand_dict[()]
        left_vert2d = hand_dict['left']['verts2d']
        right_vert2d = hand_dict['right']['verts2d']

        # img = Image.new('L', (256, 256), 0)
        # ImageDraw.Draw(img).polygon(left_vert2d, outline=1, fill=1)
        # mask_left = np.array(img)
        #
        # img1 = Image.new('L', (256, 256), 0)
        # ImageDraw.Draw(img1).polygon(right_vert2d, outline=1, fill=1)
        # mask_right = np.array(img1)

        img_left = np.zeros((256, 256))
        img_left[np.round(left_vert2d).astype(np.int16)] = 1
        img_right = np.zeros((256, 256))
        img_right[np.round(right_vert2d).astype(np.int16)] = 1
        mask_left = fill_hull(img_left)
        mask_right = fill_hull(img_right)

        iou = bm0(mask_left, mask_right)
        with open(os.path.join(self.data_path, self.split, 'iou/{}'.format(idx)), 'w') as file1:
            file1.write(str(iou))
        return img0, hand_dict

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data_path =  '/mnt/user/E-shenfei.llj-356552/data/dataset/interhand_5fps/interhand_data/'
    for split in ['test']:
        dataset = InterHand_realsubset(data_path, split)
        batch_generator = DataLoader(dataset, batch_size=512, shuffle=False,
                                     num_workers=4, pin_memory=False)
        idx = 0
        for itr, (inputs, targets) in tqdm(enumerate(batch_generator)):
        # for itr, (inputs, _, _, targets) in tqdm(enumerate(batch_generator)):
            idx += 1

