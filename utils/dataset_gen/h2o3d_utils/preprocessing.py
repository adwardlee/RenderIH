import os
import os.path as osp
import pickle
import cv2
import numpy as np
from main.config import cfg
import random
import math
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from utils.dataset_gen.h2o3d_utils.transform import transformManoParamsToCam
import tqdm


def load_img(path, order='RGB'):
    # load
    if not os.path.exists(path):
        path = path.replace('.png', '.jpg')
    if not os.path.exists(path):
        path = path.replace('.jpg', '.png')
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order == 'RGB':
        img = img[:, :, ::-1].copy()

    img = img.astype(np.float32)
    return img


def load_skeleton(path, joint_num):
    # load joint info (name, parent_id)
    skeleton = [{} for _ in range(joint_num)]

    with open(path) as fp:
        for line in fp:
            if line[0] == '#': continue
            splitted = line.split(' ')
            joint_name, joint_id, joint_parent_id = splitted
            joint_id, joint_parent_id = int(joint_id), int(joint_parent_id)
            skeleton[joint_id]['name'] = joint_name
            skeleton[joint_id]['parent_id'] = joint_parent_id
    # save child_id
    for i in range(len(skeleton)):
        joint_child_id = []
        for j in range(len(skeleton)):
            if skeleton[j]['parent_id'] == i:
                joint_child_id.append(j)
        skeleton[i]['child_id'] = joint_child_id

    return skeleton


def get_aug_config():
    trans_factor = 0.15
    scale_factor = 0.25
    if cfg.predict_type == 'vectors':
        rot_factor = 45  # no rotation for now
    else:
        rot_factor = 45
    color_factor = 0.2

    trans = [np.random.uniform(-trans_factor, trans_factor), np.random.uniform(-trans_factor, trans_factor)]
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    if cfg.predict_type == 'angles':
        # cant flip mano angles
        do_flip = False
    else:
        do_flip = random.random() <= 0.5
    if cfg.predict_type == 'angles' or cfg.dataset == 'ho3d':  # or cfg.dataset == 'h2o3d':
        # never flip, pose parameters for left and right hand are different, cant just flip them later
        do_flip = False
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])

    return trans, scale, rot, do_flip, color_scale


def rotate_mano(mano_pose, mano_trans, mano_shape, joint_cam, rot, obj_rot, obj_trans):
    rot_rad = np.pi * rot / 180
    new_rotmat = np.array([[np.cos(rot_rad), np.sin(rot_rad), 0],
                           [-np.sin(rot_rad), np.cos(rot_rad), 0],
                           [0., 0., 1.]])
    new_rotmat_rod = cv2.Rodrigues(new_rotmat)[0].squeeze()

    mano_pose_right, mano_trans_right, _ = transformManoParamsToCam(mano_pose[:48], mano_trans[:3], mano_shape[:10],
                                                                    new_rotmat_rod, np.zeros((3,)), 'right')
    mano_pose_left, mano_trans_left, _ = transformManoParamsToCam(mano_pose[48:], mano_trans[3:], mano_shape[10:],
                                                                  new_rotmat_rod, np.zeros((3,)), 'left')

    new_mano_pose = np.concatenate([mano_pose_right, mano_pose_left], axis=0)
    new_mano_trans = np.concatenate([mano_trans_right, mano_trans_left], axis=0)

    new_joint_cam = (new_rotmat.dot(joint_cam.T)).T

    if obj_rot is not None:
        new_obj_pose = cv2.Rodrigues(new_rotmat.dot(cv2.Rodrigues(obj_rot.copy())[0]))[0].squeeze()
        new_obj_trans = new_rotmat.dot(obj_trans.copy())
    else:
        new_obj_pose = new_obj_trans = None

    return new_mano_pose, np.squeeze(new_mano_trans), new_joint_cam, new_obj_pose, new_obj_trans


def augmentation(img, bbox, joint_coord, joint_valid, hand_type, mode, joint_type, mano_pose, mano_trans, mano_shape,
                 joint_cam,
                 obj_seg=None, obj_rot=None, obj_trans=None):
    img = img.copy()
    joint_coord = joint_coord.copy()
    hand_type = hand_type.copy()

    original_img_shape = img.shape
    joint_num = len(joint_coord)

    if mode == 'train':
        trans, scale, rot, do_flip, color_scale = get_aug_config()
        if mano_pose is not None:
            mano_pose, mano_trans, joint_cam, obj_rot, obj_trans = rotate_mano(mano_pose, mano_trans, mano_shape,
                                                                               joint_cam, rot, obj_rot, obj_trans)
    else:
        trans, scale, rot, do_flip, color_scale = [0, 0], 1.0, 0.0, False, np.array([1, 1, 1])

    bbox[0] = bbox[0] + bbox[2] * trans[0]
    bbox[1] = bbox[1] + bbox[3] * trans[1]
    img, trans, inv_trans, inv_trans_no_rot, seg_patch = generate_patch_image(img, bbox, do_flip, scale, rot,
                                                                              cfg.input_img_shape, obj_seg)
    img = np.clip(img * color_scale[None, None, :], 0, 255)

    if do_flip:
        joint_coord[:, 0] = original_img_shape[1] - joint_coord[:, 0] - 1
        joint_coord[joint_type['right']], joint_coord[joint_type['left']] = joint_coord[joint_type['left']].copy(), \
                                                                            joint_coord[joint_type['right']].copy()
        joint_valid[joint_type['right']], joint_valid[joint_type['left']] = joint_valid[joint_type['left']].copy(), \
                                                                            joint_valid[joint_type['right']].copy()
        hand_type[0], hand_type[1] = hand_type[1].copy(), hand_type[0].copy()
    for i in range(joint_num):
        joint_coord[i, :2] = trans_point2d(joint_coord[i, :2], trans)

    return img, joint_coord, joint_valid, hand_type, inv_trans, inv_trans_no_rot, do_flip, mano_pose, mano_trans, joint_cam, seg_patch, obj_rot, obj_trans


def transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, root_valid, root_joint_idx, joint_type,
                                    skeleton):
    # transform to output heatmap space
    joint_coord = joint_coord.copy();
    joint_valid = joint_valid.copy()

    joint_coord[:, 0] = joint_coord[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
    joint_coord[:, 1] = joint_coord[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

    if cfg.dep_rel_to == 'root':
        joint_coord[joint_type['right'], 2] = joint_coord[joint_type['right'], 2] - joint_coord[
            root_joint_idx['right'], 2]
        joint_coord[joint_type['left'], 2] = joint_coord[joint_type['left'], 2] - joint_coord[root_joint_idx['left'], 2]
    elif cfg.dep_rel_to == 'parent':
        pid = np.array([skeleton[i]['parent_id'] for i in range(42)])
        joint_coord[joint_type['right'][:-1], 2] = joint_coord[joint_type['right'][:-1], 2] - joint_coord[
            pid[joint_type['right'][:-1]], 2]
        joint_coord[joint_type['left'][:-1], 2] = joint_coord[joint_type['left'][:-1], 2] - joint_coord[
            pid[joint_type['left'][:-1]], 2]
        joint_coord[root_joint_idx['right'], 2] = joint_coord[root_joint_idx['left'], 2] = 0
    else:
        raise NotImplementedError

    joint_coord[:, 2] = (joint_coord[:, 2] / (cfg.bbox_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]
    if cfg.predict_type == 'vectors' and cfg.predict_2p5d:
        # when predicting hm, ensure all points can fit in hm, if not they are invalid
        joint_valid = joint_valid * ((joint_coord[:, 2] >= 0) * (joint_coord[:, 2] < cfg.output_hm_shape[0])).astype(
            np.float32)
        joint_valid = joint_valid * ((joint_coord[:, 0] >= 0) * (joint_coord[:, 0] < cfg.output_hm_shape[1])).astype(
            np.float32)
        joint_valid = joint_valid * ((joint_coord[:, 1] >= 0) * (joint_coord[:, 1] < cfg.output_hm_shape[2])).astype(
            np.float32)

    return joint_coord, joint_valid, rel_root_depth, root_valid


def get_bbox(joint_img, joint_valid):
    x_img = joint_img[:, 0][joint_valid == 1];
    y_img = joint_img[:, 1][joint_valid == 1];
    xmin = min(x_img);
    ymin = min(y_img);
    xmax = max(x_img);
    ymax = max(y_img);

    x_center = (xmin + xmax) / 2.;
    width = xmax - xmin;
    xmin = x_center - 0.5 * width * 1.2
    xmax = x_center + 0.5 * width * 1.2

    y_center = (ymin + ymax) / 2.;
    height = ymax - ymin;
    ymin = y_center - 0.5 * height * 1.2
    ymax = y_center + 0.5 * height * 1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox


def process_bbox(bbox, original_img_shape):
    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.
    c_y = bbox[1] + h / 2.
    aspect_ratio = cfg.input_img_shape[1] / cfg.input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * 1.25
    bbox[3] = h * 1.25
    bbox[0] = c_x - bbox[2] / 2.
    bbox[1] = c_y - bbox[3] / 2.

    return bbox


def generate_patch_image(cvimg, bbox, do_flip, scale, rot, out_shape, obj_seg):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1
        if obj_seg is not None:
            obj_seg = obj_seg[:, ::-1, :]

    trans, trans_no_rot = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0],
                                                  scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    if obj_seg is not None:
        obj_seg_patch = cv2.warpAffine(obj_seg.copy().astype(np.float32), trans, (int(out_shape[1]), int(out_shape[0])),
                                       flags=cv2.INTER_LINEAR)
    else:
        obj_seg_patch = None
    img_patch = img_patch.astype(np.float32)
    inv_trans, inv_trans_no_rot = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1],
                                                          out_shape[0], scale, rot, inv=True)

    return img_patch, trans, inv_trans, inv_trans_no_rot, obj_seg_patch


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src_no_rot = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir
    src_no_rot[0, :] = src_center
    src_no_rot[1, :] = src_center + np.array([0, src_h * 0.5], dtype=np.float32)
    src_no_rot[2, :] = src_center + np.array([src_w * 0.5, 0], dtype=np.float32)

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        trans_no_rot = cv2.getAffineTransform(np.float32(dst), np.float32(src_no_rot))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        trans_no_rot = cv2.getAffineTransform(np.float32(src_no_rot), np.float32(dst))

    trans = trans.astype(np.float32)
    trans_no_rot = trans_no_rot.astype(np.float32)
    return trans, trans_no_rot


def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


class PeakDetector():
    def __init__(self, nearest_neighbor_th=None):
        if nearest_neighbor_th is None:
            nearest_neighbor_th = cfg.nearest_neighbor_th

        X = np.arange(0, cfg.output_hm_shape[0])
        Y = np.arange(0, cfg.output_hm_shape[1])
        X, Y = np.meshgrid(X, Y)
        self.pixel_inds = np.reshape(np.stack([Y, X], axis=2), [-1, 2])  # N x 2

        self.dists = []

        filename = 'dists_dict_%d.pickle' % (cfg.output_hm_shape[0])
        if not os.path.exists(filename):
            self.dists_dict = {}
            for i in tqdm.tqdm(range(self.pixel_inds.shape[0])):
                dists = np.linalg.norm(np.expand_dims(self.pixel_inds[i:i + 1], 1) - np.expand_dims(self.pixel_inds, 0),
                                       axis=2)  # 1 x N
                dists_mask = dists <= nearest_neighbor_th
                nearest_pixel_inds = np.where(dists_mask.squeeze())[0]
                self.dists_dict[i] = nearest_pixel_inds

            with open(filename, 'wb') as f:
                pickle.dump(self.dists_dict, f)
        else:
            with open(filename, 'rb') as f:
                self.dists_dict = pickle.load(f)

    def detect_peaks_nms(self, image, num_peaks=200, nearest_neighbor_th=None, intensity_th=None):
        img = image.copy()
        if intensity_th is None:
            intensity_th = cfg.intensity_th
        if nearest_neighbor_th is None:
            nearest_neighbor_th = cfg.nearest_neighbor_th

        peaks_ind_list = []

        while np.sum(img > intensity_th) > 0:
            ind = np.unravel_index(np.argmax(img, axis=None), img.shape)
            if img[ind[0], ind[1]] <= intensity_th:
                assert False
            img[ind[0], ind[1]] = intensity_th

            nearest_pixel_inds = self.pixel_inds[
                self.dists_dict[np.ravel_multi_index(ind, (image.shape[0], image.shape[1]))]]
            img[nearest_pixel_inds[:, 0], nearest_pixel_inds[:, 1]] = intensity_th
            peaks_ind_list.append(np.array(ind))

        peak_values_list = []
        for ind in peaks_ind_list:
            peak_values_list.append(image[ind[0], ind[1]])

        sorted_inds = np.argsort(np.array(peak_values_list))[::-1]
        if len(peaks_ind_list) > num_peaks:
            peaks_ind_list = [peaks_ind_list[k] for k in sorted_inds[:num_peaks]]

        detected_peaks = np.zeros_like(img).astype(np.bool)
        for ind in peaks_ind_list:
            detected_peaks[ind[0], ind[1]] = True

        return detected_peaks, peaks_ind_list

    def detect_peaks(self, image, num_peaks=200):
        """
        Takes an image and detect the peaks usingthe local maximum filter.
        Returns a boolean mask of the peaks (i.e. 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        """

        # define an 8-connected neighborhood
        neighborhood = generate_binary_structure(2, 2)

        # apply the local maximum filter; all pixel of maximal value
        # in their neighborhood are set to 1
        local_max = maximum_filter(image, footprint=neighborhood) == image
        # local_max is a mask that contains the peaks we are
        # looking for, but also the background.
        # In order to isolate the peaks we must remove the background from the mask.

        # we create the mask of the background
        background = (image == 0)
        low_val_regions = image > 10

        # a little technicality: we must erode the background in order to
        # successfully subtract it form local_max, otherwise a line will
        # appear along the background border (artifact of the local maximum filter)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

        # we obtain the final mask, containing only peaks,
        # by removing the background from the local_max mask (xor operation)
        detected_peaks = local_max ^ eroded_background
        detected_peaks = detected_peaks * low_val_regions

        xind, yind = np.nonzero(detected_peaks)
        peaks_ind_list = list(np.stack([xind, yind], axis=1))

        peak_values_list = []
        for ind in peaks_ind_list:
            peak_values_list.append(image[ind[0], ind[1]])

        sorted_inds = np.argsort(np.array(peak_values_list))[::-1]
        if len(peaks_ind_list) > num_peaks:
            peaks_ind_list = [peaks_ind_list[k] for k in sorted_inds[:num_peaks]]

            detected_peaks = np.zeros_like(image).astype(np.bool)
            for ind in peaks_ind_list:
                detected_peaks[ind[0], ind[1]] = True

        return detected_peaks, peaks_ind_list


def load_pickle_data(f_name):
    """ Loads the pickle data """
    if not osp.exists(f_name):
        raise Exception('Unable to find annotations pickle file at %s. Aborting.' % (f_name))
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)

    return pickle_data


def swap_coord_sys(arr):
    coordChangMat = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    return arr.dot(coordChangMat.T)