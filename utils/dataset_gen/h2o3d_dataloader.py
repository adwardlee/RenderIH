# Copyright (c) 2020 Graz University of Technology All rights reserved.

import torch
import torch.utils.data
from utils.dataset_gen.h2o3d_utils.preprocessing import *
from utils.dataset_gen.h2o3d_utils.transform import world2cam, cam2pixel, pixel2cam, transformManoParamsToCam, convert_pose_to_opencv
from utils.dataset_gen.h2o3d_utils.vis import vis_keypoints, vis_3d_keypoints, vis_3d_obj_corners
import json
import matplotlib.pyplot as plt
import cv2 as cv
import smplx
import open3d as o3d
from tqdm import tqdm

ih26m_joint_regressor = np.load(cfg.joint_regr_np_path)
import torchvision.transforms as transforms
from common.utils.mano import MANO

jointsMapManoToDefault = [
    16, 15, 14, 13,
    17, 3, 2, 1,
    18, 6, 5, 4,
    19, 12, 11, 10,
    20, 9, 8, 7,
    0]

def my_print(string, f=None):
    print(string)
    if f is not None:
        f.write(string+'\n')


def get_root_rel_from_parent_rel_depths(dep):
    joint_recon_order = [3, 2, 1, 0,
                         7, 6, 5, 4,
                         11, 10, 9, 8,
                         15, 14, 13, 12,
                         19, 18, 17, 16]

    dep_root = []
    for j in range(5):
        for i in range(4):
            if i == 0:
                dep_root.append(dep[joint_recon_order[j*4+i]])
            else:
                new_dep = dep[joint_recon_order[j*4+i]] + dep_root[-1]
                dep_root.append(new_dep)

    dep_root_reorder = np.array([dep_root[i] for i in joint_recon_order]+[0])
    return dep_root_reorder

def plot_2dvert(img, verts):
    for one in verts:
        x = one[0]
        y = one[1]
        cv2.circle(img, (int(x), int(y)), 1, [255, 0, 0], 1)
    cv2.imwrite('tmp2.jpg', img)


def fix_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1


def cut_img(img_list, label2d_list, camera=None, radio=0.7, img_size=256):
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

    if camera is not None:
        camera[0, 0] = camera[0, 0] * M[0, 0]
        camera[1, 1] = camera[1, 1] * M[1, 1]
        camera[0, 2] = camera[0, 2] * M[0, 0] + M[0, 2]
        camera[1, 2] = camera[1, 2] * M[1, 1] + M[1, 2]

    return img_list_out, label2d_list_out, camera


class Dataset(torch.utils.data.Dataset):
    def __init__(self, h2o3d_anno_dir, obj_kps_dir, transform, mode, annot_subset, capture=None, camera=None, seq_name_test=None):
        self.mode = mode  # train, test, val
        if mode == 'test':
            self.mode = 'evaluation'  # train, test, val

        self.dataset_path = h2o3d_anno_dir
        self.obj_kps_dir = obj_kps_dir

        self.transform = transform
        self.joint_num = 21  # single hand
        self.root_joint_idx = {'right': 20, 'left': 41}
        self.joint_type = {'right': np.arange(0, self.joint_num), 'left': np.arange(self.joint_num, self.joint_num * 2)}
        self.skeleton = load_skeleton(cfg.skeleton_file, self.joint_num * 2)

        self.datalist = []
        self.datalist_sh = []
        self.datalist_ih = []
        self.sequence_names = []

        # load annotation
        with open(osp.join(self.dataset_path, self.mode + '.txt'), 'r') as f:
            self.filelist = f.readlines()
        self.filelist = [f.strip() for f in self.filelist]

        self.mano_layer = {
            'right': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=True, flat_hand_mean=True),
            'left': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=False, flat_hand_mean=True)}
        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if torch.sum(torch.abs(
                self.mano_layer['left'].shapedirs[:, 0, :] - self.mano_layer['right'].shapedirs[:, 0, :])) < 1:
            # print('Fix shapedirs bug of MANO')
            self.mano_layer['left'].shapedirs[:, 0, :] *= -1

        for fname in tqdm(self.filelist):
            seq_name = fname.split('/')[0]
            frame_idx = fname.split('/')[1]
            img_path = osp.join(self.dataset_path, self.mode, seq_name, 'rgb', frame_idx + '.jpg')
            anno_path = osp.join(self.dataset_path, self.mode, seq_name, 'meta', frame_idx + '.pkl')

            if not os.path.exists(img_path) or not os.path.exists(anno_path):
                continue
            if seq_name_test is not None:
                if seq_name_test != seq_name:
                    continue
            if cv2.imread(img_path) is None:
                continue
            try:
                anno = load_pickle_data(anno_path)
            except:
                print('error in ', anno_path, flush=True)
                continue

            focal, princpt = np.array([anno['camMat'][0, 0], anno['camMat'][1, 1]], dtype=np.float32), np.array(
                [anno['camMat'][0, 2], anno['camMat'][1, 2]], dtype=np.float32)
            if self.mode == 'evaluation':
                anno_hand_joints_3d_right = np.expand_dims(anno['rightHandJoints3D'], 0)
                anno_hand_joints_3d_right = np.concatenate([np.zeros((20, 3)), anno_hand_joints_3d_right], axis=0)
                anno_hand_joints_3d_left = np.zeros_like(anno_hand_joints_3d_right)
                anno_hand_joints_3d_left[-1] = anno['leftHandJoints3D']
                anno_hand_joints_3d = np.concatenate([anno_hand_joints_3d_right, anno_hand_joints_3d_left])
                anno['rightHandPose'] = np.zeros((48,)) * 1.0
                anno['rightHandTrans'] = np.zeros((3,)) * 1.0
                anno['leftHandPose'] = np.zeros((48,)) * 1.0
                anno['leftHandTrans'] = np.zeros((3,)) * 1.0
                anno['handBeta'] = np.zeros((10,)) * 1.0
            else:
                anno_hand_joints_3d = np.concatenate([anno['rightHandJoints3D'][jointsMapManoToDefault],
                                                      anno['leftHandJoints3D'][jointsMapManoToDefault]])
            joint_cam = swap_coord_sys(anno_hand_joints_3d) * 1000
            joint_img = cam2pixel(joint_cam, focal, princpt)[:, :2]

            obj_cam = swap_coord_sys(anno['objCorners3D']) * 1000
            obj_img = cam2pixel(swap_coord_sys(anno['objCorners3D']), focal, princpt)[:, :2]

            hand_type = 'interacting'
            joint_valid = np.ones((self.joint_num * 2))

            obj_rot, obj_trans = convert_pose_to_opencv(anno['objRot'].squeeze(), anno['objTrans'])

            # get mano params in current cam
            mano_hand_type = ['right', 'left']
            mano_pose = []
            mano_trans = []
            mano_shape = []
            mano_valid = np.array([True, True])  # right, left valid
            coordChangMat = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
            for ii, ht in enumerate(mano_hand_type):
                if ht == 'right':
                    pose, trans, _ = transformManoParamsToCam(anno['rightHandPose'].squeeze(),
                                                              anno['rightHandTrans'].squeeze(),
                                                              anno['handBeta'].squeeze(),
                                                              cv2.Rodrigues(coordChangMat)[0].squeeze(), np.zeros((3,)),
                                                              'right')
                    shape = anno['handBeta'].squeeze()
                    assert np.sum(joint_valid[self.joint_type[ht]]) > 0
                else:
                    pose, trans, _ = transformManoParamsToCam(anno['leftHandPose'].squeeze(),
                                                              anno['leftHandTrans'].squeeze(),
                                                              anno['handBeta'].squeeze(),
                                                              cv2.Rodrigues(coordChangMat)[0].squeeze(), np.zeros((3,)),
                                                              'left')
                    shape = anno['handBeta'].squeeze()
                    assert np.sum(joint_valid[self.joint_type[ht]]) > 0

                mano_pose.append(pose)
                mano_trans.append(trans.squeeze())
                mano_shape.append(shape)
            mano_pose = np.concatenate(mano_pose, axis=0)
            mano_trans = np.concatenate(mano_trans, axis=0)
            mano_shape = np.concatenate(mano_shape, axis=0)

            if True:
                # if both hands not in the image, skip it
                if np.sum(mano_valid) == 0:
                    continue

                if cfg.hand_type == 'right':
                    if mano_valid[0] == 0 or mano_valid[1] == 1:
                        continue
                elif cfg.hand_type == 'left':
                    if mano_valid[1] == 0 or mano_valid[0] == 1:
                        continue

            hand_type_valid = np.array([1.], dtype=np.float32)

            img_width, img_height = 640, 480
            if self.mode == 'evaluation':
                hand_bb_right = np.array(anno['rightHandBoundingBox'])
                hand_bb_left = np.array(anno['leftHandBoundingBox'])
                tl = np.minimum(hand_bb_right[:2], hand_bb_left[:2])
                br = np.maximum(hand_bb_right[2:], hand_bb_left[2:])
                tl = np.min(np.concatenate([tl[None], obj_img], axis=0), axis=0)
                br = np.max(np.concatenate([br[None], obj_img], axis=0), axis=0)
            else:
                tl = np.min(np.concatenate([joint_img[joint_valid == 1], obj_img], axis=0), axis=0)
                br = np.max(np.concatenate([joint_img[joint_valid == 1], obj_img], axis=0), axis=0)

            box_size = br - tl
            bbox = np.concatenate([tl - 10, box_size + 20], axis=0)
            bbox = process_bbox(bbox, (img_height, img_width))
            abs_depth = {'right': joint_cam[self.root_joint_idx['right'], 2],
                         'left': joint_cam[self.root_joint_idx['left'], 2]}

            obj_bb_rest = anno['objCorners3DRest']

            obj_kps_3d_rest = np.load(osp.join(self.obj_kps_dir, '%s.npy' % (anno['objName'])))
            obj_kps_3d = obj_kps_3d_rest.dot(cv2.Rodrigues(obj_rot)[0].T) + obj_trans
            obj_kps_2d = cam2pixel(obj_kps_3d, focal, princpt)[:, :2]

            if anno['objName'] in ['004_sugar_box']:
                # '004_sugar_box' is not H2O-3D training set, hence not considering it for evaluation
                obj_pose_valid = 0.
            else:
                obj_pose_valid = 1.

            if anno['objName'] == '024_bowl' and self.mode == 'train':
                # Bowl is symmetric
                obj_bb_rest[:, :2] = 0

            cam_param = {'focal': focal, 'princpt': princpt}
            joint = {'cam_coord': joint_cam.astype(np.float32), 'img_coord': joint_img.astype(np.float32),
                     'valid': joint_valid.astype(np.float32)}
            object = {'cam_coord': obj_cam.astype(np.float32), 'img_coord': obj_img.astype(np.float32),
                      'obj_bb_rest': obj_bb_rest.astype(np.float32), 'obj_kps_2d': obj_kps_2d,
                      'obj_kps_3d': obj_kps_3d, 'obj_id': int(anno['objName'][:3]),
                      'obj_kps_3d_rest': obj_kps_3d_rest}
            data = {'img_path': img_path, 'seq_name': seq_name, 'cam_param': cam_param,
                    'bbox': bbox, 'joint': joint, 'object': object, 'hand_type': hand_type,
                    'hand_type_valid': hand_type_valid,
                    'abs_depth': abs_depth, 'file_name': frame_idx + '.png', 'capture': 0, 'cam': 0,
                    'frame': frame_idx, 'mano_pose': mano_pose, 'mano_trans': mano_trans, 'mano_shape': mano_shape,
                    'mano_valid': mano_valid, 'obj_rot': obj_rot, 'obj_trans': obj_trans,
                    'obj_pose_valid': obj_pose_valid}
            if hand_type == 'right' or hand_type == 'left':
                self.datalist_sh.append(data)
            else:
                self.datalist_ih.append(data)
            if seq_name not in self.sequence_names:
                self.sequence_names.append(seq_name)

        self.datalist = self.datalist_sh + self.datalist_ih
        print('Number of annotations in single hand sequences: ' + str(len(self.datalist_sh)))
        print('Number of annotations in interacting hand sequences: ' + str(len(self.datalist_ih)))

    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1, 0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0, 1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1, 1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)

    def __len__(self):
        return len(self.datalist)

    def get_mano_vertices(self, pose, shape, transl, hand_type):
        mesh = self.mano_layer[hand_type](global_orient=torch.from_numpy(pose[:3]).float().unsqueeze(0),
                                          hand_pose=torch.from_numpy(pose[3:]).float().unsqueeze(0),
                                          betas=torch.from_numpy(shape).float().unsqueeze(0),
                                          transl=torch.from_numpy(transl).float().unsqueeze(0))

        return mesh.vertices[0].numpy(), self.mano_layer[hand_type].faces, mesh.joints[0].numpy()

    def restore_coord_cam_from_img(self, pred_joint_coord_img, inv_trans, data, do_flip):
        pred_joint_coord_img[:, 0] = pred_joint_coord_img[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
        pred_joint_coord_img[:, 1] = pred_joint_coord_img[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
        for j in range(self.joint_num * 2):
            pred_joint_coord_img[j, :2] = trans_point2d(pred_joint_coord_img[j, :2], inv_trans)
        # restore depth to original camera space

        pred_joint_coord_img[:, 2] = (pred_joint_coord_img[:, 2] / cfg.output_hm_shape[0] * 2 - 1) * (
                cfg.bbox_3d_size / 2)

        if cfg.dep_rel_to == 'parent':
            pred_joint_coord_img[:21, 2] = get_root_rel_from_parent_rel_depths(pred_joint_coord_img[:21, 2])
            pred_joint_coord_img[21:, 2] = get_root_rel_from_parent_rel_depths(pred_joint_coord_img[21:, 2])

        # add root joint depth
        if do_flip:
            pred_joint_coord_img[self.joint_type['right'], 2] += data['abs_depth']['left']
            pred_joint_coord_img[self.joint_type['left'], 2] += data['abs_depth']['right']
        else:
            pred_joint_coord_img[self.joint_type['right'], 2] += data['abs_depth']['right']
            pred_joint_coord_img[self.joint_type['left'], 2] += data['abs_depth']['left']

        # back project to camera coordinate system
        pred_joint_coord_cam = pixel2cam(pred_joint_coord_img, data['cam_param']['focal'], data['cam_param']['princpt'])

        return pred_joint_coord_cam

    def __getitem__(self, idx):
        data = self.datalist[idx]
        img_path, bbox, joint, object, hand_type, hand_type_valid = data['img_path'], data['bbox'], data['joint'], data[
            'object'], data['hand_type'], data['hand_type_valid']
        mano_pose, mano_trans, mano_shape, mano_valid = data['mano_pose'].copy(), data['mano_trans'].copy(), data[
            'mano_shape'].copy(), data['mano_valid'].copy()
        obj_rot, obj_trans = data['obj_rot'].copy(), data['obj_trans'].copy()
        joint_cam = joint['cam_coord'].copy();
        joint_img = joint['img_coord'].copy();
        joint_valid = joint['valid'].copy().astype(np.float32);
        obj_cam = object['cam_coord'].copy();
        obj_img = object['img_coord'].copy();
        obj_bb_rest = object['obj_bb_rest'];
        obj_kps_img = object['obj_kps_2d'].copy();
        obj_kps_cam = object['obj_kps_3d'].copy();
        obj_kps_rest = object['obj_kps_3d_rest'].copy()
        seq_name = data['seq_name']
        hand_type = self.handtype_str2array(hand_type)
        joint_coord = np.concatenate((joint_img, joint_cam[:, 2, None].copy()), 1)
        obj_coord = np.concatenate((obj_img, obj_cam[:, 2, None].copy()), 1)
        obj_kps_coord = np.concatenate((obj_kps_img, obj_kps_cam[:, 2, None].copy()), 1)
        focal = data['cam_param']['focal']
        princpt = data['cam_param']['princpt']

        obj_pose_valid = data['obj_pose_valid']

        # make the obj trans relative to hand, so that when augmentation is done its all good
        if cfg.predict_type == 'angles':
            obj_trans = obj_trans - mano_trans[:3]
        elif cfg.predict_type == 'vectors':
            obj_trans = obj_trans - joint_cam[self.root_joint_idx['right']] / 1000

        num_kps = obj_kps_coord.shape[0]
        num_corners = 8

        joint_obj_coord = np.concatenate([joint_coord, obj_kps_coord, obj_coord], axis=0)
        joint_obj_cam = np.concatenate([joint_cam, obj_kps_cam, obj_cam], axis=0)

        # image load
        img = load_img(img_path)
        orisize_img = img.copy()
        obj_seg = load_img(osp.join(self.dataset_path, self.mode, data['seq_name'], 'segr', data['frame'] + '.png'))
        obj_seg = cv2.resize(obj_seg.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # augmentation
        img, joint_obj_coord, joint_valid, hand_type, inv_trans, inv_trans_no_rot, do_flip, mano_pose, \
        mano_trans, joint_obj_cam, obj_seg, obj_rot, obj_trans = \
            augmentation(img, bbox, joint_obj_coord, joint_valid, hand_type, 'evaluation', self.joint_type,
                         mano_pose, mano_trans, mano_shape, joint_obj_cam, obj_seg, obj_rot, obj_trans)

        obj_seg = cv2.resize(obj_seg, (cfg.output_hm_shape[1], cfg.output_hm_shape[2]), interpolation=cv2.INTER_NEAREST)

        if do_flip:
            obj_pose_valid *= 0
            mano_valid = mano_valid[[1, 0]]

        rel_root_depth = np.array(
            [joint_coord[self.root_joint_idx['left'], 2] - joint_coord[self.root_joint_idx['right'], 2]],
            dtype=np.float32).reshape(1)

        obj_root_valid = np.array([1]) * 1.0
        if cfg.predict_type == 'vectors':
            root_valid = np.array([np.sum(joint_valid[21:]) > 0]) * 1.0
            if (mano_valid[0] == 1) and (mano_valid[1] == 1):  # if interacting hand
                root_valid *= (joint_valid[self.root_joint_idx['right']] * joint_valid[self.root_joint_idx['left']])
        elif cfg.predict_type == 'angles':
            root_valid = np.array([mano_valid[1]], dtype=np.float32)

        # transform to output heatmap space
        joint_obj_coord, joint_valid, rel_root_depth, root_valid = transform_input_to_output_space(joint_obj_coord,
                                                                                                   joint_valid,
                                                                                                   rel_root_depth,
                                                                                                   root_valid,
                                                                                                   self.root_joint_idx,
                                                                                                   self.joint_type,
                                                                                                   self.skeleton)

        joint_cam = joint_obj_cam[:42]
        obj_cam = joint_obj_cam[42:42 + num_kps]
        obj_corners_cam = joint_obj_cam[42 + num_kps:]
        joint_coord = joint_obj_coord[:42].astype(np.float32)
        obj_coord = joint_obj_coord[42:42 + num_kps].astype(np.float32)
        obj_corners_coord = joint_obj_coord[42 + num_kps:]

        # fill some dummy values in obj_coord
        dummy = np.zeros((30 - obj_coord.shape[0], 3)) + np.array([-2000, -2000, -2000])
        obj_coord = np.concatenate([obj_coord, dummy], axis=0).astype(np.float32)

        obj_coord[:, 2] = obj_coord[:, 2] - obj_coord[0, 2]  # rel depths for now, not used anywhere yet!

        oriimg = img.copy()
        img = self.transform(img.astype(np.float32)) / 255.
        obj_seg = (obj_seg[:, :, 1] > 200) * 255.
        obj_seg = self.transform(obj_seg.astype(np.float32))[0]

        if cfg.predict_type == 'angles':
            rel_trans_hands_rTol = mano_trans[3:] - mano_trans[:3]
            rel_trans_obj = obj_trans
            if root_valid and (np.sum(joint_valid[:21]) == 0):  # when its only left hand image
                # when its only left hand, shift it closer to right hand
                rel_trans_hands_rTol = np.array([0.2, 0., 0.]).astype(np.float32)

            joint_cam_no_trans = joint_cam.copy().astype(np.float32)
            joint_cam_no_trans[self.joint_type['right']] -= mano_trans[:3] * 1000  # because joints in mm
            joint_cam_no_trans[self.joint_type['left']] -= mano_trans[3:] * 1000
            joint_cam_no_trans[self.joint_type['left']] += rel_trans_hands_rTol * 1000


        elif cfg.predict_type == 'vectors':
            joint_cam_no_trans = self.restore_coord_cam_from_img(joint_coord.copy(), inv_trans_no_rot, data, do_flip)

            rel_trans_hands_rTol = (joint_cam_no_trans[self.root_joint_idx['left']] - joint_cam_no_trans[
                self.root_joint_idx['right']]) / 1000
            rel_trans_obj = obj_trans
            if root_valid and (np.sum(joint_valid[:21]) == 0):  # when its only left hand image
                # when its only left hand, shift it closer to right hand
                rel_trans_hands_rTol = np.array([0.2, 0., 0.]).astype(np.float32)

            right_root_loc = joint_cam_no_trans[self.root_joint_idx['right']].copy()
            joint_cam_no_trans[self.joint_type['right']] -= joint_cam_no_trans[self.root_joint_idx['right']]
            joint_cam_no_trans[self.joint_type['left']] -= joint_cam_no_trans[self.root_joint_idx['left']]
            joint_cam_no_trans[self.joint_type['left']] += rel_trans_hands_rTol * 1000

            if False:
                plt.ioff()
                # plt.imshow(np.array(img.permute(1,2,0)*255).astype(np.uint8))
                # print(obj_corners_coord[:,:2]*2)
                # plt.show()
                print(seq_name, data['frame'], np.sum(obj_seg.cpu().numpy()) / 128 / 128 / 255)
                ax = vis_3d_keypoints(joint_cam_no_trans, joint_valid, self.skeleton, plot=False)
                gt_obj_corners = obj_bb_rest.dot(cv2.Rodrigues(obj_rot)[0].T) + rel_trans_obj
                vis_3d_obj_corners([gt_obj_corners * 1000], ax=ax)
                img_rot = cv2.warpAffine(np.array(img.permute(1, 2, 0) * 255).astype(np.uint8), inv_trans_no_rot,
                                         (int(cfg.input_img_shape[1] * 2), int(cfg.input_img_shape[0] * 2)),
                                         flags=cv2.INTER_LINEAR)
                j_coord_rot = cam2pixel(joint_cam_no_trans + right_root_loc,
                                        data['cam_param']['focal'], data['cam_param']['princpt'])[:, :2]
                img_rot = vis_keypoints(img_rot, j_coord_rot, joint_valid, load_skeleton(cfg.skeleton_file, 42))
                plt.imshow(img_rot)
                plt.show()

        obj_kps_3d = obj_kps_rest.dot(cv2.Rodrigues(obj_rot)[0].T) + rel_trans_obj

        # use zero mask for now. Later if required put ones along padded pixels
        mask = np.zeros((img.shape[1], img.shape[2])).astype(np.bool)
        mask = self.transform(mask.astype(np.uint8))

        if cfg.predict_type == 'angles':
            # get the mano vertices
            verts_right, faces, mano_joints_right = self.get_mano_vertices(mano_pose[:48], mano_shape[:10],
                                                                           np.zeros((3,)), 'right')
            verts_left, faces, mano_joints_left = self.get_mano_vertices(mano_pose[48:], mano_shape[10:],
                                                                         rel_trans_hands_rTol,
                                                                         'left')  # add the rel trans in loss.py
            verts = np.concatenate([verts_right, verts_left], axis=0)
        else:
            verts = np.zeros((1, 3))

        # If object is feavily occluded do not consider for pose estimation both during training and evaluating
        perc_obj_seg = np.sum(obj_seg.cpu().numpy()) / 128 / 128 / 255
        if (perc_obj_seg < 0.02 and self.mode == 'train') or (perc_obj_seg < 0.02 and self.mode == 'evaluation'):
            obj_pose_valid *= 0.
        else:
            obj_pose_valid *= 1.

        # heatmap valid flag. If its interacting and only one of the hand annotations is valid, set hm_valid=0
        if (np.sum(hand_type) == 2 and np.sum(joint_valid[:21]) < 21) or (
                np.sum(hand_type) == 2 and np.sum(joint_valid[21:]) < 21):
            hm_valid = np.array([0.]).astype(np.float32)
        else:
            hm_valid = np.array([1.]).astype(np.float32)

        if False:

            rotmat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) * 1.0
            # visualize mesh in open3d
            # print(mano_valid)
            print(seq_name, data['frame'])
            mesh_list = []
            if hand_type[0] == 1:
                mesh_right_o3d = o3d.geometry.TriangleMesh()
                mesh_right_o3d.vertices = o3d.utility.Vector3dVector(verts_right)
                mesh_right_o3d.triangles = o3d.utility.Vector3iVector(faces)
                mesh_right_o3d.vertex_colors = o3d.utility.Vector3dVector(
                    np.load('/home/shreyas/docs/vertex_colors.npy')[:, ::-1])
                mesh_list.append(mesh_right_o3d.transform(rotmat))

            if hand_type[1] == 1:
                mesh_left_o3d = o3d.geometry.TriangleMesh()
                mesh_left_o3d.vertices = o3d.utility.Vector3dVector(verts_left)
                mesh_left_o3d.triangles = o3d.utility.Vector3iVector(faces)
                mesh_left_o3d.vertex_colors = o3d.utility.Vector3dVector(
                    np.load('/home/shreyas/docs/vertex_colors.npy')[:, ::-1])
                mesh_list.append(mesh_left_o3d.transform(rotmat))

            o3d.visualization.draw_geometries(mesh_list, mesh_show_back_face=True)

        if np.sum(img.numpy()) == 0:
            mano_valid *= False
            joint_valid *= 0
            root_valid *= 0
            hand_type_valid *= 0
            obj_pose_valid *= 0

        inputs = {'img': img, 'mask': mask, 'oriimg': oriimg, 'orisize_img': orisize_img}
        targets = {'joint_coord': joint_coord, 'rel_trans_hands_rTol': rel_trans_hands_rTol, 'hand_type': hand_type,
                   'mano_pose': mano_pose, 'mano_shape': mano_shape, 'joint_cam_no_trans': joint_cam_no_trans,
                   'verts': verts,
                   'obj_rot': obj_rot.astype(np.float32), 'rel_obj_trans': rel_trans_obj.astype(np.float32),
                   'obj_kps_coord': obj_coord, 'obj_seg': obj_seg, 'obj_corners_coord': obj_corners_coord,
                   'obj_kps_3d': obj_kps_3d, 'focal': focal, 'princpt': princpt, 'mano_trans': mano_trans,
                   'root_right': right_root_loc}

        meta_info = {'joint_valid': joint_valid, 'root_valid': root_valid, 'hand_type_valid': hand_type_valid,
                     'obj_root_valid': obj_root_valid,
                     'mano_valid': mano_valid, 'inv_trans': inv_trans, 'capture': int(data['capture']),
                     'cam': int(data['cam']),
                     'frame': int(data['frame']), 'seq_id': (seq_name[:9]),
                     'obj_bb_rest': obj_bb_rest.astype(np.float32),
                     'obj_pose_valid': obj_pose_valid, 'focal': data['cam_param']['focal'],
                     'princpt': data['cam_param']['princpt'],
                     'obj_id': data['object']['obj_id'], 'hm_valid': hm_valid}
        return inputs, targets, meta_info

    def dump_for_challenge(self, pred_out_path, xyz_pred_right_list, xyz_pred_left_list, verts_pred_right_list,
                           verts_pred_left_list):
        """ Save predictions into a json file. """
        # make sure its only lists
        xyz_pred_list = np.concatenate([xyz_pred_right_list, xyz_pred_left_list], axis=2)
        xyz_pred_list = [x.tolist() for x in xyz_pred_list]
        if len(verts_pred_right_list) > 0 and len(verts_pred_left_list) > 0:
            verts_pred_list = np.concatenate([verts_pred_right_list, verts_pred_left_list], axis=2)
            verts_pred_list = [x.tolist() for x in verts_pred_list]
        else:
            verts_pred_list = None

        if verts_pred_list is not None:
            # save to a json
            with open(pred_out_path, 'w') as fo:
                json.dump(
                    [
                        xyz_pred_list,
                        verts_pred_list
                    ], fo)
            print('Dumped %d joints and %d verts predictions to %s' % (
                len(xyz_pred_list), len(verts_pred_list), pred_out_path))
        else:
            # save to a json
            with open(pred_out_path, 'w') as fo:
                json.dump(
                    [
                        xyz_pred_list
                    ], fo)
            print('Dumped %d joints to %s' % (
                len(xyz_pred_list), pred_out_path))

    def get_obj_id_name(self):
        YCB_models_dir = cfg.object_models_dir
        obj_names = os.listdir(YCB_models_dir)
        self.obj_id_to_name = {int(o[:3]): o for o in obj_names}
        self.obj_id_to_vertices = {}
        self.obj_id_to_dia = {}
        for id in self.obj_id_to_name.keys():
            if id not in [3, 4, 6, 10, 11, 19, 21, 25, 35, 37, 24]:
                continue
            obj_name = self.obj_id_to_name[id]
            print(os.path.join(YCB_models_dir, obj_name, 'textured_simple_2000.obj'))
            assert os.path.exists(os.path.join(YCB_models_dir, obj_name, 'textured_simple_2000.obj'))
            verts = np.array(
                o3d.io.read_triangle_mesh(os.path.join(YCB_models_dir, obj_name, 'textured_simple_2000.obj')).vertices)
            self.obj_id_to_vertices[id] = verts
            dia = np.max(np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=2))
            self.obj_id_to_dia[id] = dia
        # print(self.obj_id_to_dia)

    def evaluate(self, preds, ckpt_path, gt=None):
        pred_verts, pred_joints, pred_rel_trans = preds['verts'], preds['joints'], preds['rel_trans']
        self.get_obj_id_name()
        num_samples = pred_joints.shape[0]

        ckpt_name = ckpt_path.split('/')[-1].split('.')[0]
        ckpt_dir = os.path.dirname(ckpt_path)

        f_log = open(osp.join(ckpt_dir, '%s.txt' % (ckpt_name)), 'w')

        if gt is not None:
            num_obj_samples = gt['obj_corners_rest'].shape[0]
            my_print('num object samples: %d' % num_obj_samples, f_log)
            all_obj_mssd_dict = {}
            for ii in tqdm(range(num_obj_samples)):
                z_rot_dir = cv2.Rodrigues(gt['obj_rot'][ii])[0].squeeze()[:3, 2] * np.pi  # N x 3
                flipped_obj_rot = np.matmul(cv2.Rodrigues(z_rot_dir)[0].squeeze(),
                                            cv2.Rodrigues(gt['obj_rot'][ii])[0].squeeze())  # 3 x 3 # flipped rot

                obj_vert_rest = self.obj_id_to_vertices[gt['obj_id'][ii]]
                obj_vert_gt = np.matmul(obj_vert_rest, cv2.Rodrigues(gt['obj_rot'][ii])[0].squeeze().T) + gt[
                                                                                                              'obj_trans'][
                                                                                                          ii:ii + 1]  # N x 8 x 3
                obj_vert_pred = np.matmul(obj_vert_rest, cv2.Rodrigues(preds['obj_rot'][ii])[0].squeeze().T) + preds[
                                                                                                                   'obj_trans'][
                                                                                                               ii:ii + 1]  # N x 8 x 3

                if gt['obj_id'][ii] in [24, 25, 19]:  # bowl, mug, pitcher base
                    # Cylindrical objects (angle of symmetry = inf)
                    z_rots = np.arange(-np.pi, np.pi, 5 * np.pi / 180)
                    obj_err = np.inf
                    for z in z_rots:
                        # print(z)
                        rot_dir_curr = cv2.Rodrigues(gt['obj_rot'][ii])[0].squeeze()[:3, 2] * z  # N x 3
                        z_obj_rot = np.matmul(cv2.Rodrigues(rot_dir_curr)[0].squeeze(),
                                              cv2.Rodrigues(gt['obj_rot'][ii])[
                                                  0].squeeze())  # 3 x 3 # flipped rot
                        obj_vert_z_rot_gt = np.matmul(obj_vert_rest, z_obj_rot.T) + gt['obj_trans'][ii:ii + 1]
                        obj_err = min(obj_err, np.max(np.linalg.norm(obj_vert_z_rot_gt - obj_vert_pred, axis=1)))
                else:
                    # Angle of symmetry = 180 degrees
                    is_rot_sym_objs = gt['obj_id'][ii] in [6, 21, 10, 4,
                                                           3]  # mustard, bleach cleanser, potted meat, sugar box, cracker box

                    obj_vert_flipped_gt = np.matmul(obj_vert_rest, flipped_obj_rot.T) + gt['obj_trans'][ii:ii + 1]
                    obj_vert_flipped_gt = obj_vert_flipped_gt * is_rot_sym_objs + obj_vert_gt * (1 - is_rot_sym_objs)

                    obj_err = min(np.max(np.linalg.norm(obj_vert_gt - obj_vert_pred, axis=1)),
                                  np.max(np.linalg.norm(obj_vert_flipped_gt - obj_vert_pred, axis=1)))

                metric_mssd = obj_err

                if gt['obj_id'][ii] not in all_obj_mssd_dict.keys():
                    all_obj_mssd_dict[gt['obj_id'][ii]] = []

                all_obj_mssd_dict[gt['obj_id'][ii]].append(metric_mssd)

            all_obj_mssd = []
            for id in all_obj_mssd_dict.keys():
                my_print('Obj MSSD %s (count = %d) = %f mts' % (
                    self.obj_id_to_name[id], len(all_obj_mssd_dict[id]), np.mean(np.array(all_obj_mssd_dict[id]))),
                         f_log)
                all_obj_mssd = all_obj_mssd + all_obj_mssd_dict[id]

            my_print('Mean obj MSSD = %f mts' % (np.mean(np.array(all_obj_mssd))), f_log)

        # dump for the challenge
        jointsNormalToManoMap = [20,
                                 7, 6, 5,
                                 11, 10, 9,
                                 19, 18, 17,
                                 15, 14, 13,
                                 3, 2, 1,
                                 0, 4, 8, 12, 16]
        pred_verts_right_list = []
        pred_verts_left_list = []
        pred_joints_right_list = []
        pred_joints_left_list = []
        if np.sum(pred_verts) == 0:
            pred_verts = None
        for i in range(num_samples):
            pred_joint_coord_cam = swap_coord_sys(pred_joints[i] / 1000)
            pred_joints_right_list.append(pred_joint_coord_cam[:21][jointsNormalToManoMap])
            pred_joints_left_list.append(
                pred_joint_coord_cam[21:][jointsNormalToManoMap] + swap_coord_sys(pred_rel_trans[i]))
            if pred_verts is not None:
                pred_verts_right_list.append(swap_coord_sys(pred_verts[i][:, :3] / 1000))
                pred_verts_left_list.append(swap_coord_sys(pred_verts[i][:, 3:] / 1000))
        self.dump_for_challenge(osp.join(ckpt_dir, 'results_%s.json' % (ckpt_name)),
                                pred_joints_right_list, pred_joints_left_list, pred_verts_right_list,
                                pred_verts_left_list)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    hand_data = ''
    object_data = ''
    dataset = Dataset(hand_data, object_data, transforms.ToTensor(), "train", 'all')
    batch_generator = DataLoader(dataset=dataset, batch_size=512, shuffle=False,
                                 num_workers=6, pin_memory=False, drop_last=False)
    idx = 0
    mano_left = MANO(hand_type='left')
    mano_right = MANO(hand_type='right')
    mano_layer = {'right': mano_right.layer,
                  'left': mano_left.layer}
    fix_shape(mano_layer)

    mano_layer = {
        'right': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=True, flat_hand_mean=True),
        'left': smplx.create(cfg.smplx_path, 'mano', use_pca=False, is_rhand=False, flat_hand_mean=True)}
    # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
    if torch.sum(torch.abs(
            mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        # print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1
    for inputs, targets, meta_infos in tqdm(batch_generator):
        batch = len(inputs['img'])
        for i in range(batch):
            img = inputs['orisize_img'][i].numpy().copy()[:, :, ::-1].astype(np.uint8)
            focal = targets['focal'][i].numpy()
            princpt = targets['princpt'][i].numpy()
            mano_pose = targets['mano_pose'][i:i + 1]
            mano_shape = targets['mano_shape'][i:i + 1]
            rel_trans_hands_rTol = targets['rel_trans_hands_rTol'][i:i + 1]
            mano_trans = targets['mano_trans'][i:i + 1]
            j2d = targets['joint_coord'][i]
            right_root = targets['root_right'][i]
            ori_joint3d = targets['joint_cam_no_trans'][i].numpy()
            camera = np.array([[focal[0], 0, princpt[0]],
                               [0, focal[1], princpt[1]],
                               [0, 0, 1]])

            right_rot = torch.from_numpy(cv2.Rodrigues(mano_pose[0, :3].numpy())[0].reshape(1, 3, 3))
            left_rot = torch.from_numpy(cv2.Rodrigues(mano_pose[0, 48:51].numpy())[0].reshape(1, 3, 3))

            # verts_right, joints_right = mano_layer['right'](right_rot.float(),
            #                                      mano_pose[:,3:48].float(),
            #                                      mano_shape[:, :10].float(),
            #                                      trans=mano_trans[:, :3].float())
            # verts_left, joints_left = mano_layer['left'](left_rot.float(),
            #                                      mano_pose[:, 51:].float(),
            #                                      mano_shape[:, 10:].float(),
            #                                      trans=mano_trans[:, 3:].float())
            mesh_right = mano_layer['right'](global_orient=mano_pose[:, :3], hand_pose=mano_pose[:, 3:48],
                                             betas=mano_shape[:, :10],
                                             transl=mano_trans[:, :3])
            mesh_left = mano_layer['left'](global_orient=mano_pose[:, 48:51], hand_pose=mano_pose[:, 51:],
                                           betas=mano_shape[:, 10:],
                                           transl=mano_trans[:, 3:],
                                           )
            verts_right = mesh_right.vertices
            verts_left = mesh_left.vertices
            # joints_right = mesh_right.joints
            # joints_left = mesh_left.joints
            joints_right = mano_right.get_3d_joints(verts_right.float())
            joints_left = mano_left.get_3d_joints(verts_left.float())

            verts_right = verts_right[0].numpy()  # - joints_right[0, 20:21] + right_root.numpy().reshape(1, 3) / 1000
            verts_left = verts_left[0].numpy()
            joint_right = joints_right[0].numpy()
            joint_left = joints_left[0].numpy()
            verts2d_right = cam2pixel(verts_right * 1000, focal, princpt)[:, :2]
            verts2d_left = cam2pixel(verts_left * 1000, focal, princpt)[:, :2]
            # joint2d_right = cam2pixel(joint_right * 1000, focal, princpt)[:, :2]
            # joint2d_left = cam2pixel(joint_left * 1000, focal, princpt)[:, :2]
            # plot_2dvert(img, joint2d_right)#j2d[:21])
            # plot_2dvert(img, joint2d_left)
            # plot_2dvert(img, j2d)
            labels2d = np.concatenate((verts2d_left, verts2d_right), axis=0)
            outimg, _, newcam = cut_img([img], [labels2d], camera)
            outimg = outimg[0]
            newjoint2d_left = cam2pixel(joint_left * 1000, [newcam[0, 0], newcam[1, 1]], [newcam[0, 2], newcam[1, 2]])
            newjoint2d_right = cam2pixel(joint_right * 1000, [newcam[0, 0], newcam[1, 1]], [newcam[0, 2], newcam[1, 2]])
            newvert2d_left = cam2pixel(verts_left * 1000, [newcam[0, 0], newcam[1, 1]], [newcam[0, 2], newcam[1, 2]])
            newvert2d_right = cam2pixel(verts_right * 1000, [newcam[0, 0], newcam[1, 1]], [newcam[0, 2], newcam[1, 2]])
            # plot_2dvert(outimg, newjoint2d_left)
            # plot_2dvert(outimg, newjoint2d_right)

            hand_dict = {}
            hand_dict['left'] = {}
            hand_dict['right'] = {}

            hand_dict['left'] = {'hms': [],
                                 'verts3d': verts_left, 'joints3d': joint_left,
                                 'verts2d': newvert2d_left, 'joints2d': newjoint2d_left,
                                 'R': torch.eye(3),
                                 'pose': mano_pose[0, 48:],
                                 'shape': mano_shape[0, 10:],
                                 'trans': mano_trans[3:],
                                 'camera': newcam
                                 }
            hand_dict['right'] = {'hms': [],
                                  'verts3d': verts_right, 'joints3d': joint_right,
                                  'verts2d': newvert2d_right, 'joints2d': newjoint2d_right,
                                  'R': torch.eye(3),
                                  'pose': mano_pose[0, :48],
                                  'shape': mano_shape[0, :10],
                                  'trans': mano_trans[:3],
                                  'camera': newcam
                                  }
            cv2.imwrite('/mnt/workspace/workgroup/lijun/hand_dataset/h2o3d/refine/img/{}.jpg'.format(idx), outimg)
            pickle.dump(hand_dict,
                        open('/mnt/workspace/workgroup/lijun/hand_dataset/h2o3d/refine/anno/' + str(idx) + '.pkl',
                             'wb'))
            idx = idx + 1