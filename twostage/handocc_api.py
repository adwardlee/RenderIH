'''
  @ Date: 2020-10-23 20:07:49
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-03-05 13:43:01
  @ FilePath: /EasyMocap/code/estimator/SPIN/spin_api.py
'''
"""
Demo code

To run our method, you need a bounding box around the person. The person needs to be centered inside the bounding box and the bounding box should be relatively tight. You can either supply the bounding box directly or provide an [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) detection file. In the latter case we infer the bounding box from the detections.

In summary, we provide 3 different ways to use our demo code and models:
1. Provide only an input image (using ```--img```), in which case it is assumed that it is already cropped with the person centered in the image.
2. Provide an input image as before, together with the OpenPose detection .json (using ```--openpose```). Our code will use the detections to compute the bounding box and crop the image.
3. Provide an image and a bounding box (using ```--bbox```). The expected format for the json file can be seen in ```examples/im1010_bbox.json```.

Example with OpenPose detection .json
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png --openpose=examples/im1010_openpose.json
```
Example with predefined Bounding Box
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png --bbox=examples/im1010_bbox.json
```
Example with cropped and centered image
```
python3 demo.py --checkpoint=data/model_checkpoint.pt --img=examples/im1010.png
```

Running the previous command will save the results in ```examples/im1010_{shape,shape_side}.png```. The file ```im1010_shape.png``` shows the overlayed reconstruction of human shape. We also render a side view, saved in ```im1010_shape_side.png```.
"""

import torch
from torchvision.transforms import Normalize
import numpy as np
import math
import cv2

from graphormer.vis_graphormer import rendering, pred_rendering


class constants:
    FOCAL_LENGTH = 5000.
    IMG_RES = 256

    # Mean and standard deviation for normalizing input image
    IMG_NORM_MEAN = [0.485, 0.456, 0.406]
    IMG_NORM_STD = [0.229, 0.224, 0.225]


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def crop(img, center, scale, res, rot=0, bias=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1,
                             res[1] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape) + bias

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                    old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]
    new_img = cv2.resize(new_img, (res[0], res[1]))
    return new_img


def process_image(img, bbox, input_res=224):
    """Read image, do preprocessing and possibly crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.
    """
    img = img[:, :, ::-1].copy()
    height, width = img.shape[:2]
    normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    l, t, r, b = bbox[:4]
    l = math.ceil(l)
    t = math.ceil(t)
    r = math.floor(r)
    b = math.floor(b)
    ### wrong one ####
    # input_img = np.zeros((b - t, r - l, 3)).astype(img.dtype)
    # input_img[:(min(height, b) - max(0, t)), :(min(width, r) - max(0, l))] = img[max(0, t):min(height, b),
    #                                                                          max(0, l):min(width, r)]
    # img = cv2.resize(input_img, (input_res, input_res))
    ## correct one ###
    center = [(l+r)/2, (t+b)/2]
    width = max(r-l, b-t)
    scale = width/200.0
    img = crop(img, center, scale, (input_res, input_res))
    norm_img = img.astype(np.float32) / 255.
    norm_img = torch.from_numpy(norm_img).permute(2, 0, 1)
    norm_img = normalize_img(norm_img.clone())#[None]
    norm_img = norm_img[None]
    return img, norm_img


def estimate_translation_np(S, joints_2d, joints_conf, K):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """
    num_joints = S.shape[0]
    # focal length
    f = np.array([K[0, 0], K[1, 1]])
    # optical center
    center = np.array([K[0, 2], K[1, 2]])

    # transformations
    Z = np.reshape(np.tile(S[:, 2], (2, 1)).T, -1)
    XY = np.reshape(S[:, 0:2], -1)
    O = np.tile(center, num_joints)
    F = np.tile(f, num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf), (2, 1)).T, -1)

    # least squares
    Q = np.array([F * np.tile(np.array([1, 0]), num_joints), F * np.tile(np.array([0, 1]), num_joints),
                  O - np.reshape(joints_2d, -1)]).T
    c = (np.reshape(joints_2d, -1) - O) * Z - F * XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W, Q)
    c = np.dot(W, c)

    # square matrix
    A = np.dot(Q.T, Q)
    b = np.dot(Q.T, c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans



def mano_to_bodyparams(pose_param, shape_param):
    results = {
        'shapes': shape_param.detach().cpu().numpy()
    }
    # rotmat = pose_param[0].detach().cpu().numpy()
    rotmat = pose_param.detach().cpu().numpy()

    # poses = np.zeros((1, rotmat.shape[0] * 3))
    # for i in range(rotmat.shape[0]):
    #     p, _ = cv2.Rodrigues(rotmat[i])
    #     poses[0, 3 * i:3 * i + 3] = p[:, 0]
    results['poses'] = rotmat  # poses

    body_params = {
        'poses': results['poses'],
        'shapes': results['shapes'],
        'Rh': results['poses'][:, :3].copy(),
        'Th': np.zeros((1, 3)),
    }
    body_params['Th'][0, 2] = 5
    body_params['poses'][:, :3] = 0
    results = body_params
    return results


def rotmat_to_bodyparams(pose_param, shape_param):
    results = {
        'shapes': shape_param.detach().cpu().numpy()
    }
    # rotmat = pose_param[0].detach().cpu().numpy()
    # # rotmat = pose_param.detach().cpu().numpy()
    #
    # poses = np.zeros((1, rotmat.shape[0] * 3))
    # for i in range(rotmat.shape[0]):
    #     p, _ = cv2.Rodrigues(rotmat[i])
    #     poses[0, 3 * i:3 * i + 3] = p[:, 0]
    results['poses'] = pose_param.detach().cpu().numpy()  # poses

    body_params = {
        'poses': results['poses'],
        'shapes': results['shapes'],
        'Rh': results['poses'][:, :3].copy(),
        'Th': np.zeros((1, 3)),
    }
    body_params['Th'][0, 2] = 5
    body_params['poses'][:, :3] = 0
    results = body_params
    return results


def init_with_spin(body_model, spin_model, img, bbox, kpts, camera, single_flag=0, combine_model=None, input_res=224, idx=0):
    if single_flag:
        div_img, norm_img0 = process_image(img, bbox[0], input_res)
        div_img, norm_img1 = process_image(img, bbox[1], input_res)
        inputs = {'img': torch.cat([norm_img0, norm_img1], dim=0).cuda()}
        targets = {}
        meta_info = {}
        if combine_model == 11:  #### intag hand ik model ####
            out = spin_model(inputs['img'], body_model, idx)
        else:
            with torch.no_grad():
                out = spin_model(inputs, targets, meta_info, 'test')
        # pred_3d_joints_left = out['joints_coord_cam_left']
        # pred_vertices_left = out['mesh_coord_cam_left']
        # pred_3d_joints_right = out['joints_coord_cam_right']
        # pred_vertices_right = out['mesh_coord_cam_right']

        pose_param_left = out['mano_pose_left'][0:1]
        pose_param_right = out['mano_pose_right'][1:2]
        shape_param_left = out['mano_shape_left'][0:1]
        shape_param_right = out['mano_shape_right'][1:2]
    else:
        div_img, norm_img = process_image(img, bbox, input_res)
        inputs = {'img': norm_img.cuda()}
        targets = {}
        meta_info = {}
        if combine_model == 11: #### intag hand ik model ####
            out = spin_model(inputs['img'], body_model, idx)
        else:
            with torch.no_grad():
                out = spin_model(inputs, targets, meta_info, 'test')
        # pred_3d_joints_left = out['joints_coord_cam_left']
        # pred_vertices_left = out['mesh_coord_cam_left']
        # pred_3d_joints_right = out['joints_coord_cam_right']
        # pred_vertices_right = out['mesh_coord_cam_right']

        pose_param_left = out['mano_pose_left']
        pose_param_right = out['mano_pose_right']
        shape_param_left = out['mano_shape_left']
        shape_param_right = out['mano_shape_right']

    body_params_left = rotmat_to_bodyparams(pose_param_left, shape_param_left)
    body_params_right = rotmat_to_bodyparams(pose_param_right, shape_param_right)
    # pred_camera = pred_camera.detach().reshape(1, 3)
    # rgb = pred_rendering(mano_model, pred_vertices, pred_camera)
    # rgb = rgb.detach().cpu().numpy()[0]
    # mask = rgb[:, :, 3:]
    # cv2.imwrite('tmp.jpg', rgb[:, :, :3][:, :, ::-1] * 255 * mask + div_img.permute(1,2,0).numpy()[:,:,::-1] * 255 * (1 - mask))

    # else:
    #     body_params = spin_model.forward(norm_img.copy())
    # body_params = body_model.check_params(body_params)
    # only use body joints to estimation translation
    nJoints = 21
    ### left ###
    vertices_left, keypoints3d_left = body_model['left'](body_params_left)
    keypoints3d_left = keypoints3d_left[0].numpy()
    vertices_right, keypoints3d_right = body_model['right'](body_params_right)
    keypoints3d_right = keypoints3d_right[0].numpy()
    if kpts.shape[-1] == 2:
        tmp = np.ones((42, 1))
        kpts = np.concatenate((kpts, tmp), axis=1)
    trans_left = estimate_translation_np(keypoints3d_left[:nJoints], kpts[:nJoints, :2], kpts[:nJoints, 2], camera['K'])
    body_params_left['Th'] += trans_left[None, :]
    # convert to world coordinate
    Rhold = cv2.Rodrigues(body_params_left['Rh'])[0]
    Thold = body_params_left['Th']
    Rh = camera['R'].T @ Rhold
    Th = (camera['R'].T @ (Thold.T - camera['T'])).T
    body_params_left['Th'] = Th
    body_params_left['Rh'] = cv2.Rodrigues(Rh)[0].reshape(1, 3)
    vertices_left, keypoints3d_left = body_model['left'](body_params_left)
    vertices_left = vertices_left[0].numpy()
    keypoints3d_left = keypoints3d_left[0].numpy()
    ### right ###
    trans_right = estimate_translation_np(keypoints3d_right[:nJoints], kpts[nJoints:, :2], kpts[nJoints:, 2], camera['K'])
    body_params_right['Th'] += trans_right[None, :]
    # convert to world coordinate
    Rhold = cv2.Rodrigues(body_params_right['Rh'])[0]
    Thold = body_params_right['Th']
    Rh = camera['R'].T @ Rhold
    Th = (camera['R'].T @ (Thold.T - camera['T'])).T
    body_params_right['Th'] = Th
    body_params_right['Rh'] = cv2.Rodrigues(Rh)[0].reshape(1, 3)
    vertices_right, keypoints3d_right = body_model['right'](body_params_right)
    vertices_right = vertices_right[0].numpy()
    keypoints3d_right = keypoints3d_right[0].numpy()

    ##### restrict translation #####
    if abs(body_params_right['Th'][0, 2] - body_params_left['Th'][0, 2]) / body_params_left['Th'][0, 2] > 0.5:
        body_params_right['Th'][0, 2] = body_params_left['Th'][0, 2]
    if body_params_right['Th'][0, 2] <= 0:
        body_params_right['Th'][0, 2] = 0.5
    if body_params_left['Th'][0, 2] <= 0:
        body_params_left['Th'][0, 2] = 0.5
    #######################################
    # height, width = img.shape[:2]
    # silhoutte, rgb = rendering(combine_model, torch.from_numpy(vertices_right).unsqueeze(0).float(), camera['K'].reshape(1, 3, 3), height, width)
    # rgb = rgb.cpu().numpy()
    # mask = rgb[:, :, 3:]
    # cv2.imwrite('tmp.jpg', rgb[:, :, :3][:,:,::-1] * 255 * mask + img * (1 - mask))
    results = {'body_params_left': body_params_left, 'vertices_left': vertices_left, 'keypoints3d_left': keypoints3d_left,
               'body_params_right': body_params_right, 'vertices_right': vertices_right, 'keypoints3d_right': keypoints3d_right}
    if combine_model == 11:
        results['orivert_left'] = out['orivert_left'].detach().cpu().numpy()
        results['orivert_right'] = out['orivert_right'].detach().cpu().numpy()
    return results


if __name__ == '__main__':
    pass