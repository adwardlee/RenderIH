import numpy as np
import torch
import pandas as pd
import os
from matplotlib import pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from AIK import adaptive_IK
from utils.mano_from_3djoint.utils import quat2aa, mat2aa, save_obj, plot_3d_hand
from manolayer import ManoLayer
from manopth.manolayer import ManoLayer as mano1
# def compute_relative_rot():
class OptimizationSMPL(torch.nn.Module):
    def __init__(self, pose):
        super(OptimizationSMPL, self).__init__()
        self.beta = torch.nn.Parameter((torch.zeros(1, 10)))#.cuda()))
        self.pose = torch.nn.Parameter((pose))

    def forward(self):
        return self.pose, self.beta

def get_3djoints(csv_data, frame_idx=0):
    out_left = []
    out_right = []
    for one in left:
        for onekey in csv_data.keys():
            if one in onekey:
                if 'End' in onekey:
                    continue
                out_left.append(csv_data[onekey][frame_idx])
    for one in right:
        for onekey in csv_data.keys():
            if one in onekey:
                if 'End' in onekey:
                    continue
                out_right.append(csv_data[onekey][frame_idx])
    return np.array(out_left).reshape(-1, 3), np.array(out_right).reshape(-1, 3)

def get_all_3djoint(input_series, node_list, left_key, right_key):
    cur_left = np.zeros((3))
    outleft = []
    outright = []
    axis_trans = [0, 9, 10, 11, 12, 17, 18, 19, 20, 13, 14, 15, 16, 5, 6, 7, 8, 1, 2, 3, 4]
    for onekey in leftprev_keys:
        idx = np.where(onekey == node_list)[0][0]
        xyz = input_series[idx * 6: idx * 6 + 3]
        cur_left += np.array(xyz)
    cur_right = 0
    for onekey in rightprev_keys:
        idx = np.where(onekey == node_list)[0][0]
        xyz = input_series[idx * 6: idx * 6 + 3]
        cur_right += np.array(xyz)
    for keyid, onekey in enumerate(right_key):
        idx_right = np.where(onekey == node_list)[0][0]
        idx_left = np.where(left_key[keyid] == node_list)[0][0]
        xyz_right = np.array(input_series[idx_right * 6: idx_right * 6 + 3])
        xyz_left =  np.array(input_series[idx_left * 6: idx_left * 6 + 3])
        if keyid == 0:
            outleft.append(cur_left + xyz_left)
            outright.append(cur_right + xyz_right)
        elif keyid == 1:
            outleft.append(outleft[-1] + xyz_left)
            outright.append(outright[-1] + xyz_right)
        elif keyid >= 2 and keyid <= 11:
            if keyid == 2 or keyid == 7:
                tmpleft = outleft[1] + xyz_left
                tmpright = outright[1] + xyz_right
            else:
                tmpleft = tmpleft + xyz_left
                tmpright = tmpright + xyz_right
                outleft.append(tmpleft)
                outright.append(tmpright)
        elif keyid >= 12 and keyid <= 15: ### 4,5,5
            if keyid == 12:
                tmpleft = outleft[0] + xyz_left
                tmpright = outright[0] + xyz_right
                outleft.append(tmpleft)
                outright.append(tmpright)
            else:
                tmpleft = tmpleft + xyz_left
                tmpright = tmpright + xyz_right
                outleft.append(tmpleft)
                outright.append(tmpright)
        elif keyid >= 16 and keyid <= 25:
            if keyid == 16 or keyid == 21:
                tmpleft = outleft[0] + xyz_left
                tmpright = outright[0] + xyz_right
            else:
                tmpleft = tmpleft + xyz_left
                tmpright = tmpright + xyz_right
                outleft.append(tmpleft)
                outright.append(tmpright)
    outleft.pop(1)
    outright.pop(1)
    outleft = np.array(outleft)[axis_trans]
    outright = np.array(outright)[axis_trans]
    return outleft, outright

rightprev_keys = ['Root_M', 'Spine1_M', 'Spine2_M', 'Spine3_M', 'Chest_M', 'Scapula_R', 'Shoulder_R', 'ShoulderPart1_R',
                  'ShoulderPart2_R','Elbow_R','ElbowPart1_R', 'ElbowPart2_R']
leftprev_keys = ['Root_M', 'Spine1_M', 'Spine2_M', 'Spine3_M', 'Chest_M', 'Scapula_L', 'Shoulder_L', 'ShoulderPart1_L',
                  'ShoulderPart2_L','Elbow_L','ElbowPart1_L', 'ElbowPart2_L']
valid_keys = ['Wrist_L','Wrist_R',
              'ThumbFinger1_L', 'ThumbFinger1_R', 'ThumbFinger2_L', 'ThumbFinger2_R', 'ThumbFinger3_L', 'ThumbFinger3_R','ThumbFinger4_L', 'ThumbFinger4_R',
              'IndexFinger1_L', 'IndexFinger1_R', 'IndexFinger2_L', 'IndexFinger2_R', 'IndexFinger3_L', 'IndexFinger3_R','IndexFinger4_L', 'IndexFinger4_R',
              'MiddleFinger1_L', 'MiddleFinger1_R', 'MiddleFinger2_L', 'MiddleFinger2_R', 'MiddleFinger3_L', 'MiddleFinger3_R','MiddleFinger4_L', 'MiddleFinger4_R',
              'RingFinger1_L', 'RingFinger1_R', 'RingFinger2_L', 'RingFinger2_R', 'RingFinger3_L', 'RingFinger3_R','RingFinger4_L', 'RingFinger4_R',
              'PinkyFinger1_L', 'PinkyFinger1_R', 'PinkyFinger2_L', 'PinkyFinger2_R', 'PinkyFinger3_L', 'PinkyFinger3_R','PinkyFinger4_L', 'PinkyFinger4_R']
# right = ['Wrist_R', 'Cup_R', 'PinkyFinger5_R', 'PinkyFinger1_R', 'PinkyFinger2_R', 'PinkyFinger3_R', 'PinkyFinger4_R',
#                              'RingFinger5_R','RingFinger1_R', 'RingFinger2_R','RingFinger3_R','RingFinger4_R',
#                     'ThumbFinger1_R', 'ThumbFinger2_R','ThumbFinger3_R','ThumbFinger4_R',
#                     'MiddleFinger5_R', 'MiddleFinger1_R','MiddleFinger2_R','MiddleFinger3_R','MiddleFinger4_R',
#                     'IndexFinger5_R', 'IndexFinger1_R','IndexFinger2_R','IndexFinger3_R','IndexFinger4_R',
#          ]#valid_keys[0::2]
# left = ['Wrist_L', 'Cup_L', 'PinkyFinger5_L', 'PinkyFinger1_L', 'PinkyFinger2_L', 'PinkyFinger3_L', 'PinkyFinger4_L',
#                             'RingFinger5_L','RingFinger1_L', 'RingFinger2_L','RingFinger3_L','RingFinger4_L',
#                     'ThumbFinger1_L', 'ThumbFinger2_L','ThumbFinger3_L','ThumbFinger4_L',
#                     'MiddleFinger5_L', 'MiddleFinger1_L','MiddleFinger2_L','MiddleFinger3_L','MiddleFinger4_L',
#                     'IndexFinger5_L', 'IndexFinger1_L','IndexFinger2_L','IndexFinger3_L','IndexFinger4_L',
#         ]
left = ['Wrist_L', 'ThumbFinger1_L', 'ThumbFinger2_L','ThumbFinger3_L','ThumbFinger4_L',
                    'IndexFinger1_L','IndexFinger2_L','IndexFinger3_L','IndexFinger4_L',
                    'MiddleFinger1_L', 'MiddleFinger2_L', 'MiddleFinger3_L', 'MiddleFinger4_L',
                    'RingFinger1_L', 'RingFinger2_L','RingFinger3_L','RingFinger4_L',
                    'PinkyFinger1_L', 'PinkyFinger2_L', 'PinkyFinger3_L', 'PinkyFinger4_L',
        ]
right = ['Wrist_R', 'ThumbFinger1_R', 'ThumbFinger2_R','ThumbFinger3_R','ThumbFinger4_R',
                    'IndexFinger1_R','IndexFinger2_R','IndexFinger3_R','IndexFinger4_R',
                    'MiddleFinger1_R','MiddleFinger2_R','MiddleFinger3_R','MiddleFinger4_R',
                    'RingFinger1_R', 'RingFinger2_R','RingFinger3_R','RingFinger4_R',
                    'PinkyFinger1_R', 'PinkyFinger2_R', 'PinkyFinger3_R', 'PinkyFinger4_R',
                    ]
frame_idx = 40
cur_mano_left = ManoLayer(manoPath='mano/MANO_LEFT.pkl')
cur_mano_left.shapedirs[:, 0, :] *= -1
cur_mano_right = ManoLayer(manoPath='mano/MANO_RIGHT.pkl')
cur_mano_layer = {'left': cur_mano_left, 'right': cur_mano_right}
ori_mano_left = mano1(mano_root='mano', side='left', use_pca=False, flat_hand_mean=False)
ori_mano_left.th_shapedirs[:, 0, :] *= -1
ori_mano_right = mano1(mano_root='mano', side='right', use_pca=False, flat_hand_mean=False)
ori_mano_layer = {'left': ori_mano_left, 'right': ori_mano_right}
# cur_mano = ManoLayer(mano_root='mano', side='right', use_pca=True, center_idx=9, flat_hand_mean=True, ncomps=15)
# joint, series, frame, time, node_list = readBVH('0809_bvh/EP04_HZ800_SC220427_HZ_shot25_363_0427_ani.bvh')#('0809_bvh/EP04_HZ800_SC220427_HZ_shot18_261_0427_ani.bvh')
# series = np.array(series)
# node_list = np.array(node_list)
# cur_series = series[0]
# leftjoints, rightjoints = get_all_3djoint(series[frame_idx], node_list, left, right)
csv_data = pd.read_csv('EP04_HZ800_SC220427_HZ_shot17_242_0427_ani_worldpos.csv')
leftjoints, rightjoints = get_3djoints(csv_data, frame_idx)
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
plot_3d_hand(ax1, rightjoints)
# plt.show()
fig.savefig('tmp1.jpg')
ori_joints = {'left': leftjoints, 'right': rightjoints}
# pose0 = torch.zeros((1, 48))

### for left joint ik ####
mano_params = {'left': [], 'right': []}
for hand in ['left', 'right']:
    pose0 = torch.eye(3).repeat(1, 16, 1, 1)#.cuda()
    joints = ori_joints[hand]
    #pre_useful_bone_len = bone.caculate_length(joints, label='useful')
    # _, j3d_p0_ops = cur_mano(pose0, torch.zeros((1, 10)))#.cuda())
    _, j3d_p0_ops = cur_mano_layer[hand](pose0[:, 0], mat2aa(pose0[:, 1:].reshape(-1, 3, 3)).reshape(1, 45), torch.zeros((1, 10)))#.cuda())
    template = j3d_p0_ops.cpu().numpy().squeeze(0)  # template, m 21*3
    ratio = np.linalg.norm(template[9] - template[0]) / np.linalg.norm(joints[9] - joints[0])
    j3d_pre_process = joints * ratio  # template, m
    j3d_pre_process = j3d_pre_process - j3d_pre_process[0] + template[0]
    pose_R = adaptive_IK(template, j3d_pre_process)

    pose_R = torch.from_numpy(pose_R).float().reshape(1, 16, 3, 3)
    # pose_R = torch.from_numpy(pose_R).float().reshape(-1, 3, 3)
    # pose_R = mat2aa(pose_R).reshape(1, 48)

    lr = 1e-1
    #### optimization ####
    parameters_smpl = OptimizationSMPL(pose_R)#.cuda()
    optimizer_smpl = torch.optim.Adam(parameters_smpl.parameters(), lr=lr)
    iterations = 200
    pred_joints = torch.FloatTensor((joints - joints[0:1])/ 100)#.cuda()
    factor_beta_reg = 0.002
    for i in range(iterations):
        pose_R, beta = parameters_smpl.forward()
        # beta = torch.clamp(beta, -1.5, 1.5)
        # vertices_smpl, joints_smpl = cur_mano(pose_R, beta)
        # vertices_smpl = vertices_smpl / 1000
        # joints_smpl = joints_smpl / 1000
        vertices_smpl, joints_smpl = cur_mano_layer[hand](pose_R[:, 0], mat2aa(pose_R[:, 1:].reshape(-1, 3, 3)).reshape(1, 45), beta)
        joints_smpl = joints_smpl - joints_smpl[:, 0:1]
        # vertices_smpl = vertices_smpl / 1000
        distances = torch.abs(joints_smpl - pred_joints)
        loss = distances.mean()

        # beta_loss = (beta ** 2).mean()
        loss = loss #+ beta_loss * factor_beta_reg

        optimizer_smpl.zero_grad()
        loss.backward()
        optimizer_smpl.step()

        for param_group in optimizer_smpl.param_groups:
            param_group['lr'] = lr * (iterations - i) / iterations



    with torch.no_grad():
        pose_R, beta = parameters_smpl.forward()
        # pose_R = torch.eye(3).repeat(1, 16, 1, 1)
        # beta = torch.clamp(beta, -1.5, 1.5)
        #beta = torch.zeros((1, 10))
        outverts, outjoints = cur_mano_layer[hand](pose_R[:, 0], mat2aa(pose_R[:, 1:].reshape(-1, 3, 3)).reshape(1, 45), beta)
        # outverts, outjoints = cur_mano(pose_R, beta)
        outverts = outverts.cpu().numpy()
        outjoints = outjoints.cpu().numpy()
    outverts = outverts - outjoints[0, 0:1]
    outjoints = outjoints - outjoints[0, 0:1]
    # outverts = outverts / 1000
    # outjoints = outjoints / 1000

    print('11')
    save_obj('cur_frame{}_{}.obj'.format(frame_idx, hand), outverts[0] * 100 + joints[0:1],
             cur_mano_layer[hand].faces)  ###  cur_mano.faces   cur_mano.th_faces.numpy()

    with torch.no_grad():
        pose_R, beta = parameters_smpl.forward()
        mean_pose = cur_mano_layer[hand].hands_mean
        axisangle = mat2aa(pose_R.reshape(-1, 3, 3)).reshape(1, 48)  ###
        hand_pose = cur_mano_layer[hand].pca2axis(axisangle[:, 3:]) - mean_pose  ### 1, 45
        root_aa = axisangle[:, :3]  ### 1, 3
        ori_pose = torch.cat((root_aa, hand_pose), dim=1)
        # pose_R = torch.eye(3).repeat(1, 16, 1, 1)
        # beta = torch.clamp(beta, -1.5, 1.5)
        #beta = torch.zeros((1, 10))
        outverts, outjoints = ori_mano_layer[hand](ori_pose, beta)
        # outverts, outjoints = cur_mano(pose_R, beta)
        outverts = outverts.cpu().numpy() / 1000
        outjoints = outjoints.cpu().numpy() / 1000
    outverts = outverts - outjoints[0, 0:1]
    outjoints = outjoints - outjoints[0, 0:1]

    #### joint length ### np.sqrt(np.sum((left_pos[4] - left_pos[0]) ** 2))
    print('11')
    save_obj('ori_frame{}_{}.obj'.format(frame_idx, hand), outverts[0] * 100 + joints[0:1],
             cur_mano_layer[hand].faces)  ###  cur_mano.faces   cur_mano.th_faces.numpy()