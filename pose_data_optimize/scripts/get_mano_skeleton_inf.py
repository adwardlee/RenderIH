import open3d as o3d
import torch
from manopth.manolayer import ManoLayer
import numpy as np
from scipy.spatial.transform import Rotation as rotation
import copy
from HandPoseConverter import HandPoseConverter
# **** joint order right hand
    #        4-3-2-1-\
    #                   \
#      8-- 7-- 6 -- 5 -----0
#   12--11 -- 10 -- 9 ----/
#    16-- 15 - 14 - 13 ---/
#    20--19-- 18 -- 17 --/

def get_mano_skeleton_inf(side, mano_shape, mano_root_path):
    euler_list = [[0.0, 0.0, 0.0] for i in range(16)]
    mano_layer = ManoLayer(
        mano_root=mano_root_path,
        use_pca=False,
        ncomps=6,
        flat_hand_mean=True,
        center_idx=0,
        return_transf=True,
        side=side,
        root_rot_mode='quat',
        joint_rot_mode='quat'
    )
    hpc = HandPoseConverter(side='right')
    faces = np.array(mano_layer.th_faces).astype(np.long)
    joint_idx = np.asarray([[0, 1, 2, 3, 4],
                            [0, 5, 6, 7, 8],
                            [0, 9, 10, 11, 12],
                            [0, 13, 14, 15, 16],
                            [0, 17, 18, 19, 20]])
    order = ['index', 'middle', 'pinky', 'ring', 'thumb']
    order = ['thumb', 'index', 'middle', 'ring', 'pinky']
    joint_name_list = {}
    for finger_name in order:
        for i in range(4):
            joint_name_list[str(len(joint_name_list.keys()) + 1)] = (finger_name + '_0' + str(i+1) + '_' + side[0])
    vec_pose = torch.from_numpy(hpc.euler_2_mano_quat(euler_list))
    vec_shape = torch.tensor(mano_shape).unsqueeze(0)
    vertices, joints, transf = mano_layer(vec_pose, vec_shape)
    joints = joints.squeeze(0).numpy()

    out = {}
    for finger_list in joint_idx:
        for idx in range(1, finger_list.__len__()):
            joint_id = finger_list[idx]
            last_joint_id = finger_list[idx - 1]
            transl = joints[joint_id] - joints[last_joint_id]
            print(joint_name_list[str(joint_id)])
            print(transl * 100)
            out[joint_name_list[str(joint_id)]] = transl
    return out

def main():
    euler_list = [[0.0, 0.0, 0.0] for i in range(16)]
    mano_layer = ManoLayer(
        mano_root="../assets/mano",
        use_pca=False,
        ncomps=6,
        flat_hand_mean=True,
        center_idx=0,
        return_transf=True,
        side='right',
        root_rot_mode='quat',
        joint_rot_mode='quat'
    )
    hpc = HandPoseConverter(side='right')
    faces = np.array(mano_layer.th_faces).astype(np.long)
    joint_idx = np.asarray([[0, 1, 2, 3, 4],
                            [0, 5, 6, 7, 8],
                            [0, 9, 10, 11, 12],
                            [0, 13, 14, 15, 16],
                            [0, 17, 18, 19, 20]])
    order = ['index', 'middle', 'pinky', 'ring', 'thumb']
    order = ['thumb', 'index', 'middle', 'ring', 'pinky']
    joint_name_list = {}
    for finger_name in order:
        for i in range(4):
            joint_name_list[str(len(joint_name_list.keys()) + 1)] = (finger_name + '_0' + str(i+1))
    vec_pose = torch.from_numpy(hpc.euler_2_mano_quat(euler_list))
    vec_shape = torch.tensor([0.5082395, -0.39488167, -1.7484332, 1.6630946, 0.34428665, -1.37387,
                              0.38293332, 1.196094, 0.6538949, -0.94331187]).unsqueeze(0)
    vertices, joints, transf = mano_layer(vec_pose, vec_shape)
    joints = joints.squeeze(0).numpy()

    out = {}
    for finger_list in joint_idx:
        for idx in range(1, finger_list.__len__()):
            joint_id = finger_list[idx]
            last_joint_id = finger_list[idx - 1]
            transl = joints[joint_id] - joints[last_joint_id]
            print(joint_name_list[str(joint_id)])
            print(transl * 100)
            out[joint_name_list[str(joint_id)]] = transl
    np.save('mano_hand_skel_relative_transl.npy', out)
    # for joint_name in list(joint_name_list.keys()):
    #     root_joint_name =
    #     joint_loc = joints[joint_name_list[joint_name]] - joints
    #     print(joint_name)
    #     print(joint_loc)

if __name__ == '__main__':
    main()