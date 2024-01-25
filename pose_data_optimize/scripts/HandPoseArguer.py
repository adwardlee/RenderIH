import numpy as np
from tqdm import *
from scipy.spatial.transform import Rotation as rotation
import os
from manopth.manolayer import ManoLayer
from .HandPoseConverter import HandPoseConverter
import copy
import torch
import time

class HandPoseArguer:
    def __init__(self, shape=None, side='right', root='assets/mano'):
        self.hand_model = ManoLayer(
            mano_root=root,
            use_pca=False,
            ncomps=6,
            flat_hand_mean=True,
            center_idx=9,
            return_transf=True,
            side=side,
            root_rot_mode='quat',
            joint_rot_mode='quat'
        )
        self.hpc = HandPoseConverter(side=side, root=root)
        self.splank_mask = []
        if shape is None:
            self.shape = torch.tensor([0.5082395, -0.39488167, -1.7484332, 1.6630946, 0.34428665, -1.37387,
                                       0.38293332, 1.196094, 0.6538949, -0.94331187]).unsqueeze(0)
        else:
            self.shape = shape
        self.faces = np.array(self.hand_model.th_faces).astype(np.long)

    def argue_pose(self, origin_vec_pose, argue_size=10, pose_type='quat', vis=False):
        vec_pose = copy.deepcopy(np.asarray(origin_vec_pose))
        if pose_type == 'quat':
            vec_pose = self.hpc.mano_quat_2_euler(vec_pose)
        if vec_pose.shape.__len__() <= 2:
            vec_pose = vec_pose[np.newaxis, :]
        batch_size = vec_pose.shape[0]
        # vec_pose = np.repeat(vec_pose, argue_size, axis=0)
        bend_angles = vec_pose[:, :, 2]
        splank_angles = vec_pose[:, :, 1]

        argued_vec_pose = np.repeat(vec_pose, argue_size, axis=0)
        bend_adjust = self.bend_argue(bend_angles, argue_size, 120) #120
        splank_adjust = self.splank_argue(splank_angles, argue_size, 30) # 30
        # print("here")
        bend_matrix = self.get_rotation_around_dyaxis(bend_adjust, axis=2)
        splank_matrix = self.get_rotation_around_dyaxis(splank_adjust, axis=1)

        argued_vec_pose = self.apply_adjustment(argued_vec_pose, bend_matrix)
        argued_vec_pose = self.apply_adjustment(argued_vec_pose, splank_matrix)
        # print(argued_vec_pose)

        if vis:
            import open3d as o3d
            idx = 0
            argued_vec_pose_quat = self.hpc.euler_2_mano_quat(argued_vec_pose)
            vertices, joints, transf = self.hand_model(torch.from_numpy(argued_vec_pose_quat), self.shape.repeat((argued_vec_pose_quat.shape[0], 1)))
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='coordinate_visualization')
            hand_mesh = o3d.geometry.TriangleMesh()
            hand_mesh.vertices = o3d.utility.Vector3dVector(vertices[0])
            hand_mesh.triangles = o3d.utility.Vector3iVector(self.faces)
            vis.add_geometry(hand_mesh)
            vis.reset_view_point(True)
            ctr = vis.get_view_control()
            ctr.rotate(0.0, -600)
            # vis.run()
            for vec in vertices:
                hand_mesh.vertices = o3d.utility.Vector3dVector(vec)
                hand_mesh.compute_vertex_normals()
                vis.update_geometry(hand_mesh)
                time.sleep(0.4)
                vis.update_renderer()
                vis.poll_events()
                # print(argued_vec_pose[idx, 4])
                idx += 1
            vis.destroy_window()
        return argued_vec_pose

    def apply_adjustment(self, vec_pose, matrix):
        '''
        :param vec_pose: euler (b, 16, 3)
        :param matrix: (b, 16, 3, 3)
        :return:
        '''
        euler = vec_pose.reshape(-1, 3)
        m = rotation.from_euler('xyz',euler, True)
        rot = m.as_matrix()
        rot = rot.reshape(-1, 16, 3, 3)
        rot = rot @ matrix

        rot = rot.reshape(-1, 3, 3)
        m = rotation.from_matrix(rot)
        return m.as_euler('xyz', degrees=True).reshape(-1, 16, 3)

    # **** axis order right hand

    #         15-14-13-\
    #                   \
    #    3-- 2 -- 1 -----0
    #   6 -- 5 -- 4 ----/
    #   12 - 11 - 10 ---/
    #    9-- 8 -- 7 --/

    def bend_argue(self, bend_angles, argue_size, random_angle=45):
        # soft_idx = [2, 5, 11, 8]  # these joints can bend 0-120 degrees
        # soft_range = [-8, 120]
        # strict_idx = [13]  # these joints can bend 0-35 degrees
        # strict_range = [-8, 35]
        # thumb_normal_idx = [14]  # these joints can bend 0-65 degrees
        # thumb_normal_range = [-8, 65]
        # normal_idx = [1, 3, 4, 6, 10, 12, 7, 9, 15]  # 0-90 degrees
        # normal_range = [-8, 90]

        soft_idx = [2, 5, 11, 8]  # these joints can bend 0-120 degrees
        soft_range = [-8, 110]
        strict_idx = [13]  # these joints can bend 0-35 degrees
        strict_range = [-20, 40]
        thumb_normal_idx = [14]  # these joints can bend 0-65 degrees
        thumb_normal_range = [-8, 50]
        normal_idx = [3, 6, 12, 9, 15]  # 0-90 degrees
        normal_range = [-8, 90]
        root_idx = [1,4,10,7]
        root_range = [-25, 70]

        batch_size = bend_angles.shape[0]
        argued_bend_angles = np.repeat(bend_angles, argue_size, axis=0)
        angle_adjust = np.zeros(argued_bend_angles.shape, dtype=argued_bend_angles.dtype)
        for i in range(batch_size):
            self.angle_random_adjust(bend_angles[i], angle_adjust[i * argue_size:(i + 1) * argue_size], soft_idx, soft_range, random_angle)
            self.angle_random_adjust(bend_angles[i], angle_adjust[i * argue_size:(i + 1) * argue_size], strict_idx,
                                     strict_range, random_angle)
            self.angle_random_adjust(bend_angles[i], angle_adjust[i * argue_size:(i + 1) * argue_size], thumb_normal_idx,
                                     thumb_normal_range, random_angle)
            self.angle_random_adjust(bend_angles[i], angle_adjust[i * argue_size:(i + 1) * argue_size], normal_idx,
                                     normal_range, random_angle)
            self.angle_random_adjust(bend_angles[i], angle_adjust[i * argue_size:(i + 1) * argue_size], root_idx,
                                     root_range, random_angle)
        return angle_adjust

    def splank_argue(self, splank_angles, argue_size, random_angle=30):
        # index, middle, ring, pinky, thumb
        finger_id = [1, 4, 10, 7, 13]
        finger_range = [[-6, 20], [-10, 10], [-15, 6], [-35, 6], [-9, 80]]
        batch_size = splank_angles.shape[0]
        argued_splank_angles = np.repeat(splank_angles, argue_size, axis=0)
        angle_adjust = np.zeros(argued_splank_angles.shape, dtype=argued_splank_angles.dtype)
        for i in range(batch_size):
            for id in range(len(finger_id)):
                self.angle_random_adjust(splank_angles[i], angle_adjust[i * argue_size:(i + 1) * argue_size],
                                         [finger_id[id]],
                                         finger_range[id], random_angle)
        return angle_adjust


    def get_rotation_around_dyaxis(self, angle, axis):
        eulers = np.zeros(angle.shape.__add__((3,)), dtype=angle.dtype)
        eulers[..., axis] = angle
        eulers = eulers.reshape(-1, 3)
        m = rotation.from_euler('xyz',eulers, True)
        rot = m.as_matrix()
        rot = rot.reshape(-1, 16, 3, 3)
        return rot
    def angle_random_adjust(self, angles, angle_adjust, idx, joint_range, max_adjust_angle):
        argue_size = angle_adjust.__len__()
        for id in idx:
            # if angles[id] < joint_range[0]:
            #     continue
            pos = max(min(max_adjust_angle, joint_range[1] - angles[id]), 0)
            neg = min(max(-max_adjust_angle, joint_range[0] - angles[id]), 0)
            angle_adjust[:, id] = (pos - neg) * np.random.rand(argue_size) + neg


if __name__ == '__main__':
    euler_list = [[[0.0, 0.0, 0.0],  # hand root
                   [-3.96, -18, 25.82],  # index
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [-1.45, 3.12, 60.07],  # middle
                   [0, 0, -1],
                   [0.0, 0.0, -0.726],
                   [-12.96, 2.41, -83.60],  # ring
                   [0.0, 0.0, -73.27],
                   [0.0, 0.0, -54.95],
                   [-17.57, 3.61, -84.38],  # pinky
                   [0.0, 0.0, -110.25],
                   [0.0, 0.0, -73.63],
                   [94.49, -24.16, -33.79],  # thumb
                   [-7.12, 3.68, -63.08],
                   [0.0, 0.0, -34.54]],
                  [[0.0, 15.0, 0.0],  # hand root
                   [-3.96, -18, -2.82],  # index
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [-1.45, 3.12, -11.07],  # middle
                   [0, 0, -1],
                   [0.0, 0.0, -0.726],
                   [-12.96, 2.41, -83.60],  # ring
                   [0.0, 0.0, -73.27],
                   [0.0, 0.0, -54.95],
                   [-17.57, 3.61, -84.38],  # pinky
                   [0.0, 0.0, -110.25],
                   [0.0, 0.0, -73.63],
                   [94.49, -24.16, -33.79],  # thumb
                   [-7.12, 3.68, -63.08],
                   [0.0, 0.0, -34.54]]
                  ]
    euler_list = np.asarray(euler_list)
    euler_list[..., 1:3] = -euler_list[..., 1:3]
    hpa = HandPoseArguer(side='right')
    hpa.argue_pose(euler_list, pose_type='euler', vis=True, argue_size=50)
