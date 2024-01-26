import open3d as o3d
import torch
from manopth.manolayer import ManoLayer
import numpy as np
from scipy.spatial.transform import Rotation as rotation
import copy
from mano.webuser.smpl_handpca_wrapper_HAND_only import ready_arguments
import os
from manopth.quatutils import quaternion_to_rotation_matrix

# **** axis order right hand

#         15-14-13-\
#                   \
#    3-- 2 -- 1 -----0
#   6 -- 5 -- 4 ----/
#   12 - 11 - 10 ---/
#    9-- 8 -- 7 --/

# **** joint order right hand
#               4-3-2-1-\
#                         \
#      8-- 7-- 6 -- 5 -----0
#   12--11 -- 10 -- 9 ----/
#    16-- 15 - 14 - 13 ---/
#    20--19-- 18 -- 17 --/

class HandPoseConverter:
    def __init__(self, shape=None, side='right', root="../assets/mano", data_type='np', device='cuda:0'):
        self.side = side
        self.mano_layer = ManoLayer(
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
        if shape is None:
            self.shape = torch.tensor([0.5082395, -0.39488167, -1.7484332, 1.6630946, 0.34428665, -1.37387,
                                       0.38293332, 1.196094, 0.6538949, -0.94331187]).unsqueeze(0)
        else:
            self.shape = shape
        self.faces = np.array(self.mano_layer.th_faces).astype(np.long)
        self.joint_idx = np.asarray([[0, 1, 2, 3, 4],
                                     [0, 5, 6, 7, 8],
                                     [0, 9, 10, 11, 12],
                                     [0, 13, 14, 15, 16],
                                     [0, 17, 18, 19, 20]])
        self.axis_idx = np.asarray([[0, 13, 14, 15],
                                    [0, 1, 2, 3],
                                    [0, 4, 5, 6],
                                    [0, 10, 11, 12],
                                    [0, 7, 8, 9]])
        vec_pose = torch.from_numpy(np.asarray([[1.0, 0.0, 0.0, 0.0]] * 16, dtype=np.float32)).unsqueeze(0)
        vertices, joints, transf = self.mano_layer(vec_pose, self.shape)
        joints = joints.squeeze(0).numpy()
        vertices = vertices.squeeze(0).numpy()
        transf = transf.squeeze(0).numpy()

        converted_transf = self.axis_convert(joints, transf)
        self.init_convert_matrix(converted_transf, transf, self.axis_idx)
        self.data_type = data_type
        if data_type == 'tensor':
            self.invU_M_n_1 = torch.from_numpy(self.invU_M_n_1).to(device)
            self.invM_U_n_0 = torch.from_numpy(self.invM_U_n_0).to(device)

        # hand_mean
        if side == "right":
            self.mano_path = os.path.join(root, "MANO_RIGHT.pkl")
        elif side == "left":
            self.mano_path = os.path.join(root, "MANO_LEFT.pkl")
        smpl_data = ready_arguments(self.mano_path)
        # hands_components = smpl_data["hands_components"]
        hands_mean = np.asarray(smpl_data["hands_mean"], dtype=np.float32)
        self.th_hands_mean = hands_mean.reshape(1, 15, 3)

    def axis_convert(self, joints, transf):
        joints_mapping = [5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]
        converted_num = joints_mapping.__len__()
        if self.side == 'right':
            x_axis = (joints[joints_mapping] - joints[[i + 1 for i in joints_mapping]])
            up_axis_base = np.vstack(
                (np.array([[0, 1, 0]]).repeat(12, axis=0), np.array([[1, 1, 1]]).repeat(3, axis=0)))
        else:
            x_axis = -(joints[joints_mapping] - joints[[i + 1 for i in joints_mapping]])
            up_axis_base = np.vstack(
                (np.array([[0, -1, 0]]).repeat(12, axis=0), np.array([[-1, -1, -1]]).repeat(3, axis=0))) # some thing may be wrong here
            up_axis_base = np.vstack(
                (np.array([[0, -1, 0]]).repeat(12, axis=0), np.array([[1, -1, - 1]]).repeat(3, axis=0)))
        loc = transf[:, 0:3, 3]
        x_axis = transf[1:, :3, :3].swapaxes(2, 1) @ x_axis[:, :, np.newaxis]
        x_axis = x_axis.squeeze(-1)
        z_axis = np.cross(x_axis, up_axis_base)
        y_axis = np.cross(z_axis, x_axis)

        x_axis = x_axis / np.linalg.norm(x_axis, axis=-1)[:, np.newaxis]
        y_axis = y_axis / np.linalg.norm(y_axis, axis=-1)[:, np.newaxis]
        z_axis = z_axis / np.linalg.norm(z_axis, axis=-1)[:, np.newaxis]

        output = np.tile(np.eye(4)[np.newaxis, :, :], (converted_num + 1, 1, 1))
        converted_axis = np.hstack([x_axis, y_axis, z_axis]).reshape(converted_num, 3, 3)
        converted_axis = np.vstack([np.eye(3)[np.newaxis, :, :], converted_axis])
        output[:, 0:3, 0:3] = converted_axis.swapaxes(2, 1)
        if self.side == 'left':
            output[0, 0:3, 0:3] = np.asarray(([1, 0, 0], [0, -1, 0], [0, 0, -1]))
        output[:, 0:3, 3] = loc
        output[:, 0:3, 0:3] = transf[:, 0:3, 0:3] @ output[:, 0:3, 0:3]
        return output

    def init_convert_matrix(self, converted_matrix, transf, axis_idx):
        # inv(R_M_n) @ R_U_n @ P_U_n @ inv(R_U_(n+1)) @ R_M_(n+1) = P_M_n
        #          invM_U_n_0                   invU_M_n_1
        self.invM_U_n_0 = np.tile(np.eye(3)[np.newaxis, :, :], (16, 1, 1)).astype(np.float32)
        self.invU_M_n_1 = np.tile(np.eye(3)[np.newaxis, :, :], (16, 1, 1)).astype(np.float32)
        u = converted_matrix[:, 0:3, 0:3]
        m = transf[:, 0:3, 0:3]
        for finger_id in range(axis_idx.__len__()):
            for id in range(axis_idx[finger_id].__len__()):
                if id == 0:
                    axis_id = 0
                    prev_axis_id = 0
                else:
                    prev_axis_id = axis_idx[finger_id][id - 1]
                    axis_id = axis_idx[finger_id][id]
                self.invU_M_n_1[axis_id] = u[axis_id].transpose() @ m[axis_id]
                self.invM_U_n_0[axis_id] = m[prev_axis_id].transpose() @ u[prev_axis_id]

    def rotation_2_euler(self, rot, order='xyz', degree=True):
        mat = rotation.from_matrix(rot)
        return mat.as_euler(order, degree)

    def euler_2_rotation(self, euler, order='xyz', degree=True):
        mat = rotation.from_euler(order, euler, degrees=degree)
        return mat.as_matrix()

    def quat_2_rotation(self, quat):
        quat = quat[:, [1, 2, 3, 0]]
        mat = rotation.from_quat(quat)
        return mat.as_matrix()

    def rotation_2_quat(self, rot, order='wxyz'):
        '''
        :param rot:(N, 3, 3)
        :return: (N, (x,y,z,w）)
        '''
        mat = rotation.from_matrix(rot)
        quat = mat.as_quat()
        if quat.shape.__len__() <= 1:
            quat = quat[np.newaxis, :]
        if order == 'wxyz':
            return quat[:, [3, 0, 1, 2]]
        else:
            return quat

    def euler_2_mano_quat(self, euler_list):
        '''
        :param euler_list: (N, 16, 3) or (16, 3)
        :return: (N, 16, 4) w,x,y,z
        '''
        euler_list = np.asarray(euler_list)
        if euler_list.shape.__len__() == 2:
            euler_list = euler_list[np.newaxis, :, :]
        # if self.side == 'left':
        #     euler_list[:, :, 0] = euler_list[:, :, 0]
        #     euler_list[:, :, 1] = - euler_list[:, :, 1]
        #     euler_list[:, :, 2] = - euler_list[:, :, 2]
        joint_num = euler_list.shape[1]
        batch_num = euler_list.shape[0]
        batch_mat = np.zeros([batch_num, joint_num, 3, 3])
        for joint_id in range(joint_num):
            eulers = euler_list[:, joint_id, :]
            mats = self.euler_2_rotation(eulers)
            batch_mat[:, joint_id, :, :] = mats
        batch_mat = self.invM_U_n_0 @ batch_mat @ self.invU_M_n_1
        if self.side == 'left':
            batch_mat[:, 0, :, :] = np.tile(np.asarray(([1, 0, 0], [0, -1, 0], [0, 0, -1]))[np.newaxis, :, :],
                                            (batch_mat.__len__(), 1, 1)) @ batch_mat[:, 0, :, :]
        batch_mat = batch_mat.reshape([-1, 3, 3])
        batch_quat = self.rotation_2_quat(batch_mat)
        batch_quat = batch_quat.reshape([-1, joint_num, 4])
        if self.side == 'left':
            batch_quat[:, :, 2] = -batch_quat[:, :, 2]
            batch_quat[:, :, 3] = -batch_quat[:, :, 3]
        return batch_quat.astype(np.float32)

    def mano_quat_2_euler(self, batch_quat):
        if batch_quat.shape.__len__() <= 2:
            batch_quat = batch_quat[np.newaxis, :, :]
        joint_num = batch_quat.shape[1]
        batch_num = batch_quat.shape[0]
        if self.side == 'left':
            batch_quat[:, :, 2] = -batch_quat[:, :, 2]
            batch_quat[:, :, 3] = -batch_quat[:, :, 3]
        batch_quat = batch_quat.reshape([-1, 4])
        batch_mat = self.quat_2_rotation(batch_quat)
        batch_mat = batch_mat.reshape([-1, joint_num, 3, 3])
        if self.side == 'left':
            batch_mat[:, 0, :, :] = np.tile(np.asarray(([1, 0, 0], [0, -1, 0], [0, 0, -1]))[np.newaxis, :, :],
                                            (batch_mat.__len__(), 1, 1)) @ batch_mat[:, 0, :, :]
        batch_mat = self.invM_U_n_0.swapaxes(1, 2) @ batch_mat @ self.invU_M_n_1.swapaxes(1, 2)  # N, 16, 3, 3
        euler_list = np.zeros([batch_num, joint_num, 3])
        for joint_id in range(joint_num):
            euler_list[:, joint_id, :] = self.rotation_2_euler(batch_mat[:, joint_id, :, :])
        # if self.side == 'left':
        #     euler_list[:, :, 0] = euler_list[:, :, 0]
        #     euler_list[:, :, 1] = - euler_list[:, :, 1]
        #     euler_list[:, :, 2] = - euler_list[:, :, 2]
        return euler_list

    def mano_quat_2_mat_tensor(self, batch_quat):
        assert self.data_type == 'tensor'
        assert batch_quat.shape.__len__() == 3
        bq = batch_quat.clone()
        joint_num = bq.shape[1]
        batch_num = bq.shape[0]
        if self.side == 'left':
            bq[:, :, 2] = -bq[:, :, 2]
            bq[:, :, 3] = -bq[:, :, 3]
        # batch_quat = batch_quat.reshape([-1, 4])
        batch_mat = quaternion_to_rotation_matrix(bq)
        # batch_mat = batch_mat.reshape([-1, joint_num, 3, 3])
        if self.side == 'left':
            batch_mat[:, 0, 1:, 1:] = -batch_mat[:, 0, 1:, 1:]
            # batch_mat[:, 0, :, :] = np.tile(np.asarray(([1, 0, 0], [0, -1, 0], [0, 0, -1]))[np.newaxis, :, :],
            #                                 (batch_mat.__len__(), 1, 1)) @ batch_mat[:, 0, :, :]
        batch_mat = self.invM_U_n_0.transpose(1, 2) @ batch_mat @ self.invU_M_n_1.transpose(1, 2)  # N, 16, 3, 3
        return batch_mat


    def vis_unreal_pose(self, euler_list):
        batch_pose = self.euler_2_mano_quat(euler_list)
        for pose in batch_pose:
            pose = torch.from_numpy(pose).unsqueeze(0)
            vertices, joints, transf = self.mano_layer(pose, self.shape)
            joints = joints.squeeze(0).numpy()
            vertices = vertices.squeeze(0).numpy()
            transf = transf.squeeze(0).numpy()
            transf = self.axis_convert(joints, transf)


            #
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='coordinate_visualization')

            for trans in transf:
                mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02,
                                                                               origin=[trans[0, 3], trans[1, 3],
                                                                                       trans[2, 3]])  # 添加坐标系
                mesh_frame.rotate(trans[0:3, 0:3])
                vis.add_geometry(mesh_frame)

            test_mesh = o3d.open3d.geometry.PointCloud()
            test_mesh.points = o3d.utility.Vector3dVector(joints)
            test_mesh.paint_uniform_color([1, 0, 1])
            vis.add_geometry(test_mesh)

            hand_mesh = o3d.geometry.TriangleMesh()
            hand_mesh.triangles = o3d.utility.Vector3iVector(self.faces)
            hand_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            # hand_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 1.0]] * len(vertices.squeeze(0).numpy())))
            hand_mesh.compute_vertex_normals()
            vis.add_geometry(hand_mesh)
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])  # 添加坐标系
            vis.add_geometry(mesh_frame)
            vis.run()
            vis.destroy_window()

    def axis_angle_2_mano_quat(self, baa_list, flat_hand=False, order='wxyz'):
        '''
        :param
        baa_list: (N, 16, 3) or (48) or (N, 48) or (16, 3) AXIS ANGLE
        flat_hand:
        :return: (N, 16, 4)
        '''
        aa_list = np.asarray(baa_list, dtype=np.float32)
        aa_list = aa_list.reshape((-1, 16, 3))

        if not flat_hand:
            root_list = aa_list[:, 0:1, :]
            aa_list = aa_list[:, 1:, :]
            aa_list = aa_list + self.th_hands_mean
            aa_list = np.hstack([root_list, aa_list])
        aa_list_shape = aa_list.shape
        r = rotation.from_rotvec(aa_list.reshape(-1, 3))
        quat = r.as_quat()
        quat = quat.reshape(aa_list_shape[0], aa_list_shape[1], 4)
        quat = np.asarray(quat, dtype=np.float32)

        if quat.shape.__len__() <= 1:
            quat = quat[np.newaxis, :]
        if order == 'wxyz':
            quat = quat[..., [3, 0, 1, 2]]

        if self.side == 'left':
            quat[:, :, 2] = -quat[:, :, 2]
            quat[:, :, 3] = -quat[:, :, 3]
        return quat

    def mano_quat_flat_2_normal(self, batch_quat, order='wxyz', return_type='quat'):
        '''
        :param batch_quat: w,x,y,z

        :return:
        '''
        quat_vec = np.asarray(batch_quat, dtype=np.float32)
        quat_vec = quat_vec.reshape((-1, 16, 4))
        if self.side == 'left':
            quat_vec[:, :, 2] = -quat_vec[:, :, 2]
            quat_vec[:, :, 3] = -quat_vec[:, :, 3]
        bts, js, _ = quat_vec.shape
        if order == 'wxyz':
            quat_vec = quat_vec[..., [1,2,3,0]]
        r = rotation.from_quat(quat_vec.reshape(-1, 4))
        aa = r.as_rotvec()
        aa = aa.reshape((bts, js, 3))

        if js == 16:
            root_list = aa[:, 0:1, :]
            aa = aa[:, 1:, :]
        else:
            root_list = []
        aa = aa - self.th_hands_mean
        if js == 16:
            aa = np.hstack([root_list, aa])
        if return_type == 'quat':
            return self.axis_angle_2_mano_quat(aa, flat_hand=True)
        else:
            return aa

    def trans_2_loc(self, vec, shape, trans, loc_center_idx):
        '''
        :param vec: B*16*4
        :param shape: B*10
        :return: B*3
        '''
        center_idx = self.mano_layer.center_idx
        self.mano_layer.center_idx = None
        v, j, _ = self.mano_layer(vec, shape)
        self.mano_layer.center_idx = center_idx
        return j[:, loc_center_idx] + trans

    def loc_2_trans(self, vec_, shape_, loc, loc_center_idx):
        vec = self.check_array_torch(vec_)
        shape = self.check_array_torch(shape_)
        center_idx = self.mano_layer.center_idx
        self.mano_layer.center_idx = None
        v, j, _ = self.mano_layer(vec, shape)
        trans = loc - j[:, loc_center_idx].numpy()
        self.mano_layer.center_idx = center_idx
        return trans

    def check_array_torch(self, input):
        return torch.tensor(input)

if __name__ == '__main__':
    HPC = HandPoseConverter(side='left')
    aa_list = np.asarray([[0.5979, 0.1381, 2.4408, 0.0309, 0.0983, 0.3270, 0.1328, -0.0663,
               0.1130, -0.0201, -0.1653, -0.0123, -0.1827, 0.0461, 0.8238, 0.0742,
               0.0355, -0.2552, -0.0886, -0.0771, -0.1113, -0.2175, 0.1242, 0.5137,
               -0.2043, 0.1445, 0.0441, -0.1066, -0.0485, -0.0105, -0.1660, 0.0220,
               0.7614, -0.0259, 0.1191, -0.2536, -0.0649, -0.0954, -0.0777, 0.1043,
               -0.1216, -0.0908, 0.0176, -0.1388, -0.0498, 0.1067, -0.0967, 0.4821],
               [0.5979, 0.1381, 2.4408, 0.0309, 0.0983, 0.3270, 0.1328, -0.0663,
               0.1130, -0.0201, -0.1653, -0.0123, -0.1827, 0.0461, 0.8238, 0.0742,
               0.0355, -0.2552, -0.0886, -0.0771, -0.1113, -0.2175, 0.1242, 0.5137,
               -0.2043, 0.1445, 0.0441, -0.1066, -0.0485, -0.0105, -0.1660, 0.0220,
               0.7614, -0.0259, 0.1191, -0.2536, -0.0649, -0.0954, -0.0777, 0.1043,
               -0.1216, -0.0908, 0.0176, -0.1388, -0.0498, 0.1067, -0.0967, 0.4821]])
    euler_list = [[[0.0, 0.0, 0.0],  # hand root
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
                   [0.0, 0.0, -34.54]],
                  [[0.0, 0.0, 0.0],  # hand root
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
    euler_list = [[ 5.16023985e+01,  7.63481741e+01, -1.51932464e+02],
       [ 5.91851022e+00, -1.10311521e+01,  4.05892598e+01],
       [ 7.71760719e-01, -1.70774565e+00,  4.76603433e+01],
       [-5.97887602e-01,  9.80578011e+00,  9.70633185e+00],
       [-1.32887584e+01, -1.31078169e+01,  6.02917304e+01],
       [ 9.64055150e-02, -5.53733942e-01,  4.74285214e+01],
       [-9.19674738e-01,  3.75067554e+00,  1.55790797e+01],
       [-2.13288045e+01,  8.17067381e+00,  6.47145005e+01],
       [ 9.47768269e+00,  4.71513016e+00,  3.98509707e+01],
       [-2.56880542e+01, -2.85849684e-01,  1.22832719e+01],
       [-1.25830692e+01, -6.33000427e+00,  5.68664011e+01],
       [-7.27590088e+00,  4.01358000e+00,  4.87501262e+01],
       [-7.83264054e+00,  4.60803228e+00,  2.20374680e+01],
       [ 3.80302080e+01,  2.90164684e+01,  2.15022014e+01],
       [ 8.61589558e-01, -1.47067579e+01, -1.82728282e+01],
       [ 1.69759865e+01, -4.57765710e+00,  2.10447822e+01]]
    # e = HPC.mano_quat_2_euler(HPC.euler_2_mano_quat(euler_list))
    # e = HPC.axis_angle_2_mano_quat(aa_list, False)
    # print(e)
    euler_list = np.asarray(euler_list)
    # euler_list[:, :, 1:3] = -euler_list[:, :, 1:3]
    HPC.vis_unreal_pose(euler_list)
