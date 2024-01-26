import open3d as o3d
import torch
from manopth.manolayer import ManoLayer
import numpy as np
from scipy.spatial.transform import Rotation as rotation
import copy

# **** axis order right hand

#         15-14-13-\
#                   \
#    3-- 2 -- 1 -----0
#   6 -- 5 -- 4 ----/
#   12 - 11 - 10 ---/
#    9-- 8 -- 7 --/

# **** joint order right hand
    #        4-3-2-1-\
    #                   \
#      8-- 7-- 6 -- 5 -----0
#   12--11 -- 10 -- 9 ----/
#    16-- 15 - 14 - 13 ---/
#    20--19-- 18 -- 17 --/
def axis_convert(joints, transf):
    joints_mapping = [5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3]
    converted_num = joints_mapping.__len__()
    up_axis_base = np.vstack((np.array([[0, 1, 0]]).repeat(12, axis=0), np.array([[1, 1, 1]]).repeat(3, axis=0)))
    loc = transf[:, 0:3, 3]
    x_axis = (joints[joints_mapping] - joints[[i + 1 for i in joints_mapping]])
    # for id in range(x_axis.__len__()):
    #     x_axis[id, :] = transf[1:, :3, :3][id].swapaxes(1, 0).dot(x_axis[id, :])

    x_axis = transf[1:, :3, :3].swapaxes(2, 1) @ x_axis[:,:,np.newaxis]
    x_axis = x_axis.squeeze(-1)
    z_axis = np.cross(x_axis, up_axis_base)
    y_axis = np.cross(z_axis, x_axis)

    x_axis = x_axis / np.linalg.norm(x_axis, axis=-1)[:, np.newaxis]
    y_axis = y_axis / np.linalg.norm(y_axis, axis=-1)[:, np.newaxis]
    z_axis = z_axis / np.linalg.norm(z_axis, axis=-1)[:, np.newaxis]

    output = np.tile(np.eye(4)[np.newaxis, :,:], (converted_num + 1,1,1))
    converted_axis = np.hstack([x_axis, y_axis, z_axis]).reshape(converted_num, 3, 3)
    converted_axis = np.vstack([np.eye(3)[np.newaxis, :, :], converted_axis])
    output[:, 0:3, 0:3] = converted_axis.swapaxes(2,1)
    output[:, 0:3, 3] = loc
    output[:, 0:3, 0:3] = transf[:, 0:3, 0:3]  @ output[:, 0:3, 0:3]
    return output

def euler_2_rotation(R, P, Y, order='xyz', degree=True):
    mat = rotation.from_euler(order,[R,P,Y], degrees=degree)
    return mat.as_matrix()
def rotation_2_quat(rot, order='wxyz'):
    '''

    :param rot:
    :return: x,y,z,w
    '''
    mat = rotation.from_matrix(rot[0:3, 0:3])
    quat = mat.as_quat()
    if order =='wxyz':
        return quat[[3,0,1,2]]
    else:
        return quat

def vis_coordinate(vis, trans):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[trans[0, 3], trans[1, 3], trans[2, 3]])  # 添加坐标系
    mesh_frame.rotate(trans[0:3, 0:3])
    vis.add_geometry(mesh_frame)

def vis_skeleton(vis, joints, joint_order):
    points = []
    correspond = []
    correspond_start_id = 0
    for finger_id in range(joint_order.__len__()):
        finger = joint_order[finger_id]
        for id in range(finger.__len__()):
            joint_id = finger[id]
            points.append(joints[joint_id])
        for id in range(finger.__len__() - 1):
            correspond.append([correspond_start_id + id, correspond_start_id + id + 1])
        correspond_start_id += finger.__len__()
    points = np.asarray(points)
    correspond = np.asarray(correspond)
    color = [[1, 0, 0] for i in range(len(correspond))]

    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(points)
    points_pcd.paint_uniform_color([0, 0.3, 0])  # 点云颜色

    # 绘制线条
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(correspond)
    lines_pcd.colors = o3d.utility.Vector3dVector(color)  # 线条颜色
    lines_pcd.points = o3d.utility.Vector3dVector(points)
    vis.add_geometry(lines_pcd)
    vis.add_geometry(points_pcd)

def convert_to_relative_matrix(matrix, axis_idx):
    converted_matrix = copy.deepcopy(matrix)
    for finger_id in range(axis_idx.__len__()):
        for id in range(axis_idx[finger_id].__len__()):
            if id == 0:
                continue
            prev_axis_id = axis_idx[finger_id][id - 1]
            axis_id = axis_idx[finger_id][id]
            converted_matrix[axis_id] = (matrix[prev_axis_id].transpose().dot(matrix[axis_id]))
    return converted_matrix

def find_convert_matrix(converted_matrix, transf, axis_idx):
    bts_2_mano = np.tile(np.eye(3)[np.newaxis, :,:], (16,1,1)).astype(np.float32)
    mano_2_bts = np.tile(np.eye(3)[np.newaxis, :, :], (16, 1, 1)).astype(np.float32)
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
            # bts_2_mano[axis_id] = (m[axis_id] @ u[axis_id].transpose() @ u[prev_axis_id] @ m[prev_axis_id].transpose())[0:3, 0:3]
            # bts_2_mano[axis_id] = (u[prev_axis_id] @ u[axis_id].transpose() @ m[axis_id] @ m[prev_axis_id].transpose())[
            #                       0:3, 0:3]
            bts_2_mano[axis_id] = u[axis_id].transpose() @ m[axis_id]
            mano_2_bts[axis_id] = m[prev_axis_id].transpose() @ u[prev_axis_id]
    # for i in range(1, 16):
    #     bts_2_mano[i] = (transf[i] @ converted_matrix[i].transpose() @ converted_matrix[i-1] @ transf[i-1].transpose())[0:3, 0:3]
    # converted_matrix_rel = convert_to_relative_matrix(converted_matrix, axis_idx)
    # transf_rel = convert_to_relative_matrix(transf, axis_idx)
    # mano_2_bts = (transf_rel.swapaxes(1, 2) @ converted_matrix_rel)[:, :3, :3]
    # bts_2_mano = (converted_matrix_rel.swapaxes(1, 2) @ transf_rel)[:, :3, :3]
    # mano_2_bts = bts_2_mano.swapaxes(1, 2)
    return mano_2_bts, bts_2_mano
def main():
    euler_list = [[0.0, 0.0, 0.0],  # hand root
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
    euler_list = [[0.0, 0.0, 0.0] for i in range(16)]
    mano_layer = ManoLayer(
        mano_root="../assets/mano",
        use_pca=False,
        ncomps=6,
        flat_hand_mean=True,
        center_idx=9,
        return_transf=True,
        side='right',
        root_rot_mode='quat',
        joint_rot_mode='quat'
    )
    faces = np.array(mano_layer.th_faces).astype(np.long)
    joint_idx = np.asarray([[0, 1, 2, 3, 4],
                            [0, 5, 6, 7, 8],
                            [0, 9, 10, 11, 12],
                            [0, 13, 14, 15, 16],
                            [0, 17, 18, 19, 20]])
    axis_idx = np.asarray([[0, 13, 14, 15],
                            [0, 1, 2, 3],
                            [0, 4, 5, 6],
                            [0, 10, 11, 12],
                            [0, 7, 8, 9]])
    # gen random shape
    vec_shape = torch.tensor([ 0.5082395,  -0.39488167, -1.7484332 ,  1.6630946 ,  0.34428665, -1.37387,
               0.38293332,  1.196094 ,   0.6538949,  -0.94331187]).unsqueeze(0)
    vec_shape = torch.tensor([-0.3832, -0.1149,  0.1231, -0.0044,  0.0175, -0.1340, -0.1389,  0.0633,
          0.0087, -0.0235]).unsqueeze(0)
    # vec_shape = torch.tensor([0.39090609550476074, -5.10849937051534653, -0.30046647787094116, -0.03732258453965187, -0.09943775832653046, -0.023937644436955452, 0.04136071354150772, 0.10307897627353668, 0.04249461740255356, 0.16355504095554352]).unsqueeze(0)

    # vec_shape = torch.rand([10]).unsqueeze(0)
    # gen zero pose
    vec_pose = torch.from_numpy(np.asarray([[1.0, 0.0, 0.0, 0.0]] * 16, dtype=np.float32)).unsqueeze(0)
    # data = np.load('../y.npy', allow_pickle=True).item()
    # vec_pose = torch.from_numpy(data['hand_pose']).unsqueeze(0)
    # vec_pose[0][0] = torch.tensor([1.0, 0, 0, 0])
    # vec_pose = torch.rand((1,16, 4))
    # vec_pose[0][1] = torch.tensor([0.707, 0.0, 0.0, 0.707])
    # gen hand
    vertices, joints, transf = mano_layer(vec_pose, vec_shape)
    joints = joints.squeeze(0).numpy()
    vertices = vertices.squeeze(0).numpy()
    transf = transf.squeeze(0).numpy()

    converted_transf = axis_convert(joints, transf)
    mano_2_bts, bts_2_mano = find_convert_matrix(converted_transf, transf, axis_idx)



    # init test angel
    euler_list = np.asarray(euler_list)

    # euler_list = 0.0 * euler_list
    euler_list[:, 1:3] = -euler_list[:, 1:3]
    # euler_list[:, 0:2] = -euler_list[:, 0:2]
    euler_list[:, 0] = euler_list[:, 0]
    euler_list[:, 1] = euler_list[:, 1]
    euler_list[:, 2] = euler_list[:, 2]
    batch_mat = []
    for euler in euler_list:
        mat = euler_2_rotation(euler[0], euler[1], euler[2])
        batch_mat.append(mat)
    batch_mat = np.asarray(batch_mat)
    batch_mat = mano_2_bts @ batch_mat @ bts_2_mano
    pose = np.asarray([], dtype=np.float32)
    for mat in batch_mat:
        pose = np.append(pose, (rotation_2_quat(mat)))
    pose = pose.astype(np.float32)
    pose = pose.reshape((16, 4))
    vec_pose = torch.from_numpy(pose).unsqueeze(0)
    # vec_pose[0][3] = torch.tensor([0.707, 0.0, 0.707, 0.0])
    # vec_pose[:, :, 2] = -vec_pose[:, :, 2]
    # vec_pose[:, :, 3] = -vec_pose[:, :, 3]
    vertices, joints, transf = mano_layer(vec_pose, vec_shape)
    joints = joints.squeeze(0).numpy()
    vertices = vertices.squeeze(0).numpy()
    transf = transf.squeeze(0).numpy()
    transf = axis_convert(joints, transf)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='coordinate_visualization')
    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.triangles = o3d.utility.Vector3iVector(faces)
    hand_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # hand_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.0, 0.0, 1.0]] * len(vertices.squeeze(0).numpy())))
    hand_mesh.compute_vertex_normals()
    vis.add_geometry(hand_mesh)
    for trans in transf:
        vis_coordinate(vis, trans)
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])  # 添加坐标系
    # vis.add_geometry(mesh_frame)
    vis_skeleton(vis, joints, joint_idx)
    vis.reset_view_point(True)
    ctr = vis.get_view_control()
    ctr.rotate(0.0, 600)
    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    main()