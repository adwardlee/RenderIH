from manopth.manolayer import ManoLayer
import torch
import numpy as np
import open3d as o3d
import copy
smplx_model_path = '../smpl/smplx_model/mano_man.obj'
smplx_model = o3d.io.read_triangle_mesh(smplx_model_path)


smplx_model__ = copy.deepcopy(smplx_model)
model_v = np.asarray(smplx_model.vertices)
model_v[:, 0] = -model_v[:, 0]
# left_mano_layer = ManoLayer(
#         mano_root="../assets/mano",
#         use_pca=False,
#         ncomps=6,
#         flat_hand_mean=True,
#         center_idx=0,
#         return_transf=False,
#         side='left',
#         root_rot_mode='quat',
#         joint_rot_mode='quat'
#     )
# right_mano_layer = ManoLayer(
#     mano_root="../assets/mano",
#     use_pca=False,
#     ncomps=6,
#     flat_hand_mean=True,
#     center_idx=0,
#     return_transf=False,
#     side='right',
#     root_rot_mode='quat',
#     joint_rot_mode='quat'
# )
# vec = torch.tensor([[[ 0.9999, -0.0117,  0.0078, -0.0060],
#          [ 0.9988,  0.0436,  0.0069,  0.0212],
#          [ 0.9987,  0.0051, -0.0245,  0.0447],
#          [ 0.9993, -0.0361,  0.0018, -0.0050],
#          [ 0.9989,  0.0185, -0.0169,  0.0401],
#          [ 0.9996,  0.0260,  0.0112, -0.0037],
#          [ 0.9997, -0.0224,  0.0067, -0.0077],
#          [ 0.9987, -0.0061, -0.0421,  0.0285],
#          [ 0.9987,  0.0385, -0.0129,  0.0311],
#          [ 0.9987,  0.0043,  0.0222, -0.0459],
#          [ 0.9987,  0.0189, -0.0354,  0.0316],
#          [ 0.9990,  0.0237,  0.0384,  0.0073],
#          [ 0.9995,  0.0237,  0.0199, -0.0064],
#          [ 0.9987,  0.0186,  0.0474, -0.0052],
#          [ 0.9987, -0.0211,  0.0460, -0.0078],
#          [ 0.9987, -0.0237, -0.0385, -0.0238]]])
# shape = torch.tensor([-0.3832, -0.1149,  0.1231, -0.0044,  0.0175, -0.1340, -0.1389,  0.0633,
#           0.0087, -0.0235]).unsqueeze(0)
# faces = np.array(left_mano_layer.th_faces).astype(np.long)
# l_vertices, l_joints = left_mano_layer(vec, shape)
# r_vertices, r_joints = right_mano_layer(vec, shape)
# l_vertices[:, :, 0] = -l_vertices[:, :, 0]
vis_cur = o3d.visualization.VisualizerWithKeyCallback()
vis_cur.create_window(window_name="Runtime Hand", width=1080, height=1080)
#
# l_hand_mesh = o3d.geometry.TriangleMesh()
# r_hand_mesh = o3d.geometry.TriangleMesh()
# vis_cur.add_geometry(l_hand_mesh)
# vis_cur.add_geometry(r_hand_mesh)
# l_hand_mesh.triangles = o3d.utility.Vector3iVector(faces)
# r_hand_mesh.triangles = o3d.utility.Vector3iVector(faces)
# mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])  # 添加坐标系
# vis_cur.add_geometry(mesh_frame)
# l_hand_mesh.vertices = o3d.utility.Vector3dVector(
#             np.array(l_vertices.detach().cpu().squeeze(0)))
# r_hand_mesh.vertices = o3d.utility.Vector3dVector(
#             np.array(r_vertices.detach().cpu().squeeze(0)))
# l_hand_mesh.vertex_colors = o3d.utility.Vector3dVector(
#             np.array([[1.0, 1.0, 1.0]] * l_vertices.shape[1]))
# r_hand_mesh.vertex_colors = o3d.utility.Vector3dVector(
#     np.array([[1.0, 0.0, 0.0]] * l_vertices.shape[1]))
#
# l_hand_mesh.compute_vertex_normals()
# r_hand_mesh.compute_vertex_normals()
# vis_cur.update_geometry(l_hand_mesh)
# vis_cur.update_geometry(r_hand_mesh)
vis_cur.add_geometry(smplx_model)
vis_cur.add_geometry(smplx_model__)
vis_cur.run()