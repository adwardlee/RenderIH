import open3d as o3d
import torch
from manopth.manolayer import ManoLayer
from HandPoseConverter import HandPoseConverter
import numpy as np
from scripts.get_mano_skeleton_inf import get_mano_skeleton_inf
from scipy.spatial.transform import Rotation as rotation
import copy

def vis_coordinate(vis, trans):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=[trans[0, 3], trans[1, 3], trans[2, 3]])  # 添加坐标系
    mesh_frame.rotate(trans[0:3, 0:3])
    vis.add_geometry(mesh_frame)

sides = ['left', 'right']
visual = True
if visual:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='coordinate_visualization')
mano_root_path = "../assets/mano"
skel_tsl_infs = dict()
for side in sides:

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
    hpc = HandPoseConverter(side=side, root="../assets/mano")
    faces = np.array(mano_layer.th_faces).astype(np.long)

    # root_loc

    wrist_loc = np.asarray([-0.71126, 0.042163, -0.06351])
    tsl = np.asarray([-0.0188, -0.0084, -0.0017])
    if side == 'left':
        wrist_loc = np.asarray([0.71077, 0.038536, -0.064438])
        tsl[0] = -tsl[0]
    root_loc = wrist_loc + tsl
    # tensor([[[-0.0188, -0.0084, -0.0017]]], device='cuda:0', requires_grad=True)
    # [[[-0.73002172  1.39272803 -0.0652555 ]]]

    # gen random shape
    vec_shape = torch.tensor([-0.3832, -0.1149,  0.1231, -0.0044,  0.0175, -0.1340, -0.1389,  0.0633,
              0.0087, -0.0235]).unsqueeze(0)
    skel_tsl_infs[side] = get_mano_skeleton_inf(side=side, mano_shape=vec_shape.squeeze(0), mano_root_path=mano_root_path)
    # gen zero pose
    vec_pose = torch.zeros(1, 48)
    vec_pose = torch.tensor([[[ 0.9999, -0.0117,  0.0078, -0.0060],
             [ 0.9988,  0.0436,  0.0069,  0.0212],
             [ 0.9987,  0.0051, -0.0245,  0.0447],
             [ 0.9993, -0.0361,  0.0018, -0.0050],
             [ 0.9989,  0.0185, -0.0169,  0.0401],
             [ 0.9996,  0.0260,  0.0112, -0.0037],
             [ 0.9997, -0.0224,  0.0067, -0.0077],
             [ 0.9987, -0.0061, -0.0421,  0.0285],
             [ 0.9987,  0.0385, -0.0129,  0.0311],
             [ 0.9987,  0.0043,  0.0222, -0.0459],
             [ 0.9987,  0.0189, -0.0354,  0.0316],
             [ 0.9990,  0.0237,  0.0384,  0.0073],
             [ 0.9995,  0.0237,  0.0199, -0.0064],
             [ 0.9987,  0.0186,  0.0474, -0.0052],
             [ 0.9987, -0.0211,  0.0460, -0.0078],
             [ 0.9987, -0.0237, -0.0385, -0.0238]]])
    # vec_pose = torch.rand(1, 45 + 3)
    # vec_pose = torch.from_numpy(np.asarray([[1.0, 0.0, 0.0, 0.0]] * 16, dtype=np.float32)).unsqueeze(0)
    vertices, joints, transf = mano_layer(vec_pose, vec_shape)
    axis_transf = hpc.axis_convert(joints.squeeze(0).numpy(), transf.squeeze(0).numpy())

    joints_mapping = [0, 13, 14, 15, 15, 1, 2, 3, 3, 4, 5, 6, 6, 10, 11, 12, 12, 7, 8, 9, 9]
    fingers_name = ['hand', 'thumb', 'index', 'middle', 'ring', 'pinky']
    joints_name = ['hand']
    for finger in fingers_name:
        if finger != 'hand':
            for id in range(4):
                joints_name.append(finger + '_' + str(id))

    convert_trans = np.asarray([[1.0, 0.0, 0.0],
                               [0.0, 0.0, -1.0],
                               [0.0, 1.0, 0.0]])
    # convert_trans = np.eye(3)

    joints_transf = axis_transf[joints_mapping]
    joints_transf[:, :3, :3] = convert_trans[np.newaxis, ...] @ (joints_transf[:, :3, :3])
    joints = np.array(joints[0])
    # vertices = np.array(vertices[0])
    transf = np.array(transf[0])
    n_vertex = vertices.shape[0]
    # hand_mesh = o3d.geometry.TriangleMesh()
    # hand_mesh.triangles = o3d.utility.Vector3iVector(faces)
    # hand_mesh.vertices = o3d.utility.Vector3dVector(
    #             np.array(vertices.detach().cpu().squeeze(0)))
    # hand_mesh.compute_vertex_normals()
    joints += root_loc
    # root_loc = np.asarray([-0.7295253 ,  0.05896984,  0.02769433]) # -0.712 0.0595 0.0359
    joints = convert_trans.dot(joints.transpose()).transpose()
    # joints *= np.asarray([1, 1, -1])
    # joints = joints[:, [0, 2, 1]]
    # joints = joints + root_loc
    print(joints)
    out_dict = dict()
    joints_transf[:, :3, -1] = joints

    if visual:
        vis.clear_geometries()
        hand_mesh = o3d.geometry.TriangleMesh()
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])  # 添加坐标系
        vis.add_geometry(mesh_frame)

        hand_mesh.triangles = o3d.utility.Vector3iVector(faces)
        vertices = vertices.squeeze(0).numpy()
        vertices = (convert_trans @ vertices.transpose()).transpose() + convert_trans @ root_loc
        hand_mesh.vertices = o3d.utility.Vector3dVector(vertices)

        hand_mesh.compute_vertex_normals()
        vis.add_geometry(hand_mesh)

        test_mesh = o3d.open3d.geometry.PointCloud()
        test_mesh.points = o3d.utility.Vector3dVector(joints)
        test_mesh.paint_uniform_color([1, 0, 1])
        vis.add_geometry(test_mesh)

        for trans in joints_transf:
            vis_coordinate(vis, trans)
        vis.reset_view_point(True)
        ctr = vis.get_view_control()
        ctr.rotate(0.0, 600)
        vis.run()
        # vis.destroy_window()
    for i in range(joints.__len__()):
        out_dict[joints_name[i]] = joints_transf[i]
    np.save(side + '_joint_axis.npy', out_dict)
    o3d.io.write_triangle_mesh(side + "_hand_mano.obj",
                                   hand_mesh,
                                   write_triangle_uvs=True)
np.save('ske_tsl_inf.npy', skel_tsl_infs)
print('done')