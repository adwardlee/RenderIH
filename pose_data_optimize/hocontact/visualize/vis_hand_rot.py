import trimesh
import pyrender
import torch
from manopth.manolayer import ManoLayer
from manopth.axislayer import AxisLayer
from manopth import demo
import argparse
import open3d as o3d
from manopth.anchorutils import recover_anchor, anchor_load_driver, get_rev_anchor_mapping
from manopth.anchorutils import anchor_load

import numpy as np


def create_hand_vertex_color(hand_verts, anchor_path):
    face_vert_idx, anchor_weight, merged_vertex_assignment, anchor_mapping = anchor_load(anchor_path)
    n_verts = hand_verts.shape[0]
    vertex_color = np.zeros((n_verts, 3))
    contact_region = merged_vertex_assignment
    vertex_color[contact_region == 0] = np.array([207, 56, 112]) / 255.0
    vertex_color[contact_region == 1] = np.array([226, 53, 74]) / 255.0
    vertex_color[contact_region == 2] = np.array([231, 91, 84]) / 255.0

    vertex_color[contact_region == 3] = np.array([236, 132, 87]) / 255.0
    vertex_color[contact_region == 4] = np.array([236, 105, 95]) / 255.0
    vertex_color[contact_region == 5] = np.array([202, 67, 99]) / 255.0

    vertex_color[contact_region == 6] = np.array([240, 162, 62]) / 255.0
    vertex_color[contact_region == 7] = np.array([244, 192, 99]) / 255.0
    vertex_color[contact_region == 8] = np.array([239, 179, 145]) / 255.0

    vertex_color[contact_region == 9] = np.array([224, 231, 243]) / 255.0
    vertex_color[contact_region == 10] = np.array([175, 186, 242]) / 255.0
    vertex_color[contact_region == 11] = np.array([195, 212, 240]) / 255.0

    vertex_color[contact_region == 12] = np.array([50, 115, 173]) / 255.0
    vertex_color[contact_region == 13] = np.array([82, 148, 200]) / 255.0
    vertex_color[contact_region == 14] = np.array([124, 191, 239]) / 255.0

    vertex_color[contact_region == 15] = np.array([144, 78, 150]) / 255.0
    vertex_color[contact_region == 16] = np.array([40, 76, 121]) / 255.0

    vertex_color[contact_region == 17] = np.array([255, 232, 246]) / 255.0

    return vertex_color


def caculate_align_mat(vec):
    vec = vec / np.linalg.norm(vec)
    z_unit_Arr = np.array([0, 0, 1])

    z_mat = np.array(
        [[0, -z_unit_Arr[2], z_unit_Arr[1]], [z_unit_Arr[2], 0, -z_unit_Arr[0]], [-z_unit_Arr[1], z_unit_Arr[0], 0],]
    )

    z_c_vec = np.matmul(z_mat, vec)
    z_c_vec_mat = np.array([[0, -z_c_vec[2], z_c_vec[1]], [z_c_vec[2], 0, -z_c_vec[0]], [-z_c_vec[1], z_c_vec[0], 0],])

    if np.dot(z_unit_Arr, vec) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, vec) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat, z_c_vec_mat) / (1 + np.dot(z_unit_Arr, vec))

    return qTrans_Mat


def draw_axis(axis, transf, scene, color):
    end_points = np.concatenate([axis[np.newaxis] * 20.0, np.zeros((1, 3))], axis=0)
    end_points = np.concatenate([end_points, np.ones((2, 1))], axis=1)
    end_points = (transf @ end_points.T).T[:, :3]

    rot_matrix = np.concatenate([caculate_align_mat(axis), np.zeros((3, 1))], axis=1)
    rot_matrix = np.concatenate([rot_matrix, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

    cylinder = trimesh.creation.cylinder(radius=0.9, segment=end_points)
    cylinder.visual.vertex_colors = color
    cylinder = pyrender.Mesh.from_trimesh(cylinder, smooth=False)
    cone = trimesh.creation.cone(
        radius=1.8, height=2.5, transform=trimesh.transformations.translation_matrix(end_points[0]) @ rot_matrix
    )
    cone.visual.vertex_colors = color
    cone = pyrender.Mesh.from_trimesh(cone, smooth=False)
    scene.add(cylinder)
    scene.add(cone)


def main(args):
    batch_size = args.batch_size

    mano_layer = ManoLayer(
        mano_root="assets/mano", use_pca=False, flat_hand_mean=True, center_idx=9, return_transf=True,
    )
    axis_layer = AxisLayer()
    faces = np.array(mano_layer.th_faces).astype(np.long)
    face_vertex_index, anchor_weight, merged_vertex_assignment, anchor_mapping = anchor_load_driver("assets")

    random_shape = torch.ones(batch_size, 10) * 0.1

    random_pose = torch.zeros(batch_size, 48)

    # Forward pass through MANO layer
    vertices, joints, transf = mano_layer(random_pose, random_shape)
    b_axis, u_axis, l_axis = axis_layer(joints, transf)
    b_axis, u_axis, l_axis = b_axis.squeeze(0).numpy(), u_axis.squeeze(0).numpy(), l_axis.squeeze(0).numpy()

    if args.render == "plt":
        demo.display_hand(
            {"verts": vertices, "joints": joints}, mano_faces=mano_layer.th_faces,
        )
    elif args.render == "pyrender":
        # =========================== Viewer Options >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        scene = pyrender.Scene()
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
        node_cam = pyrender.Node(camera=cam, matrix=np.eye(4))
        scene.add_node(node_cam)
        # same pose in our paper
        pose = np.array(
            [
                [3.58605842e-01, -9.28514808e-01, 9.62398091e-02, 2.31115278e01],
                [4.39218554e-01, 2.58801518e-01, 8.60295784e-01, 1.53212718e02],
                [-8.23704383e-01, -2.66236785e-01, 5.00628668e-01, -7.77342471e01],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )

        scene.set_pose(node_cam, pose=pose)
        vertex_colors = np.array([150, 150, 150, 200])
        joint_colors = np.array([164, 0, 102, 255])
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        transl = np.array([0, 0, -200.0])
        transl = transl[np.newaxis, :]
        joints = np.array(joints[0])
        vertices = np.array(vertices[0])
        transf = np.array(transf[0])

        joints = joints * 1000.0 + transl
        vertices = vertices * 1000.0 + transl
        transf[:, :3, 3] = transf[:, :3, 3] * 1000.0 + transl

        # Add other
        for k in range(1, 6):
            hand_pose = random_pose * (0.618 ** k)
            v, _, _ = mano_layer(hand_pose, random_shape)
            v = np.array(v[0])
            v = v * 1000.0 + transl
            other_colors = np.array([150, 150, 150, 210 * (0.618 ** k)])
            tri_mesh = trimesh.Trimesh(v, faces, vertex_colors=other_colors)
            mesh = pyrender.Mesh.from_trimesh(tri_mesh)
            scene.add(mesh)

        tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        scene.add(mesh)

        # Add Joints
        for j in range(21):
            sm = trimesh.creation.uv_sphere(radius=2.8)
            sm.visual.vertex_colors = joint_colors
            tfs = np.tile(np.eye(4), (1, 1, 1))
            tfs[0, :3, 3] = joints[j]
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

        for i in range(15):
            draw_axis(b_axis[i], transf[i + 1], scene, color=np.array([255, 42, 34, 255]))
            draw_axis(u_axis[i], transf[i + 1], scene, color=np.array([190, 255, 0, 255]))
            draw_axis(l_axis[i], transf[i + 1], scene, color=np.array([23, 217, 255, 255]))

        pyrender.Viewer(scene, use_raymond_lighting=True, viewport_size=(1280, 960))
    elif args.render == "open3d":
        hand_mesh = o3d.geometry.TriangleMesh()
        vertices = np.array(vertices[0])
        hand_mesh.triangles = o3d.utility.Vector3iVector(faces)
        hand_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        hand_mesh.vertex_colors = o3d.utility.Vector3dVector(create_hand_vertex_color(vertices, "assets/anchor"))

        hand_mesh.compute_vertex_normals()
        vis_gt = o3d.visualization.Visualizer()
        vis_gt.create_window(window_name="Ground-Truth Hand", width=2080, height=2080)
        vis_gt.add_geometry(hand_mesh)

        anchor_pos = recover_anchor(vertices, face_vertex_index, anchor_weight)
        print(anchor_pos.shape)

        for anchors in anchor_pos:
            b = o3d.geometry.TriangleMesh.create_box(width=0.003, height=0.003, depth=0.003)
            b.translate(anchors)
            b.translate(np.array([-0.0015, -0.0015, -0.0015]))
            b.paint_uniform_color([221 / 255, 175 / 255, 39 / 255])
            b.compute_vertex_normals()
            vis_gt.add_geometry(b)

        while True:
            vis_gt.update_geometry(hand_mesh)
            vis_gt.update_renderer()

            if not vis_gt.poll_events():
                break
    else:
        raise ValueError(f"Unknown renderer: {args.render}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument(
        "--flat_hand_mean", action="store_true", help="Use flat hand as mean instead of average hand pose"
    )
    parser.add_argument("--use_pca", action="store_true", help="Use PCA")
    parser.add_argument("--render", choices=["plt", "pyrender", "open3d"], default="pyrender", type=str)

    main(parser.parse_args())
