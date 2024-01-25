import trimesh
import pyrender
import torch
from manopth.manolayer import ManoLayer
import numpy as np
from matplotlib import cm
from manopth.anchorutils import anchor_load_driver, recover_anchor
import open3d as o3d
from HandPoseConverter import HandPoseConverter
def get_colors(num_color, alpha=1.0, shuffle=False):
    cmap = cm.get_cmap("rainbow")
    x = np.linspace(0.0, 1.0, num_color)
    res = cmap(x)
    res[:, 3] = alpha
    if shuffle:
        np.random.shuffle(res)
    return res


def main(render=False, face_vertex_index=None, anchor_weight=None, anchor_class_type=None):
    # init mano
    mano_layer = ManoLayer(
        mano_root="../assets/mano",
        use_pca=False,
        ncomps=6,
        flat_hand_mean=False,
        center_idx=9,
        return_transf=True,
        side='right',
        root_rot_mode='quat',
        joint_rot_mode='quat'
    )
    hpc = HandPoseConverter(root='../assets/mano', side='right')
    faces = np.array(mano_layer.th_faces).astype(np.long)

    # gen random shape
    vec_shape = torch.tensor([ 0.5082395,  -0.39488167, -1.7484332 ,  1.6630946 ,  0.34428665, -1.37387,
               0.38293332,  1.196094 ,   0.6538949,  -0.94331187]).unsqueeze(0)
    # gen zero pose
    vec_pose = torch.zeros(1, 48)
    # vec_pose = torch.rand(1, 45 + 3)
    vec_pose = torch.from_numpy(np.asarray([[1.0, 0.0, 0.0, 0.0]] * 16, dtype=np.float32)).unsqueeze(0)
    euler_list = np.asarray([[0.0, 0.0, 0.0] for i in range(16)])
    # euler_list[1] = np.asarray([0.0, 0.0, 60])
    euler_list[1] = np.asarray([0.0, 30.0, 00])
    pose = hpc.euler_2_mano_quat(euler_list)
    pose[0][13:15] = vec_pose[0][13:15].numpy()
    # vec_pose[0][1] = torch.tensor([0.707, 0.0, 0.0, 0.707])
    # gen hand
    # vec_pose[0][0] = np.pi/2
    # vec_pose = vec_pose.reshape([1, -1])
    vertices, joints, transf = mano_layer(vec_pose, vec_shape)

    joints = np.array(joints[0])
    vertices = np.array(vertices[0])
    transf = np.array(transf[0])
    n_vertex = vertices.shape[0]

    transf = hpc.axis_convert(joints, transf)

    # ========= MAIN >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    anchors = recover_anchor(vertices, face_vertex_index, anchor_weight)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # set up viewer
    if render:
        scene = pyrender.Scene()
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
        node_cam = pyrender.Node(camera=cam, matrix=np.eye(4))
        scene.add_node(node_cam)
        scene.set_pose(node_cam, pose=np.eye(4))
        vertex_colors = np.array([0.5, 0.5, 0.6, 1])
        vertex_colors = np.expand_dims(vertex_colors, 0).repeat(n_vertex, axis=0)
        joint_colors = np.array([10, 73, 233, 255])
        transl = np.array([-0, -0, -0.0])
        transl = transl[np.newaxis, :]

        joints = joints * 1000.0 + transl
        vertices = vertices * 1000.0 + transl
        transf[:, :3, 3] = transf[:, :3, 3] * 1000.0 + transl

        tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        scene.add(mesh)

        predefined_color = np.asarray([np.random.rand(3) for i in range(anchors.shape[0])])
        predefined_color = np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

        for trans in transf:
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25, origin=[trans[0, 3], trans[1, 3],
                                                                                              trans[2, 3]])  # 添加坐标系
            mesh_frame.rotate(trans[0:3, 0:3])
            cor_mesh = trimesh.Trimesh(np.asarray(mesh_frame.vertices), np.asarray(mesh_frame.triangles), vertex_colors=np.asarray(mesh_frame.vertex_colors))
            mesh = pyrender.Mesh.from_trimesh(cor_mesh)
            # scene.add(mesh)


        anchor_colors = get_colors(anchors.shape[0], alpha=1.0)
        anchor_colors[:, 0:3] = predefined_color[act]
        anchor_colors[100, 0:3] = np.asarray([1.0, 1.0, 1.0])
        anchor_colors[[ 115,  83, 123,  63], 0:3] = np.asarray([1.0, 0.0, 0.0])
        # anchor_colors = np.asarray([[0.8, 0.0, 0.0, 0.9] for i in range(anchor_colors.__len__())])
        for k in range(len(anchors)):
            anchor_sphere = trimesh.creation.uv_sphere(radius=2)
            anchor_sphere.visual.vertex_colors = anchor_colors[k]
            tfs = np.tile(np.eye(4), (1, 1, 1))
            tfs[0, :3, 3] = anchors[k] * 1000 + transl
            anchor_mesh = pyrender.Mesh.from_trimesh(anchor_sphere, poses=tfs)
            scene.add(anchor_mesh)

        # pyrender.Viewer(scene, use_raymond_lighting=True, render_flags={"all_wireframe": True})
        plot_origin = False
        if plot_origin:
            euler_list = np.asarray([[0.0, 0.0, 0.0] for i in range(16)])
            pose = hpc.euler_2_mano_quat(euler_list)
            pose[0][13:15] = vec_pose[0][13:15].numpy()
            vertices, joints, transf = mano_layer(torch.from_numpy(pose), vec_shape)

            joints = np.array(joints[0])
            vertices = np.array(vertices[0])
            transf = np.array(transf[0])
            n_vertex = vertices.shape[0]
            joints = joints * 1000.0 + transl
            vertices = vertices * 1000.0 + transl
            vertex_colors = np.array([0.8, 0.8, 0.8, 0.5])
            vertex_colors = np.expand_dims(vertex_colors, 0).repeat(n_vertex, axis=0)
            tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
            mesh = pyrender.Mesh.from_trimesh(tri_mesh)

            scene.add(mesh)
        pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--inpath", type=str, default="../assets")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    fvi, aw, act, _ = anchor_load_driver(args.inpath)
    main(render=True, face_vertex_index=fvi, anchor_weight=aw, anchor_class_type=act)
