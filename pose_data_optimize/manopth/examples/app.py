import trimesh
import pyrender
import torch
from manopth.manolayer import ManoLayer
from manopth import demo
import argparse

import numpy as np


def main(args):
    batch_size = args.batch_size
    if args.use_pca:
        ncomps = 12
    else:
        ncomps = 45

    # Initialize MANO layer
    mano_layer = ManoLayer(
        mano_root="mano/models",
        use_pca=args.use_pca,
        ncomps=ncomps,
        flat_hand_mean=args.flat_hand_mean,
        center_idx=9,
        return_transf=True,
    )
    faces = np.array(mano_layer.th_faces).astype(np.long)

    # Generate random shape parameters
    random_shape = torch.rand(batch_size, 10)
    # Generate random pose parameters, including 3 values for global axis-angle rotation
    if args.use_pca:
        random_pose = torch.rand(batch_size, ncomps + 3)
    else:
        random_pose = torch.zeros(batch_size, ncomps + 3)

    # Forward pass through MANO layer
    vertices, joints, transf = mano_layer(random_pose, random_shape)

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
        scene.set_pose(node_cam, pose=np.eye(4))
        vertex_colors = np.array([200, 200, 200, 150])
        joint_colors = np.array([10, 73, 233, 255])
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        transl = np.array([0, 0, -200.0])
        transl = transl[np.newaxis, :]
        joints = np.array(joints[0])
        vertices = np.array(vertices[0])
        transf = np.array(transf[0])

        joints = joints * 1000.0 + transl
        vertices = vertices * 1000.0 + transl
        transf[:, :3, 3] = transf[:, :3, 3] * 1000.0 + transl

        tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
        mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        scene.add(mesh)

        # Add Joints
        for j in range(21):
            sm = trimesh.creation.uv_sphere(radius=2)
            sm.visual.vertex_colors = joint_colors
            tfs = np.tile(np.eye(4), (1, 1, 1))
            tfs[0, :3, 3] = joints[j]
            joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
            scene.add(joints_pcl)

        # Add Transformation
        for tf in range(16):
            axis = trimesh.creation.axis(transform=transf[tf], origin_size=3, axis_length=15)
            axis = pyrender.Mesh.from_trimesh(axis, smooth=False)
            scene.add(axis)

        pyrender.Viewer(scene, use_raymond_lighting=True)
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
    parser.add_argument("--render", choices=["plt", "pyrender"], default="pyrender", type=str)

    main(parser.parse_args())
