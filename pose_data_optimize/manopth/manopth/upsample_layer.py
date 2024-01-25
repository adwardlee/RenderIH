import torch
import numpy as np
from torch.nn import Module

from manopth.manolayer import ManoLayer
import open3d as o3d
from functools import lru_cache

# class UpSampleLayer(Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, vertices, faces):
# """
#    *
#   / \
#  /   \
# * --- *
#    |
#    *
#   /|\
#  / o \
# * --- *
# """
#         device = vertices.device
#         batch_size = vertices.shape[0]
#         expand_vertices = vertices.unsqueeze(1).expand(-1, faces.shape[1], -1, -1)
#         expand_faces = faces.unsqueeze(-1).expand(-1, -1, -1, 3)
#         new_verts = torch.mean(torch.gather(expand_vertices, 2, expand_faces), dim=-2)
#         new_idx_head = torch.cat([faces[..., [0, 1]], faces[..., [1, 2]], faces[..., [2, 0]]], dim=-2)
#         new_idx_tail = (
#             torch.arange(vertices.shape[-2], vertices.shape[-2] + faces.shape[-2])
#             .unsqueeze(0)
#             .expand(3, -1)
#             .reshape(3 * faces.shape[-2])
#             .unsqueeze(0)
#             .expand(batch_size, -1)
#             .to(device)
#         )

#         new_faces = torch.cat([new_idx_head, new_idx_tail[:, :, None]], dim=-1)
#         new_verts = torch.cat([vertices, new_verts], dim=-2)
#         return new_verts, new_faces


class UpSampleLayer(Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    @lru_cache(maxsize=256)
    def calculate_faces(faces, vn):
        edges = {}
        new_faces = []

        def get_edge_id(e):
            if e not in edges:
                edges[e] = len(edges)
            return edges[e]

        for f in faces:
            a, b, c = f[0], f[1], f[2]
            e1, e2, e3 = tuple(sorted([a, b])), tuple(sorted([b, c])), tuple(sorted([c, a]))
            x = get_edge_id(e1) + vn
            y = get_edge_id(e2) + vn
            z = get_edge_id(e3) + vn
            new_faces.append(np.array([x, y, z]))
            new_faces.append(np.array([a, x, z]))
            new_faces.append(np.array([b, y, x]))
            new_faces.append(np.array([c, z, y]))

        new_faces = np.vstack(new_faces)
        new_vertices_idx = np.vstack([np.array(list(k)) for k in edges.keys()])
        return new_vertices_idx, new_faces

    def forward(self, vertices, faces):
        """
            *
           / \
          /   \
         /     \
        * ----- *
            |
            *
           / \
          o - o
         / \ / \
        * --o-- *
        """
        device = vertices.device
        new_vertices_idx_list, new_faces_list = [], []
        for i, fs in enumerate(faces):
            tuple_faces = tuple([tuple(i) for i in fs.cpu().numpy()])
            new_vertices_idx, new_faces = self.calculate_faces(tuple_faces, len(vertices[i]))
            new_vertices_idx_list.append(np.expand_dims(new_vertices_idx, axis=0))
            new_faces_list.append(np.expand_dims(new_faces, axis=0))
        # TODO check the gradient
        new_vertices_idx_list = torch.from_numpy(np.vstack(new_vertices_idx_list)).long().to(device)
        new_faces_list = torch.from_numpy(np.vstack(new_faces_list)).long().to(device)

        expand_vertices = vertices.unsqueeze(1).expand(-1, new_vertices_idx_list.shape[1], -1, -1)
        expand_vertices_idx = new_vertices_idx_list.unsqueeze(-1).expand(-1, -1, -1, 3)
        new_verts = torch.mean(torch.gather(expand_vertices, 2, expand_vertices_idx), dim=-2)
        new_verts = torch.cat([vertices, new_verts], dim=1)
        return new_verts, new_faces_list


if __name__ == "__main__":
    ncomps = 6

    mano_layer = ManoLayer(mano_root="assets/mano/", use_pca=True, ncomps=ncomps, flat_hand_mean=False)
    bs = 5
    random_shape = torch.rand(bs, 10)
    random_pose = torch.rand(bs, ncomps + 3)

    # Forward pass through MANO layer
    hand_verts, _ = mano_layer(random_pose, random_shape)
    hand_faces = mano_layer.th_faces

    # print(hand_verts.shape, hand_faces.shape)

    # hand_mesh = o3d.geometry.TriangleMesh()
    # hand_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(hand_faces))
    # hand_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(hand_verts.squeeze(0)))
    # hand_mesh.compute_vertex_normals()

    vis_pred = o3d.visualization.Visualizer()
    vis_pred.create_window(window_name="Predicted Hand", width=1080, height=1080)
    # vis_pred.add_geometry(hand_mesh)

    up_sample_layer = UpSampleLayer()
    up_verts, up_faces = hand_verts, hand_faces.unsqueeze(0).expand(bs, -1, -1)
    for _ in range(4):
        print(up_verts.shape, up_faces.shape)
        up_verts, up_faces = up_sample_layer(up_verts, up_faces)

    print(up_verts.shape, up_faces.shape)

    hand_up = o3d.geometry.TriangleMesh()
    # np.random.shuffle(np.asarray(up_faces))
    hand_up.triangles = o3d.utility.Vector3iVector(np.asarray(up_faces[3]))
    hand_up.vertices = o3d.utility.Vector3dVector(np.asarray(up_verts[3]))
    hand_up.compute_vertex_normals()

    vis_up = o3d.visualization.Visualizer()
    vis_up.create_window(window_name="Up Hand", width=1080, height=1080)
    vis_up.add_geometry(hand_up)

    while True:
        # vis_pred.update_geometry(hand_mesh)
        vis_up.update_geometry(hand_up)
        vis_up.update_renderer()
        vis_pred.update_renderer()

        if not vis_pred.poll_events() or not vis_up.poll_events():
            break
