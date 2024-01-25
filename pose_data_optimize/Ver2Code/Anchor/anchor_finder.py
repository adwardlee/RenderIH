from HandPoseConverter import HandPoseConverter
import numpy as np
import open3d as o3d
from manopth.manolayer import ManoLayer
import torch
import time
shape_dim = 10
from tqdm import tqdm
class point_inf:
    def __init__(self, id):
        self.neighours = set()
        self.faces = []
        self.id = id

def data_loader(data_path, mode, scale):
    mocap_data = np.load(data_path, allow_pickle=True).item()
    left_rot = mocap_data['left']['rot']
    left_loc = mocap_data['left']['loc'] * scale / 100
    right_rot = mocap_data['right']['rot']
    right_loc = mocap_data['right']['loc'] * scale / 100

    if mode == 1:
        left_rot[:, 0, 1:3] = -left_rot[:, 0, 1:3]
        left_loc[:, 1] = -left_loc[:, 1]
        right_rot[:, 0, 1:3] = -right_rot[:, 0, 1:3]
        right_loc[:, 1] = -right_loc[:, 1]
    return right_rot, right_loc, left_rot, left_loc, mocap_data

def dfs(points_inf, id, mask, dis, search_dis, region_mask):
    if (not region_mask[id]) or dis > search_dis:
        return
    mask[id] = True
    for nv in points_inf[id].neighours:
        if(mask[nv]):
            continue
        dfs(points_inf, nv, mask, dis + 1, search_dis, region_mask)


def get_anchor(points_inf, point_count, region_v, num_anchor, mask, search_dis = 2):
    res = []
    region_mask = np.ones(778, dtype=bool)
    region_count = point_count[region_v]
    sorted_v = np.asarray(region_v)[region_count.argsort()[::-1]]
    region_mask[region_v] = True
    for v in sorted_v:
        if mask[v] == 0:
            dfs(points_inf, v, mask, 0, search_dis, region_mask)
            res.append(v)
        if(res.__len__() >= num_anchor):
            return res
    return res


def main():
    point_count = np.zeros(778)
    origin_data_path = '/home/tallery/CodeSpace/myCPF/dataset/InterHand26M_annotations/discrete_hand_interaction_poses.npy'
    optimized_data_path = origin_data_path
    right_rot, right_loc, left_rot, left_loc, raw_data = data_loader(optimized_data_path, 0, 1.0)
    mano_root = '../../assets/mano'
    lhpc = HandPoseConverter(side='left', root=mano_root)
    rhpc = HandPoseConverter(side='right', root=mano_root)

    if 'shape' in raw_data['right'].keys():
        shape = torch.from_numpy(raw_data['right']['shape'].astype(np.float32)).squeeze(1)
        sub_shape = torch.from_numpy(raw_data['left']['shape'].astype(np.float32)).squeeze(1)
        shape = torch.cat([shape, sub_shape], 1)
    else:
        shape = torch.tensor([0.5082395, -0.39488167, -1.7484332, 1.6630946, 0.34428665, -1.37387,
                              0.38293332, 1.196094, 0.6538949, -0.94331187]).unsqueeze(0).repeat((len(right_rot), 2))

    # shape = torch.tensor([-0.7443165183067322, 0.11856709420681, 0.11352552473545074, 0.026025250554084778, 0.1905139982700348, -0.06649057567119598, 0.02787777967751026, -0.0026709693484008312, -0.25168120861053467, -0.08985370397567749]).unsqueeze(0)
    # shape = torch.tensor([0.39090609550476074, -5.10849937051534653, -0.30046647787094116, -0.03732258453965187, -0.09943775832653046, -0.023937644436955452, 0.04136071354150772, 0.10307897627353668, 0.04249461740255356, 0.16355504095554352]).unsqueeze(0)
    left_mano_layer = ManoLayer(
        mano_root=mano_root,
        use_pca=False,
        ncomps=6,
        flat_hand_mean=True,
        center_idx=0,
        return_transf=False,
        side='left',
        root_rot_mode='quat',
        joint_rot_mode='quat'
    )
    right_mano_layer = ManoLayer(
        mano_root=mano_root,
        use_pca=False,
        ncomps=6,
        flat_hand_mean=True,
        center_idx=0,
        return_transf=False,
        side='right',
        root_rot_mode='quat',
        joint_rot_mode='quat'
    )
    faces = np.array(left_mano_layer.th_faces).astype(np.long)
    num_points = 778
    points_inf = []
    for i in range(num_points):
        points_inf.append(point_inf(i))
    for face in faces:
        for i in range(3):
            j = i - 1 if i > 0 else 2
            k = i + 1 if i < 2 else 0
            points_inf[face[i]].faces.append(face)
            points_inf[face[i]].neighours.add(face[j])
            points_inf[face[i]].neighours.add(face[k])

    l_hand_mesh = o3d.geometry.TriangleMesh()
    r_hand_mesh = o3d.geometry.TriangleMesh()
    l_hand_mesh.triangles = o3d.utility.Vector3iVector(faces)
    r_hand_mesh.triangles = o3d.utility.Vector3iVector(faces)

    left_quat = lhpc.euler_2_mano_quat(left_rot)
    right_quat = rhpc.euler_2_mano_quat(right_rot)
    left_quat = torch.from_numpy(left_quat)
    right_quat = torch.from_numpy(right_quat)
    left_loc = torch.from_numpy(left_loc)
    right_loc = torch.from_numpy(right_loc)
    num_frame = left_quat.__len__()

    use_prev_data = True
    if not use_prev_data:
        for i in tqdm(range(num_frame)):
            lq = left_quat[i:i + 1]
            rq = right_quat[i:i + 1]
            l_vertices, l_joints = left_mano_layer(lq, shape[i:i + 1, shape_dim:])
            r_vertices, r_joints = right_mano_layer(rq, shape[i:i + 1, :shape_dim])

            r_vertices += right_loc[i] - left_loc[i]  # + torch.tensor([0.5, 0.0, 0.0])

            l_hand_mesh.vertices = o3d.utility.Vector3dVector(
                np.array(l_vertices.detach().cpu().squeeze(0)))
            r_hand_mesh.vertices = o3d.utility.Vector3dVector(
                np.array(r_vertices.detach().cpu().squeeze(0)))
            l_hand_mesh.compute_vertex_normals()
            r_hand_mesh.compute_vertex_normals()

            r_v = np.asarray(r_hand_mesh.vertices)
            r_n = np.asarray(r_hand_mesh.vertex_normals)

            l_v = np.asarray(l_hand_mesh.vertices)
            l_n = np.asarray(l_hand_mesh.vertex_normals)

            r_v = r_v[np.newaxis, :].repeat(778, 0)
            l_v = l_v[:, np.newaxis]

            dis = np.linalg.norm(r_v - l_v, axis=-1)

            against = l_n.dot(r_n.transpose()) < -0.6

            pair = (dis < 0.02) * against

            paired_l, paired_r = np.where(pair > 0)
            paired_p = np.concatenate([paired_l, paired_r])
            ok_points = np.unique(paired_p)
            point_count[ok_points]+=1
        np.save('point_count.npy', point_count)
    else:
        point_count = np.load('point_count.npy')
    region_inf = np.load("/home/tallery/CodeSpace/23_02_02_cpf/code_sdf/part_vert.npy", allow_pickle=True).item()

    f = []
    w = []
    c = []
    num_anchor = 16 * 600
    mask = np.zeros(778, dtype=bool)
    # for region_id in region_inf.keys():
    #     region_v = list(region_inf[region_id])
    #
    #     region_count = point_count[region_v]
    #
    #     # soted_v_id = np.asarray(region_v)[region_count.argsort()[::-1]][0:6]
    #     res = get_anchor(points_inf, point_count, region_v, int(num_anchor * (region_v.__len__() / 778.)), mask, 1)
    #     # if res.__len__() == 0:
    #     print("{}_{}".format(region_id, res.__len__()))
    #     for v in res:
    #         pi = points_inf[v]
    #         f.append(pi.faces[0])
    #         cw = [0.0, 0.0]
    #         if (pi.faces[0][1] == v):
    #             cw = [1.0, 0.0]
    #         elif (pi.faces[0][2] == v):
    #             cw = [0.0, 1.0]
    #         w.append(cw)
    #         c.append(0)
    # region_v = list(region_inf[region_id])

    region_v = np.arange(0, 778, 1)
    region_count = point_count[region_v]
    res = get_anchor(points_inf, point_count, region_v, int(num_anchor * (region_v.__len__() / 778.)), mask, 2)
    for v in res:
        pi = points_inf[v]
        f.append(pi.faces[0])
        cw = [0.0, 0.0]
        if (pi.faces[0][1] == v):
            cw = [1.0, 0.0]
        elif (pi.faces[0][2] == v):
            cw = [0.0, 1.0]
        w.append(cw)
        c.append(0)

    file_handle = open('/home/tallery/CodeSpace/23_02_02_cpf/assets/anchor/face_vertex_idx.txt', mode='w')
    for face in f:
        file_handle.write('{} {} {}\n'.format(face[0], face[1], face[2]))
    file_handle.close()
    file_handle = open('/home/tallery/CodeSpace/23_02_02_cpf/assets/anchor/anchor_weight.txt', mode='w')
    for weight in w:
        file_handle.write('{} {}\n'.format(weight[0], weight[1]))
    file_handle.close()
    file_handle = open('/home/tallery/CodeSpace/23_02_02_cpf/assets/anchor/merged_vertex_assignment.txt', mode='w')
    for class_type in c:
        file_handle.write('{}\n'.format(class_type))
    file_handle.close()


if __name__ == '__main__':
    main()