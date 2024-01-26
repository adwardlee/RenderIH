import argparse
import torch
import numpy as np
import random
from tqdm import *
from chumpy import optimization_internal
from matplotlib import pyplot as plt
from hocontact.postprocess.geo_optimizer_both_batch import GeOptimizer, init_runtime_viz_hand, update_runtime_viz_hand
from hocontact.utils.anatomyutils import AnatomyMetric
from manopth.anchorutils import anchor_load, get_rev_anchor_mapping, masking_load_driver, anchor_load_driver, \
    recover_anchor
from manopth.quatutils import angle_axis_to_quaternion, quaternion_to_angle_axis, quaternion_inv, quaternion_mul
from manopth import manolayer
import trimesh
# from liegroups import SO3
import sys
import open3d as o3d
from scripts.Interpolation import *

plt.switch_backend("agg")
from scripts.HandPoseConverter import HandPoseConverter
from collections import Counter

shape_dim = 10
def set_all_seeds(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


viz_initialized = False
runtime_viz = None


def load_mocap_data(data_path):
    mocap_data = np.load(data_path, allow_pickle=True).item()

    left_rot = mocap_data['left']['rot']
    # left_rot[:, 0, 1:3] = -left_rot[:, 0, 1:3]
    left_loc = mocap_data['left']['loc'] / 100
    # left_loc[:, 1] = -left_loc[:, 1]

    right_rot = mocap_data['right']['rot']
    # right_rot[:, 0, 1:3] = -right_rot[:, 0, 1:3]
    right_loc = mocap_data['right']['loc'] / 100
    # right_loc[:, 1] = -right_loc[:, 1]
    return right_rot, right_loc, left_rot, left_loc, mocap_data


def check_contact_type(cls_tar, cls_array):
    for id, cls in enumerate(cls_array):
        if 2 in [cls_tar, cls]:
            cls_array[id] = 2
        elif cls_tar == 1 and cls == 1:
            cls_array[id] = 1
        else:
            cls_array[id] = 0
    return cls_array


def search_anchors(main_anchors, sub_anchors, main_normals, sub_normals, class_types=None, prev_anchor_id=None,
                   judge_hand_contact=False):
    # when class_types is not None, we only ensure 0-0 0-1 2-*  or   1-1 2-*
    # we set contact type (0-0 0-1) as 0, (1-1) as 1, 2-* as 2
    tip_anchor_list = [2, 3, 4, 9, 10, 11, 15, 16, 17, 22, 23, 24, 29, 30, 31]
    dim = 4
    against = sub_normals.dot(main_normals.transpose()) > -0.6
    anchor_id = np.zeros([sub_anchors.shape[0], dim], dtype=np.int64)
    vertex_contact = np.zeros(sub_anchors.shape[0], dtype=np.int64)
    anchor_elasti = np.zeros([sub_anchors.shape[0], dim], dtype=np.float32)
    anchor_padding_mask = np.zeros([sub_anchors.shape[0], dim], dtype=np.int64)
    idx = 0

    if prev_anchor_id is not None:
        for sub_anchor in sub_anchors:
            dis = np.linalg.norm(sub_anchor - main_anchors, axis=-1)
            dis_ = 1000 * np.ones(dis.shape, dtype=dis.dtype)
            prev_valid_anchor_id = prev_anchor_id[idx]
            dis_[prev_valid_anchor_id[prev_valid_anchor_id != -1]] = dis[
                prev_valid_anchor_id[prev_valid_anchor_id != -1]]
            dis = dis_
            # dis[against[idx, :]] = 1000.0
            contact_able = (dis < 0.02)
            # print(dis.min())
            k = contact_able * (0.5 * np.cos(np.pi * dis / 0.02) + 0.5)
            vertex_contact[idx] = contact_able.sum() > 0
            anchor_id[idx] = prev_anchor_id[idx]
            anchor_elasti[idx] = k[anchor_id[idx]]
            anchor_padding_mask[idx] = anchor_elasti[idx] > 0.0
            idx += 1
    else:
        if judge_hand_contact:
            ks = []
            contact_type = []
            for sub_anchor in sub_anchors:
                sub_class_type = class_types[idx]
                sub_contactable_type = -1
                dis = np.linalg.norm(sub_anchor - main_anchors, axis=-1)
                dis[against[idx, :]] = 1000.0
                contact_able = (dis < 0.015)
                # print(dis.min())
                k = contact_able * (0.5 * np.cos(np.pi * dis / 0.015) + 0.5)
                ks.append(k)
                sorted_id = np.argsort(dis)
                sorted_class_type = class_types[sorted_id]
                sorted_contact_type = check_contact_type(sub_class_type, sorted_class_type)
                sorted_contactable_type = sorted_contact_type[contact_able]
                if sorted_contactable_type.__len__() > 0:
                    sub_contactable_type = Counter(sorted_contactable_type).most_common(1)[0][0]
                contact_type.append(sub_contactable_type)
                idx += 1
            contact_type = np.asarray(contact_type)
            total_contact_type = 1
        else:
            for sub_anchor in sub_anchors:
                dis = np.linalg.norm(sub_anchor - main_anchors, axis=-1)
                dis[against[idx, :]] = 1000.0
                contact_able = (dis < 0.015)
                # print(dis.min())
                k = contact_able * (0.5 * np.cos(np.pi * dis / 0.015) + 0.5)
                sorted_id = np.argsort(dis)
                vertex_contact[idx] = contact_able.sum() > 0
                anchor_id[idx] = sorted_id[0:dim]
                anchor_elasti[idx] = k[anchor_id[idx]]
                anchor_padding_mask[idx] = anchor_elasti[idx] > 0.0
                idx += 1
    # anchor_elasti[(class_types.repeat(dim).reshape([-1, dim])==4) | (class_types[anchor_id] == 4)] *= 1.5
    anchor_elasti[(class_types.repeat(dim).reshape([-1, dim]) != 4) * (class_types[anchor_id] != 4)] *= 0.3
    return vertex_contact, anchor_id, anchor_elasti, anchor_padding_mask


def vis_anchor_corresponding(main_anchors, sub_anchors, vertex_contact, anchor_id,
                             face, main_vertices, sub_vertices, fvi, main_normals, sub_normals):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='anchor_corresponding')
    points = []
    correspond = []
    cor_id = 0
    for id in range(sub_anchors.__len__()):
        if vertex_contact[id] <= 0:
            continue
        sub_point = sub_anchors[id]
        main_point = main_anchors[anchor_id[id][0]]
        points.append(sub_point)
        points.append(main_point)
        correspond.append([cor_id * 2, cor_id * 2 + 1])

        translate = sub_anchors[id]
        sub_axis = (sub_vertices[fvi[id, 1]] - sub_vertices[fvi[id, 0]])
        sub_axis = sub_axis / np.linalg.norm(sub_axis)
        sub_sub_axis = np.cross(sub_normals[id], sub_axis)
        rot = np.asarray([sub_axis, sub_sub_axis, sub_normals[id]])
        rot = rot.transpose()
        b = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.00015,
            cone_radius=0.0002,
            cylinder_height=0.005,
            cone_height=0.003,
            resolution=20,
            cylinder_split=4,
            cone_split=1,
        )
        b.compute_vertex_normals()
        b.paint_uniform_color([0.1, 0.9, 0.0])
        b.rotate(rot, center=(0, 0, 0))
        b.translate(translate)
        vis.add_geometry(b)

        correspond_id = anchor_id[id][0]
        translate = main_anchors[correspond_id]
        sub_axis = (main_vertices[fvi[correspond_id, 1]] - main_vertices[fvi[correspond_id, 0]])
        sub_axis = sub_axis / np.linalg.norm(sub_axis)
        sub_sub_axis = np.cross(main_normals[correspond_id], sub_axis)
        rot = np.asarray([sub_axis, sub_sub_axis, main_normals[correspond_id]])
        rot = rot.transpose()
        b = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.00015,
            cone_radius=0.0002,
            cylinder_height=0.005,
            cone_height=0.003,
            resolution=20,
            cylinder_split=4,
            cone_split=1,
        )
        b.compute_vertex_normals()
        b.paint_uniform_color([0.0, 0.0, 0.99])
        b.rotate(rot, center=(0, 0, 0))
        b.translate(translate)
        vis.add_geometry(b)

        cor_id += 1
    points = np.asarray(points)
    correspond = np.asarray(correspond)
    color = [[1, 0, 0] for i in range(len(correspond))]

    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(points)
    points_pcd.paint_uniform_color([1.0, 0, 0])  # 点云颜色

    # 绘制线条
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(correspond)
    lines_pcd.colors = o3d.utility.Vector3dVector(color)  # 线条颜色
    lines_pcd.points = o3d.utility.Vector3dVector(points)
    vis.add_geometry(lines_pcd)
    vis.add_geometry(points_pcd)

    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.triangles = o3d.utility.Vector3iVector(face)
    hand_mesh.vertices = o3d.utility.Vector3dVector(sub_vertices)
    hand_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.8, 0.8, 0.9]] * len(sub_vertices)))
    hand_mesh.compute_vertex_normals()
    # vis.add_geometry({'name': 'box', 'geometry': hand_mesh, 'material': mat_box})
    vis.add_geometry(hand_mesh)
    hand_mesh_ = o3d.geometry.TriangleMesh()
    hand_mesh_.triangles = o3d.utility.Vector3iVector(face)
    hand_mesh_.vertices = o3d.utility.Vector3dVector(main_vertices)
    hand_mesh_.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.8, 0.9, 1]] * len(sub_vertices)))
    hand_mesh_.compute_vertex_normals()
    vis.add_geometry(hand_mesh_)
    # vis.update_geometry(points)
    vis.run()
    vis.destroy_window()


def update_scene(inf=None, optimized_hand_pose=None, optimized_hand_tsl=None, optimized_sub_hand_pose=None,
                 optimized_sub_hand_tsl=None, hand_shape=None, anchor_id=None, get_anchor_similarity=False,
                 main_hand=None, sub_hand=None):
    if inf is not None:
        main_hand_loc = optimized_hand_tsl.numpy()
        main_hand_pose = optimized_hand_pose.unsqueeze(0)
        hand_shape = torch.from_numpy(inf['hand_shape']).unsqueeze(0)
        if optimized_sub_hand_pose is not None:
            sub_hand_loc = optimized_sub_hand_tsl.numpy()
            hand_pose = optimized_sub_hand_pose.unsqueeze(0)
        else:
            sub_hand_loc = inf['obj_tsl']
            hand_pose = torch.from_numpy(inf['obj_pose']).unsqueeze(0)
    else:
        main_hand_loc = optimized_hand_tsl
        sub_hand_loc = optimized_sub_hand_tsl
        main_hand_pose = torch.from_numpy(optimized_hand_pose).unsqueeze(0)
        hand_pose = torch.from_numpy(optimized_sub_hand_pose).unsqueeze(0)
    fvi, aw, anchor_class_type, _ = anchor_load_driver('./assets')
    main_vertices, main_joints, main_transf = main_hand(main_hand_pose, hand_shape[:, :shape_dim])
    sub_vertices, sub_joints, sub_transf = sub_hand(hand_pose, hand_shape[:, shape_dim:])
    if main_hand_pose.shape.__len__() >= 3:
        main_hand_pose = main_hand_pose.squeeze(0)
    if hand_pose.shape.__len__() >= 3:
        hand_pose = hand_pose.squeeze(0)
    main_hand_pose = np.asarray(main_hand_pose)
    hand_pose = np.asarray(hand_pose)
    sub_vertices = np.asarray(sub_vertices[0])
    main_vertices = np.asarray(main_vertices[0])
    main_vertices += main_hand_loc
    sub_vertices += sub_hand_loc
    main_joints += main_hand_loc
    sub_joints += sub_hand_loc
    sub_anchors = recover_anchor(sub_vertices, fvi, aw)
    main_anchors = recover_anchor(main_vertices, fvi, aw)

    # compute obj/sub_hand normals
    obj_normals = np.cross((sub_vertices[fvi[:, 1]] - sub_vertices[fvi[:, 0]]),
                           (sub_vertices[fvi[:, 2]] - sub_vertices[fvi[:, 0]]))
    obj_normals = -obj_normals / np.linalg.norm(obj_normals, axis=-1)[:, np.newaxis]

    hand_normals = np.cross((main_vertices[fvi[:, 1]] - main_vertices[fvi[:, 0]]),
                            (main_vertices[fvi[:, 2]] - main_vertices[fvi[:, 0]]))
    hand_normals = hand_normals / np.linalg.norm(hand_normals, axis=-1)[:, np.newaxis]

    vertex_contact, anchor_id, anchor_elasti, anchor_padding_mask = \
        search_anchors(main_anchors, sub_anchors, hand_normals, obj_normals, anchor_class_type, anchor_id)
    # if not get_anchor_similarity:
    #     vis_anchor_corresponding(main_anchors, sub_anchors, vertex_contact, anchor_id,
    #                              sub_hand.th_faces.numpy().astype(np.long), main_vertices, sub_vertices, fvi, hand_normals, obj_normals)
    if get_anchor_similarity:
        num_anchor = main_anchors.shape[0]
        anchor_similarity = np.zeros([num_anchor, num_anchor], dtype=np.float32)
        for i in range(num_anchor):
            dis = np.linalg.norm(main_anchors[i] - main_anchors, axis=-1)
            dis[dis >= 0.04] = 0.04
            anchor_similarity[i, :] = -np.exp(dis * 17.3) + 2
            normals_similarity = (hand_normals[i] * hand_normals).sum(-1)
            normals_similarity[normals_similarity < 0] = 0.0
            normals_similarity = np.sin(normals_similarity * np.pi / 2)
            anchor_similarity[i, :] *= normals_similarity
        # anchor_similarity = np.eye(num_anchor, dtype=np.float32)
    out_inf = {'mode': 'both',
               'vertex_contact': vertex_contact,  # num_of_obj_vertex //
               'hand_region': np.zeros(sub_anchors.shape[0], dtype=np.int64),
               # num_of_obj_vertex // which region the vertex belongs to
               'anchor_id': anchor_id,  # num_of_obj_vertex * 4 //which anchor the vertext contact
               'anchor_elasti': anchor_elasti,  # num_of_obj_vertex * 4 // the energy
               'anchor_padding_mask': anchor_padding_mask,  # num_of_obj_vertex * 4 // equal to anchor_elastic != 0.0
               'hand_verts_3d': main_vertices,  # num_of_hand_vertex * 3(778 * 3)
               'hand_joints_3d': main_joints,  # 21 * 3
               'hand_pose': main_hand_pose,  # 16 * 4 (w,x,y,z)
               'hand_shape': hand_shape.squeeze(0).numpy(),
               'hand_tsl': main_hand_loc,  # 3
               'obj_verts_3d': sub_vertices,  # num_of_obj_vertex * 3
               'obj_tsl': sub_hand_loc,  # 3
               'obj_rot': np.asarray([0.0, 0.0, 0.0]),
               # both:
               'obj_joints_3d': sub_joints,
               'obj_pose': hand_pose,
               'hand_normals': hand_normals,
               'fvi': fvi,

               'hand_face': np.asarray(main_hand.th_faces).astype(np.long),
               'obj_face': np.asarray(sub_hand.th_faces).astype(np.long),
               'hand_anchors': main_anchors,
               'obj_anchors': sub_anchors,
               'obj_normals': obj_normals}
    if get_anchor_similarity:
        return out_inf, anchor_similarity
    return out_inf


def combine_frame_inf(main_inf, frame_inf):
    for key_name in frame_inf.keys():
        if key_name in ['vertex_contact', 'anchor_id', 'anchor_elasti', 'anchor_padding_mask',
                        'hand_verts_3d', 'hand_joints_3d', 'hand_pose',
                        'hand_tsl', 'obj_verts_3d', 'obj_tsl', 'obj_joints_3d', 'obj_pose',
                        'hand_normals', 'hand_anchors', 'obj_anchors', 'obj_normals', 'optimize_it', 'hand_shape']:
            if main_inf is None:
                frame_inf[key_name] = frame_inf[key_name][np.newaxis]
            else:
                if main_inf[key_name].shape == frame_inf[key_name].shape:
                    main_inf[key_name] = np.stack((main_inf[key_name], frame_inf[key_name]))
                elif main_inf[key_name].shape.__len__() == frame_inf[key_name].shape.__len__():
                    main_inf[key_name] = np.row_stack((main_inf[key_name], frame_inf[key_name]))
                else:
                    main_inf[key_name] = np.row_stack((main_inf[key_name], frame_inf[key_name][np.newaxis]))
    return main_inf if main_inf is not None else frame_inf


def main(args):
    optimized_range = [0, -1]
    discrete_optimize = True
    # load mocap data and process
    optimize_interval = args.optimize_interval
    fps = args.fps
    mocap_data_path = args.mocap_path
    mocap_scale = args.mocap_scale

    lhpc = HandPoseConverter(side='left', root=args.mano_root)
    rhpc = HandPoseConverter(side='right', root=args.mano_root)

    if args.reoptimize_raw != '':
        raw_data = np.load(args.reoptimize_raw, allow_pickle=True).item()
        op_right_quat = raw_data['right']['rot']
        op_right_loc = raw_data['right']['loc']
        op_left_quat = raw_data['left']['rot']
        op_left_loc = raw_data['left']['loc']

        if not discrete_optimize:
            op_right_quat, op_right_loc = filter_quat_loc(op_right_quat, op_right_loc, 8)
            op_left_quat, op_left_loc = filter_quat_loc(op_left_quat, op_left_loc, 8)

        op_right_euler = rhpc.mano_quat_2_euler(op_right_quat)
        op_left_euler = lhpc.mano_quat_2_euler(op_left_quat)

        right_dict = {'rot': op_right_euler,
                      'loc': op_right_loc * 100}
        left_dict = {'rot': op_left_euler,
                     'loc': op_left_loc * 100}
        out_put_dict = {'right': right_dict,
                        'left': left_dict}
        np.save(args.optimized_mocap_path, out_put_dict)
        return

    rr, rl, lr, ll, raw_data = load_mocap_data(mocap_data_path)
    num_total_frame = rr.__len__()
    chosen_frame_list = []
    for i in range(num_total_frame):
        if i % optimize_interval == 0:
            chosen_frame_list.append(i)
    rr = rr[chosen_frame_list]
    rl = rl[chosen_frame_list]
    lr = lr[chosen_frame_list]
    ll = ll[chosen_frame_list]
    num_sample_frame = rr.__len__()

    left_quat = lhpc.euler_2_mano_quat(lr)
    right_quat = rhpc.euler_2_mano_quat(rr)
    ll *= mocap_scale
    rl *= mocap_scale

    if discrete_optimize:
        hand_shape = torch.from_numpy(raw_data['right']['shape'].astype(np.float32)).squeeze(1)
        sub_hand_shape = torch.from_numpy(raw_data['left']['shape'].astype(np.float32)).squeeze(1)
        hand_shape = torch.cat([hand_shape, sub_hand_shape], 1)
    else:
        hand_shape = torch.tensor([0.5082395, -0.39488167, -1.7484332, 1.6630946, 0.34428665, -1.37387,
                               0.38293332, 1.196094, 0.6538949, -0.94331187]).unsqueeze(0).repeat((left_quat.__len__(), 1))
    raw_data['right']['shape'] = hand_shape.numpy()[:, :shape_dim]
    raw_data['left']['shape'] = hand_shape.numpy()[:, shape_dim:]

    rank = args.gpu
    target_device = f"cuda:{rank}"
    set_all_seeds(args.manual_seed)
    hoptim = GeOptimizer(
        target_device,
        lr=1e-2,
        n_iter=100,
        verbose=False,
        mano_root=args.mano_root,
        anchor_path=args.anchor_root,
        # values to initialize coef_val
        lambda_contact_loss=args.lambda_contact_loss,
        lambda_repulsion_loss=args.lambda_repulsion_loss,
        repulsion_query=args.repulsion_query,
        repulsion_threshold=args.repulsion_threshold,
        mode='both'
    )
    lambda_repulsion = hoptim.coef_val['lambda_repulsion_loss']
    lambda_contact = hoptim.coef_val['lambda_contact_loss']

    # run optimize
    hands_ineraction = False
    op_left_quat = []
    op_right_quat = []
    op_left_loc = []
    op_right_loc = []
    frame_inf = None
    anchor_id = None

    main_hand = manolayer.ManoLayer(
        mano_root="assets/mano",
        use_pca=False,
        ncomps=6,
        flat_hand_mean=True,
        center_idx=args.center_idx,
        return_transf=True,
        root_rot_mode='quat',
        joint_rot_mode='quat',
        side='right'
    )
    sub_hand = manolayer.ManoLayer(
        mano_root="assets/mano",
        use_pca=False,
        ncomps=6,
        flat_hand_mean=True,
        center_idx=args.center_idx,
        return_transf=True,
        root_rot_mode='quat',
        joint_rot_mode='quat',
        side='left'
    )

    right_quat_var = copy.deepcopy(right_quat)
    rl_var = copy.deepcopy(rl)
    ll_var = copy.deepcopy(ll)
    left_quat_var = copy.deepcopy(left_quat)
    # right_quat_var, rl_var = filter_quat_loc(right_quat_var, rl_var, 4)
    # left_quat_var, ll_var = filter_quat_loc(left_quat_var, ll_var, 4)

    optimized_range[1] = num_sample_frame if optimized_range[1] == -1 else optimized_range[1]
    for attempt_id in range(4):
        # set the optimize parameter
        if attempt_id == 0:
            hoptim.coef_val['lambda_repulsion_loss'] = lambda_repulsion * 1
            hoptim.coef_val['lambda_contact_loss'] = lambda_contact * 1
            hoptim.n_iter = 50
        elif attempt_id == 1:
            hoptim.coef_val['lambda_repulsion_loss'] = lambda_repulsion * 0.1
            hoptim.coef_val['lambda_contact_loss'] = lambda_contact * 15
            hoptim.n_iter = 40

        elif attempt_id == 2:
            hoptim.coef_val['lambda_repulsion_loss'] = lambda_repulsion * 30 #15
            hoptim.coef_val['lambda_contact_loss'] = lambda_contact * 0.1
            hoptim.n_iter = 75
            # anchor_id = None
        elif attempt_id == 3:
            hoptim.coef_val['lambda_repulsion_loss'] = lambda_repulsion * 1
            hoptim.coef_val['lambda_contact_loss'] = lambda_contact * 5
            hoptim.n_iter = 50

        # filter the anchor corresponding

        hand_anchor_similaritys = []
        if attempt_id == 0 or attempt_id == 3:
            anchor_ids = []
            for frame_id in tqdm(
                    range(optimized_range[0], optimized_range[1] if optimized_range[1] != -1 else num_sample_frame)):
                frame_inf, hand_anchor_similarity = update_scene(
                    optimized_hand_pose=right_quat_var[frame_id],
                    optimized_hand_tsl=rl_var[frame_id],
                    optimized_sub_hand_pose=left_quat_var[frame_id],
                    optimized_sub_hand_tsl=ll_var[frame_id],
                    hand_shape=hand_shape[frame_id:frame_id+1],
                    anchor_id=None,
                    get_anchor_similarity=True,
                    main_hand=main_hand,
                    sub_hand=sub_hand)
                anchor_ids.append(frame_inf['anchor_id'])
                hand_anchor_similaritys.append(hand_anchor_similarity)
            anchor_ids = np.asarray(anchor_ids)
            hand_anchor_similaritys = np.asarray(hand_anchor_similaritys)

            if not discrete_optimize:
                anchor_ids = filter_anchor_id(anchor_ids, hand_anchor_similaritys, window_length=4)
        if attempt_id == 0:
            # we have to find the changes of the joints
            right_joint_change = get_joint_change(right_quat_var)
            left_joint_change = get_joint_change(left_quat_var)
            if discrete_optimize:
                right_joint_change *= 0.0
                left_joint_change *= 0.0

        start_id = optimized_range[0]
        while start_id < (optimized_range[1] if optimized_range[1] != -1 else num_sample_frame):
            # get batch inf
            batch_frame_inf = None
            cur_batch_size = min(args.batch_size,
                                 (optimized_range[1] if optimized_range[1] != -1 else num_sample_frame) - start_id)
            print(cur_batch_size)
            for sub_id in range(cur_batch_size):
                frame_id = start_id + sub_id
                frame_inf = update_scene(
                    optimized_hand_pose=right_quat_var[frame_id],
                    optimized_hand_tsl=rl_var[frame_id],
                    optimized_sub_hand_pose=left_quat_var[frame_id],
                    optimized_sub_hand_tsl=ll_var[frame_id],
                    hand_shape=hand_shape[frame_id:frame_id+1],
                    anchor_id=anchor_ids[frame_id - optimized_range[0]],
                    get_anchor_similarity=False,
                    main_hand=main_hand,
                    sub_hand=sub_hand
                )
                vertex_contact = frame_inf['vertex_contact']
                if vertex_contact.sum() == 0:
                    hands_ineraction = False
                else:
                    hands_ineraction = True
                frame_inf['optimize_it'] = np.asarray([hands_ineraction])
                batch_frame_inf = combine_frame_inf(batch_frame_inf, frame_inf)
            batch_frame_inf['batch_size'] = cur_batch_size
            batch_frame_inf['hand_pose_consistent_mask'] = right_joint_change.numpy()[
                                                           start_id:start_id + cur_batch_size]
            batch_frame_inf['sub_hand_pose_consistent_mask'] = left_joint_change.numpy()[
                                                               start_id:start_id + cur_batch_size]
            # batch optimize
            # vis = False if args.batch_size > 1 else args.vis
            optimized_res = run_sample(
                target_device,
                hoptim,
                batch_frame_inf,
                args.hand_closed_path,
                args.vis
            )

            right_quat_var[start_id:start_id + cur_batch_size] = optimized_res['optimized_hand_pose']
            rl_var[start_id:start_id + cur_batch_size] = optimized_res['optimized_hand_tsl']
            left_quat_var[start_id:start_id + cur_batch_size] = optimized_res['optimized_sub_hand_pose']
            ll_var[start_id:start_id + cur_batch_size] = optimized_res['optimized_sub_hand_tsl']

            start_id = start_id + cur_batch_size

    op_right_quat = right_quat_var[optimized_range[0]:optimized_range[1]]
    op_right_loc = rl_var[optimized_range[0]: optimized_range[1]]
    op_left_quat = left_quat_var[optimized_range[0]:optimized_range[1]]
    op_left_loc = ll_var[optimized_range[0]:optimized_range[1]]

    right_dict = {'rot': op_right_quat,
                  'loc': op_right_loc}
    left_dict = {'rot': op_left_quat,
                 'loc': op_left_loc}
    out_put_dict = {'right': right_dict,
                    'left': left_dict}
    # np.save(args.optimized_mocap_path.replace('.', '_temp.'), out_put_dict)

    if not discrete_optimize:
        op_right_quat, op_right_loc = filter_quat_loc(op_right_quat, op_right_loc, 6)
        op_left_quat, op_left_loc = filter_quat_loc(op_left_quat, op_left_loc, 6)

    op_right_euler = rhpc.mano_quat_2_euler(op_right_quat)
    op_left_euler = lhpc.mano_quat_2_euler(op_left_quat)

    raw_data['left']['rot'] = op_left_euler
    raw_data['left']['loc'] = op_left_loc * 100
    raw_data['right']['rot'] = op_right_euler
    raw_data['right']['loc'] = op_right_loc * 100
    # right_dict = {'rot': op_right_euler,
    #               'loc': op_right_loc * 100}
    # left_dict = {'rot': op_left_euler,
    #              'loc': op_left_loc * 100}
    # out_put_dict = {'right': right_dict,
    #                 'left': left_dict}

    np.save(args.optimized_mocap_path, raw_data)


def get_joint_change(quat_var):
    # we have to find the changes of the joints
    num_frame = quat_var.shape[0]
    prev_index = torch.arange(num_frame) - 1
    prev_index = torch.clamp_min_(prev_index, 0)

    prev_prev_index = torch.arange(num_frame) - 3
    prev_prev_index = torch.clamp_min_(prev_prev_index, 0)

    quat_var_tensor = torch.from_numpy(quat_var)
    quat_var_inv = quaternion_inv(quat_var_tensor)
    joint_change = torch.abs(
        1 - torch.abs(quaternion_mul(quat_var_tensor, quat_var_inv[prev_index])[..., 0]))
    joint_change += torch.abs(
        1 - torch.abs(quaternion_mul(quat_var_tensor, quat_var_inv[prev_prev_index])[..., 0]))
    joint_change[:, 1:] = joint_change[:, 1:] / (2 * joint_change[:, 1:].mean(0))
    joint_change[:, 1:] = torch.clamp_max(joint_change[:, 1:], 1)
    joint_change = torch.sigmoid(joint_change * 12 - 6)
    joint_change = 1 - joint_change
    joint_change[:, 0] = 0.0

    out = copy.deepcopy(joint_change)
    half_window = 1
    for idx in range(num_frame):
        left_id = max(0, idx - half_window)
        right_id = min(num_frame - 1, idx + half_window)
        out[idx] = joint_change[left_id:right_id + 1].mean(0)
    return out


def filter_anchor_id(anchor_ids, hand_anchor_similaritys, window_length=12):
    num_frame = anchor_ids.shape[0]
    num_anchor = anchor_ids.shape[1]
    dim = anchor_ids.shape[2]
    half_window = window_length // 2
    new_anchor_ids = np.zeros(anchor_ids.shape, dtype=anchor_ids.dtype)
    for idx in range(num_frame):
        left_id = max(0, idx - half_window)
        right_id = min(num_frame - 1, idx + half_window)
        anchor = anchor_ids[left_id:right_id + 1]
        similarity = hand_anchor_similaritys[left_id:right_id + 1]
        sub_window_length = right_id - left_id + 1

        anchor_prob = np.zeros([num_anchor, num_anchor], dtype=np.float32)
        for frame_id in range(sub_window_length):
            anchor_prob += similarity[frame_id][anchor[frame_id], :].sum(axis=1)
        new_anchor_ids[idx] = anchor_prob.argsort()[:, ::-1][:, 0:dim]
    return new_anchor_ids


def filter_optimized_res(quats, locs):
    '''
    :param quats: (w,x,y,z) N * 16 * 4
    :param locs: N * 3
    :return:
    '''


def run_sample(
        device,
        hoptim,
        info,
        hand_closed_path="assets/closed_hand/hand_mesh_close.obj",
        vis=True,
):
    global viz_initialized, runtime_viz
    obj_verts_3d_np = np.asarray(info["obj_verts_3d"])
    # rot_matrix = SO3.exp(info["obj_rot"]).as_matrix()
    # obj_rot_np = np.asarray(info["obj_rot"])
    # obj_tsl_np = np.asarray(info["obj_tsl"])
    # obj_faces_np = np.asarray(info['obj_face'])
    obj_pose_np = np.asarray(info['obj_pose'])
    obj_pose = torch.from_numpy(obj_pose_np).float().to(device)

    hand_faces_np = np.asarray(info["hand_face"])

    hand_pose_np = np.asarray(info["hand_pose"])
    hand_pose = torch.from_numpy(hand_pose_np).float().to(device)  # 16 * 4
    hand_pose_axisang_np = quaternion_to_angle_axis(hand_pose).detach().cpu().numpy()

    # hand: verts & joints dumped
    hand_verts_np = np.asarray(info["hand_verts_3d"])
    hand_joints_np = np.asarray(info["hand_joints_3d"])

    # hand: close faces => "data/info/closed_hand/hand_mesh_close.obj"
    hand_closed_trimesh = trimesh.load(hand_closed_path, process=False)
    hand_close_faces_np = np.array(hand_closed_trimesh.faces)

    # no viz required
    runtime_viz = (None if vis else 9527) if runtime_viz is None else runtime_viz
    if runtime_viz != 9527:
        if not viz_initialized:
            runtime_viz = init_runtime_viz_hand(
                hand_verts_np[0],
                obj_verts_3d_np[0],
                hand_faces_np,
                hand_faces_np,
                contact_info=info,
            )
            viz_initialized = True
        else:
            update_runtime_viz_hand(
                runtime_viz,
                hand_verts_np[0],
                obj_verts_3d_np[0],
                hand_faces_np,
                hand_faces_np,
            )
    # ==================== optimize engine >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # prepare kwargs according to mode
    opt_val_kwargs = dict(
        # static
        vertex_contact=torch.from_numpy(info["vertex_contact"]).long().to(device),
        contact_region=torch.from_numpy(info["hand_region"]).long().to(device),
        anchor_id=torch.from_numpy(info["anchor_id"]).long().to(device),
        anchor_elasti=torch.from_numpy(info["anchor_elasti"]).float().to(device),
        anchor_padding_mask=torch.from_numpy(info["anchor_padding_mask"]).long().to(device),
        # hand
        hand_shape_init=torch.from_numpy(info["hand_shape"]).float().to(device),
        hand_tsl_init=torch.from_numpy(info["hand_tsl"]).float().to(device),
        hand_pose_gt=([0], hand_pose[:, 0:1, :]),
        hand_pose_init=(list(range(1, 16)), hand_pose[:, 1:, :]),
        # viz
        runtime_vis=runtime_viz,

        # both
        obj_tsl_init=torch.from_numpy(info["obj_tsl"]).float().to(device),
        obj_pose_gt=([0], obj_pose[:, 0:1, :]),
        obj_pose_init=(list(range(1, 16)), obj_pose[:, 1:, :]),

        # obj
        obj_anchors=torch.from_numpy(info["obj_anchors"]).float().to(device),
        obj_normals=torch.from_numpy(info["obj_normals"]).float().to(device),

        optimize_it=torch.from_numpy(info["optimize_it"]).bool().to(device),
        batch_size=info['batch_size'],
        consistent_mask=[torch.from_numpy(info['hand_pose_consistent_mask']).float().to(device),
                         torch.from_numpy(info['sub_hand_pose_consistent_mask']).float().to(device)]
    )

    hoptim.set_opt_val(**opt_val_kwargs)
    res_dict = hoptim.optimize(progress=False)
    return res_dict


if __name__ == "__main__":
    # ==================== argument parser >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser = argparse.ArgumentParser()
    # environment arguments
    parser.add_argument(
        "--gpu",
        type=str,
        default=0,
    )

    # exp arguments
    parser.add_argument("--exp_keyword", type=str, default=None)

    # Dataset params
    parser.add_argument("--data_root", type=str, default="data", help="hodata root")
    parser.add_argument("--center_idx", default=0, type=int)
    parser.add_argument("--mano_root", default="assets/mano")
    parser.add_argument("--anchor_root", default="assets/anchor")


    # Training parameters
    parser.add_argument("--manual_seed", type=int, default=0)

    # Dump parameters
    parser.add_argument("--vertex_contact_thresh", type=float, default=0.7)

    # GEO
    parser.add_argument("--palm_path", type=str, default="assets/hand_palm_full.txt")
    parser.add_argument("--hand_closed_path", type=str, default="assets/closed_hand/hand_mesh_close.obj")
    parser.add_argument("--lambda_contact_loss", type=float, default=10.0)
    parser.add_argument("--lambda_repulsion_loss", type=float, default=1.6 * 2)
    parser.add_argument("--repulsion_query", type=float, default=0.009) # 0.020
    parser.add_argument("--repulsion_threshold", type=float, default=0.050)

    # MOCAP
    parser.add_argument("--mocap_scale", type=float, default=1.0)
    parser.add_argument("--mocap_path", type=str, default='',
                        help="Path to the pose to be optimized")
    parser.add_argument("--optimized_mocap_path", type=str, default='',
                        help="Path to save the optimized pose"
                        )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--optimize_interval", type=int, default=1)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--reoptimize_raw", type=str, default='')
    parser.add_argument("--batch_size", type=int, default=2048 * 3)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # ==================== setup environment & run >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    args = parser.parse_args()
    args.mocap_path = 'test_hand_pose.npy'
    args.mocap_path = 'data/template_argued_data0.npy'
    args.optimized_mocap_path = 'data/template_optimized_data0.npy'
    main(args)
    # main(args)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
