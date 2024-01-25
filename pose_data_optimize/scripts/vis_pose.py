from HandPoseConverter import HandPoseConverter
import numpy as np
import open3d as o3d
from manopth.manolayer import ManoLayer
import torch
import time
import os
shape_dim = 10
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


def main():
    # origin_data_path = '../data/10.npy'
    # optimized_data_path = '../data/batch_optimeized.npy'
    fps = 300
    pic_root = "/home/tallery/CodeSpace/23_02_02_cpf/pic"
    pic_premix = '3792_a'
    pic_dir = None
    if pic_premix is not None:
        pic_dir = os.path.join(pic_root, pic_premix)
        if os.path.exists(pic_dir):
            for file_name in os.listdir(pic_dir):
                os.remove(os.path.join(pic_dir,file_name))
            os.removedirs(pic_dir)
        os.mkdir(pic_dir)
    target_id = [1, 3, 5, 6, 8, 10, 13, 14, 15, 16, 18, 23, 24, 30, 31, 32, 34, 35, 37, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 64, 65, 68, 69, 70, 71, 72, 75, 76, 77, 78, 80, 81, 83, 84, 85, 86, 88, 91, 94, 95, 97, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 175, 176, 177, 180, 182, 190, 191, 194, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 247, 253, 255, 256, 260, 261, 262, 263, 264, 265, 266, 268, 269, 272, 273, 274, 275, 276, 278, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299]
    target_id = []
    origin_data_path = '/home/tallery/CodeSpace/myCPF/dataset/36W/data/36w_data.npy'
    origin_data_path = '/home/tallery/CodeSpace/myCPF/dataset/DiscreteDataset/ArguedData/discrete_hand_pose_0.npy'
    optimized_data_path = '/home/tallery/CodeSpace/myCPF/dataset/DiscreteDataset/temp_data/discrete_hand_pose_0_optimized.npy'
    # origin_data_path = '/home/tallery/CodeSpace/myCPF/dataset/36W/data/36w_data.npy'
    origin_data_path = '/home/tallery/CodeSpace/myCPF/dataset/36W/data/36w_data.npy'
    # origin_data_path = '/home/tallery/CodeSpace/myCPF/dataset/DiscreteDataset/temp_data/argue.npy'
    origin_data_path = '/home/tallery/CodeSpace/myCPF/dataset/DiscreteDataset/temp_data/sdf/full.npy'
    # origin_data_path = '/home/tallery/CodeSpace/myCPF/dataset/DiscreteDataset/temp_data/sdf/origin_full.npy'
    optimized_data_path = origin_data_path
    origin_data_path= None
    optimized_start_idx = 0 #15 #33
    right_rot, right_loc, left_rot, left_loc, raw_data = data_loader(optimized_data_path, 0, 1.0)
    # temp_rot = right_rot
    # right_rot = left_rot
    # left_rot = temp_rot
    #
    # # temp_loc = right_loc
    # #
    # # left_loc = left_loc - right_loc
    # # right_loc *= 0
    # right_loc[:, 1] = -right_loc[:, 1]
    # left_loc[:, 1] = -left_loc[:, 1]

    lhpc = HandPoseConverter(side='left')
    rhpc = HandPoseConverter(side='right')

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
        mano_root="../assets/mano",
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
        mano_root="../assets/mano",
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
    vis_cur = o3d.visualization.VisualizerWithKeyCallback()
    vis_cur.create_window(window_name="Runtime Hand", width=1080, height=1080)
    rop = vis_cur.get_render_option()
    rop.mesh_show_back_face = True
    left_quat = lhpc.euler_2_mano_quat(left_rot)
    right_quat = rhpc.euler_2_mano_quat(right_rot)
    left_quat = torch.from_numpy(left_quat)
    right_quat = torch.from_numpy(right_quat)
    left_loc = torch.from_numpy(left_loc)
    right_loc = torch.from_numpy(right_loc)
    num_frame = left_quat.__len__()

    l_hand_mesh = o3d.geometry.TriangleMesh()
    r_hand_mesh = o3d.geometry.TriangleMesh()
    vis_cur.add_geometry(l_hand_mesh)
    vis_cur.add_geometry(r_hand_mesh)
    l_hand_mesh.triangles = o3d.utility.Vector3iVector(faces)
    r_hand_mesh.triangles = o3d.utility.Vector3iVector(faces)

    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])  # 添加坐标系
    # vis_cur.add_geometry(mesh_frame)
    frame_read = 0
    if origin_data_path is not None:
        o_right_rot, o_right_loc, o_left_rot, o_left_loc, _ = data_loader(origin_data_path, 0, 1.0)
        o_left_quat = lhpc.euler_2_mano_quat(o_left_rot)
        o_right_quat = rhpc.euler_2_mano_quat(o_right_rot)
        o_left_quat = torch.from_numpy(o_left_quat)
        o_right_quat = torch.from_numpy(o_right_quat)
        o_left_loc = torch.from_numpy(o_left_loc)
        o_left_loc[:, 2] += 0.24
        o_right_loc = torch.from_numpy(o_right_loc)
        o_right_loc[:, 2] += 0.24
        o_l_hand_mesh = o3d.geometry.TriangleMesh()
        o_r_hand_mesh = o3d.geometry.TriangleMesh()
        vis_cur.add_geometry(o_l_hand_mesh)
        vis_cur.add_geometry(o_r_hand_mesh)
        o_l_hand_mesh.triangles = o3d.utility.Vector3iVector(faces)
        o_r_hand_mesh.triangles = o3d.utility.Vector3iVector(faces)
        num_frame = len(o_left_loc)
    for i in range(optimized_start_idx, num_frame):
        if target_id.__len__() > 0:

            frame_read = target_id[i]
            i = target_id[i]
    # while True:
    #     i = 1700
    #     frame_read = 45
        # i = 666
        print(i)
        zero_time = time.time()
        lq = left_quat[i:i + 1]
        rq = right_quat[i:i + 1]
        l_vertices, l_joints = left_mano_layer(lq, shape[i:i+1, shape_dim:])
        r_vertices, r_joints = right_mano_layer(rq, shape[i:i+1, :shape_dim])

        # #get normal mano param
        # # right_trans = rhpc.loc_2_trans(rq, shape[i:i+1, :shape_dim], right_loc[i], 0)
        # # left_trans = lhpc.loc_2_trans(lq, shape[i:i+1, shape_dim:], left_loc[i] , 0)
        # #
        # # right_aa = rhpc.mano_quat_flat_2_normal(rq, return_type='aa')
        # # left_aa = lhpc.mano_quat_flat_2_normal(lq, return_type='aa')
        # right_trans = rhpc.loc_2_trans(right_quat, shape[:, :shape_dim], right_loc, 0)
        # left_trans = lhpc.loc_2_trans(left_quat, shape[:, shape_dim:], left_loc, 0)
        #
        # right_aa = rhpc.mano_quat_flat_2_normal(right_quat, return_type='aa')
        # left_aa = lhpc.mano_quat_flat_2_normal(left_quat, return_type='aa')
        #
        # hand_data = dict(right_trans=right_trans, left_trans=left_trans,right_aa=right_aa, left_aa=left_aa, right_shape= shape[:, :shape_dim],
        #                  left_shape=shape[:, shape_dim:])
        # ######################

        # l_vertices += left_loc[frame_read]
        # l_vertices += torch.tensor([0.5, 0.0, 0.0])
        r_vertices += right_loc[i] - left_loc[i] #+ torch.tensor([0.5, 0.0, 0.0])
        l_hand_mesh.vertices = o3d.utility.Vector3dVector(
            np.array(l_vertices.detach().cpu().squeeze(0)))
        r_hand_mesh.vertices = o3d.utility.Vector3dVector(
            np.array(r_vertices.detach().cpu().squeeze(0)))

        l_hand_mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.array([[0.8, 0.7, 0.9]] * l_vertices.shape[1]))
        r_hand_mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.array([[0.8, 0.8, 0.9]] * l_vertices.shape[1]))

        l_hand_mesh.compute_vertex_normals()
        r_hand_mesh.compute_vertex_normals()

        vis_cur.update_geometry(l_hand_mesh)
        vis_cur.update_geometry(r_hand_mesh)

        if origin_data_path is not None:
            lq = o_left_quat[i:i + 1]
            rq = o_right_quat[i:i + 1]
            l_vertices, l_joints = left_mano_layer(lq, shape[i:i+1, shape_dim:])
            r_vertices, r_joints = right_mano_layer(rq, shape[i:i+1, :shape_dim])
            print(shape[i:i+1, shape_dim:])
            # l_vertices += o_left_loc[i]
            # r_vertices += o_right_loc[i]
            l_vertices += torch.tensor([0.0, 0.0, 0.0])
            r_vertices += o_right_loc[i] - o_left_loc[i] + torch.tensor([0.0, 0.0, 0.0])

            o_l_hand_mesh.vertices = o3d.utility.Vector3dVector(
                np.array(l_vertices.detach().cpu().squeeze(0)))
            o_r_hand_mesh.vertices = o3d.utility.Vector3dVector(
                np.array(r_vertices.detach().cpu().squeeze(0)))

            o_l_hand_mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.array([[1.0, 1.0, 0.0]] * len(l_vertices)))
            o_r_hand_mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.array([[1.0, 1.0, 0.0]] * len(r_vertices)))

            o_l_hand_mesh.compute_vertex_normals()
            o_r_hand_mesh.compute_vertex_normals()

            vis_cur.update_geometry(o_l_hand_mesh)
            vis_cur.update_geometry(o_r_hand_mesh)

        vis_cur.update_renderer()

        if True or frame_read <= 0:
            vis_cur.reset_view_point(True)
            ctr = vis_cur.get_view_control()
            # ctr.set_lookat(np.asarray([0.0, -1.0, 0.0], dtype=np.float64)[:, np.newaxis])
            # ctr.set_up(np.asarray([0.0, -0.0, 1.0], dtype=np.float64)[:, np.newaxis])
            ctr.rotate(0.0, -200 + 1400 )
            ctr.rotate(-0, 0.0)
            ctr.translate(-50, 0)
            ctr.set_zoom(1.5)
            ctr.set_zoom(0.55)
            ctr.change_field_of_view(-300)
        # if frame_read == 0:


        print(ctr.get_field_of_view())
        vis_cur.poll_events()
        cur_time = time.time()
        # o3d.io.write_triangle_mesh("/home/tallery/CodeSpace/myCPF/dataset/DiscreteDataset/left.obj", l_hand_mesh)
        # o3d.io.write_triangle_mesh("/home/tallery/CodeSpace/myCPF/dataset/DiscreteDataset/right.obj", r_hand_mesh)
        if pic_dir is not  None:
            vis_cur.capture_screen_image(os.path.join(pic_dir, "{}.png".format(i)))
        while (cur_time - zero_time) < 1 / fps:
            cur_time = time.time()
            vis_cur.poll_events()
            # time.sleep(1 / 1.0 - cur_time + zero_time)
        print(i)
        frame_read += 1


if __name__ == '__main__':
    main()
