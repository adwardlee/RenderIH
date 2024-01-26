from HandPoseConverter import HandPoseConverter
import numpy as np
import open3d as o3d
import torch
import time

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
    return right_rot, right_loc, left_rot, left_loc


def main():
    # origin_data_path = '../data/10.npy'
    # optimized_data_path = '../data/batch_optimeized.npy'
    origin_data_path = '/home/tallery/CodeSpace/myCPF/data/10.npy'
    optimized_start_idx = 0

    vis_cur = o3d.visualization.VisualizerWithKeyCallback()
    vis_cur.create_window(window_name="Runtime Hand", width=1080, height=1080)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])  # 添加坐标系

    r_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0, 0, 0])  # 添加坐标系
    l_mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0, 0, 0])  # 添加坐标系
    vis_cur.add_geometry(mesh_frame)
    vis_cur.add_geometry(r_mesh_frame)
    vis_cur.add_geometry(l_mesh_frame)
    frame_read = 0

    o_right_rot, o_right_loc, o_left_rot, o_left_loc = data_loader(origin_data_path, 0, 1.0)
    o_right_rot = o_right_rot * np.pi/180
    o_left_rot = o_left_rot * np.pi / 180
    num_frame = o_left_loc.__len__()
    # o_right_loc = o_right_loc - o_left_loc
    for i in range(optimized_start_idx, num_frame):
        zero_time = time.time()
        # i = 666
        # if i >= 0:
        #     o_right_loc[i] = o_right_loc[i] - o_right_loc[i-1]
        #     o_left_loc[i] = o_left_loc[i] - o_left_loc[i - 1]
        r_mesh_frame.translate(o_right_loc[i]/10, False)
        l_mesh_frame.translate(o_left_loc[i]/10, False)

        mat_r = np.asarray(r_mesh_frame.get_rotation_matrix_from_xyz(o_right_rot[i][0]))
        mat_l = np.asarray(l_mesh_frame.get_rotation_matrix_from_xyz(o_left_rot[i][0]))
        if i > 0:
            mat_rr = np.asarray(r_mesh_frame.get_rotation_matrix_from_xyz(o_right_rot[i-1][0]))
            mat_ll = np.asarray(l_mesh_frame.get_rotation_matrix_from_xyz(o_left_rot[i-1][0]))
            mat_r = mat_rr.transpose().dot(mat_r)
            mat_l = mat_ll.transpose().dot(mat_l)
        r_mesh_frame.rotate(mat_r)
        l_mesh_frame.rotate(mat_l)
        print(r_mesh_frame.get_rotation_matrix_from_xyz(o_right_rot[i][0]))
        vis_cur.update_geometry(l_mesh_frame)
        vis_cur.update_geometry(r_mesh_frame)
        vis_cur.update_renderer()
        if frame_read <= 0:
            vis_cur.reset_view_point(True)
            ctr = vis_cur.get_view_control()
            print('')
            # ctr.set_lookat(np.asarray([0.0, -1.0, 0.0], dtype=np.float64)[:, np.newaxis])
            # ctr.set_up(np.asarray([0.0, -0.0, 1.0], dtype=np.float64)[:, np.newaxis])
            # ctr.rotate(0.0, -500)
            # ctr.translate(-50, 50)
            # ctr.set_zoom(0.7)
        vis_cur.poll_events()
        cur_time = time.time()
        if (cur_time - zero_time) < 1 / 20.0:
            time.sleep(1 / 20.0 - cur_time + zero_time)
        print(i)
        frame_read += 1


if __name__ == '__main__':
    main()
