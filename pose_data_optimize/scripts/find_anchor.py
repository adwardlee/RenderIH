import trimesh
import pyrender
import torch
from manopth.manolayer import ManoLayer
from manopth.axislayer import AxisLayer
import numpy as np
from matplotlib import cm
from manopth.anchorutils import anchor_load_driver, recover_anchor
import pyvista as pv


def ray_triangle_intersection(ray_start, ray_vec, triangle):
    """Moeller–Trumbore intersection algorithm.

    Parameters
    ----------
    ray_start[n, 3] : np.ndarray
        Length three numpy array representing start of point.

    ray_vec : np.ndarray
        Direction of the ray.

    triangle : np.ndarray
        ``3 x 3`` numpy array containing the three vertices of a
        triangle.

    Returns
    -------
    bool
        ``True`` when there is an intersection.

    tuple
        Length three tuple containing the distance ``t``, and the
        intersection in unit triangle ``u``, ``v`` coordinates.  When
        there is no intersection, these values will be:
        ``[np.nan, np.nan, np.nan]``

    """
    # define a null intersection
    null_inter = np.array([np.nan, np.nan, np.nan])

    # break down triangle into the individual points
    v1, v2, v3 = triangle
    eps = 0.000001

    # compute edges
    edge1 = v2 - v1
    edge2 = v3 - v1
    pvec = np.cross(ray_vec, edge2)
    det = edge1.dot(pvec)

    if abs(det) < eps:  # no intersection
        return False, null_inter
    inv_det = 1.0 / det
    tvec = ray_start - v1
    u = tvec.dot(pvec) * inv_det

    if u < 0.0 or u > 1.0:  # if not intersection
        return False, null_inter

    qvec = np.cross(tvec, edge1)
    v = ray_vec.dot(qvec) * inv_det
    if v < 0.0 or u + v > 1.0:  # if not intersection
        return False, null_inter

    t = edge2.dot(qvec) * inv_det
    if t < eps:
        return False, null_inter

    return True, np.array([t, u, v])


def find_control_points(tip_idx, joints, axiss):
    control_points = []
    control_points_axis = []
    for finger_id in range(tip_idx.__len__()):
        for idx in range(tip_idx[finger_id].__len__() - 1):
            joint_id = tip_idx[finger_id][idx]
            next_joint_id = tip_idx[finger_id][idx + 1]

            twist_axis = axiss[0][joint_id]
            splay_axis = axiss[1][joint_id]
            bend_axis = axiss[2][joint_id]

            next_twist_axis = axiss[0][next_joint_id]
            next_splay_axis = axiss[1][next_joint_id]
            next_bend_axis = axiss[2][next_joint_id]
            if idx == 0 and finger_id > 0:
                new_twist_axis = twist_axis
                new_bend_axis = bend_axis / 5 + next_bend_axis * 4 / 5
                new_bend_axis = new_bend_axis / np.linalg.norm(new_bend_axis)
                new_splay_axis = np.cross(new_bend_axis, new_twist_axis)
                new_splay_axis = new_splay_axis / np.linalg.norm(new_splay_axis)

                axis = np.asarray([new_twist_axis, new_splay_axis, new_bend_axis])
                control_points_axis.append(axis.transpose())

                new_bend_axis = bend_axis * 2 / 3 + next_bend_axis/ 3
                new_bend_axis = new_bend_axis / np.linalg.norm(new_bend_axis)
                new_splay_axis = np.cross(new_bend_axis, new_twist_axis)
                new_splay_axis = new_splay_axis / np.linalg.norm(new_splay_axis)

                axis = np.asarray([new_twist_axis, new_splay_axis, new_bend_axis])
                control_points_axis.append(axis.transpose())

                control_points.append((joints[next_joint_id] - joints[joint_id]) * 4 / 5 + joints[joint_id])
                control_points.append((joints[next_joint_id] - joints[joint_id]) / 3 + joints[joint_id])
            else:
                new_twist_axis = twist_axis
                new_bend_axis = bend_axis + next_bend_axis
                new_bend_axis = new_bend_axis / np.linalg.norm(new_bend_axis)
                new_splay_axis = np.cross(new_bend_axis, new_twist_axis)
                new_splay_axis = new_splay_axis / np.linalg.norm(new_splay_axis)

                axis = np.asarray([new_twist_axis, new_splay_axis, new_bend_axis])
                control_points_axis.append(axis.transpose())

                control_points.append((joints[next_joint_id] + joints[joint_id]) / 2)
    padding_control_points(control_points, control_points_axis, joints, tip_idx, axiss)
    return np.asarray(control_points), np.asarray(control_points_axis)

def axis_average(axis_1, axis_2):
    axis = axis_1 + axis_2
    return axis/np.linalg.norm(axis, axis=-1)

def padding_control_points(control_points, control_points_axis, joints, tip_idx, axiss):
    '''

    :param control_points: [24,3]
    :param control_points_axis: [24, 3, 3]
    :return:
    '''
    joints_axis = []
    for i in range(axiss[0].shape[0]):
        twist_axis = axiss[0][i]
        splay_axis = axiss[1][i]
        bend_axis = axiss[2][i]
        axis = np.asarray([twist_axis, splay_axis, bend_axis]).transpose()
        joints_axis.append(axis)

    # 24th tiger mouth
    point = (joints[2] + joints[5]) / 2
    axis = axis_average(control_points_axis[2], control_points_axis[6])
    point = point + axis[0:3, 0] * np.linalg.norm(joints[2] - joints[3])/2
    control_points.append(point)
    control_points_axis.append(axis)

    # 25-27th between finger
    for finger_id in range(1,4):
        root_joint_id = tip_idx[finger_id][1]
        mid_finger_joint_id = tip_idx[finger_id][2]
        next_root_joint_id = tip_idx[finger_id + 1][1]
        next_mid_finger_joint_id = tip_idx[finger_id + 1][2]
        point = (joints[root_joint_id] + joints[next_root_joint_id]) / 2
        axis = axis_average(joints_axis[root_joint_id], joints_axis[next_root_joint_id])
        offset = (np.linalg.norm(joints[mid_finger_joint_id] - joints[root_joint_id]) +
                  np.linalg.norm(joints[next_mid_finger_joint_id] - joints[next_root_joint_id]))/4
        point = point + axis[0:3, 0] * offset
        control_points.append(point)
        control_points_axis.append(axis)

    # 28-31th on the root joints
    for finger_id in range(1, 5):
        root_joint_id = tip_idx[finger_id][1]
        point = joints[root_joint_id]
        axis = joints_axis[root_joint_id]
        control_points.append(point)
        control_points_axis.append(axis)

    # 32-39
    for finger_id in range(1, 5):
        for id in range(2, 4):
            joint_id = tip_idx[finger_id][id]
            point = joints[joint_id]
            axis = joints_axis[joint_id]
            control_points.append(point)
            control_points_axis.append(axis)

def search_hand_intersection(start_point, direction, faces, vertices):
    intersection_points = []
    inf = {'face': [],
           'weight': []}
    for face in faces:
        tri = vertices[face]
        inter, tuv = ray_triangle_intersection(start_point, direction, vertices[face])
        if inter:
            t, u, v, = tuv
            a, b, c = (1 - u - v), u, v
            inter_point = tri[0] * a + tri[1] * b + tri[2] * c
            intersection_points.append(inter_point)
            inf['face'].append(face)
            inf['weight'].append([u, v])

    if intersection_points.__len__() > 0:
        intersection_points = np.asarray(intersection_points)
        dis = np.linalg.norm(intersection_points - start_point, axis=-1)
        choosen_id = dis.argmin()
        return inf['face'][choosen_id], inf['weight'][choosen_id]
    else:
        return None, None


def add_anchor_res(anchor_inf, face, weight, class_type):
    if face is not None:
        anchor_inf['face'].append(face)
        anchor_inf['weight'].append(weight)
        anchor_inf['class'].append(class_type)

def get_radius_ray(axis):
    rays = []
    splay = np.asarray([0.0, 1.0, 0.0])
    radius = 1.3
    rays.append([-axis[0:3, 1], 4])
    for theta in [np.pi * (-45/180), np.pi * (-135/180), np.pi * (90/180)]:
        ray = splay + radius * np.asarray([-np.sin(theta), 0.0, np.cos(theta)])
        ray = axis.dot(ray)
        rays.append([ray, 1])
        ray = -splay + radius * np.asarray([-np.sin(theta), 0.0, np.cos(theta)])
        ray = axis.dot(ray)
        rays.append([ray, 4])
    return rays

def search_anchor_ponint(control_points, control_points_axis, faces, vertices):
    anchor_inf = {'face': [], 'weight': [], 'class': []} # 0-inside 1-outside 2-undefine 4-tip
    for control_id in range(control_points.__len__()):
        point = control_points[control_id]
        axis = control_points_axis[control_id]
        twist = axis[0:3, 0]
        splay = axis[0:3, 1]
        bend = axis[0:3, 2]
        if control_id in [15]:
            continue
        elif control_id in [1]:
            f, w = search_hand_intersection(point, bend, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 2)
            f, w = search_hand_intersection(point, splay, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 1)
            f, w = search_hand_intersection(point, -splay, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 0)

        elif control_id == 4:
            # f, w = search_hand_intersection(point, bend, faces, vertices)
            # add_anchor_res(anchor_inf, f, w)
            f, w = search_hand_intersection(point, splay, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 1)
            f, w = search_hand_intersection(point, -splay, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 0)
        elif control_id in [19, 20]:
            f, w = search_hand_intersection(point, -bend, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 2)
            f, w = search_hand_intersection(point, splay, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 1)
            f, w = search_hand_intersection(point, -splay, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 0)
        elif control_id in [9, 14, 5, 10]:
            f, w = search_hand_intersection(point, splay, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 1)
            f, w = search_hand_intersection(point, -splay, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 0)
        elif control_id in [3, 8, 13, 18, 23]:
            rays = get_radius_ray(axis)
            for ray_inf in rays:
                ray, class_type = ray_inf
                f, w = search_hand_intersection(point, ray, faces, vertices)
                add_anchor_res(anchor_inf, f, w, class_type)
            #tip
            f, w = search_hand_intersection(point, -twist, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 4)
            f, w = search_hand_intersection(point, -bend, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 2)
            f, w = search_hand_intersection(point, bend, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 2)
            pass
        elif control_id in [24, 25, 26, 27]:
            f, w = search_hand_intersection(point, -twist, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 2)
            f, w = search_hand_intersection(point, splay, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 1)
            f, w = search_hand_intersection(point, -splay, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 0)
        elif control_id in [28, 29, 30 ,31]:
            f, w = search_hand_intersection(point, -splay, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 0)
            f, w = search_hand_intersection(point, splay, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 1)
            if control_id == 28:
                f, w = search_hand_intersection(point, bend, faces, vertices)
                add_anchor_res(anchor_inf, f, w, 2)
            if control_id == 31:
                f, w = search_hand_intersection(point, -bend, faces, vertices)
                add_anchor_res(anchor_inf, f, w, 2)
        else:
            f, w = search_hand_intersection(point, bend, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 2)
            f, w = search_hand_intersection(point, -bend, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 2)
            f, w = search_hand_intersection(point, splay, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 1)
            f, w = search_hand_intersection(point, -splay, faces, vertices)
            add_anchor_res(anchor_inf, f, w, 0)

    return anchor_inf



if __name__ == '__main__':
    mano_layer = ManoLayer(
        mano_root="../assets/mano",
        use_pca=False,
        ncomps=6,
        flat_hand_mean=False,
        center_idx=9,
        return_transf=True,
        root_rot_mode='quat',
        joint_rot_mode='quat',
        side='right'
    )
    data = np.load('../y.npy', allow_pickle=True).item()
    hand_pose = torch.from_numpy(data['hand_pose']).unsqueeze(0)
    hand_pose[0][0] = torch.tensor([1.0, 0, 0, 0])
    hand_shape = torch.from_numpy(data['hand_shape']).unsqueeze(0)
    faces = np.array(mano_layer.th_faces).astype(np.long)
    vec_pose = torch.from_numpy(np.asarray([[1.0, 0.0, 0.0, 0.0]] * 16, dtype=np.float32)).unsqueeze(0)
    vec_shape = torch.tensor([0.5082395, -0.39488167, -1.7484332, 1.6630946, 0.34428665, -1.37387,
                              0.38293332, 1.196094, 0.6538949, -0.94331187]).unsqueeze(0)
    vertices, joints, transf = mano_layer(vec_pose, vec_shape)
    axis_layer = AxisLayer()
    twist_axiss, splay_axiss, bend_axiss = axis_layer(joints, transf)
    joints_remapping = [0, 12, 13, 14, 14, 0, 1, 2, 2, 3, 4, 5, 5, 9, 10, 11, 11, 6, 7, 8, 8]
    # in order to align with the joints size
    twist_axiss = twist_axiss[0][joints_remapping].numpy()
    twist_axiss[0] = np.asarray([1.0, 0.0, 0.0])
    splay_axiss = splay_axiss[0][joints_remapping].numpy()
    splay_axiss[0] = np.asarray([0.0, 1.0, 0.0])
    bend_axiss = bend_axiss[0][joints_remapping].numpy()
    bend_axiss[0] = np.asarray([0.0, 0.0, 1.0])

    vertices = vertices.squeeze(0).numpy()
    joints = joints.squeeze(0).numpy()
    tip_idx = np.asarray([[0, 1, 2, 3, 4],
                          [0, 5, 6, 7, 8],
                          [0, 9, 10, 11, 12],
                          [0, 13, 14, 15, 16],
                          [0, 17, 18, 19, 20]])
    control_points, control_points_axis = find_control_points(tip_idx, joints, [twist_axiss, splay_axiss, bend_axiss])
    # search_hand_intersection(control_points[11], np.asarray([-1, 0, 0]), faces, vertices)
    anchor_inf = search_anchor_ponint(control_points, control_points_axis, faces, vertices)
    file_handle = open('1.txt', mode='w')
    for face in anchor_inf['face']:
        file_handle.write('{} {} {}\n'.format(face[0], face[1], face[2]))
    file_handle.close()
    file_handle = open('2.txt', mode='w')
    for weight in anchor_inf['weight']:
        file_handle.write('{} {}\n'.format(weight[0], weight[1]))
    file_handle.close()
    file_handle = open('3.txt', mode='w')
    for class_type in anchor_inf['class']:
        file_handle.write('{}\n'.format(class_type))
    file_handle.close()
