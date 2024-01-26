import copy

import numpy as np
from .util import *
from scipy.spatial.transform import rotation as R
def InterpolationQuat(q1, q2, t):
    if q1.dot(q2) < 0:
        q1 = -q1
    angle = np.arccos((q1.dot(q2))/(np.linalg.norm(q1) * np.linalg.norm(q2))) # * 180/np.pi
    if angle < 10e-5:
        q = t*q1 + (1-t)*q2
        q = q/np.linalg.norm(q)
    else:
        q = (np.sin((1-t) * angle) * q1 + np.sin((t) * angle) * q2)/np.sin(t * angle)
    return q

def FindNearestIdx(target_time, time_stramp):
    time_len = time_stramp.__len__()
    mid_idx = time_len//2
    left_idx = 0
    right_idx = time_len - 1
    if target_time > time_stramp[-1]:
        print("Error, the data length is not enough")
        return right_idx
    elif target_time < time_stramp[0]:
        return left_idx
    while True:
        center_idx = (left_idx + right_idx)//2
        if time_stramp[center_idx] > target_time:
            right_idx = center_idx - 1
        elif time_stramp[center_idx] <= target_time:
            if center_idx < time_len and time_stramp[center_idx + 1] > target_time:
                return center_idx
            left_idx = center_idx + 1

def quatWAvgMarkley(Q, weights):
    '''
    Averaging Quaternions.

    Arguments:
        Q(ndarray): an Mx4 ndarray of quaternions.
        weights(list): an M elements list, a weight for each quaternion.
    '''

    # Form the symmetric accumulator matrix
    A = np.zeros((4, 4))
    M = Q.shape[0]
    if weights is None:
        weights = np.ones(M)/M
    wSum = 0

    for i in range(M):
        q = Q[i, :]
        w_i = weights[i]
        A += w_i * (np.outer(q, q)) # rank 1 update
        wSum += w_i

    # scale
    A /= wSum

    # Get the eigenvector corresponding to largest eigen value
    return np.linalg.eigh(A)[1][:, -1]

def filter_quat_loc(quats, locs, window_length=5):
    data_len = quats.__len__()
    data_quat = np.asarray(quats)
    data_loc = np.asarray(locs)
    new_data_quat = copy.deepcopy(data_quat)
    new_data_loc = copy.deepcopy(data_loc)
    start_idx = 0
    half_window = window_length // 2
    for start_idx in range(half_window, data_len - half_window - 1):
    # while start_idx < (data_len - half_window - 1) and start_idx >= half_window:
        data_sample_quat = data_quat[(start_idx-half_window):(start_idx + half_window + 1), :] # wl * 16 * 4
        data_sample_loc = data_loc[(start_idx-half_window):(start_idx + half_window + 1), :]

        new_data_loc[start_idx] = np.mean(data_sample_loc, axis=0)
        for joint_id in range(data_sample_quat.shape[1]):
            filtered_quat = quatWAvgMarkley(data_sample_quat[:, joint_id, :],None)
            new_data_quat[start_idx, joint_id] = filtered_quat
        # start_idx += 1
    return new_data_quat, new_data_loc

def filter(data, window_length = 5):
    data_len = data.__len__()
    data = np.asarray(data)
    new_data = copy.deepcopy(data)
    start_idx = 0
    while start_idx < (data_len - window_length - 1):
        data_loc = data[start_idx:(start_idx + window_length), 0:3]
        data_euler = data[start_idx:(start_idx + window_length), 3:6]
        new_data[start_idx][0:3] = np.mean(data_loc, axis=0)
        quats = []
        for euler in data_euler:
            rotation = R.from_euler('xyz', [euler[2], euler[1], euler[0]], True)
            quat = rotation.as_quat()
            quats.append(quat)
        filtered_quat = quatWAvgMarkley(np.asarray(quats),None)
        filtered_euler = quat2euler(filtered_quat)
        new_data[start_idx][3:6] = [filtered_euler[2], filtered_euler[1], filtered_euler[0]]
        start_idx += 1
    return new_data.tolist()

def InterpolationViveEuler(data_euler, time_stramp, fps=30):
    interpolated_data_euler = []
    order = 'xyz'
    num_target_frame = int((time_stramp[-1] - time_stramp[0]) * fps) + 1
    time_interval = 1.0/fps
    for i in range(num_target_frame):
        target_time = i * time_interval
        left_idx =FindNearestIdx(target_time, time_stramp)
        if left_idx == 0 or left_idx == time_stramp.__len__()-1:
            interpolated_data_euler.append(data_euler[left_idx])
            continue
        right_idx = left_idx + 1
        t = (target_time - time_stramp[left_idx])/(time_stramp[right_idx] - time_stramp[left_idx])

        vive_euler = data_euler[left_idx]
        left_euler = [vive_euler[5], vive_euler[4], vive_euler[3]]
        rotation = R.from_euler(seq=order, angles=left_euler, degrees=True)
        left_quat = rotation.as_quat()

        vive_euler = data_euler[right_idx]
        right_euler = [vive_euler[5], vive_euler[4], vive_euler[3]]
        rotation = R.from_euler(seq=order, angles=right_euler, degrees=True)
        right_quat = rotation.as_quat()

        quat = InterpolationQuat(left_quat, right_quat, t)
        inter_rotation = R.from_quat(quat)
        euler = inter_rotation.as_euler(seq=order,degrees=True)

        loc = []
        for i in range(3):
            loc.append(data_euler[left_idx][i] * t + data_euler[right_idx][i] * (1-t))
        interploted_data = [loc[0], loc[1], loc[2], euler[2], euler[1], euler[0]]
        interpolated_data_euler.append(interploted_data)
    return filter(interpolated_data_euler)



if __name__ == '__main__':
    data = np.load('record_data.npy', allow_pickle=True)
    left_data_euler = data[0]
    right_data_euler = data[1]
    time_stramp = data[2]
    InterpolationViveEuler(right_data_euler, time_stramp)
